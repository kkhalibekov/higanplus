import os
from itertools import chain
from munch import Munch
import numpy as np
import torch
from torch.nn import CTCLoss, CrossEntropyLoss
import torch.distributed as dist
import torch.nn.functional as F
from networks.utils import (
    set_requires_grad,
    get_scheduler,
    idx_to_words,
    extract_all_patches,
)
from networks.BigGAN_networks import (
    Generator,
    Discriminator,
    PatchDiscriminator,
)
from networks.module import (
    Recognizer,
    WriterIdentifier,
    StyleEncoder,
    StyleBackbone,
)
from networks.rand_dist import prepare_z_dist, prepare_y_dist
from networks.loss import recn_l1_loss, CXLoss, KLloss
from networks.model_adversarial import AdversarialModel
from lib.utils import AverageMeterManager, plot_heatmap


class GlobalLocalAdversarialModel(AdversarialModel):
    def __init__(self, opt, log_root="./"):
        super().__init__(opt, log_root)

        device = self.device

        generator = Generator(**opt.GenModel).to(device)
        style_backbone = StyleBackbone(**opt.StyBackbone).to(device)
        style_encoder = StyleEncoder(**opt.EncModel).to(device)
        writer_identifier = WriterIdentifier(**opt.WidModel).to(device)
        discriminator = Discriminator(**opt.DiscModel).to(device)
        patch_discriminator = PatchDiscriminator(**opt.PatchDiscModel).to(
            device
        )
        recognizer = Recognizer(**opt.OcrModel).to(device)

        self.models = Munch(
            G=generator,
            D=discriminator,
            P=patch_discriminator,
            R=recognizer,
            E=style_encoder,
            W=writer_identifier,
            B=style_backbone,
        )

        self.ctc_loss = CTCLoss(zero_infinity=True, reduction="mean")
        self.classify_loss = CrossEntropyLoss()
        self.contextual_loss = CXLoss()

    def train(self):
        self.info()

        opt = self.opt
        self.z = prepare_z_dist(
            opt.training.batch_size,
            opt.EncModel.style_dim,
            self.device,
            seed=self.opt.seed,
        )
        self.y = prepare_y_dist(
            opt.training.batch_size,
            len(self.lexicon),
            self.device,
            seed=self.opt.seed,
        )

        self.eval_z = prepare_z_dist(
            opt.training.eval_batch_size,
            opt.EncModel.style_dim,
            self.device,
            seed=self.opt.seed,
        )
        self.eval_y = prepare_y_dist(
            opt.training.eval_batch_size,
            len(self.lexicon),
            self.device,
            seed=self.opt.seed,
        )

        self.optimizers = Munch(
            G=torch.optim.Adam(
                chain(self.models.G.parameters(), self.models.E.parameters()),
                lr=opt.training.lr,
                betas=(opt.training.adam_b1, opt.training.adam_b2),
            ),
            D=torch.optim.Adam(
                chain(self.models.D.parameters(), self.models.P.parameters()),
                lr=opt.training.lr,
                betas=(opt.training.adam_b1, opt.training.adam_b2),
            ),
        )

        self.lr_schedulers = Munch(
            G=get_scheduler(self.optimizers.G, opt.training),
            D=get_scheduler(self.optimizers.D, opt.training),
        )

        epoch_done = 1
        if os.path.exists(self.opt.training.pretrained_ckpt):
            epoch_done = self.load(
                self.opt.training.pretrained_ckpt, self.device
            )
            self.validate(style_guided=True)
        else:
            if os.path.exists(self.opt.training.pretrained_w):
                w_dict = torch.load(
                    self.opt.training.pretrained_w, self.device
                )
                self.models.W.load_state_dict(w_dict["WriterIdentifier"])
                self.models.B.load_state_dict(w_dict["StyleBackbone"])
                print(
                    "load pretrained writer_identifier: ",
                    self.opt.training.pretrained_w,
                )
                # self.validate_wid()
            if os.path.exists(self.opt.training.pretrained_r):
                r_dict = torch.load(self.opt.training.pretrained_r)[
                    "Recognizer"
                ]
                self.models.R.load_state_dict(r_dict, self.device)
                print(
                    "load pretrained recognizer: ",
                    self.opt.training.pretrained_r,
                )
                # self.validate_ocr()

        # multi-gpu
        if self.local_rank > -1:
            for key in self.models.keys():
                self.models[key] = torch.nn.parallel.DistributedDataParallel(
                    self.models[key],
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    broadcast_buffers=False,
                )

        self.averager_meters = AverageMeterManager(
            [
                "adv_loss",
                "fake_disc_loss",
                "real_disc_loss",
                "adv_loss_patch",
                "fake_disc_loss_patch",
                "real_disc_loss_patch",
                "recn_loss",
                "fake_ctc_loss",
                "info_loss",
                "fake_wid_loss",
                "ctx_loss",
                "kl_loss",
                "gp_ctc",
                "gp_info",
                "gp_wid",
                "gp_recn",
            ]
        )
        device = self.device

        if self.local_rank > -1:
            ctc_len_scale = self.models.R.module.len_scale
        else:
            ctc_len_scale = self.models.R.len_scale

        best_fid = np.inf
        iter_count = 0
        for epoch in range(epoch_done, self.opt.training.epochs):
            for i, batch in enumerate(self.train_loader):
                #############################
                # Prepare inputs & Network Forward
                #############################
                self.set_mode("train")
                real_imgs, real_img_lens, real_wids = (
                    batch["style_imgs"].to(device),
                    batch["style_img_lens"].to(device),
                    batch["wids"].to(device),
                )
                real_aug_imgs, real_aug_img_lens = batch["aug_imgs"].to(
                    device
                ), batch["aug_img_lens"].to(device)
                real_lbs, real_lb_lens = batch["lbs"].to(device), batch[
                    "lb_lens"
                ].to(device)
                max_label_len = real_lbs.size(-1)

                #############################
                # Optimizing Discriminator
                #############################
                self.optimizers.D.zero_grad()
                set_requires_grad(
                    [
                        self.models.G,
                        self.models.E,
                        self.models.R,
                        self.models.W,
                        self.models.B,
                    ],
                    False,
                )
                set_requires_grad([self.models.D, self.models.P], True)
                # self.models.B.frozen_bn()

                with torch.no_grad():
                    self.y.sample_()
                    sampled_words = idx_to_words(
                        self.y,
                        self.lexicon,
                        max_label_len,
                        self.opt.training.capitalize_ratio,
                        self.opt.training.blank_ratio,
                    )
                    fake_lbs, fake_lb_lens = self.label_converter.encode(
                        sampled_words, max_label_len
                    )
                    fake_lbs, fake_lb_lens = (
                        fake_lbs.to(device).detach(),
                        fake_lb_lens.to(device).detach(),
                    )

                    self.z.sample_()
                    fake_imgs = self.models.G(self.z, fake_lbs, fake_lb_lens)

                    if self.vae_mode:
                        enc_z, _, _ = self.models.E(
                            real_imgs,
                            real_img_lens,
                            self.models.B,
                            vae_mode=True,
                        )
                    else:
                        enc_z = self.models.E(
                            real_imgs,
                            real_img_lens,
                            self.models.B,
                            vae_mode=False,
                        )

                    style_imgs = self.models.G(enc_z, fake_lbs, fake_lb_lens)
                    recn_imgs = self.models.G(enc_z, real_lbs, real_lb_lens)

                    cat_fake_imgs = torch.cat(
                        [fake_imgs, style_imgs, recn_imgs], dim=0
                    )
                    cat_fake_lb_lens = torch.cat(
                        [fake_lb_lens, fake_lb_lens, real_lb_lens], dim=0
                    )
                    cat_fake_img_lens = cat_fake_lb_lens * self.opt.char_width

                # Compute discriminative loss for real & fake samples
                fake_disc = self.models.D(
                    cat_fake_imgs.detach(), cat_fake_img_lens, cat_fake_lb_lens
                )
                fake_disc_loss = torch.mean(F.relu(1.0 + fake_disc))

                fake_img_patches = extract_all_patches(
                    cat_fake_imgs, cat_fake_img_lens
                )
                fake_disc_patches = self.models.P(fake_img_patches.detach())
                fake_disc_loss_patch = torch.mean(
                    F.relu(1.0 + fake_disc_patches)
                )

                # real_imgs.requires_grad_()
                real_disc = self.models.D(
                    real_imgs, real_img_lens, real_lb_lens
                )
                real_disc_aug = self.models.D(
                    real_aug_imgs, real_aug_img_lens, real_lb_lens
                )
                real_disc_loss = (
                    torch.mean(F.relu(1.0 - real_disc))
                    + torch.mean(F.relu(1.0 - real_disc_aug))
                ) / 2

                real_img_patches = extract_all_patches(
                    real_imgs, real_img_lens, plot=False
                )
                real_aug_imgs_patches = extract_all_patches(
                    real_aug_imgs, real_aug_img_lens
                )
                real_disc_patches = self.models.P(
                    torch.cat(
                        [real_img_patches, real_aug_imgs_patches], dim=0
                    ).detach()
                )
                real_disc_loss_patch = torch.mean(
                    F.relu(1.0 - real_disc_patches)
                )

                disc_loss = (
                    real_disc_loss
                    + fake_disc_loss
                    + real_disc_loss_patch
                    + fake_disc_loss_patch
                )
                self.averager_meters.update(
                    "real_disc_loss", real_disc_loss.item()
                )
                self.averager_meters.update(
                    "fake_disc_loss", fake_disc_loss.item()
                )
                self.averager_meters.update(
                    "real_disc_loss_patch", real_disc_loss_patch.item()
                )
                self.averager_meters.update(
                    "fake_disc_loss_patch", fake_disc_loss_patch.item()
                )

                disc_loss.backward()
                self.optimizers.D.step()

                #############################
                # Optimizing Generator
                #############################
                if iter_count % self.opt.training.num_critic_train == 0:
                    self.optimizers.G.zero_grad()
                    set_requires_grad(
                        [
                            self.models.D,
                            self.models.P,
                            self.models.R,
                            self.models.W,
                            self.models.B,
                        ],
                        False,
                    )
                    set_requires_grad([self.models.G, self.models.E], True)
                    # self.models.B.frozen_bn()

                    ##########################
                    # Prepare Fake Inputs
                    ##########################
                    self.y.sample_()
                    sampled_words = idx_to_words(
                        self.y,
                        self.lexicon,
                        max_label_len,
                        self.opt.training.capitalize_ratio,
                        self.opt.training.blank_ratio,
                        sort=True,
                    )

                    fake_lbs, fake_lb_lens = self.label_converter.encode(
                        sampled_words, max_label_len
                    )
                    fake_lbs, fake_lb_lens = (
                        fake_lbs.to(device).detach(),
                        fake_lb_lens.to(device).detach(),
                    )

                    self.z.sample_()
                    fake_imgs = self.models.G(self.z, fake_lbs, fake_lb_lens)

                    if self.vae_mode:
                        (enc_z, mu, logvar), real_img_feats = self.models.E(
                            real_imgs,
                            real_img_lens,
                            self.models.B,
                            ret_feats=True,
                            vae_mode=True,
                        )
                    else:
                        enc_z, real_img_feats = self.models.E(
                            real_imgs,
                            real_img_lens,
                            self.models.B,
                            ret_feats=True,
                            vae_mode=False,
                        )
                    style_imgs = self.models.G(enc_z, fake_lbs, fake_lb_lens)
                    recn_imgs = self.models.G(enc_z, real_lbs, real_lb_lens)

                    ###################################################
                    # Calculating G Losses
                    ####################################################
                    # deal with fake samples
                    # Compute Adversarial loss
                    cat_fake_imgs = torch.cat(
                        [fake_imgs, style_imgs, recn_imgs], dim=0
                    )
                    cat_fake_lb_lens = torch.cat(
                        [fake_lb_lens, fake_lb_lens, real_lb_lens], dim=0
                    )
                    cat_fake_disc = self.models.D(
                        cat_fake_imgs,
                        cat_fake_lb_lens * self.opt.char_width,
                        cat_fake_lb_lens,
                    )
                    adv_loss = -torch.mean(cat_fake_disc)

                    fake_img_patches = extract_all_patches(
                        cat_fake_imgs, cat_fake_lb_lens * self.opt.char_width
                    )
                    fake_disc_patches = self.models.P(fake_img_patches)
                    adv_loss_patch = -torch.mean(fake_disc_patches)

                    # CTC Auxiliary loss
                    # self.models.R.frozen_bn()
                    fake_img_lens = fake_lb_lens * self.opt.char_width
                    fake_ctc_rand = self.models.R(fake_imgs, fake_img_lens)
                    fake_ctc_loss_rand = self.ctc_loss(
                        fake_ctc_rand,
                        fake_lbs,
                        fake_img_lens // ctc_len_scale,
                        fake_lb_lens,
                    )

                    style_img_lens = fake_lb_lens * self.opt.char_width
                    fake_ctc_style = self.models.R(style_imgs, style_img_lens)
                    fake_ctc_loss_style = self.ctc_loss(
                        fake_ctc_style,
                        fake_lbs,
                        style_img_lens // ctc_len_scale,
                        fake_lb_lens,
                    )

                    recn_img_lens = real_lb_lens * self.opt.char_width
                    fake_ctc_recn = self.models.R(recn_imgs, recn_img_lens)
                    fake_ctc_loss_recn = self.ctc_loss(
                        fake_ctc_recn,
                        real_lbs,
                        recn_img_lens // ctc_len_scale,
                        real_lb_lens,
                    )

                    fake_ctc_loss = (
                        fake_ctc_loss_rand
                        + fake_ctc_loss_recn
                        + fake_ctc_loss_style
                    )

                    # Style Reconstruction
                    styles = self.models.E(
                        fake_imgs,
                        fake_lb_lens * self.opt.char_width,
                        self.models.B,
                    )
                    info_loss = torch.mean(torch.abs(styles - self.z.detach()))

                    # Content Restruction
                    recn_loss = recn_l1_loss(
                        recn_imgs, real_imgs, real_img_lens
                    )

                    # Writer Identify Loss
                    cat_style_imgs = torch.cat([style_imgs, recn_imgs], dim=0)
                    cat_style_img_lens = (
                        torch.cat([fake_lb_lens, real_lb_lens], dim=0)
                        * self.opt.char_width
                    )
                    recn_wid_logits, fake_imgs_feats = self.models.W(
                        cat_style_imgs,
                        cat_style_img_lens,
                        self.models.B,
                        ret_feats=True,
                    )
                    fake_wid_loss = self.classify_loss(
                        recn_wid_logits, real_wids.repeat(2)
                    )

                    #  Contextual Loss and Gram Loss for non-aligned data
                    ctx_loss = torch.FloatTensor([0.0]).to(self.device)
                    for real_img_feat, fake_img_feat in zip(
                        real_img_feats, fake_imgs_feats
                    ):
                        fake_feat = fake_img_feat.chunk(2, dim=0)
                        # ctx_loss for style_imgs
                        ctx_loss += self.contextual_loss(
                            real_img_feat, fake_feat[0]
                        )
                        # ctx_loss for recn_imgs
                        ctx_loss += self.contextual_loss(
                            real_img_feat, fake_feat[1]
                        )

                    # KL-Divergency loss
                    kl_loss = (
                        KLloss(mu, logvar)
                        if self.vae_mode
                        else torch.FloatTensor([0.0]).to(self.device)
                    )

                    grad_fake_adv = torch.autograd.grad(
                        adv_loss,
                        cat_fake_imgs,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    grad_fake_OCR = torch.autograd.grad(
                        fake_ctc_loss_rand,
                        fake_ctc_rand,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    grad_fake_info = torch.autograd.grad(
                        info_loss,
                        fake_imgs,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    grad_fake_wid = torch.autograd.grad(
                        fake_wid_loss,
                        recn_wid_logits,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    grad_fake_recn = torch.autograd.grad(
                        recn_loss, enc_z, create_graph=True, retain_graph=True
                    )[0]

                    std_grad_adv = torch.std(grad_fake_adv)
                    gp_ctc = (
                        torch.div(
                            std_grad_adv, torch.std(grad_fake_OCR) + 1e-8
                        ).detach()
                        + 1
                    )
                    gp_ctc.clamp_max_(100)
                    gp_info = (
                        torch.div(
                            std_grad_adv, torch.std(grad_fake_info) + 1e-8
                        ).detach()
                        + 1
                    )
                    gp_wid = (
                        torch.div(
                            std_grad_adv, torch.std(grad_fake_wid) + 1e-8
                        ).detach()
                        + 1
                    )
                    gp_wid.clamp_max_(10)
                    gp_recn = (
                        torch.div(
                            std_grad_adv, torch.std(grad_fake_recn) + 1e-8
                        ).detach()
                        + 1
                    )

                    self.averager_meters.update("gp_ctc", gp_ctc.item())
                    self.averager_meters.update("gp_info", gp_info.item())
                    self.averager_meters.update("gp_wid", gp_wid.item())
                    self.averager_meters.update("gp_recn", gp_recn.item())

                    g_loss = (
                        adv_loss
                        + adv_loss_patch
                        + gp_ctc * fake_ctc_loss
                        + gp_info * info_loss
                        + gp_wid * fake_wid_loss
                        + gp_recn * recn_loss
                        + self.opt.training.lambda_ctx * ctx_loss
                        + self.opt.training.lambda_kl * kl_loss
                    )
                    g_loss.backward()
                    self.averager_meters.update("adv_loss", adv_loss.item())
                    self.averager_meters.update(
                        "adv_loss_patch", adv_loss_patch.item()
                    )
                    self.averager_meters.update(
                        "fake_ctc_loss", fake_ctc_loss.item()
                    )
                    self.averager_meters.update("info_loss", info_loss.item())
                    self.averager_meters.update(
                        "fake_wid_loss", fake_wid_loss.item()
                    )
                    self.averager_meters.update("recn_loss", recn_loss.item())
                    self.averager_meters.update("ctx_loss", ctx_loss.item())
                    self.averager_meters.update("kl_loss", kl_loss.item())
                    self.optimizers.G.step()

                if iter_count % self.opt.training.print_iter_val == 0:
                    meter_vals = self.averager_meters.eval_all()
                    self.averager_meters.reset_all()
                    info = (
                        "[%3d|%3d]-[%4d|%4d] G:%.4f G-p:%.4f D-fake:%.4f D-real:%.4f "
                        "D-fake-p:%.4f D-real-p:%.4f CTC-fake:%.4f Wid-fake:%.4f "
                        "Recn-z:%.4f Recn-c:%.4f Ctx:%.4f Kl:%.4f"
                        % (
                            epoch,
                            self.opt.training.epochs,
                            iter_count % len(self.train_loader),
                            len(self.train_loader),
                            meter_vals["adv_loss"],
                            meter_vals["adv_loss_patch"],
                            meter_vals["fake_disc_loss"],
                            meter_vals["real_disc_loss"],
                            meter_vals["fake_disc_loss_patch"],
                            meter_vals["real_disc_loss_patch"],
                            meter_vals["fake_ctc_loss"],
                            meter_vals["fake_wid_loss"],
                            meter_vals["info_loss"],
                            meter_vals["recn_loss"],
                            meter_vals["ctx_loss"],
                            meter_vals["kl_loss"],
                        )
                    )
                    if self.local_rank < 1:
                        self.print(info)

                    if self.writer:
                        for key, val in meter_vals.items():
                            if self.local_rank < 1:
                                self.writer.add_scalar(
                                    f"loss/{key}", val, iter_count + 1
                                )
                        try:
                            lr = self.lr_schedulers.G.get_last_lr()[0]
                        except Exception:
                            lr = self.lr_schedulers.G.get_lr()[0]
                        if self.local_rank < 1:
                            self.writer.add_scalar(
                                "loss/lr", lr, iter_count + 1
                            )

                        info_attns = self.models.G._info_attention()
                        for i_, info in enumerate(info_attns):
                            if self.local_rank < 1:
                                self.writer.add_scalar(
                                    f"loss/gamma{i_}",
                                    info["gamma"],
                                    iter_count + 1,
                                )
                            heatmap = plot_heatmap(info["out"])
                            self.writer.add_image(
                                f"attention/{i_}",
                                heatmap.transpose((2, 0, 1)),
                            )

                if (iter_count + 1) % self.opt.training.sample_iter_val == 0:
                    if not (self.logger and self.writer):
                        self.create_logger() if self.local_rank < 1 else None

                    sample_root = os.path.join(
                        self.log_root, self.opt.training.sample_dir
                    )
                    if not os.path.exists(sample_root):
                        if self.local_rank < 1:
                            os.makedirs(sample_root)
                    if self.local_rank < 1:
                        self.sample_images(iter_count + 1)

                iter_count += 1

            if epoch:
                ckpt_root = os.path.join(
                    self.log_root, self.opt.training.ckpt_dir
                )
                if not os.path.exists(ckpt_root):
                    os.makedirs(ckpt_root)

                self.save("last", epoch)
                if (
                    epoch >= self.opt.training.start_save_epoch_val
                    and epoch % self.opt.training.save_epoch_val == 0
                ):
                    if self.local_rank < 1:
                        self.print("Calculate FID_KID")
                    scores = self.validate()

                    if "fid" in scores and scores["fid"] < best_fid:
                        best_fid = scores["fid"]
                        if self.local_rank < 1:
                            self.save("best", epoch, **scores)

                    if self.writer:
                        for key, val in scores.items():
                            if self.local_rank < 1:
                                self.writer.add_scalar(
                                    f"valid/{key}", val, epoch
                                )

                if self.local_rank > -1:
                    dist.barrier()

            for scheduler in self.lr_schedulers.values():
                scheduler.step(epoch)
