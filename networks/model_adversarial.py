import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from distance import levenshtein
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from metric.fid_kid_is import calculate_fid_kid_is
from metric.mssim_psnr import calculate_mssim_psnr
from networks.utils import (
    idx_to_words,
    rescale_images,
    rescale_images2,
    words_to_images,
    ctc_greedy_decoder,
)
from networks.module import (
    Recognizer,
    WriterIdentifier,
    StyleBackbone,
)
from networks.rand_dist import prepare_z_dist, prepare_y_dist
from networks.model_base import BaseModel
from lib.datasets import get_dataset, get_collect_fn, Hdf5Dataset
from lib.alphabet import get_lexicon, get_true_alphabet
from lib.utils import draw_image


class AdversarialModel(BaseModel):
    def __init__(self, opt, log_root="./"):
        super().__init__(opt, log_root)

        self.lexicon = get_lexicon(
            self.opt.training.lexicon,
            get_true_alphabet(opt.dataset),
            max_length=self.opt.training.max_word_len,
        )
        self.max_valid_image_width = (
            self.opt.char_width * self.opt.training.max_word_len
        )
        self.vae_mode = self.opt.training.vae_mode
        self.collect_fn = get_collect_fn(
            self.opt.training.sort_input, sort_style=True
        )
        self.train_loader = DataLoader(
            get_dataset(
                opt.dataset,
                opt.training.dset_split,
                recogn_aug=True,
                wid_aug=True,
                process_style=True,
            ),
            batch_size=opt.training.batch_size,
            shuffle=True,
            collate_fn=self.collect_fn,
            num_workers=4,
            drop_last=True,
        )

        self.tst_loader = DataLoader(
            get_dataset(
                opt.dataset,
                opt.valid.dset_split,
                recogn_aug=False,
                wid_aug=False,
                process_style=True,
            ),
            batch_size=opt.training.eval_batch_size // 2,
            shuffle=True,
            collate_fn=self.collect_fn,
        )

        self.tst_loader2 = DataLoader(
            get_dataset(
                opt.dataset,
                opt.training.dset_split,
                recogn_aug=False,
                wid_aug=False,
                process_style=True,
            ),
            batch_size=opt.training.eval_batch_size // 2,
            shuffle=True,
            collate_fn=self.collect_fn,
        )

        self.models = None

    def train(self):
        raise NotImplementedError()

    def sample_images(self, iteration_done=0):
        self.set_mode("eval")

        device = self.device
        batchA = next(iter(self.tst_loader))
        batchB = next(iter(self.tst_loader2))
        batch = Hdf5Dataset.merge_batch(batchA, batchB, device)

        real_imgs, real_img_lens = batch["style_imgs"].to(device), batch[
            "style_img_lens"
        ].to(device)
        real_lbs, real_lb_lens = batch["lbs"].to(device), batch["lb_lens"].to(
            device
        )

        with torch.no_grad():
            self.eval_z.sample_()
            recn_imgs = None
            if "E" in self.models:
                enc_z = self.models.E(real_imgs, real_img_lens, self.models.B)
                recn_imgs = self.models.G(enc_z, real_lbs, real_lb_lens)

            fake_real_imgs = self.models.G(self.eval_z, real_lbs, real_lb_lens)

            self.eval_y.sample_()
            sampled_words = idx_to_words(
                self.eval_y,
                self.lexicon,
                self.opt.training.capitalize_ratio,
                self.opt.training.blank_ratio,
            )
            sampled_words[-2] = sampled_words[-1]
            fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words)
            fake_lbs, fake_lb_lens = fake_lbs.to(device), fake_lb_lens.to(
                device
            )
            fake_imgs = self.models.G(self.eval_z, fake_lbs, fake_lb_lens)
            style_imgs = self.models.G(enc_z, fake_lbs, fake_lb_lens)

            max_img_len = max(
                [
                    real_imgs.size(-1),
                    fake_real_imgs.size(-1),
                    fake_imgs.size(-1),
                ]
            )
            img_shape = [real_imgs.size(2), max_img_len, real_imgs.size(1)]

            real_imgs = F.pad(
                real_imgs,
                [0, max_img_len - real_imgs.size(-1), 0, 0],
                value=-1.0,
            )
            fake_real_imgs = F.pad(
                fake_real_imgs,
                [0, max_img_len - fake_real_imgs.size(-1), 0, 0],
                value=-1.0,
            )
            fake_imgs = F.pad(
                fake_imgs,
                [0, max_img_len - fake_imgs.size(-1), 0, 0],
                value=-1.0,
            )
            recn_imgs = (
                F.pad(
                    recn_imgs,
                    [0, max_img_len - recn_imgs.size(-1), 0, 0],
                    value=-1.0,
                )
                if recn_imgs is not None
                else None
            )
            style_imgs = F.pad(
                style_imgs,
                [0, max_img_len - recn_imgs.size(-1), 0, 0],
                value=-1.0,
            )

            real_words = self.label_converter.decode(real_lbs, real_lb_lens)
            real_labels = words_to_images(real_words, *img_shape)
            rand_labels = words_to_images(sampled_words, *img_shape)

            try:
                sample_img_list = [
                    real_labels.cpu(),
                    real_imgs.cpu(),
                    fake_real_imgs.cpu(),
                    fake_imgs.cpu(),
                    style_imgs.cpu(),
                    rand_labels.cpu(),
                ]
                if recn_imgs is not None:
                    sample_img_list.insert(2, recn_imgs.cpu())
                sample_imgs = torch.cat(sample_img_list, dim=2).repeat(
                    1, 3, 1, 1
                )
                res_img = draw_image(
                    1 - sample_imgs.data,
                    nrow=self.opt.training.sample_nrow,
                    normalize=True,
                )
                save_path = os.path.join(
                    self.log_root,
                    self.opt.training.sample_dir,
                    f"iter_{iteration_done}.png",
                )
                im = Image.fromarray(res_img)
                im.save(save_path)
                if self.writer:
                    self.writer.add_image(
                        "Image", res_img.transpose((2, 0, 1)), iteration_done
                    )
            except RuntimeError as e:
                print(e)

    def image_generator(
        self,
        style_dloader,
        use_rand_corpus=False,
        style_guided=True,
        n_repeats=1,
    ):
        device = self.device
        word_idx_sampler = None
        if use_rand_corpus:
            word_idx_sampler = prepare_y_dist(
                style_dloader.batch_size,
                len(self.lexicon),
                self.device,
                seed=self.opt.seed,
            )

        if style_guided and not use_rand_corpus:
            n_repeats = 1

        with torch.no_grad():
            for _ in range(n_repeats):
                for batch in style_dloader:
                    fake_batch = {}
                    style_imgs, style_img_lens = batch["style_imgs"].to(
                        device
                    ), batch["style_img_lens"].to(device)
                    style_lbs, style_lb_lens = batch["lbs"].to(device), batch[
                        "lb_lens"
                    ].to(device)
                    if use_rand_corpus:
                        word_idx_sampler.sample_()
                        sampled_words = idx_to_words(
                            word_idx_sampler[: style_imgs.size(0)],
                            self.lexicon,
                            self.opt.training.capitalize_ratio,
                            blank_ratio=0,
                        )
                        content_lbs, content_lb_lens = (
                            self.label_converter.encode(sampled_words)
                        )
                    else:
                        content_lbs, content_lb_lens = style_lbs, style_lb_lens

                    fake_batch["lbs"], fake_batch["lb_lens"] = content_lbs.to(
                        device
                    ), content_lb_lens.to(device)

                    if style_guided:
                        enc_z = self.models.E(
                            style_imgs.to(device),
                            style_img_lens.to(device),
                            self.models.B,
                        )
                    else:
                        enc_z = torch.randn(
                            style_lb_lens.size(0), self.models.G.style_dim
                        ).to(device)

                    fake_batch["style_imgs"] = self.models.G(
                        enc_z, content_lbs, content_lb_lens
                    )
                    fake_batch["style_img_lens"] = (
                        fake_batch["lb_lens"] * self.opt.char_width
                    )
                    fake_batch["wids"] = batch["wids"]

                    fake_batch["org_imgs"], fake_batch["org_img_lens"] = (
                        rescale_images(
                            fake_batch["style_imgs"],
                            fake_batch["style_img_lens"],
                            batch["org_img_lens"],
                        )
                    )

                    yield fake_batch

    def validate(self, style_guided=True, test_stage=False, *args, **kwargs):
        self.set_mode("eval")
        # style images are resized
        eval_dloader = DataLoader(
            get_dataset(
                self.opt.valid.dset_name,
                self.opt.valid.dset_split,
                process_style=True,
            ),
            collate_fn=self.collect_fn,
            batch_size=self.opt.valid.batch_size,
            shuffle=False,
            num_workers=4,
        )

        if "E" not in self.models:
            style_guided = False
            n_rand_repeat = 1
        else:
            n_rand_repeat = (
                1
                if style_guided and not self.opt.valid.use_rand_corpus
                else self.opt.valid.n_rand_repeat
            )

        def get_generator():
            generator = self.image_generator(
                eval_dloader,
                self.opt.valid.use_rand_corpus,
                style_guided,
                n_rand_repeat,
            )
            return generator

        if test_stage:
            res = calculate_fid_kid_is(
                self.opt.valid,
                eval_dloader,
                get_generator(),
                n_rand_repeat,
                self.device,
            )
        else:
            res = calculate_fid_kid_is(
                self.opt.valid,
                eval_dloader,
                get_generator(),
                n_rand_repeat,
                self.device,
                crop=True,
            )

        if test_stage:
            if not self.opt.valid.use_rand_corpus:
                psnr_mssim = calculate_mssim_psnr(
                    eval_dloader, get_generator()
                )
                res["psnr"] = psnr_mssim["psnr"]
                res["mssim"] = psnr_mssim["mssim"]
            res["cer"], res["wer"] = self.validate_ocr(
                get_generator(), n_iters=len(eval_dloader) * n_rand_repeat
            )
            if style_guided:
                wier = self.validate_wid(
                    get_generator(),
                    real_dloader=eval_dloader,
                    split=self.opt.valid.dset_split,
                )
                res["wier"] = wier

        return res

    def validate_ocr(self, dloader, n_iters):
        self.set_mode("eval")
        recognizer = Recognizer(**self.opt.OcrModel).to(self.device)
        r_dict = torch.load(self.opt.training.pretrained_r)["Recognizer"]
        recognizer.load_state_dict(r_dict, self.device)
        recognizer.eval()
        print("load pretrained recognizer: ", self.opt.training.pretrained_r)
        ctc_len_scale = self.models.R.len_scale
        char_trans = 0
        total_chars = 0
        word_trans = 0
        total_words = 0

        with torch.no_grad():
            for i, batch in tqdm(enumerate(dloader), total=n_iters):
                real_imgs, real_img_lens = batch["style_imgs"].to(
                    self.device
                ), batch["style_img_lens"].to(self.device)
                logits = recognizer(real_imgs, real_img_lens)
                logits = torch.nn.functional.softmax(logits, dim=2).detach()

                logits = logits.cpu().numpy()
                word_preds = []
                for logit, img_len in zip(
                    logits, batch["style_img_lens"].cpu().numpy()
                ):
                    label = ctc_greedy_decoder(
                        logit[: img_len // ctc_len_scale]
                    )
                    word_preds.append(self.label_converter.decode(label))
                word_reals = self.label_converter.decode(
                    batch["lbs"], batch["lb_lens"]
                )
                for word_pred, word_real in zip(word_preds, word_reals):
                    char_tran = levenshtein(word_pred, word_real)
                    char_trans += char_tran
                    total_chars += len(word_real)
                    total_words += 1
                    if char_tran > 0:
                        word_trans += 1

        for model in self.models.values():
            model.train()
        cer = char_trans * 1.0 / total_chars
        wer = word_trans * 1.0 / total_words
        print(f"CER:{cer:.4f}  WER:{wer:.4f}")
        return cer, wer

    def validate_wid(self, generator, real_dloader, split="test"):
        if split == "test":
            assert os.path.exists(self.opt.valid.pretrained_test_w)
            w_dict = torch.load(self.opt.valid.pretrained_test_w, self.device)
            test_writer = WriterIdentifier(**self.opt.valid.test_wid_model).to(
                self.device
            )
            test_writer.load_state_dict(w_dict["WriterIdentifier"])
            test_writer_backbone = StyleBackbone(**self.opt.StyBackbone).to(
                self.device
            )
            test_writer_backbone.load_state_dict(w_dict["StyleBackbone"])
            writer_identifier = test_writer
            writer_backbone = test_writer_backbone
            print(
                "load pretrained test_writer_identifier: ",
                self.opt.valid.pretrained_test_w,
            )
        else:
            writer_identifier = WriterIdentifier(**self.opt.WidModel).to(
                self.device
            )
            writer_backbone = StyleBackbone(**self.opt.StyBackbone).to(
                self.device
            )
            print(
                "load pretrained writer_identifier: ",
                self.opt.training.pretrained_w,
            )
            w_dict = torch.load(self.opt.training.pretrained_w, self.device)
            writer_identifier.load_state_dict(w_dict["WriterIdentifier"])
            writer_backbone.load_state_dict(w_dict["StyleBackbone"])

        writer_identifier.eval(), writer_backbone.eval()
        with torch.no_grad():
            n_iters = len(real_dloader)

            acc_counts = 0.0
            total_counts = 0.0
            for i, (batch_real, batch_fake) in tqdm(
                enumerate(zip(real_dloader, generator)), total=n_iters
            ):
                # predicting pesudo labels
                real_wid_logits = writer_identifier(
                    batch_real["style_imgs"].to(self.device),
                    batch_real["style_img_lens"].to(self.device),
                    writer_backbone,
                )
                _, real_preds = torch.max(real_wid_logits.data, dim=1)

                # predicting pesudo labels
                fake_wid_logits = writer_identifier(
                    batch_fake["style_imgs"].to(self.device),
                    batch_fake["style_img_lens"].to(self.device),
                    writer_backbone,
                )
                _, fake_preds = torch.max(fake_wid_logits.data, dim=1)
                acc_counts += (
                    real_preds.eq(fake_preds.to(self.device)).sum().item()
                )
                total_counts += real_preds.size(0)

            wier = 1 - acc_counts * 1.0 / total_counts

        for model in self.models.values():
            model.train()
        print(f"WID_wier:{wier:.2f}")
        return wier

    def plot_interp(
        self, nrow: int, ncol: int, gen_imgs: np.ndarray, text: str
    ):
        plt.figure()
        for i in range(nrow * ncol):
            plt.subplot(nrow, ncol, i + 1)
            plt.imshow(gen_imgs[i], cmap="gray")
            plt.axis("off")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"interp_{text}.png")

    def plot_style(
        self,
        nrow: int,
        ncol: int,
        gen_imgs: np.ndarray,
        real_imgs: np.ndarray,
        text: str,
    ):
        plt.figure()
        for i in range(nrow):
            plt.subplot(nrow, 1 + ncol, i * (1 + ncol) + 1)
            # plt.imshow(real_imgs[i, :, :real_img_lens[i]], cmap='gray')
            plt.imshow(real_imgs[i], cmap="gray")
            plt.axis("off")
            for j in range(ncol):
                plt.subplot(nrow, 1 + ncol, i * (1 + ncol) + 2 + j)
                # plt.imshow(gen_imgs[i * ncol + j, :, :gen_img_lens[i * ncol + j]], cmap='gray')
                plt.imshow(gen_imgs[i * ncol + j], cmap="gray")
                plt.axis("off")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"style_{text}.png")

    def plot_rand(self, nrow: int, ncol: int, gen_imgs: np.ndarray, text: str):
        plt.figure()
        for i in range(nrow):
            for j in range(ncol):
                ax = plt.subplot(nrow, ncol, i * ncol + 1 + j)
                gen_img = gen_imgs[i * ncol + j]
                ax.imshow(gen_img, cmap="gray")
                ax.axis("off")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"rand_{text}.png")

    def plot_text(
        self,
        nrow: int,
        gen_imgs: np.ndarray,
        real_imgs: np.ndarray,
        gen_img_lens,
        real_img_lens,
        text: str,
    ):
        plt.figure()

        for i in range(nrow):
            plt.subplot(nrow * 2, 1, i * 2 + 1)
            plt.imshow(real_imgs[i, :, : real_img_lens[i]], cmap="gray")
            plt.axis("off")
            plt.subplot(nrow * 2, 1, i * 2 + 2)
            plt.imshow(gen_imgs[i, :, : gen_img_lens[i]], cmap="gray")
            plt.axis("off")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"text_{text}.png")

    def eval_interp(self, text: str):
        """Style interpolation"""
        if not text:
            return

        self.set_mode("eval")

        with torch.no_grad():
            interp_num = self.opt.test.interp_num
            nrow, ncol = 1, interp_num

            # while True:

            fake_lbs = self.label_converter.encode(text)
            fake_lbs = torch.LongTensor(fake_lbs)
            fake_lb_lens = torch.IntTensor([len(text)])

            # style0 = torch.zeros((1, self.opt.GenModel.style_dim)) + 1e-1
            # style1 = torch.ones_like(style0) - 1e-1
            style0 = torch.randn((1, self.opt.EncModel.style_dim))
            style1 = torch.randn(style0.size())

            styles = [
                torch.lerp(style0, style1, i / (interp_num - 1))
                for i in range(interp_num)
            ]
            styles = torch.cat(styles, dim=0).float().to(self.device)

            fake_lbs, fake_lb_lens = fake_lbs.repeat(nrow * ncol, 1).to(
                self.device
            ), fake_lb_lens.repeat(nrow * ncol).to(self.device)
            gen_imgs = self.models.G(styles, fake_lbs, fake_lb_lens)
            gen_imgs = (1 - gen_imgs).squeeze().cpu().numpy() * 127

            self.plot_interp(nrow, ncol, gen_imgs, text)

    def eval_style(self, text: str):
        """Reference-guided synthesis"""
        if not text:
            return

        self.set_mode("eval")

        tst_loader = DataLoader(
            get_dataset(
                self.opt.dataset,
                self.opt.training.dset_split,
                process_style=True,
            ),
            batch_size=self.opt.test.nrow,
            shuffle=True,
            collate_fn=self.collect_fn,
            drop_last=False,
        )

        with torch.no_grad():
            # while True:

            texts = text.split(" ")
            ncol = len(texts)
            batch = next(iter(tst_loader))
            imgs, img_lens, lbs, lb_lens = (
                batch["style_imgs"],
                batch["style_img_lens"],
                batch["lbs"],
                batch["lb_lens"],
            )
            real_imgs, real_img_lens = imgs.to(self.device), img_lens.to(
                self.device
            )
            if len(texts) == 1:
                fake_lbs = self.label_converter.encode(texts)
                fake_lbs = torch.LongTensor(fake_lbs)
                fake_lb_lens = torch.IntTensor([len(texts[0])])
            else:
                fake_lbs, fake_lb_lens = self.label_converter.encode(texts)

            nrow = batch["style_imgs"].size(0)
            fake_lbs = fake_lbs.repeat(nrow, 1).to(self.device)
            fake_lb_lens = fake_lb_lens.repeat(
                nrow,
            ).to(self.device)
            enc_styles = (
                self.models.E(real_imgs, real_img_lens, self.models.B)
                .unsqueeze(1)
                .repeat(1, ncol, 1)
                .view(nrow * ncol, self.opt.EncModel.style_dim)
            )

            gen_imgs = self.models.G(enc_styles, fake_lbs, fake_lb_lens)
            gen_imgs, gen_img_lens = rescale_images2(
                gen_imgs,
                fake_lb_lens * self.opt.char_width,
                fake_lb_lens,
                batch["org_img_lens"].repeat_interleave(ncol).to(self.device),
                batch["lb_lens"].repeat_interleave(ncol).to(self.device),
            )
            gen_imgs = (1 - gen_imgs).squeeze().cpu().numpy() * 127
            real_imgs = torch.nn.functional.pad(
                batch["org_imgs"],
                [0, gen_imgs.shape[-1] - batch["org_imgs"].size(-1), 0, 0],
                mode="constant",
                value=-1,
            )
            real_imgs = (1 - real_imgs).squeeze().cpu().numpy() * 127

            self.plot_style(nrow, ncol, gen_imgs, real_imgs, text)

    def eval_rand(self, text: str):
        """Latent-guided synthesis"""

        if not text:
            return

        self.set_mode("eval")

        with torch.no_grad():
            nrow, ncol = self.opt.test.nrow, 2
            rand_z = prepare_z_dist(
                nrow, self.opt.EncModel.style_dim, self.device
            )

            # while True:

            texts = text.split(" ")
            ncol = len(texts)
            if len(texts) == 1:
                fake_lbs = self.label_converter.encode(texts)
                fake_lbs = torch.LongTensor(fake_lbs)
                fake_lb_lens = torch.IntTensor([len(texts[0])])
            else:
                fake_lbs, fake_lb_lens = self.label_converter.encode(texts)

            fake_lbs = fake_lbs.repeat(nrow, 1).to(self.device)
            fake_lb_lens = fake_lb_lens.repeat(
                nrow,
            ).to(self.device)

            rand_z.sample_()
            rand_styles = (
                rand_z.unsqueeze(1)
                .repeat(1, ncol, 1)
                .view(nrow * ncol, self.opt.GenModel.style_dim)
            )
            gen_imgs = self.models.G(rand_styles, fake_lbs, fake_lb_lens)
            gen_imgs = (1 - gen_imgs).squeeze().cpu().numpy() * 127

            self.plot_rand(nrow, ncol, gen_imgs, text)

    def eval_text(self, text: str):
        """Text synthesis"""
        if not text:
            return

        self.set_mode("eval")

        tst_loader = DataLoader(
            get_dataset(
                self.opt.dataset,
                self.opt.training.dset_split,
                process_style=True,
            ),
            batch_size=self.opt.test.nrow,
            shuffle=True,
            collate_fn=self.collect_fn,
            drop_last=False,
        )

        with torch.no_grad():
            # while True:

            batch = next(iter(tst_loader))
            real_imgs = batch["style_imgs"].to(self.device)
            real_img_lens = batch["style_img_lens"].to(self.device)

            fake_lbs = self.label_converter.encode(text)
            fake_lbs = torch.LongTensor(fake_lbs)
            fake_lb_lens = torch.IntTensor([len(text)])

            nrow = real_imgs.size(0)
            fake_lbs = fake_lbs.repeat(nrow, 1).to(self.device)
            fake_lb_lens = fake_lb_lens.repeat(
                nrow,
            ).to(self.device)
            enc_styles = self.models.E(real_imgs, real_img_lens, self.models.B)

            real_imgs = (1 - real_imgs).squeeze().cpu().numpy() * 127
            gen_imgs = self.models.G(enc_styles, fake_lbs, fake_lb_lens)
            space_indexs = [i for i, ch in enumerate(text) if ch == " "]
            for idx in space_indexs:
                gen_imgs[
                    :,
                    :,
                    idx
                    * self.opt.char_width : (idx + 1)
                    * self.opt.char_width,
                ] = -1
            gen_imgs, gen_img_lens = rescale_images2(
                gen_imgs,
                fake_lb_lens * self.opt.char_width,
                fake_lb_lens,
                batch["org_img_lens"].to(self.device),
                batch["lb_lens"].to(self.device),
            )
            gen_imgs = (1 - gen_imgs).squeeze().cpu().numpy() * 127

            self.plot_text(
                nrow, gen_imgs, real_imgs, real_img_lens, gen_img_lens, text
            )
