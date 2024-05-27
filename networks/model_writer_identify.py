import os
from itertools import chain
import torch
from munch import Munch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss
from networks.utils import get_scheduler
from networks.module import WriterIdentifier, StyleBackbone
from networks.model_base import BaseModel
from lib.datasets import get_dataset, get_collect_fn
from lib.utils import AverageMeter


class WriterIdentifyModel(BaseModel):
    def __init__(self, opt, log_root="./"):
        super().__init__(opt, log_root)

        device = self.device

        style_backbone = StyleBackbone(**opt.StyBackbone).to(device)
        if os.path.exists(opt.training.pretrained_backbone):
            ckpt = torch.load(opt.training.pretrained_backbone, device)

            if "Recognizer" in ckpt:
                ckpt = ckpt["Recognizer"]
                new_ckpt = {}
                for key, val in ckpt.items():
                    if key.startswith("cnn_backbone") or key.startswith(
                        "cnn_ctc"
                    ):
                        new_ckpt[key] = val
                style_backbone.load_state_dict(new_ckpt)
            else:
                ckpt = ckpt["StyleBackbone"]
                style_backbone.load_state_dict(ckpt)

            print(
                "Load style_backbone from ", opt.training.pretrained_backbone
            )

        identifier = WriterIdentifier(**opt.WidModel).to(device)
        self.models = Munch(W=identifier, B=style_backbone)

        self.tst_loader = DataLoader(
            get_dataset(opt.dataset, opt.valid.dset_split),
            batch_size=opt.valid.batch_size,
            shuffle=False,
            collate_fn=get_collect_fn(sort_input=False),
        )

        self.wid_loss = CrossEntropyLoss()

    def train(self):
        self.info()

        trainset_info = (
            self.opt.training.dset_name,
            self.opt.training.dset_split,
            self.opt.training.random_clip,
            False,
            self.opt.training.process_style,
        )
        self.print("Trainset: {} [{}]".format(*trainset_info))
        self.train_loader = DataLoader(
            get_dataset(*trainset_info),
            batch_size=self.opt.training.batch_size,
            shuffle=True,
            collate_fn=get_collect_fn(sort_input=True, sort_style=False),
            num_workers=4,
        )

        if self.opt.training.frozen_backbone:
            print("frozen_backbone")
            self.optimizers = Munch(
                W=torch.optim.Adam(self.models.W.parameters()),
                lr=self.opt.training.lr,
            )
        else:
            self.optimizers = Munch(
                W=torch.optim.Adam(
                    chain(
                        self.models.W.parameters(), self.models.B.parameters()
                    ),
                    lr=self.opt.training.lr,
                )
            )

        self.lr_schedulers = Munch(
            W=get_scheduler(self.optimizers.W, self.opt.training)
        )

        epoch_done = 1
        if self.opt.training.resume:
            epoch_done = self.load(self.opt.training.resume)
            self.print(self.validate())

        device = self.device
        wid_loss_meter = AverageMeter()
        best_wrr = 0
        iter_count = 0
        for epoch in range(epoch_done, self.opt.training.epochs):
            for i, batch in enumerate(self.train_loader):
                #############################
                # Prepare inputs
                #############################
                self.set_mode("train")
                real_imgs, real_img_lens, real_wids = (
                    batch["aug_imgs"].to(device),
                    batch["aug_img_lens"].to(device),
                    batch["wids"].to(device),
                )

                if self.opt.training.frozen_backbone:
                    self.models.B.frozen_bn()

                #############################
                # OptimizingRecognizer
                #############################
                self.optimizers.W.zero_grad()
                # Compute CTC loss for real samples
                wid_logits = self.models.W(
                    real_imgs, real_img_lens, self.models.B
                )
                wid_loss = self.wid_loss(wid_logits, real_wids)
                wid_loss_meter.update(wid_loss.item())
                wid_loss.backward()
                self.optimizers.W.step()

                if iter_count % self.opt.training.print_iter_val == 0:
                    if epoch > 1 and not (self.logger and self.writer):
                        self.create_logger()

                    try:
                        lr = self.lr_schedulers.W.get_last_lr()[0]
                    except Exception:
                        lr = self.lr_schedulers.W.get_lr()[0]

                    wid_loss_avg = wid_loss_meter.eval()
                    wid_loss_meter.reset()
                    info = "[%3d|%3d]-[%4d|%4d] WID: %.5f  Lr: %.6f" % (
                        epoch,
                        self.opt.training.epochs,
                        iter_count % len(self.train_loader),
                        len(self.train_loader),
                        wid_loss_avg,
                        lr,
                    )
                    self.print(info)

                iter_count += 1

            if epoch:
                ckpt_root = os.path.join(
                    self.log_root, self.opt.training.ckpt_dir
                )
                if not os.path.exists(ckpt_root):
                    os.makedirs(ckpt_root)

                self.save("last", epoch)
                if epoch >= self.opt.training.start_save_epoch_val and (
                    epoch % self.opt.training.save_epoch_val == 0
                    or epoch >= self.opt.training.epochs
                ):
                    self.print("Calculate WRR")
                    ckpt_root = os.path.join(
                        self.log_root, self.opt.training.ckpt_dir
                    )
                    if not os.path.exists(ckpt_root):
                        os.makedirs(ckpt_root)

                    wrr = self.validate()
                    self.print(f"WRR:{wrr:.2f}")
                    if wrr > best_wrr:
                        best_wrr = wrr
                        self.save("best", epoch, WRR=wrr)
                    if self.writer:
                        self.writer.add_scalar("valid/WRR", wrr, epoch)

            for scheduler in self.lr_schedulers.values():
                scheduler.step(epoch)

    def validate(self, *args, **kwargs):
        self.set_mode("eval")

        with torch.no_grad():
            acc_counts = 0.0
            total_counts = 0.0
            for i, batch in tqdm(
                enumerate(self.tst_loader), total=len(self.tst_loader)
            ):
                wid_logits = self.models.W(
                    batch["style_imgs"].to(self.device),
                    batch["style_img_lens"].to(self.device),
                    self.models.B,
                )
                _, preds = torch.max(wid_logits.data, dim=1)

                acc_counts += (
                    preds.eq(batch["wids"].to(self.device)).sum().item()
                )
                total_counts += wid_logits.size(0)

            wrr = acc_counts * 100.0 / total_counts
            wier = 1 - acc_counts * 1.0 / total_counts
            print("wier: ", wier)

        for model in self.models.values():
            model.train()

        return wrr
