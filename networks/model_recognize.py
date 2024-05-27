import os
import numpy as np
from munch import Munch
from distance import levenshtein
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import CTCLoss
from networks.utils import get_scheduler, ctc_greedy_decoder
from networks.module import Recognizer
from networks.model_base import BaseModel
from lib.datasets import get_dataset, get_collect_fn
from lib.utils import AverageMeter


class RecognizeModel(BaseModel):
    def __init__(self, opt, log_root="./"):
        super().__init__(opt, log_root)

        device = self.device
        self.collect_fn = get_collect_fn(
            sort_input=opt.training.sort_input, sort_style=False
        )
        recognizer = Recognizer(**opt.OcrModel).to(device)
        # print(recognizer.cnn_backbone)
        if os.path.exists(opt.training.pretrained_backbone):
            ckpt = torch.load(opt.training.pretrained_backbone, device)[
                "Recognizer"
            ]
            new_ckpt = {}
            for key, val in ckpt.items():
                if not key.startswith("ctc_cls"):
                    new_ckpt[key] = val
            recognizer.load_state_dict(new_ckpt, strict=False)
            print(
                "load pretrained backbone from ",
                opt.training.pretrained_backbone,
            )

        if os.path.exists(opt.training.resume):
            ckpt = torch.load(opt.training.resume, device)["Recognizer"]
            recognizer.load_state_dict(ckpt)
            print("load pretrained model from ", opt.training.resume)

        self.models = Munch(R=recognizer)

        self.tst_loader = DataLoader(
            get_dataset(
                self.opt.valid.dset_name,
                self.opt.valid.dset_split,
                process_style=True,
            ),
            batch_size=opt.valid.batch_size,
            shuffle=False,
            collate_fn=get_collect_fn(sort_input=True, sort_style=True),
        )

        self.ctc_loss = CTCLoss(zero_infinity=True, reduction="mean")

    def train(self):
        self.info()

        trainset_info = (
            self.opt.training.dset_name,
            self.opt.training.dset_split,
            False,
            self.opt.training.augment,
            True,
        )
        self.print("Trainset: {} [{}]".format(*trainset_info))
        self.train_loader = DataLoader(
            get_dataset(*trainset_info),
            batch_size=self.opt.training.batch_size,
            shuffle=True,
            collate_fn=self.collect_fn,
            num_workers=4,
        )

        self.optimizers = Munch(
            R=torch.optim.Adam(
                self.models.R.parameters(), lr=self.opt.training.lr
            )
        )
        self.lr_schedulers = Munch(
            R=get_scheduler(self.optimizers.R, self.opt.training)
        )

        epoch_done = 1
        if self.opt.training.resume:
            epoch_done = self.load(self.opt.training.resume)
            self.print(self.validate())

        device = self.device
        ctc_loss_meter = AverageMeter()
        ctc_len_scale = self.models.R.len_scale
        best_cer = np.inf
        iter_count = 0
        for epoch in range(epoch_done, self.opt.training.epochs):
            for i, batch in enumerate(self.train_loader):
                #############################
                # Prepare inputs
                #############################
                self.set_mode("train")
                real_imgs, real_img_lens = batch["aug_imgs"].to(device), batch[
                    "aug_img_lens"
                ].to(device)
                real_lbs, real_lb_lens = batch["lbs"].to(device), batch[
                    "lb_lens"
                ].to(device)

                #############################
                # OptimizingRecognizer
                #############################
                self.optimizers.R.zero_grad()
                # Compute CTC loss for real samples
                real_ctc = self.models.R(real_imgs, real_img_lens)
                real_ctc_lens = real_img_lens // ctc_len_scale
                real_ctc_loss = self.ctc_loss(
                    real_ctc, real_lbs, real_ctc_lens, real_lb_lens
                )
                ctc_loss_meter.update(real_ctc_loss.item())
                real_ctc_loss.backward()
                self.optimizers.R.step()

                if iter_count % self.opt.training.print_iter_val == 0:
                    if epoch > 1 and not (self.logger and self.writer):
                        self.create_logger()

                    try:
                        lr = self.lr_schedulers.R.get_last_lr()[0]
                    except Exception:
                        lr = self.lr_schedulers.R.get_lr()[0]

                    ctc_loss_avg = ctc_loss_meter.eval()
                    ctc_loss_meter.reset()
                    info = "[%3d|%3d]-[%4d|%4d] CTC: %.5f  Lr: %.6f" % (
                        epoch,
                        self.opt.training.epochs,
                        iter_count % len(self.train_loader),
                        len(self.train_loader),
                        ctc_loss_avg,
                        lr,
                    )
                    self.print(info)

                    if self.writer:
                        self.writer.add_scalar(
                            "loss/ctc_loss", ctc_loss_avg, iter_count + 1
                        )
                        self.writer.add_scalar("loss/lr", lr, iter_count + 1)

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
                    self.print("Calculate CER_WER")
                    ckpt_root = os.path.join(
                        self.log_root, self.opt.training.ckpt_dir
                    )
                    if not os.path.exists(ckpt_root):
                        os.makedirs(ckpt_root)

                    scores = self.validate()
                    wer, cer = scores["WER"], scores["CER"]
                    self.print(f"WER:{wer} CER:{cer}")
                    if cer < best_cer:
                        best_cer = cer
                        self.save("best", epoch, WER=wer, CER=cer)
                    if self.writer:
                        self.writer.add_scalar("valid/WER", wer, epoch)
                        self.writer.add_scalar("valid/CER", cer, epoch)

            for scheduler in self.lr_schedulers.values():
                scheduler.step(epoch)

    def validate(self, *args, **kwargs):
        self.set_mode("eval")
        ctc_len_scale = self.models.R.len_scale
        char_trans = 0
        total_chars = 0
        word_trans = 0
        total_words = 0
        print(self.tst_loader.dataset.file_path)
        with torch.no_grad():
            for i, batch in tqdm(
                enumerate(self.tst_loader), total=len(self.tst_loader)
            ):
                real_imgs, real_img_lens = batch["style_imgs"].to(
                    self.device
                ), batch["style_img_lens"].to(self.device)
                logits = self.models.R(real_imgs, real_img_lens)
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

        cer = char_trans / total_chars
        wer = word_trans / total_words
        return {"CER": cer, "WER": wer}
