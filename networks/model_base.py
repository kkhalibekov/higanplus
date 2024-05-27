import os
import torch
from torch.utils.tensorboard import SummaryWriter
from munch import Munch
from lib.alphabet import Alphabets, StrLabelConverter
from lib.utils import option_to_string, get_logger
from networks.utils import _info


class BaseModel:
    def __init__(self, opt, log_root="./"):
        self.opt = opt
        self.local_rank = opt.local_rank if "local_rank" in opt else -1
        self.device = torch.device(opt.device)
        self.models = Munch()
        self.models_ema = Munch()
        self.optimizers = Munch()
        self.log_root = log_root
        self.logger = None
        self.writer = None

        alphabet_key = (
            "rimes_word" if opt.dataset.startswith("rimes") else "all"
        )
        self.alphabet = Alphabets[alphabet_key]
        self.label_converter = StrLabelConverter(alphabet_key)

    def print(self, info):
        if self.logger is None:
            print(info)
        else:
            self.logger.info(info)

    def create_logger(self):
        if self.logger or self.writer:
            return

        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        self.writer = SummaryWriter(log_dir=self.log_root)

        opt_str = option_to_string(self.opt)
        with open(
            os.path.join(self.log_root, "config.txt"), "w", encoding="utf-8"
        ) as f:
            f.writelines(opt_str)
        print("log_root:", self.log_root)
        self.logger = get_logger(self.log_root)

    def info(self, extra=None):
        self.print(f"RUNDIR: {self.log_root}")
        opt_str = option_to_string(self.opt)
        self.print(opt_str)
        for model in self.models.values():
            self.print(_info(model, ret=True))
        if extra is not None:
            self.print(extra)
        self.print("=" * 20)

    def save(self, tag="best", epoch_done=0, **kwargs):
        ckpt = {
            type(model).__name__: model.state_dict()
            for model in self.models.values()
        }
        ckpt.update(
            {
                f"OPT.{key}": optim.state_dict()
                for key, optim in self.optimizers.items()
            }
        )
        ckpt.update(kwargs)

        ckpt["Epoch"] = epoch_done
        ckpt_save_path = os.path.join(
            self.log_root, self.opt.training.ckpt_dir, f"{tag}.pth"
        )
        torch.save(ckpt, ckpt_save_path)

    def load(self, ckpt: dict, map_location=None, modules=None):
        if modules is None:
            modules = []
        elif not isinstance(modules, list):
            modules = [modules]

        print("load checkpoint from ", ckpt)
        if map_location is None:
            ckpt = torch.load(ckpt)
        else:
            ckpt = torch.load(ckpt, map_location=map_location)

        if ckpt is None:
            return

        models = self.models.values() if not modules else modules
        for model in models:
            model_name = type(model).__name__
            print(type(model))
            print(*ckpt.keys())
            try:
                if model_name in ckpt:
                    model.load_state_dict(ckpt.pop(model_name))
            except Exception as e:
                print(f"{type(e).__name__}: {e}: Load {model_name} failed")
            print()

        for key in self.optimizers:
            try:
                self.optimizers[key].load_state_dict(ckpt.pop(f"OPT.{key}"))
            except Exception:
                print(f"Load OPT.{key} failed")

        ckpt["Epoch"] = 0 if "Epoch" not in ckpt else ckpt["Epoch"]
        return ckpt["Epoch"]

    def set_mode(self, mode="eval"):
        for model in self.models.values():
            if mode == "eval":
                model.eval()
            elif mode == "train":
                model.train()
            else:
                raise NotImplementedError()

    def validate(self, *args, **kwargs):
        yield NotImplementedError()

    def train(self):
        yield NotImplementedError()
