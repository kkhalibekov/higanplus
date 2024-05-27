import os
import logging
import datetime
import numpy
from munch import Munch
import matplotlib.pyplot as plt
import cv2
import yaml
import munch
from torchvision.utils import make_grid


def get_logger(logdir):
    logger = logging.getLogger("gan")
    ts = str(datetime.datetime.now())
    ts = ts.split(".", maxsplit=1)[0]
    ts = ts.replace(" ", "_")
    ts = ts.replace(":", "_")
    ts = ts.replace("-", "_")
    file_path = os.path.join(logdir, f"run_{ts}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=file_path,
        filemode="w",
    )
    # define a new Handler to log to console as well
    console = logging.StreamHandler()
    # optional, set the logging level
    console.setLevel(logging.INFO)
    # set a format which is the same for console use
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger("").addHandler(console)
    return logger


def yaml2config(yml_path):
    with open(yml_path) as fp:
        json = yaml.load(fp, Loader=yaml.FullLoader)

    def to_munch(json):
        for key, val in json.items():
            if isinstance(val, dict):
                json[key] = to_munch(val)
        return munch.Munch(json)

    cfg = to_munch(json)
    return cfg


def draw_image(
    tensor,
    nrow=8,
    padding=2,
    normalize=False,
    value_range=None,
    scale_each=False,
    pad_value=0,
):
    grid = make_grid(
        tensor,
        nrow=nrow,
        padding=padding,
        pad_value=pad_value,
        normalize=normalize,
        value_range=value_range,
        scale_each=scale_each,
    )
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = (
        grid.mul_(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .astype(numpy.uint8)
    )
    return ndarr


def plot_heatmap(arr):
    heatmapshow = cv2.normalize(
        arr,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    return heatmapshow


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def eval(self):
        return self.avg


class AverageMeterManager:
    def __init__(self, keys):
        self.meters = {key: AverageMeter() for key in keys}

    def reset(self, key):
        self.meters[key].reset()

    def reset_all(self):
        for meter in self.meters.values():
            meter.reset()

    def update(self, key, val, n=1):
        self.meters[key].update(val, n)

    def eval(self, keys):
        if isinstance(keys, str):
            keys = [keys]

        return {key: self.meters[key].eval() for key in keys}

    def eval_all(self):
        return {key: meter.eval() for key, meter in self.meters.items()}


def option_to_string(opt, num_row_blanks=20):
    blanks = "-" * num_row_blanks

    def opt_to_str(opt, depth=0) -> str:
        res = []
        indent = "|" + "-" * depth
        for key, val in opt.items():
            if isinstance(val, (Munch, dict)):
                res.append(f"{blanks}\n{key}\n")
                res.append(opt_to_str(val, depth + 2))
            else:
                res.append(f"{indent}{key}: {val}\n")
        return "".join(res)

    body = opt_to_str(opt)
    footer = "=" * num_row_blanks
    header = f"{footer}\nRoot\n{blanks}\n"
    return f"{header}{body}{footer}"


def get_corpus(corpus_path):
    items = []
    with open(corpus_path, "r") as f:
        for line in f.readlines():
            items.append(line.strip())
    return items


def show_image_pair(img1, img2, title1="", title2=""):
    plt.subplot(211)
    plt.imshow(img1, cmap="binary")
    plt.title(title1)
    plt.subplot(212)
    plt.imshow(img2, cmap="binary")
    plt.title(title2)
    plt.show()
