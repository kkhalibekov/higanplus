import argparse
import random
import ast
from lib.utils import yaml2config
from networks import get_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/gan_iam.yml",
        help="Configuration file to use",
    )

    parser.add_argument(
        "--ckpt",
        nargs="?",
        type=str,
        default="./pretrained/deploy_HiGAN+.pth",
        help="checkpoint for evaluation",
    )

    parser.add_argument(
        "--split",
        nargs="?",
        type=str,
        default="test",
        help="checkpoint for evaluation",
    )

    parser.add_argument(
        "--guided",
        dest="guided",
        default="True",
        type=ast.literal_eval,
    )

    args = parser.parse_args()
    cfg = yaml2config(args.config)
    cfg.seed = random.randint(0, 10000)
    cfg.valid.dset_split = args.split
    print("cfg.model =", cfg.model)
    model = get_model(cfg.model)
    model = model(cfg, args.config)
    model.load(args.ckpt, cfg.device)

    print("guided: ", args.guided)
    print(model.validate(args.guided, test_stage=True))
