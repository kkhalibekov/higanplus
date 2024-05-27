import argparse
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
        # default="./pretrained/HiGAN+.pth",
        default="./pretrained/deploy_HiGAN+.pth",
        help="checkpoint for evaluation",
    )

    parser.add_argument(
        "--mode",
        nargs="?",
        type=str,
        # default="text",
        default="style",
        help="mode: [rand] [style] [text] [interp]",
    )

    args = parser.parse_args()
    cfg = yaml2config(args.config)
    print("cfg.model =", cfg.model)
    model = get_model(cfg.model)
    model = model(cfg, args.config)
    model.load(args.ckpt, cfg.device)
    model.set_mode("eval")

    text = input("input text: ")

    if args.mode == "style":
        model.eval_style(text)
    elif args.mode == "rand":
        model.eval_rand(text)
    elif args.mode == "interp":
        model.eval_interp(text)
    elif args.mode == "text":
        model.eval_text(text)
    else:
        print(f"Unsupported mode: {cfg.mode} | [rand] [style] [text] [interp]")
