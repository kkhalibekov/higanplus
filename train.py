import os
from datetime import datetime
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

    args = parser.parse_args()
    cfg = yaml2config(args.config)
    run_id = datetime.strftime(datetime.now(), "%m-%d-%H-%M")
    logdir = os.path.join(
        "runs", f"{os.path.basename(args.config)[:-4]}-{run_id}"
    )
    print("cfg.model =", cfg.model)
    model = get_model(cfg.model)
    model = model(cfg, logdir)
    model.train()
