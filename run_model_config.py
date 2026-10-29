"""Train and test PRISM under LOSO for one run configuration.

usage: python run_model_config.py --config Configs/<run>.yaml

A run configuration names the model and dataset and overrides the defaults of Configs/config.yaml.
"""

import argparse

from main import main
from Utils.config import build_config, load_config


def run(config_path: str):
    run_cfg = load_config(config_path)
    cfg = build_config(run_cfg["model"], run_cfg["dataset"])
    cfg.update(run_cfg.get("overrides") or {})
    main(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a run configuration in Configs/.")
    run(parser.parse_args().config)
