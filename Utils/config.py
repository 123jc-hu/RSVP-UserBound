import logging
import os
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml

DEFAULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Configs", "config.yaml")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_config(model_name: str, dataset_name: str) -> Dict[str, Any]:
    """Shared defaults of Configs/config.yaml, then its `models.<model>` and `datasets.<dataset>` entries."""
    full_config = load_config(DEFAULTS)
    shared = {k: v for k, v in full_config.items() if k not in ("models", "datasets")}
    merged = {
        **shared,
        **full_config["models"].get(model_name, {}),
        **full_config["datasets"][dataset_name],
    }
    merged["model"] = model_name
    merged["dataset"] = dataset_name
    return merged


def set_random_seed(
    seed: int = 2026,
    *,
    deterministic: bool = True,
    benchmark: bool = False,
    matmul_precision: str = "highest",
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = bool(benchmark)
    cudnn.deterministic = bool(deterministic)
    torch.set_float32_matmul_precision(str(matmul_precision))


def get_logger(name: str) -> logging.Logger:
    """Logger writing to the console and to `<name>.log`."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(f"{name}.log", mode="a", encoding="utf-8"), logging.StreamHandler()):
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
