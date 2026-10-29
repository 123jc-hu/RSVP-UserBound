import os
from pathlib import Path
from typing import Any, Dict


def resolve_dataset_dir(config: Dict[str, Any]) -> Path:
    """Directory of the preprocessed NPZ files: <root>/<dataset>/Standard_<fs>Hz.

    The root is `dataset_root` from the configuration, else $RSVP_DATASET_ROOT, else ./data.
    """
    root = config.get("dataset_root") or os.environ.get("RSVP_DATASET_ROOT") or Path(__file__).resolve().parents[1] / "data"
    return Path(root) / str(config["dataset"]) / f"Standard_{int(config['fs'])}Hz"
