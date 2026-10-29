"""Shared locations of data and results.

RSVP_DATASET_ROOT  preprocessed data, <root>/<THU|CAS|THINGS>/Standard_128Hz (default: ./data)
RSVP_RESULTS_ROOT  experiment results and analysis outputs (default: ./Experiments)
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RSVP_DATASET_ROOT", ROOT / "data"))
EXP = Path(os.environ.get("RSVP_RESULTS_ROOT", ROOT / "Experiments"))
ANALYSIS = EXP / "analysis"

DATA_DIRS = {ds: DATA / ds / "Standard_128Hz" for ds in ("THU", "CAS", "THINGS")}
# THINGS-EEG: participants 1, 6, 18, 23 are flagged in the source metadata, and 49, 50 use a different montage.
THINGS_EXCLUDED = (1, 6, 18, 23, 49, 50)
SUBJECTS = {
    "THU": list(range(1, 65)),
    "CAS": list(range(1, 15)),
    "THINGS": [s for s in range(1, 51) if s not in THINGS_EXCLUDED],
}
