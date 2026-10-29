"""Choice of PRISM's spatial width D in {8, 16} from source-validation BA (Section III-C).

Dataset-level rule: per dataset, the D with the higher mean (over LOSO folds) of the best
source-validation BA. Per-fold rule: each fold uses the D with the higher validation BA in
that fold only; its held-out BA is reported for comparison.

Input: Experiments/analysis/fold_validation_ba.csv (dataset, D, subject, val_ba, test_ba), shipped
with the results archive. `--export` rebuilds that file from the TensorBoard logs of retrained runs.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from paths import ANALYSIS, EXP

RUNS = {
    "THU": {8: "PRISM/THU/cross-subject/prism_thu64_d8_bestsofar_prewsdrop010_regionconcat_lmr_m8",
            16: "PRISM/THU/cross-subject/prism_thu64_d16_bestsofar_prewsdrop010_regionconcat_lmr_m8"},
    "CAS": {8: "PRISM/CAS/cross-subject/prism_cas14_d8_bestsofar_prewsdrop010_regionconcat_lmr_m8",
            16: "PRISM/CAS/cross-subject/prism_cas14_d16_bestsofar_prewsdrop010_regionconcat_lmr_m8"},
    "THINGS": {8: "PRISM/THINGS/cross-subject/prism_things_strict44_d8_bestsofar_sb32_prewsdrop010_regionconcat_lmr_m8_montagefix",
               16: "PRISM/THINGS/cross-subject/prism_things_strict44_d16_bestsofar_sb32_prewsdrop010_regionconcat_lmr_m8_montagefix"},
}
TABLE = ANALYSIS / "fold_validation_ba.csv"


def best_val_ba(log_dir) -> float:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    versions = sorted(log_dir.glob("version_*"))
    acc = EventAccumulator(str(versions[-1]))
    acc.Reload()
    tag = next(t for t in acc.Tags()["scalars"] if re.search(r"val.*/BA$", t))
    return max(e.value for e in acc.Scalars(tag))


def export() -> None:
    rows = []
    for ds, runs in RUNS.items():
        for d, rel in runs.items():
            res = pd.read_csv(EXP / rel / "results.csv")
            res = res[res.SUB.str.fullmatch(r"SUB\d+")]
            for _, r in res.iterrows():
                sid = int(r.SUB[3:])
                rows.append({"dataset": ds, "D": d, "subject": sid, "test_ba": float(r.BA),
                             "val_ba": best_val_ba(EXP / rel / "logs" / f"sub{sid}" / f"metrics_sub{sid}")})
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLE, index=False)
    print(f"wrote {TABLE}")


def select() -> None:
    t = pd.read_csv(TABLE)
    for ds in RUNS:
        w = t[t.dataset == ds].pivot(index="subject", columns="D", values=["val_ba", "test_ba"])
        val8, val16 = w["val_ba"][8].mean(), w["val_ba"][16].mean()
        pick = np.where(w["val_ba"][8] > w["val_ba"][16], 8, 16)
        per_fold = np.where(pick == 8, w["test_ba"][8], w["test_ba"][16])
        print(f"{ds}: mean val BA D8={val8:.4f} D16={val16:.4f} -> D{8 if val8 > val16 else 16} | "
              f"test BA D8={w['test_ba'][8].mean():.4f} D16={w['test_ba'][16].mean():.4f} | "
              f"per-fold rule: D8 in {int((pick == 8).sum())}/{len(pick)} folds, test BA {per_fold.mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", action="store_true", help="rebuild the table from TensorBoard logs first")
    args = parser.parse_args()
    if args.export:
        export()
    select()
