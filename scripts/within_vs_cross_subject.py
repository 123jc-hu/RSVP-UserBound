#!/usr/bin/env python3
"""Does calibration on a user's own data rescue the users that calibration-free decoders find hard?

Compares second-half BA of the within-subject reference (scripts/run_within_subject_calibration.py)
with second-half BA of the eleven LOSO decoders on the same trials
(analysis/diagnostic_circularity_reliability/subject_ba_by_half.csv). ERP strength is taken from
the first half only (erp_A), i.e. from the calibration data, so it never sees the test trials.
Run from the repo root.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from paths import ANALYSIS, EXP

DCR = ANALYSIS / "diagnostic_circularity_reliability"
OUT = ANALYSIS / "within_vs_cross_subject"
DATASETS = ("THU", "CAS", "THINGS")
WITHIN = {
    ("sLDA", "half"): "SLDA/{ds}/within-subject/slda_{low}_within_half_bin10_lw_s2026",
    ("sLDA", "K40"): "SLDA/{ds}/within-subject/slda_{low}_within_k40_bin10_lw_s2026",
    ("xDAWN-RG", "half"): "XdawnRiemann/{ds}/within-subject/xdawnrg_{low}_within_half_nf4_lwf_neg4_s2026",
    ("xDAWN-RG", "K40"): "XdawnRiemann/{ds}/within-subject/xdawnrg_{low}_within_k40_nf4_lwf_neg4_s2026",
    ("PRISM-FT", "half"): "PRISM/{ds}/within-subject/prism_{low}_finetune_half_lr0.0001_s2026",
    ("PRISM-FT", "K40"): "PRISM/{ds}/within-subject/prism_{low}_finetune_k40_lr0.0001_s2026",
}
CROSS_MODELS = [
    "PRISM", "PPNN", "DC-tCNN-DG", "SCLDGN", "EEG-DG", "EEGConformer", "EEGNet",
    "EEGTransferNetDG", "PLNet", "KnIFE", "xDAWN-RG",
]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def read_within(method: str, cond: str, ds: str) -> pd.Series:
    path = EXP / WITHIN[(method, cond)].format(ds=ds, low=ds.lower()) / "results.csv"
    df = pd.read_csv(path)
    df = df[df.SUB.str.fullmatch(r"SUB\d+")]
    return pd.Series(df.BA.to_numpy(float), index=df.SUB.str[3:].astype(int).to_numpy())


def fe_regression(y: np.ndarray, x: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """Slope of y on x with dataset fixed effects; returns (beta per SD of x, p)."""
    dummies = pd.get_dummies(groups, drop_first=True).to_numpy(float)
    design = np.column_stack([np.ones(len(y)), dummies, x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = len(y) - design.shape[1]
    cov = np.linalg.inv(design.T @ design) * (resid @ resid) / dof
    t = beta[-1] / np.sqrt(cov[-1, -1])
    return float(beta[-1]), float(2 * stats.t.sf(abs(t), dof))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    axes = pd.read_csv(DCR / "subject_axes_by_split.csv")
    halves = pd.read_csv(DCR / "subject_ba_by_half.csv")
    frames = []
    for ds in DATASETS:
        d = halves[halves.dataset == ds].set_index("subject_id")
        tab = pd.DataFrame({m: d[f"{m}|B"] for m in CROSS_MODELS})
        tab["erp_A"] = axes[axes.dataset == ds].set_index("subject_id")["erp_A"]
        for (method, cond) in WITHIN:
            try:
                tab[f"{method}|{cond}"] = read_within(method, cond, ds)
            except FileNotFoundError:
                pass
        tab["dataset"] = ds
        frames.append(tab.reset_index().rename(columns={"index": "subject_id"}))
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(OUT / "per_subject.csv", index=False)
    within_cols = [c for c in data.columns if "|" in c]

    rows = []
    for ds in DATASETS + ("pooled",):
        d = data if ds == "pooled" else data[data.dataset == ds]
        for col in within_cols:
            dd = d.dropna(subset=[col])
            gain = dd[col] - dd["PRISM"]
            row = {"dataset": ds, "within": col, "n": len(dd),
                   "within_BA": dd[col].mean(), "PRISM_BA": dd["PRISM"].mean(),
                   "gain_mean": gain.mean(), "gain_wins": int((gain > 0).sum()),
                   "gain_p": stats.wilcoxon(gain).pvalue if len(dd) > 5 else np.nan}
            if ds == "pooled":
                z = dd.groupby("dataset")["erp_A"].transform(lambda v: (v - v.mean()) / v.std()).to_numpy()
                for scale, f in (("BA", lambda v: v), ("logit", logit)):
                    g = f(dd[col].to_numpy()) - f(dd["PRISM"].to_numpy())
                    row[f"gain_vs_sERP_beta_{scale}"], row[f"gain_vs_sERP_p_{scale}"] = fe_regression(g, z, dd.dataset.to_numpy())
                wz = dd[col] - dd.groupby("dataset")[col].transform("mean")
                row["within_vs_sERP_r"], row["within_vs_sERP_p"] = stats.pearsonr(z, wz)
                cz = dd["PRISM"] - dd.groupby("dataset")["PRISM"].transform("mean")
                row["within_vs_PRISM_r"], row["within_vs_PRISM_p"] = stats.pearsonr(wz, cz)
            else:
                row["spearman_within_vs_PRISM"] = stats.spearmanr(dd[col], dd["PRISM"])[0]
                q = int(np.ceil(len(dd) / 4))
                hard_cross = set(dd.nsmallest(q, "PRISM").subject_id)
                hard_within = set(dd.nsmallest(q, col).subject_id)
                row["q1_overlap"] = len(hard_cross & hard_within) / q
                row["q1_chance"] = q / len(dd)
                hc = dd[dd.subject_id.isin(hard_cross)]
                row["PRISM_q1_users_PRISM_BA"] = hc["PRISM"].mean()
                row["PRISM_q1_users_within_BA"] = hc[col].mean()
                rest = dd[~dd.subject_id.isin(hard_cross)]
                row["rest_gain"] = (rest[col] - rest["PRISM"]).mean()
                row["q1_gain"] = (hc[col] - hc["PRISM"]).mean()
                row["within_q1_BA"] = dd.nsmallest(q, col)[col].mean()
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(summary.round(4).to_string(index=False))
    (OUT / "summary.json").write_text(json.dumps(summary.to_dict("records"), indent=1, default=float))


if __name__ == "__main__":
    main()
