#!/usr/bin/env python3
"""THU three-seed analysis for PRISM, PPNN, EEGNet and DC-tCNN (training seeds 2026-2028)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from paths import ANALYSIS, EXP as XS

OUT = ANALYSIS / "multiseed_four_models"
SEEDS = (2026, 2027, 2028)
SUBJECTS = tuple(range(1, 65))
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2031

BASES = {
    "PRISM": "PRISM/THU/cross-subject/prism_thu64_d16_bestsofar_prewsdrop010_regionconcat_lmr_m8",
    "PPNN": "PPNN/THU/cross-subject/ppnn_thu64_k32_bgds1_sb32_ce",
    "EEGNet": "EEGNet/THU/cross-subject/eegnet_thu64_k32_bgds1_sb32_ce",
    "DC-tCNN": "DCTCNNDG/THU/cross-subject/dctcnndg_thu64_k32_bgds1_sb32_cdan1",
}
METRICS = ["AUC", "BA", "F1", "TPR", "FPR"]


def run_dir(model: str, seed: int) -> Path:
    suffix = "" if seed == 2026 else f"_trainseed{seed}"
    return XS / f"{BASES[model]}{suffix}"


def read_csv(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data[data["SUB"].astype(str).str.fullmatch(r"SUB\d+")].copy()
    data["subject"] = data["SUB"].str.extract(r"(\d+)")[0].astype(int)
    return data[["subject", *METRICS]]


def read_run(model: str, seed: int) -> pd.DataFrame:
    data = read_csv(run_dir(model, seed) / "results.csv")
    if data.subject.duplicated().any() or tuple(sorted(data.subject)) != SUBJECTS:
        raise ValueError(f"{model} {seed}: incomplete or duplicated ({len(data)} rows)")
    data["model"], data["seed"] = model, seed
    return data.sort_values("subject").reset_index(drop=True)


def matrix(long: pd.DataFrame, model: str, metric: str = "BA") -> np.ndarray:
    return long[long.model == model].pivot(index="subject", columns="seed", values=metric).loc[
        list(SUBJECTS), list(SEEDS)
    ].to_numpy()


def crossed_bootstrap(delta: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_REPLICATES)
    for i in range(BOOTSTRAP_REPLICATES):
        s = rng.integers(0, delta.shape[0], size=delta.shape[0])
        k = rng.integers(0, delta.shape[1], size=delta.shape[1])
        values[i] = delta[np.ix_(s, k)].mean()
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), float(
        (1 + np.sum(values <= 0)) / (BOOTSTRAP_REPLICATES + 1)
    )


def icc_seed(m: np.ndarray) -> float:
    """ICC(3,1) across seeds treated as raters."""
    n, k = m.shape
    grand = m.mean()
    ss_r = k * ((m.mean(1) - grand) ** 2).sum()
    ss_c = n * ((m.mean(0) - grand) ** 2).sum()
    ss_e = ((m - grand) ** 2).sum() - ss_r - ss_c
    ms_r, ms_e = ss_r / (n - 1), ss_e / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    long = pd.concat([read_run(m, s) for m in BASES for s in SEEDS], ignore_index=True)
    long.to_csv(OUT / "per_subject_by_seed.csv", index=False)

    summary = []
    for model in BASES:
        m = matrix(long, model)
        rho = [stats.spearmanr(m[:, a], m[:, b])[0] for a, b in ((0, 1), (0, 2), (1, 2))]
        summary.append({
            "model": model,
            **{f"BA_seed{s}": float(m[:, j].mean()) for j, s in enumerate(SEEDS)},
            "BA_3seed_mean": float(m.mean()),
            "seed_mean_range": float(np.ptp(m.mean(0))),
            # Manuscript definition: mean over participants of the across-seed SD
            "within_subject_sd": float(m.std(axis=1, ddof=1).mean()),
            "between_subject_sd": float(m.mean(1).std(ddof=1)),
            "subject_var_share": float(m.mean(1).var(ddof=1) / (m.mean(1).var(ddof=1) + m.var(axis=1, ddof=1).mean())),
            "icc31_seeds": icc_seed(m),
            "mean_pairwise_spearman": float(np.mean(rho)),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "seed_summary.csv", index=False)

    contrasts = []
    prism = matrix(long, "PRISM")
    for other in ("PPNN", "EEGNet", "DC-tCNN"):
        delta = prism - matrix(long, other)
        lo, hi, p_boot = crossed_bootstrap(delta)
        per_subject = delta.mean(1)
        contrasts.append({
            "contrast": f"PRISM-{other}",
            "delta_3seed": float(delta.mean()),
            **{f"delta_seed{s}": float(delta[:, j].mean()) for j, s in enumerate(SEEDS)},
            "ci95_low": lo,
            "ci95_high": hi,
            "p_boot_one_sided": p_boot,
            "wins_seedavg": int((per_subject > 0).sum()),
            "wilcoxon_p_seedavg": float(stats.wilcoxon(per_subject).pvalue),
        })
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(OUT / "prism_contrasts.csv", index=False)

    # Rank consistency of participants across decoders, per seed and on seed means
    rank_rows = []
    for j, s in enumerate(SEEDS):
        cols = np.column_stack([matrix(long, m)[:, j] for m in BASES])
        r = stats.spearmanr(cols)[0]
        rank_rows.append({"seed": s, "mean_pairwise_spearman_across_models": float(r[np.triu_indices(4, 1)].mean())})
    pd.DataFrame(rank_rows).to_csv(OUT / "cross_model_rank_by_seed.csv", index=False)

    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    print(contrasts.round(4).to_string(index=False))
    print(pd.DataFrame(rank_rows).round(3).to_string(index=False))
    (OUT / "summary.json").write_text(json.dumps({
        "seed_summary": summary.to_dict("records"),
        "contrasts": contrasts.to_dict("records"),
        "cross_model_rank_by_seed": rank_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
