"""Circularity, reliability and SNR checks for the two ERP diagnostic axes.

Reviewer concerns addressed (no retraining; uses saved trial-level predictions):
  1. Circularity: s_ERP / r_topo are computed from the same labelled trials on
     which BA is measured.  Here the axes come from one temporal half of each
     held-out subject and BA from the other (both directions).  For CAS the
     halves coincide with the two recording sessions.
  2. Calibration: axes estimated from only the first K target trials (with the
     non-targets in the same time span), BA on the second half.
  3. Reliability: split-half reliability of each subject's |d'| map; r_topo
     disattenuated, and r_topo tested after flexible (spline) control of s_ERP
     and of map reliability.
  4. SNR simulation: in spatially typical subjects, the target-class mean shift
     is scaled by lambda in [0, 1] while each subject's real noise is kept, to
     map how far r_topo can fall from SNR loss alone.  Observed r_topo is then
     expressed as a residual from that SNR-only curve.

Axes follow the manuscript: per-channel pooled-SD |d'| of window-mean amplitude
(300-700 ms THU/CAS, 250-600 ms THINGS) on the standardized epochs;
s_ERP = mean |d'|, r_topo = Pearson r with the leave-one-subject-out mean map.
All pooled models use dataset fixed effects with axes z-scored within dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_subject_bound_difficulty import logit, loso_r2, ols, partial_r  # noqa: E402

from paths import ANALYSIS, DATA_DIRS, EXP, SUBJECTS  # noqa: E402

PRED_DIR = ANALYSIS / "positive_class_metrics/trial_predictions"
FS = 128
WINDOW_MS = {"THU": (300, 700), "CAS": (300, 700), "THINGS": (250, 600)}
DATASETS = {
    ds: {"root": DATA_DIRS[ds], "subjects": SUBJECTS[ds], "window_ms": WINDOW_MS[ds]}
    for ds in ("THU", "CAS", "THINGS")
}

# Trial-prediction directory name -> manuscript label.
MODELS = {
    "PRISM": "PRISM",
    "PPNN": "PPNN",
    "DCTCNNDG": "DC-tCNN-DG",
    "SCLDGN": "SCLDGN",
    "EEGDG": "EEG-DG",
    "EEGConformer": "EEGConformer",
    "EEGNet": "EEGNet",
    "EEGTransferNetDG": "EEGTransferNetDG",
    "PLNet": "PLNet",
    "KnIFE": "KnIFE",
}

# CAS PRISM uses D = 8 (selected by source-validation BA), whose predictions are in PRED_DIR.
# For the D = 16 sensitivity analysis, point ("PRISM", "CAS") at
# ANALYSIS / "prism_cas_d16_predictions/prism_cas14_d16_bestsofar_prewsdrop010_regionconcat_lmr_m8".
PRED_DIR_OVERRIDE: dict = {}

# Models whose trial predictions live with their own experiment rather than in PRED_DIR.
EXTRA_PRED_DIRS = {
    "xDAWN-RG": {
        ds: EXP / f"XdawnRiemann/{ds}/cross-subject/xdawnrg_{ds.lower()}_nf4_lwf_neg4_s2026/trial_predictions"
        for ds in ("THU", "CAS", "THINGS")
    },
}

CALIBRATION_K = (10, 20, 40, 80)
LAMBDAS = np.linspace(0.0, 1.0, 21)


# --------------------------------------------------------------------------- io


def load_window_amplitudes(dataset: str, cache_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{subject: (amp [N, C] window-mean amplitude, y [N])}, cached per dataset."""
    cache = cache_dir / f"{dataset}_window_amp.npz"
    cfg = DATASETS[dataset]
    if cache.exists():
        z = np.load(cache)
        return {s: (z[f"amp{s}"], z[f"y{s}"]) for s in cfg["subjects"]}
    t0, t1 = (int(round(ms / 1000.0 * FS)) for ms in cfg["window_ms"])
    out = {}
    for s in cfg["subjects"]:
        d = np.load(cfg["root"] / f"sub{s}.npz")
        amp = d["x_data"][:, :, t0:t1].mean(axis=2, dtype=np.float64)
        out[s] = (amp.astype(np.float32), d["y_data"].astype(np.int64).ravel())
        print(f"[{dataset}] sub{s}: {amp.shape}", flush=True)
    np.savez(cache, **{f"amp{s}": a for s, (a, _) in out.items()}, **{f"y{s}": y for s, (_, y) in out.items()})
    return out


def load_predictions(dataset: str, data: dict[int, tuple[np.ndarray, np.ndarray]]) -> dict[str, dict[int, np.ndarray]]:
    preds: dict[str, dict[int, np.ndarray]] = {}
    dirs = {label: PRED_DIR_OVERRIDE.get((label, dataset), PRED_DIR / key / dataset) for key, label in MODELS.items()}
    dirs.update({label: d[dataset] for label, d in EXTRA_PRED_DIRS.items() if d[dataset].exists()})
    for label, pred_dir in dirs.items():
        preds[label] = {}
        for s, (_, y) in data.items():
            z = np.load(pred_dir / f"sub{s}.npz")
            if not np.array_equal(z["target"], y):
                raise RuntimeError(f"Trial order mismatch: {label} {dataset} sub{s}")
            preds[label][s] = z["prediction"]
    return preds


# ------------------------------------------------------------------- measures


def dprime_map(amp: np.ndarray, y: np.ndarray, signed: bool = False) -> np.ndarray:
    pos, neg = amp[y == 1].astype(np.float64), amp[y == 0].astype(np.float64)
    n_p, n_n = len(pos), len(neg)
    pooled = (pos.var(0, ddof=1) * (n_p - 1) + neg.var(0, ddof=1) * (n_n - 1)) / max(n_p + n_n - 2, 1)
    d = (pos.mean(0) - neg.mean(0)) / np.sqrt(np.maximum(pooled, 1e-10))
    return d if signed else np.abs(d)


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return 0.5 * (np.mean(pred[y == 1] == 1) + np.mean(pred[y == 0] == 0))


def loo_templates(maps: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    ids = list(maps)
    total = np.sum([maps[s] for s in ids], axis=0)
    return {s: (total - maps[s]) / (len(ids) - 1) for s in ids}


def axes_from(map_: np.ndarray, abs_map: np.ndarray, template: np.ndarray) -> tuple[float, float]:
    """s_ERP is always mean |d'|; r_topo uses `map_` (abs or signed) vs matching template."""
    return float(abs_map.mean()), float(np.corrcoef(map_, template)[0, 1])


def first_k_target_span(y: np.ndarray, k: int) -> np.ndarray:
    pos = np.flatnonzero(y == 1)
    return np.arange(pos[k - 1] + 1) if len(pos) >= k else np.arange(0)


def spearman_brown(r: np.ndarray) -> np.ndarray:
    return 2 * r / (1 + r)


# ---------------------------------------------------------------- statistics


def zwithin(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = df.groupby("dataset")[c].transform(lambda v: (v - v.mean()) / v.std(ddof=1))
    return out


def rcs_basis(x: np.ndarray, n_knots: int = 4) -> np.ndarray:
    """Harrell restricted cubic spline basis (linear term + n_knots-2 nonlinear terms)."""
    knots = np.quantile(x, np.linspace(0.05, 0.95, n_knots))
    k, kk = knots, len(knots)
    pos3 = lambda v: np.maximum(v, 0.0) ** 3
    cols = [x]
    for j in range(kk - 2):
        cols.append(
            (pos3(x - k[j]) - pos3(x - k[kk - 2]) * (k[kk - 1] - k[j]) / (k[kk - 1] - k[kk - 2])
             + pos3(x - k[kk - 1]) * (k[kk - 2] - k[j]) / (k[kk - 1] - k[kk - 2])) / (k[kk - 1] - k[0]) ** 2
        )
    return np.column_stack(cols)


def increment_test(
    y: np.ndarray, base: np.ndarray, add: np.ndarray, groups: np.ndarray, n_perm: int, rng: np.random.Generator
) -> dict[str, float]:
    """Out-of-sample R2 gained by `add` over `base`; null permutes `add` within dataset."""
    full = np.column_stack([base, add])
    r2_base, r2_full = loso_r2(base, y), loso_r2(full, y)
    gain = r2_full - r2_base
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    null = np.empty(n_perm)
    for i in range(n_perm):
        a = add.copy()
        for g in idx:
            a[g] = add[rng.permutation(g)]
        null[i] = loso_r2(np.column_stack([base, a]), y) - r2_base
    beta, se, _ = ols(full, y)
    pr, pr_p = partial_r(y, add, base[:, 1:])
    return {
        "R2_LOSO_base": r2_base,
        "R2_LOSO_full": r2_full,
        "dR2_LOSO": gain,
        "dR2_perm_p": float((1 + np.sum(null >= gain)) / (1 + n_perm)),
        "beta_per_sd": beta[-1],
        "p_beta": float(2 * stats.t.sf(abs(beta[-1] / se[-1]), len(y) - full.shape[1])),
        "partial_r": pr,
        "partial_r_p": pr_p,
    }


def fe_matrix(datasets: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(datasets)), pd.get_dummies(datasets, drop_first=True).to_numpy(dtype=float)])


def topo_increment_over_models(
    table: pd.DataFrame, ba: pd.DataFrame, erp_col: str, topo_col: str, n_perm: int, rng, extra_base=None,
    scale: str = "BA",
) -> pd.DataFrame:
    """For every model: BA ~ FE + z(erp) [+ extra] + z(topo); returns increment of topo.

    `table` holds one row per (dataset, subject_id) with the axes; `ba` the
    matching BA per model in columns.
    """
    t = zwithin(table, [erp_col, topo_col])
    groups = t["dataset"].to_numpy()
    base = np.column_stack([fe_matrix(groups), t[erp_col].to_numpy()])
    if extra_base is not None:
        base = np.column_stack([base, extra_base(t)])
    rows = []
    for model in ba.columns:
        y = ba[model].to_numpy()
        if scale == "logit":
            y = logit(np.clip(y, 1e-3, 1 - 1e-3))
        rows.append({"model": model, "n": len(y), **increment_test(y, base, t[topo_col].to_numpy(), groups, n_perm, rng)})
    return pd.DataFrame(rows)


def erp_increment_over_models(table: pd.DataFrame, ba: pd.DataFrame, erp_col: str, n_perm: int, rng) -> pd.DataFrame:
    """For every model: BA ~ FE + z(erp); returns increment of s_ERP over dataset FE alone."""
    t = zwithin(table, [erp_col])
    groups = t["dataset"].to_numpy()
    base = fe_matrix(groups)
    rows = []
    for model in ba.columns:
        y = ba[model].to_numpy()
        rows.append({"model": model, "n": len(y), **increment_test(y, base, t[erp_col].to_numpy(), groups, n_perm, rng)})
    return pd.DataFrame(rows)


def summarize_models(df: pd.DataFrame, label: dict) -> dict:
    prism = df[df["model"] == "PRISM"].iloc[0]
    return {
        **label,
        "PRISM_dR2_LOSO": prism["dR2_LOSO"],
        "PRISM_dR2_perm_p": prism["dR2_perm_p"],
        "PRISM_partial_r": prism["partial_r"],
        "PRISM_partial_r_p": prism["partial_r_p"],
        "median_dR2_LOSO": df["dR2_LOSO"].median(),
        "median_partial_r": df["partial_r"].median(),
        "min_partial_r": df["partial_r"].min(),
        "n_models_perm_p_lt_0.05": int((df["dR2_perm_p"] < 0.05).sum()),
        "n_models": len(df),
    }


# -------------------------------------------------------------------- analyses


def build_tables(all_data, all_preds) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Per-subject axes for every split condition plus per-model BA on each half."""
    axes_rows, ba_rows, sim_inputs = [], [], {}
    for dataset, data in all_data.items():
        abs_full = {s: dprime_map(a, y) for s, (a, y) in data.items()}
        sgn_full = {s: dprime_map(a, y, signed=True) for s, (a, y) in data.items()}
        abs_tpl, sgn_tpl = loo_templates(abs_full), loo_templates(sgn_full)
        halves = {s: (np.arange(len(y) // 2), np.arange(len(y) // 2, len(y))) for s, (_, y) in data.items()}
        abs_half = {h: {s: dprime_map(data[s][0][halves[s][h]], data[s][1][halves[s][h]]) for s in data} for h in (0, 1)}
        tpl_half = {h: loo_templates(abs_half[h]) for h in (0, 1)}
        sim_inputs[dataset] = (abs_tpl, abs_full)

        for s, (amp, y) in data.items():
            row = {"dataset": dataset, "subject_id": s}
            row["erp_full"], row["topo_full"] = axes_from(abs_full[s], abs_full[s], abs_tpl[s])
            _, row["topo_signed_full"] = axes_from(sgn_full[s], abs_full[s], sgn_tpl[s])
            for h, name in ((0, "A"), (1, "B")):
                idx = halves[s][h]
                a_map = abs_half[h][s]
                s_map = dprime_map(amp[idx], y[idx], signed=True)
                row[f"erp_{name}"], row[f"topo_{name}"] = axes_from(a_map, a_map, abs_tpl[s])
                _, row[f"topo_signed_{name}"] = axes_from(s_map, a_map, sgn_tpl[s])
            for k in CALIBRATION_K:
                idx = first_k_target_span(y[: len(y) // 2], k)
                a_map = dprime_map(amp[idx], y[idx])
                row[f"erp_K{k}"], row[f"topo_K{k}"] = axes_from(a_map, a_map, abs_tpl[s])
                row[f"span_K{k}"] = len(idx)
            # Split-half reliability of the subject map and of its LOO template.
            r_sub = np.corrcoef(abs_half[0][s], abs_half[1][s])[0, 1]
            r_tpl = np.corrcoef(tpl_half[0][s], tpl_half[1][s])[0, 1]
            row["map_split_r"] = r_sub
            row["map_reliability_SB"] = spearman_brown(r_sub)
            row["template_reliability_SB"] = spearman_brown(r_tpl)
            axes_rows.append(row)

            ba = {"dataset": dataset, "subject_id": s}
            for model, p in all_preds[dataset].items():
                for h, name in ((0, "A"), (1, "B")):
                    idx = halves[s][h]
                    ba[f"{model}|{name}"] = balanced_accuracy(y[idx], p[s][idx])
                ba[f"{model}|full"] = balanced_accuracy(y, p[s])
            ba_rows.append(ba)
    axes = pd.DataFrame(axes_rows)
    ba = pd.DataFrame(ba_rows)
    rel = axes["map_reliability_SB"].clip(lower=0.05)
    axes["topo_disattenuated"] = (axes["topo_full"] / np.sqrt(rel * axes["template_reliability_SB"])).clip(-1, 1)
    return axes, ba, sim_inputs


def ba_cols(ba: pd.DataFrame, part: str) -> pd.DataFrame:
    cols = [c for c in ba.columns if c.endswith(f"|{part}")]
    return ba[cols].rename(columns=lambda c: c.split("|")[0])


def check_full_ba(ba: pd.DataFrame) -> pd.DataFrame:
    """Recomputed full-data BA should equal the BA reported in results.csv."""
    from analyze_subject_bound_difficulty import load_architecture_panel

    rows = []
    for dataset in DATASETS:
        ref = load_architecture_panel(dataset)
        mine = ba_cols(ba[ba["dataset"] == dataset].set_index("subject_id"), "full")
        for model in mine.columns:
            diff = (mine[model] - ref.loc[mine.index, model]).abs()
            rows.append({"dataset": dataset, "model": model, "max_abs_BA_diff": float(diff.max())})
    return pd.DataFrame(rows)


def snr_simulation(all_data, sim_inputs, axes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sim_rows = []
    for dataset, data in all_data.items():
        abs_tpl, _ = sim_inputs[dataset]
        a_ds = axes[axes["dataset"] == dataset]
        typical = set(a_ds.loc[a_ds["topo_full"] >= a_ds["topo_full"].median(), "subject_id"])
        for s in typical:
            amp, y = data[s]
            amp = amp.astype(np.float64)
            shift = amp[y == 1].mean(0) - amp[y == 0].mean(0)
            for lam in LAMBDAS:
                sim = amp.copy()
                sim[y == 1] -= (1.0 - lam) * shift
                m = dprime_map(sim, y)
                erp, topo = axes_from(m, m, abs_tpl[s])
                sim_rows.append({"dataset": dataset, "subject_id": s, "lambda": lam, "erp": erp, "topo": topo})
    sim = pd.DataFrame(sim_rows)

    resid_rows = []
    for dataset, g in sim.groupby("dataset"):
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(g["erp"], g["topo"])
        # Spread of the SNR-only cloud: 5th percentile of residuals around the curve.
        lo = np.quantile(g["topo"] - iso.predict(g["erp"]), 0.05)
        a_ds = axes[axes["dataset"] == dataset]
        expected = iso.predict(a_ds["erp_full"])
        for (_, r), e in zip(a_ds.iterrows(), expected):
            resid_rows.append(
                {
                    "dataset": dataset,
                    "subject_id": r["subject_id"],
                    "erp_full": r["erp_full"],
                    "topo_full": r["topo_full"],
                    "topo_expected_from_snr": e,
                    "topo_snr_residual": r["topo_full"] - e,
                    "below_snr_band": bool(r["topo_full"] - e < lo),
                }
            )
    return sim, pd.DataFrame(resid_rows)


# ---------------------------------------------------------------------- report


def fmt(v, nd=3):
    if isinstance(v, (float, np.floating)):
        return f"{v:.1e}" if (abs(v) < 1e-3 and v != 0) else f"{v:.{nd}f}"
    return str(v)


def md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    cols = cols or list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for _, r in df[cols].iterrows()]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS / "diagnostic_circularity_reliability")
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    out = args.out_dir
    (out / "cache").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_data = {ds: load_window_amplitudes(ds, out / "cache") for ds in DATASETS}
    all_preds = {ds: load_predictions(ds, all_data[ds]) for ds in DATASETS}
    axes, ba, sim_inputs = build_tables(all_data, all_preds)
    axes.to_csv(out / "subject_axes_by_split.csv", index=False)
    ba.to_csv(out / "subject_ba_by_half.csv", index=False)
    ba_check = check_full_ba(ba)
    ba_check.to_csv(out / "ba_recompute_check.csv", index=False)

    detail, summary = [], []

    def run(label: dict, erp_col: str, topo_col: str, ba_part: str, rows=None, extra_base=None, scale="BA"):
        sel = axes.index if rows is None else rows
        res = topo_increment_over_models(
            axes.loc[sel].reset_index(drop=True), ba_cols(ba.loc[sel], ba_part).reset_index(drop=True),
            erp_col, topo_col, args.n_perm, rng, extra_base=extra_base, scale=scale,
        )
        detail.append(res.assign(**label))
        summary.append(summarize_models(res, label))

    # s_ERP alone over dataset FE, same trials vs disjoint halves.
    for erp_col, part, axes_lbl, ba_lbl in (
        ("erp_full", "full", "full", "full"),
        ("erp_A", "B", "first half", "second half"),
        ("erp_B", "A", "second half", "first half"),
    ):
        res = erp_increment_over_models(axes, ba_cols(ba, part), erp_col, args.n_perm, rng)
        label = {"analysis": "s_ERP over dataset FE", "axes": axes_lbl, "BA": ba_lbl}
        detail.append(res.assign(**label))
        summary.append(summarize_models(res, label))

    # Reference: full-data axes and BA (reproduces the circular analysis).
    run({"analysis": "0 reference (same trials)", "axes": "full", "BA": "full"}, "erp_full", "topo_full", "full")
    # 1. Circularity removed.
    run({"analysis": "1 disjoint halves", "axes": "first half", "BA": "second half"}, "erp_A", "topo_A", "B")
    run({"analysis": "1 disjoint halves", "axes": "second half", "BA": "first half"}, "erp_B", "topo_B", "A")
    run({"analysis": "1 disjoint halves", "axes": "first half", "BA": "second half (logit)"}, "erp_A", "topo_A", "B", scale="logit")
    run({"analysis": "1 disjoint halves, signed r_topo", "axes": "first half", "BA": "second half"}, "erp_A", "topo_signed_A", "B")
    no_things = axes.index[axes["dataset"] != "THINGS"]
    run({"analysis": "1 disjoint halves, THU+CAS", "axes": "first half", "BA": "second half"}, "erp_A", "topo_A", "B", rows=no_things)
    # 2. Calibration budget.
    for k in CALIBRATION_K:
        run({"analysis": "2 calibration", "axes": f"first {k} targets", "BA": "second half"}, f"erp_K{k}", f"topo_K{k}", "B")
    # 3. Reliability.
    run({"analysis": "3 disattenuated r_topo", "axes": "full", "BA": "full"}, "erp_full", "topo_disattenuated", "full")
    spline = lambda t: np.column_stack([rcs_basis(t["erp_full"].to_numpy())[:, 1:], rcs_basis(t["map_reliability_SB"].to_numpy())])
    run({"analysis": "3 spline(s_ERP) + spline(map reliability)", "axes": "full", "BA": "full"}, "erp_full", "topo_full", "full", extra_base=spline)
    spline_a = lambda t: np.column_stack([rcs_basis(t["erp_A"].to_numpy())[:, 1:], rcs_basis(t["map_reliability_SB"].to_numpy())])
    run({"analysis": "3 spline controls + disjoint halves", "axes": "first half", "BA": "second half"}, "erp_A", "topo_A", "B", extra_base=spline_a)

    # 4. SNR simulation.
    sim, resid = snr_simulation(all_data, sim_inputs, axes)
    sim.to_csv(out / "snr_simulation_curves.csv", index=False)
    resid.to_csv(out / "snr_residuals.csv", index=False)
    axes = axes.merge(resid[["dataset", "subject_id", "topo_snr_residual"]], on=["dataset", "subject_id"])
    run({"analysis": "4 r_topo residual from SNR-only curve", "axes": "full", "BA": "full"}, "erp_full", "topo_snr_residual", "full")

    detail_df = pd.concat(detail, ignore_index=True)
    summary_df = pd.DataFrame(summary)
    detail_df.to_csv(out / "topo_increment_by_model.csv", index=False)
    summary_df.to_csv(out / "topo_increment_summary.csv", index=False)

    # Reliability descriptives.
    rel_rows = []
    for dataset, g in axes.groupby("dataset", sort=False):
        rel_rows.append(
            {
                "dataset": dataset,
                "n": len(g),
                "s_ERP_halfA_vs_halfB_r": g["erp_A"].corr(g["erp_B"]),
                "r_topo_halfA_vs_halfB_r": g["topo_A"].corr(g["topo_B"]),
                "map_reliability_SB_median": g["map_reliability_SB"].median(),
                "map_reliability_SB_min": g["map_reliability_SB"].min(),
                "template_reliability_SB_median": g["template_reliability_SB"].median(),
                "spearman_map_reliability_vs_s_ERP": stats.spearmanr(g["map_reliability_SB"], g["erp_full"]).statistic,
                "spearman_r_topo_vs_s_ERP": stats.spearmanr(g["topo_full"], g["erp_full"]).statistic,
            }
        )
    rel_df = pd.DataFrame(rel_rows)
    rel_df.to_csv(out / "reliability_summary.csv", index=False)

    snr_rows = []
    for dataset, g in resid.groupby("dataset", sort=False):
        q25 = g["topo_full"].quantile(0.25)
        low = g[g["topo_full"] <= q25]
        s = sim[sim["dataset"] == dataset]
        snr_rows.append(
            {
                "dataset": dataset,
                "n_typical_simulated": s["subject_id"].nunique(),
                "obs_s_ERP_min": g["erp_full"].min(),
                "sim_r_topo_at_obs_s_ERP_min": float(np.interp(g["erp_full"].min(), *zip(*sorted(zip(s["erp"], s["topo"]))))),
                "sim_r_topo_lambda0_median": s.loc[s["lambda"] == 0, "topo"].median(),
                "n_low_r_topo_q25": len(low),
                "n_low_r_topo_below_snr_band": int(low["below_snr_band"].sum()),
                "n_all_below_snr_band": int(g["below_snr_band"].sum()),
                "spearman_snr_residual_vs_r_topo": stats.spearmanr(g["topo_snr_residual"], g["topo_full"]).statistic,
            }
        )
    snr_df = pd.DataFrame(snr_rows)
    snr_df.to_csv(out / "snr_simulation_summary.csv", index=False)

    report = [
        "# Diagnostic-axis circularity, reliability and SNR checks",
        "",
        "Pooled over THU, CAS, THINGS (n=122) unless noted; dataset fixed effects; axes z-scored within dataset. "
        "Each row tests the out-of-sample increment of r_topo over FE + s_ERP (+ listed controls) for all 10 models; "
        "null permutes r_topo within dataset. Halves are temporal (CAS: the two sessions). "
        "Rows `s_ERP over dataset FE` instead test s_ERP alone over FE (PRISM_* columns then refer to s_ERP).",
        "",
        f"BA recomputed from trial predictions vs results.csv: max |diff| = {ba_check['max_abs_BA_diff'].max():.1e}.",
        "",
        "## Summary over models",
        "",
        md_table(summary_df),
        "",
        "## Reliability",
        "",
        md_table(rel_df),
        "",
        "## SNR simulation (target mean shift scaled by lambda, real noise kept)",
        "",
        "`below_snr_band`: observed r_topo lower than the 5th percentile of the SNR-only cloud at the same s_ERP.",
        "",
        md_table(snr_df),
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(report))
    print(f"Wrote {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
