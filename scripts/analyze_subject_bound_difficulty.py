"""Is strict-DG RSVP difficulty a property of the subject or of the model?

Uses only existing per-subject LOSO results (no retraining) to test whether
architecture changes and source-alignment objectives shift accuracy uniformly
while leaving the identity of hard subjects unchanged, and whether the two
post-hoc ERP axes (s_ERP, r_topo) describe that persistent difficulty.

Analyses, per dataset (THU / CAS / THINGS):
  A. Subject x model variance decomposition of BA (ICC(3,1), ICC(2,1)),
     separately for the architecture panel and the alignment panel.
  B. Rank stability: pairwise Spearman of per-subject BA and bottom-quartile
     overlap across models.
  C. BA ~ z(s_ERP) + z(r_topo) for every model: LOSO R2, partial r(r_topo|s_ERP).
  D. Gain analysis: dBA (PRISM - baseline, aligned - unaligned) ~ axes.
  E. Pooled gain analysis across datasets with dataset fixed effects.
  F. THU three-seed PRISM/PPNN: subject vs training-seed variance.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from paths import ANALYSIS, EXP

MAIN_COMPARISON = ANALYSIS / "main_comparison/subject_level_metrics.csv"
MULTISEED = ANALYSIS / "multiseed_prism_ppnn/subject_seed_metrics.csv"
# Per-participant s_ERP and r_topo computed from all trials by analyze_diagnostic_circularity_reliability.py.
DESCRIPTORS = ANALYSIS / "diagnostic_circularity_reliability/subject_axes_by_split.csv"

PRISM_FINAL = {
    "THU": "PRISM/THU/cross-subject/prism_thu64_d16_bestsofar_prewsdrop010_regionconcat_lmr_m8",
    "CAS": "PRISM/CAS/cross-subject/prism_cas14_d8_bestsofar_prewsdrop010_regionconcat_lmr_m8",
    # Manuscript THINGS row uses the montage-fixed run (BA 0.8261).
    "THINGS": "PRISM/THINGS/cross-subject/prism_things_strict44_d16_bestsofar_sb32_prewsdrop010_regionconcat_lmr_m8_montagefix",
}

EEGNET_BASE = {
    "THU": "EEGNet/THU/cross-subject/eegnet_thu64_k32_bgds1_sb32_ce",
    "CAS": "EEGNet/CAS/cross-subject/eegnet_cas14_k13_bgds1_sb32_ce",
    "THINGS": "EEGNet/THINGS/cross-subject/eegnet_things_strict44_spb32_sb32_bgds1_ce",
}

# Manuscript alignment settings: lambda=10 for CORAL/MMD, lambda=0.1 for DANN.
XDAWN_RG = {ds: f"XdawnRiemann/{ds}/cross-subject/xdawnrg_{ds.lower()}_nf4_lwf_neg4_s2026" for ds in ("THU", "CAS", "THINGS")}

ALIGN_SUFFIX = {
    "CORAL": "_sourcebatch_coral_lam10",
    "MMD": "_sourcebatch_mmd_lam10",
    "DANN": "_sourcebatch_dann_lam01",
}


# --------------------------------------------------------------------------- io


def read_results_csv(rel_dir: str) -> pd.Series:
    path = EXP / rel_dir / "results.csv"
    df = pd.read_csv(path)
    df = df[df["SUB"].astype(str).str.startswith("SUB")]
    subj = df["SUB"].str.replace("SUB", "", regex=False).astype(int)
    return pd.Series(df["BA"].astype(float).to_numpy(), index=subj.to_numpy(), name="BA")


def load_architecture_panel(dataset: str) -> pd.DataFrame:
    """Wide subject x model BA table; PRISM replaced by the final LMR run."""
    df = pd.read_csv(MAIN_COMPARISON)
    df = df[(df["dataset"] == dataset) & (df["method"] != "PRISM")].copy()
    df["subject_id"] = df["subject"].str.replace("SUB", "", regex=False).astype(int)
    wide = df.pivot(index="subject_id", columns="method", values="BA")
    wide.insert(0, "PRISM", read_results_csv(PRISM_FINAL[dataset]))
    if (EXP / XDAWN_RG[dataset] / "results.csv").exists():
        wide["xDAWN-RG"] = read_results_csv(XDAWN_RG[dataset])
    return wide.sort_index()


def load_alignment_panel(dataset: str) -> pd.DataFrame:
    cols = {}
    for backbone, base in (("PRISM", PRISM_FINAL[dataset]), ("EEGNet", EEGNET_BASE[dataset])):
        cols[f"{backbone}"] = read_results_csv(base)
        for name, suffix in ALIGN_SUFFIX.items():
            cols[f"{backbone}+{name}"] = read_results_csv(base + suffix)
    return pd.DataFrame(cols).sort_index()


def load_axes(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(DESCRIPTORS)
    df = df[df["dataset"] == dataset].rename(columns={"erp_full": "erp_strength", "topo_full": "topo_corr_LOO"})
    return df.set_index("subject_id")[["erp_strength", "topo_corr_LOO"]].sort_index()


# ------------------------------------------------------------------ statistics


def two_way_decomposition(y: np.ndarray) -> dict[str, float]:
    """Two-way ANOVA without replication on an (n_subjects, k_models) matrix."""
    n, k = y.shape
    grand = y.mean()
    row = y.mean(axis=1, keepdims=True)
    col = y.mean(axis=0, keepdims=True)
    ss_subj = k * np.sum((row - grand) ** 2)
    ss_model = n * np.sum((col - grand) ** 2)
    ss_res = np.sum((y - row - col + grand) ** 2)
    ss_tot = ss_subj + ss_model + ss_res
    ms_subj = ss_subj / (n - 1)
    ms_model = ss_model / (k - 1)
    ms_res = ss_res / ((n - 1) * (k - 1))
    # Shrout & Fleiss ICC(3,1) consistency and ICC(2,1) absolute agreement.
    icc31 = (ms_subj - ms_res) / (ms_subj + (k - 1) * ms_res)
    icc21 = (ms_subj - ms_res) / (ms_subj + (k - 1) * ms_res + k * (ms_model - ms_res) / n)
    return {
        "share_subject": ss_subj / ss_tot,
        "share_model": ss_model / ss_tot,
        "share_interaction": ss_res / ss_tot,
        "ICC31_consistency": icc31,
        "ICC21_agreement": icc21,
    }


def bootstrap_decomposition(y: np.ndarray, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    n = y.shape[0]
    rows = [two_way_decomposition(y[rng.integers(0, n, n)]) for _ in range(n_boot)]
    return pd.DataFrame(rows)


def ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns coefficients, standard errors and hat-matrix diagonal."""
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    dof = max(x.shape[0] - x.shape[1], 1)
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    hat = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    return beta, se, hat


def loso_r2(x: np.ndarray, y: np.ndarray) -> float:
    beta, _, hat = ols(x, y)
    loo_resid = (y - x @ beta) / (1.0 - hat)
    return 1.0 - np.sum(loo_resid**2) / np.sum((y - y.mean()) ** 2)


def partial_r(y: np.ndarray, target: np.ndarray, covars: np.ndarray) -> tuple[float, float]:
    c = np.column_stack([np.ones_like(y), covars])
    ry = y - c @ np.linalg.lstsq(c, y, rcond=None)[0]
    rt = target - c @ np.linalg.lstsq(c, target, rcond=None)[0]
    r = float(np.corrcoef(ry, rt)[0, 1])
    dof = len(y) - c.shape[1] - 1
    t = r * np.sqrt(dof / max(1e-12, 1 - r**2))
    return r, float(2 * stats.t.sf(abs(t), dof))


def axes_regression(
    y: np.ndarray, erp: np.ndarray, topo: np.ndarray, n_perm: int, rng: np.random.Generator
) -> dict[str, float]:
    z = lambda v: (v - v.mean()) / v.std(ddof=1)
    x = np.column_stack([np.ones_like(y), z(erp), z(topo)])
    beta, se, _ = ols(x, y)
    dof = len(y) - x.shape[1]
    p = 2 * stats.t.sf(np.abs(beta / se), dof)
    r2_in = 1.0 - np.sum((y - x @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
    r2_cv = loso_r2(x, y)
    perm = np.array([loso_r2(x, rng.permutation(y)) for _ in range(n_perm)])
    pr, pr_p = partial_r(y, topo, erp[:, None])
    return {
        "n": len(y),
        "mean": float(y.mean()),
        "R2_in": r2_in,
        "R2_LOSO": r2_cv,
        "R2_LOSO_perm_p": float((1 + np.sum(perm >= r2_cv)) / (1 + n_perm)),
        "beta_erp_per_sd": beta[1],
        "p_erp": p[1],
        "beta_topo_per_sd": beta[2],
        "p_topo": p[2],
        "partial_r_topo_given_erp": pr,
        "partial_r_p": pr_p,
        "spearman_erp": stats.spearmanr(y, erp).statistic,
        "spearman_topo": stats.spearmanr(y, topo).statistic,
    }


def rank_stability(wide: pd.DataFrame, reference: str) -> tuple[pd.DataFrame, dict[str, float]]:
    rho = wide.corr(method="spearman")
    off = rho.to_numpy()[np.triu_indices(len(rho), 1)]
    q = np.ceil(0.25 * len(wide)).astype(int)
    bottom = {m: set(wide[m].nsmallest(q).index) for m in wide.columns}
    jacc = [len(bottom[a] & bottom[b]) / len(bottom[a] | bottom[b]) for a, b in itertools.combinations(wide.columns, 2)]
    ref_overlap = [len(bottom[reference] & bottom[m]) / q for m in wide.columns if m != reference]
    # Chance overlap of two random size-q subsets of n: q/n.
    summary = {
        "spearman_mean": float(off.mean()),
        "spearman_min": float(off.min()),
        "bottom_quartile_size": int(q),
        "bottom_jaccard_mean": float(np.mean(jacc)),
        f"bottom_overlap_with_{reference}_mean": float(np.mean(ref_overlap)),
        "bottom_overlap_chance": float(q / len(wide)),
    }
    return rho, summary


# -------------------------------------------------------------------- analyses


def analyse_panel(
    dataset: str, panel_name: str, wide: pd.DataFrame, axes: pd.DataFrame, args, rng
) -> dict[str, list]:
    common = wide.dropna().index.intersection(axes.index)
    wide = wide.loc[common]
    ax = axes.loc[common]
    out: dict[str, list] = {"decomp": [], "rank": [], "axes": [], "gain": []}

    y = wide.to_numpy()
    dec = two_way_decomposition(y)
    boot = bootstrap_decomposition(y, args.n_boot, rng)
    row = {"dataset": dataset, "panel": panel_name, "n_subjects": len(wide), "n_models": wide.shape[1]}
    for key, val in dec.items():
        row[key] = val
        row[f"{key}_ci_low"] = float(boot[key].quantile(0.025))
        row[f"{key}_ci_high"] = float(boot[key].quantile(0.975))
    row["subject_sd_of_mean_BA"] = float(wide.mean(axis=1).std(ddof=1))
    row["model_range_of_mean_BA"] = float(wide.mean(axis=0).max() - wide.mean(axis=0).min())
    out["decomp"].append(row)

    rho, rsum = rank_stability(wide, reference=wide.columns[0])
    out["rank"].append({"dataset": dataset, "panel": panel_name, **rsum})
    rho.to_csv(args.out_dir / f"spearman_{dataset}_{panel_name}.csv")

    erp = ax["erp_strength"].to_numpy()
    topo = ax["topo_corr_LOO"].to_numpy()
    for model in wide.columns:
        res = axes_regression(wide[model].to_numpy(), erp, topo, args.n_perm, rng)
        out["axes"].append({"dataset": dataset, "panel": panel_name, "model": model, **res})

    # Gains: architecture panel -> PRISM minus every baseline;
    # alignment panel -> aligned minus its own unaligned backbone.
    if panel_name == "architecture":
        pairs = [("PRISM", m) for m in wide.columns if m != "PRISM"]
    else:
        pairs = [(c, c.split("+")[0]) for c in wide.columns if "+" in c]
    for new, ref in pairs:
        delta = (wide[new] - wide[ref]).to_numpy()
        res = axes_regression(delta, erp, topo, args.n_perm, rng)
        # Difficulty defined by models outside the pair limits regression to the mean.
        others = wide.drop(columns=[new, ref]).mean(axis=1).to_numpy()
        hard = others <= np.quantile(others, 0.25)
        res.update(
            {
                "dataset": dataset,
                "panel": panel_name,
                "comparison": f"{new} - {ref}",
                "delta_mean": float(delta.mean()),
                "delta_sd": float(delta.std(ddof=1)),
                "delta_hard_q1": float(delta[hard].mean()),
                "delta_rest": float(delta[~hard].mean()),
                "spearman_delta_vs_other_models_BA": stats.spearmanr(delta, others).statistic,
                "spearman_delta_vs_other_models_p": stats.spearmanr(delta, others).pvalue,
            }
        )
        out["gain"].append(res)
    return out, wide, ax


def logit(p: pd.Series) -> pd.Series:
    return np.log(p / (1.0 - p))


def pooled_gain(
    per_dataset: dict[str, tuple[pd.DataFrame, pd.DataFrame]], comparisons: list[tuple[str, str]], scale: str = "BA"
):
    """dBA ~ z(erp) + z(topo) + dataset fixed effects, axes z-scored within dataset.

    scale="logit" differences logit(BA) to remove ceiling compression near BA=1,
    which otherwise makes gains shrink mechanically for strong-ERP subjects.
    """
    f = logit if scale == "logit" else (lambda v: v)
    rows = []
    for new, ref in comparisons:
        ys, xs, ds = [], [], []
        for di, (dataset, (wide, ax)) in enumerate(per_dataset.items()):
            if new not in wide or ref not in wide:
                continue
            z = (ax - ax.mean()) / ax.std(ddof=1)
            ys.append((f(wide[new]) - f(wide[ref])).to_numpy())
            xs.append(z[["erp_strength", "topo_corr_LOO"]].to_numpy())
            ds.append(np.full(len(wide), dataset))
        if len(ys) < 2:
            continue
        y = np.concatenate(ys)
        x_ax = np.vstack(xs)
        d = np.concatenate(ds)
        dummies = pd.get_dummies(d, drop_first=True).to_numpy(dtype=float)
        x = np.column_stack([np.ones_like(y), x_ax, dummies])
        beta, se, _ = ols(x, y)
        p = 2 * stats.t.sf(np.abs(beta / se), len(y) - x.shape[1])
        pr, pr_p = partial_r(y, x_ax[:, 1], np.column_stack([x_ax[:, 0], dummies]))
        rows.append(
            {
                "scale": scale,
                "comparison": f"{new} - {ref}",
                "n": len(y),
                "datasets": ",".join(pd.unique(d)),
                "beta_erp_per_sd": beta[1],
                "p_erp": p[1],
                "beta_topo_per_sd": beta[2],
                "p_topo": p[2],
                "partial_r_topo": pr,
                "partial_r_topo_p": pr_p,
            }
        )
    return pd.DataFrame(rows)


def pooled_axes_regression(
    per_dataset: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    datasets: tuple[str, ...],
    n_perm: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """BA ~ dataset FE + z(erp) + z(topo), axes z-scored within dataset, per model.

    Incremental value of r_topo is tested out of sample: LOSO R2 of
    FE + erp + topo minus FE + erp, with r_topo permuted within dataset for the null.
    Slope homogeneity is tested with an F-test on topo x dataset interactions.
    """
    models = [m for m in per_dataset[datasets[0]][0].columns if all(m in per_dataset[d][0] for d in datasets)]
    rows = []
    for scale in ("BA", "logit"):
        f = logit if scale == "logit" else (lambda v: v)
        for model in models:
            ys, xs, ds = [], [], []
            for dataset in datasets:
                wide, ax = per_dataset[dataset]
                z = (ax - ax.mean()) / ax.std(ddof=1)
                ys.append(f(wide[model]).to_numpy())
                xs.append(z[["erp_strength", "topo_corr_LOO"]].to_numpy())
                ds.append(np.full(len(wide), dataset))
            y = np.concatenate(ys)
            erp, topo = np.vstack(xs).T
            d = np.concatenate(ds)
            fe = np.column_stack([np.ones_like(y), pd.get_dummies(d, drop_first=True).to_numpy(dtype=float)])

            x_fe, x_erp = fe, np.column_stack([fe, erp])
            x_full = np.column_stack([fe, erp, topo])
            beta, se, _ = ols(x_full, y)
            dof = len(y) - x_full.shape[1]
            p = 2 * stats.t.sf(np.abs(beta / se), dof)
            r2_fe, r2_erp, r2_full = (loso_r2(x, y) for x in (x_fe, x_erp, x_full))
            gain = r2_full - r2_erp

            groups = [np.flatnonzero(d == ds_) for ds_ in datasets]
            null = np.empty(n_perm)
            for i in range(n_perm):
                tp = topo.copy()
                for g in groups:
                    tp[g] = topo[rng.permutation(g)]
                null[i] = loso_r2(np.column_stack([fe, erp, tp]), y) - r2_erp

            # Slope homogeneity: add topo x dataset (and erp x dataset) interactions.
            inter = np.column_stack([fe[:, 1:] * topo[:, None], fe[:, 1:] * erp[:, None]])
            x_int = np.column_stack([x_full, inter])
            rss_full = np.sum((y - x_full @ ols(x_full, y)[0]) ** 2)
            rss_int = np.sum((y - x_int @ ols(x_int, y)[0]) ** 2)
            df_num, df_den = inter.shape[1], len(y) - x_int.shape[1]
            f_int = ((rss_full - rss_int) / df_num) / (rss_int / df_den)

            pr, pr_p = partial_r(y, topo, np.column_stack([fe[:, 1:], erp]))
            rows.append(
                {
                    "datasets": "+".join(datasets),
                    "scale": scale,
                    "model": model,
                    "n": len(y),
                    "R2_LOSO_FE": r2_fe,
                    "R2_LOSO_FE_erp": r2_erp,
                    "R2_LOSO_FE_erp_topo": r2_full,
                    "dR2_LOSO_topo": gain,
                    "dR2_LOSO_topo_perm_p": float((1 + np.sum(null >= gain)) / (1 + n_perm)),
                    "beta_erp_per_sd": beta[-2],
                    "p_erp": p[-2],
                    "beta_topo_per_sd": beta[-1],
                    "p_topo": p[-1],
                    "partial_r_topo": pr,
                    "partial_r_topo_p": pr_p,
                    "slope_heterogeneity_F": f_int,
                    "slope_heterogeneity_p": float(stats.f.sf(f_int, df_num, df_den)),
                }
            )
    return pd.DataFrame(rows)


def seed_decomposition(axes: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(MULTISEED)
    rows = []
    for model, g in df.groupby("model"):
        wide = g.pivot(index="subject", columns="seed", values="BA").dropna()
        dec = two_way_decomposition(wide.to_numpy())
        seed_sd_within_subject = float(wide.std(axis=1, ddof=1).mean())
        rows.append(
            {
                "dataset": "THU",
                "model": model,
                "n_subjects": len(wide),
                "n_seeds": wide.shape[1],
                "share_subject": dec["share_subject"],
                "share_seed": dec["share_model"],
                "share_residual": dec["share_interaction"],
                "ICC31_across_seeds": dec["ICC31_consistency"],
                "mean_within_subject_seed_sd": seed_sd_within_subject,
                "between_subject_sd": float(wide.mean(axis=1).std(ddof=1)),
            }
        )
    wide = df.pivot_table(index="subject", columns=["model", "seed"], values="BA")
    delta = (wide["PRISM"] - wide["PPNN"]).mean(axis=1)
    ax = axes.loc[delta.index]
    res = axes_regression(delta.to_numpy(), ax["erp_strength"].to_numpy(), ax["topo_corr_LOO"].to_numpy(), 0, np.random.default_rng(0))
    rows.append({"dataset": "THU", "model": "seed-averaged PRISM - PPNN vs axes", **{k: res[k] for k in ("n", "mean", "beta_erp_per_sd", "p_erp", "beta_topo_per_sd", "p_topo", "partial_r_topo_given_erp", "partial_r_p")}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- report


def fmt(v, nd=3):
    if isinstance(v, (float, np.floating)):
        if abs(v) < 1e-3 and v != 0:
            return f"{v:.1e}"
        return f"{v:.{nd}f}"
    return str(v)


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df[cols].iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report(out_dir: Path, decomp, rank, axes_df, gain, pooled, seeds, pooled_axes) -> None:
    parts = [
        "# Subject-bound difficulty analysis",
        "",
        "Existing strict-LOSO per-subject BA only; no retraining. Axes: pooled-SD |d'| "
        "ERP strength and LOO topographic correlation, 300-700 ms (manuscript primary definition).",
        "",
        "## A. Subject x model variance decomposition of BA",
        "",
        md_table(
            decomp,
            ["dataset", "panel", "n_subjects", "n_models", "share_subject", "share_model", "share_interaction",
             "ICC31_consistency", "ICC31_consistency_ci_low", "ICC31_consistency_ci_high", "ICC21_agreement",
             "subject_sd_of_mean_BA", "model_range_of_mean_BA"],
        ),
        "",
        "## B. Rank stability across models",
        "",
        md_table(rank, list(rank.columns)),
        "",
        "## C. BA ~ z(s_ERP) + z(r_topo), per model",
        "",
        md_table(
            axes_df,
            ["dataset", "panel", "model", "mean", "R2_LOSO", "R2_LOSO_perm_p", "beta_erp_per_sd", "p_erp",
             "beta_topo_per_sd", "p_topo", "partial_r_topo_given_erp", "partial_r_p"],
        ),
        "",
        "## D. Gains vs axes (per dataset)",
        "",
        "`delta_hard_q1`: mean gain in subjects whose BA, averaged over all models outside the pair, "
        "is in the bottom quartile.",
        "",
        md_table(
            gain,
            ["dataset", "comparison", "delta_mean", "delta_hard_q1", "delta_rest", "beta_erp_per_sd", "p_erp",
             "beta_topo_per_sd", "p_topo", "partial_r_topo_given_erp", "partial_r_p",
             "spearman_delta_vs_other_models_BA"],
        ),
        "",
        "## E. Pooled gains vs axes (dataset fixed effects, axes z-scored within dataset)",
        "",
        "`scale=logit` differences logit(BA); compare with `scale=BA` to separate a targeted gain "
        "from ceiling compression near BA=1.",
        "",
        md_table(pooled, list(pooled.columns)),
        "",
        "## F. THU training-seed variance (PRISM, PPNN; seeds 2026-2028)",
        "",
        md_table(seeds, [c for c in seeds.columns if seeds[c].notna().any()]).replace("nan", ""),
        "",
        "## G. Pooled BA ~ dataset FE + z(s_ERP) + z(r_topo), per model",
        "",
        "Axes z-scored within dataset. `dR2_LOSO_topo`: out-of-sample R2 gained by adding r_topo to "
        "FE + s_ERP; null permutes r_topo within dataset. `slope_heterogeneity_p`: F-test of axis x "
        "dataset interactions (small p = slopes differ across datasets).",
        "",
        md_table(
            pooled_axes,
            ["datasets", "scale", "model", "n", "R2_LOSO_FE_erp", "R2_LOSO_FE_erp_topo", "dR2_LOSO_topo",
             "dR2_LOSO_topo_perm_p", "beta_erp_per_sd", "p_erp", "beta_topo_per_sd", "p_topo",
             "partial_r_topo", "partial_r_topo_p", "slope_heterogeneity_p"],
        ),
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS / "subject_bound_difficulty")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    decomp, rank, axes_rows, gain = [], [], [], []
    arch_by_ds, align_by_ds = {}, {}
    for dataset in ("THU", "CAS", "THINGS"):
        axes = load_axes(dataset)
        for panel_name, loader, store in (
            ("architecture", load_architecture_panel, arch_by_ds),
            ("alignment", load_alignment_panel, align_by_ds),
        ):
            res, wide, ax = analyse_panel(dataset, panel_name, loader(dataset), axes, args, rng)
            store[dataset] = (wide, ax)
            decomp += res["decomp"]
            rank += res["rank"]
            axes_rows += res["axes"]
            gain += res["gain"]

    decomp = pd.DataFrame(decomp)
    rank = pd.DataFrame(rank)
    axes_df = pd.DataFrame(axes_rows)
    gain = pd.DataFrame(gain)

    arch_models = [c for c in arch_by_ds["THU"][0].columns if c != "PRISM"]
    pooled = pd.concat(
        [
            pooled_gain(arch_by_ds, [("PRISM", m) for m in arch_models], scale)
            for scale in ("BA", "logit")
        ]
        + [
            pooled_gain(align_by_ds, [(f"{b}+{a}", b) for b in ("PRISM", "EEGNet") for a in ALIGN_SUFFIX], scale)
            for scale in ("BA", "logit")
        ],
        ignore_index=True,
    )
    # Merge the two panels per dataset so alignment variants sit beside the architectures.
    merged = {
        ds: (arch_by_ds[ds][0].join(align_by_ds[ds][0].drop(columns=["PRISM", "EEGNet"]), how="inner"), arch_by_ds[ds][1])
        for ds in arch_by_ds
    }
    pooled_axes = pd.concat(
        [
            pooled_axes_regression(merged, ("THU", "CAS", "THINGS"), args.n_perm, rng),
            pooled_axes_regression(merged, ("THU", "CAS"), args.n_perm, rng),
        ],
        ignore_index=True,
    )
    seeds = seed_decomposition(load_axes("THU"))

    decomp.to_csv(args.out_dir / "variance_decomposition.csv", index=False)
    rank.to_csv(args.out_dir / "rank_stability.csv", index=False)
    axes_df.to_csv(args.out_dir / "axes_regression_per_model.csv", index=False)
    gain.to_csv(args.out_dir / "gain_vs_axes.csv", index=False)
    pooled.to_csv(args.out_dir / "gain_vs_axes_pooled.csv", index=False)
    seeds.to_csv(args.out_dir / "seed_variance_thu.csv", index=False)
    pooled_axes.to_csv(args.out_dir / "axes_regression_pooled.csv", index=False)
    write_report(args.out_dir, decomp, rank, axes_df, gain, pooled, seeds, pooled_axes)
    print(f"Wrote {args.out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
