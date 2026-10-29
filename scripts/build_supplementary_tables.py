"""LaTeX tables of the supplementary material.

No model is trained. Every table is generated from existing LOSO results and the
analysis outputs of analyze_subject_bound_difficulty.py,
analyze_diagnostic_circularity_reliability.py and
scripts/multiseed_four_models.py and scripts/within_vs_cross_subject.py.
Writes supplementary_tables/*.tex.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_subject_bound_difficulty as sbd  # noqa: E402

from paths import EXP  # noqa: E402

AN = EXP / "analysis"
SBD = AN / "subject_bound_difficulty"
DCR = AN / "diagnostic_circularity_reliability"
SEEDS = AN / "multiseed_four_models"
CALIB = AN / "within_vs_cross_subject"
POSITIVE = AN / "positive_class_metrics/positive_class_metrics_summary.csv"
OUT = ROOT / "supplementary_tables"

DATASETS = ("THU", "CAS", "THINGS")
DISPLAY = {"THU": "THU", "CAS": "CAS", "THINGS": "THINGS-EEG"}
# Display names follow the main text.
NAME = {
    "DC-tCNN-DG": "DC-tCNN",
    "EEGTransferNetDG": "EEGTransferNet",
}
DECODER_ORDER = [
    "xDAWN-RG", "EEGNet", "PLNet", "PPNN", "DC-tCNN-DG", "EEGConformer",
    "EEGTransferNetDG", "EEG-DG", "SCLDGN", "KnIFE", "PRISM",
]
ALIGN_ORDER = ["EEGNet+CORAL", "EEGNet+DANN", "EEGNet+MMD", "PRISM+CORAL", "PRISM+DANN", "PRISM+MMD"]

ABLATION = {
    "THU": {
        "w/o grouped expansion": "PRISM/THU/cross-subject/prism_thu64_d16_bestsofar_prewsdrop_010",
        "w/o $\\mathbf{W}_s$": "PRISM/THU/cross-subject/prism_thu64_d16_bestsofar_prewsdrop010_regionconcat_lmr_m8_ablate_nows",
        "single-scale Stage~(iii)": "PRISM/THU/cross-subject/prism_thu64_d16_bestsofar_prewsdrop010_regionconcat_lmr_m8_ablate_stage2single",
        "single-scale Stage~(i)": "PRISM/THU/cross-subject/prism_thu64_d16_bestsofar_prewsdrop010_regionconcat_lmr_m8_ablate_stage1single",
        "single-scale both stages": "PRISM/THU/cross-subject/prism_thu64_d16_bestsofar_prewsdrop010_regionconcat_lmr_m8_ablate_bothsingle",
    },
    "CAS": {
        "w/o grouped expansion": "PRISM/CAS/cross-subject/prism_cas14_d8_bestsofar_prewsdrop_010",
        "w/o $\\mathbf{W}_s$": "PRISM/CAS/cross-subject/prism_cas14_d8_bestsofar_prewsdrop010_regionconcat_lmr_m8_ablate_nows",
        "single-scale Stage~(iii)": "PRISM/CAS/cross-subject/prism_cas14_d8_bestsofar_prewsdrop010_regionconcat_lmr_m8_ablate_stage2single",
        "single-scale Stage~(i)": "PRISM/CAS/cross-subject/prism_cas14_d8_bestsofar_prewsdrop010_regionconcat_lmr_m8_ablate_stage1single",
        "single-scale both stages": "PRISM/CAS/cross-subject/prism_cas14_d8_bestsofar_prewsdrop010_regionconcat_lmr_m8_ablate_bothsingle",
    },
    "THINGS": {
        "w/o grouped expansion": "PRISM/THINGS/cross-subject/prism_things_strict44_d16_bestsofar_sb32_prewsdrop_010",
        "w/o $\\mathbf{W}_s$": "PRISM/THINGS/cross-subject/prism_things_strict44_d16_bestsofar_sb32_prewsdrop010_regionconcat_lmr_m8_montagefix_ablate_nows",
        "single-scale Stage~(iii)": "PRISM/THINGS/cross-subject/prism_things_strict44_d16_bestsofar_sb32_prewsdrop010_regionconcat_lmr_m8_montagefix_ablate_stage2single",
        "single-scale Stage~(i)": "PRISM/THINGS/cross-subject/prism_things_strict44_d16_bestsofar_sb32_prewsdrop010_regionconcat_lmr_m8_montagefix_ablate_stage1single",
        "single-scale both stages": "PRISM/THINGS/cross-subject/prism_things_strict44_d16_bestsofar_sb32_prewsdrop010_regionconcat_lmr_m8_montagefix_ablate_bothsingle",
    },
}


# ------------------------------------------------------------------ helpers


def name(model: str) -> str:
    return NAME.get(model, model).replace("+", "$+$")


def fmt_p(p: float) -> str:
    if p < 1e-3:
        mantissa, exponent = f"{p:.1e}".split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"
    return f"{p:.3f}"


def signed(x: float, digits: int = 4) -> str:
    return f"${x:+.{digits}f}$"


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    q = np.empty(m)
    q[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def tabular(caption: str, label: str, spec: str, header: str, rows: list[str], small: str = "\\footnotesize") -> str:
    body = "\n".join(rows)
    return (
        "\\begin{table}[H]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n{small}\n"
        f"\\begin{{adjustbox}}{{max width=\\textwidth}}\n"
        f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n"
        "\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n"
    )


def write(stem: str, text: str) -> None:
    (OUT / f"{stem}.tex").write_text(text)
    print(f"wrote {stem}.tex")


# ------------------------------------------------------------------- tables


def benchmark_tests(panels: dict[str, pd.DataFrame]) -> None:
    rows = []
    for ds in DATASETS:
        w = panels[ds]
        comps = [m for m in DECODER_ORDER if m != "PRISM"]
        deltas = {m: w["PRISM"] - w[m] for m in comps}
        p = np.array([stats.wilcoxon(deltas[m], alternative="greater").pvalue for m in comps])
        q = bh(p)
        for i, m in enumerate(comps):
            d = deltas[m]
            prefix = f"\\multirow{{{len(comps)}}}{{*}}{{{DISPLAY[ds]}}}" if i == 0 else ""
            rows.append(
                f"{prefix} & {name(m)} & {signed(d.mean())} & {signed(d.median())} & "
                f"{d.mean() / d.std(ddof=1):.2f} & {int((d > 0).sum())}/{len(d)} & {fmt_p(q[i])} \\\\"
            )
        if ds != DATASETS[-1]:
            rows.append("\\midrule")
    write("benchmark_tests", tabular(
        "PRISM versus every other decoder (BA). One-sided participant-paired Wilcoxon tests with "
        "Benjamini--Hochberg correction over the ten comparisons within each dataset. $d_z$: mean "
        "paired difference divided by its standard deviation.",
        "tab:supp-benchmark-tests", "@{}llrrrrr@{}",
        "Dataset & Comparator & $\\Delta$BA & Median $\\Delta$ & $d_z$ & Wins & BH $q$", rows))


def xdawn_positive_metrics() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        pred_dir = EXP / sbd.XDAWN_RG[ds] / "trial_predictions"
        per = []
        for f in sorted(pred_dir.glob("sub*.npz")):
            z = np.load(f)
            t, yhat, s = z["target"].astype(int), z["prediction"].astype(int), z["positive_probability"]
            fpr, tpr, _ = roc_curve(t, s)
            per.append({
                "positive_precision": precision_score(t, yhat, zero_division=0),
                "positive_recall": recall_score(t, yhat, zero_division=0),
                "positive_F1": f1_score(t, yhat, zero_division=0),
                "AUPRC": average_precision_score(t, s),
                "sensitivity_at_FPR_0.05": tpr[fpr <= 0.05 + 1e-12].max(),
                "sensitivity_at_FPR_0.10": tpr[fpr <= 0.10 + 1e-12].max(),
                "false_positives_per_1000_nontarget": 1000.0 * ((t == 0) & (yhat == 1)).sum() / max((t == 0).sum(), 1),
            })
        means = pd.DataFrame(per).mean()
        rows.append({"model": "xDAWN-RG", "dataset": ds, **{f"{k}_mean": v for k, v in means.items()}})
    return pd.DataFrame(rows)


def positive_metrics() -> None:
    deep = pd.read_csv(POSITIVE)
    data = pd.concat([deep, xdawn_positive_metrics()], ignore_index=True)
    data["dataset"] = data["dataset"].replace({"THINGS-EEG": "THINGS"})
    rows = []
    for ds in DATASETS:
        sub = data[data.dataset == ds].set_index("model")
        for i, m in enumerate(DECODER_ORDER):
            r = sub.loc[m]
            prefix = f"\\multirow{{{len(DECODER_ORDER)}}}{{*}}{{{DISPLAY[ds]}}}" if i == 0 else ""
            rows.append(
                f"{prefix} & {name(m)} & {r.AUPRC_mean:.3f} & {r.positive_F1_mean:.3f} & "
                f"{r.positive_precision_mean:.3f} & {r.positive_recall_mean:.3f} & "
                f"{r['sensitivity_at_FPR_0.05_mean']:.3f} & {r['sensitivity_at_FPR_0.10_mean']:.3f} & "
                f"{r.false_positives_per_1000_nontarget_mean:.1f} \\\\"
            )
        if ds != DATASETS[-1]:
            rows.append("\\midrule")
    write("positive_metrics", tabular(
        "Positive-class and operating-point metrics (participant means) at the default threshold of "
        "0.5. Sens.@5\\%: sensitivity at a false-positive rate of 5\\%. FP/1k: false positives per "
        "1{,}000 non-target trials.",
        "tab:supp-positive", "@{}llrrrrrrr@{}",
        "Dataset & Decoder & AUPRC & $F_1^+$ & Prec.$^+$ & Rec.$^+$ & Sens.@5\\% & Sens.@10\\% & FP/1k", rows))


def seeds() -> None:
    summary = pd.read_csv(SEEDS / "seed_summary.csv").set_index("model")
    contrasts = pd.read_csv(SEEDS / "prism_contrasts.csv").set_index("contrast")
    rows = []
    for m in ("PRISM", "DC-tCNN", "PPNN", "EEGNet"):
        r = summary.loc[m]
        if m == "PRISM":
            lead = "---"
        else:
            c = contrasts.loc[f"PRISM-{m}"]
            lead = f"{signed(c.delta_3seed)} [{c.ci95_low:+.4f}, {c.ci95_high:+.4f}]"
        rows.append(
            f"{m} & {r.BA_seed2026:.4f} & {r.BA_seed2027:.4f} & {r.BA_seed2028:.4f} & {r.BA_3seed_mean:.4f} & "
            f"{r.within_subject_sd:.4f} & {r.between_subject_sd:.4f} & {r.icc31_seeds:.3f} & {lead} \\\\"
        )
    write("seeds", tabular(
        "THU training-seed robustness. Source splits and background downsampling were fixed; only the "
        "training seed changed. Within SD: mean over participants of the across-seed standard deviation. "
        "Between SD: standard deviation of seed-averaged participant BA. ICC(3,1) treats seeds as raters. "
        "PRISM lead: mean paired difference with a 20{,}000-replicate crossed participant--seed bootstrap "
        "95\\% interval.",
        "tab:supp-seeds", "@{}lrrrrrrrl@{}",
        "Decoder & 2026 & 2027 & 2028 & Mean & Within SD & Between SD & ICC & PRISM lead", rows, "\\scriptsize"))


def userbound() -> None:
    vd = pd.read_csv(SBD / "variance_decomposition.csv")
    rk = pd.read_csv(SBD / "rank_stability.csv")
    m = vd.merge(rk, on=["dataset", "panel"])
    rows = []
    for panel, label in (("architecture", "Eleven decoders"), ("alignment", "Alignment (8)")):
        for ds in DATASETS:
            r = m[(m.dataset == ds) & (m.panel == panel)].iloc[0]
            rows.append(
                f"{label if ds == 'THU' else ''} & {DISPLAY[ds]} & "
                f"{r.share_subject:.3f} [{r.share_subject_ci_low:.3f}, {r.share_subject_ci_high:.3f}] & "
                f"{r.share_model:.3f} & {r.share_interaction:.3f} & "
                f"{r.ICC31_consistency:.3f} [{r.ICC31_consistency_ci_low:.3f}, {r.ICC31_consistency_ci_high:.3f}] & "
                f"{r.ICC21_agreement:.3f} & {r.spearman_mean:.3f} ({r.spearman_min:.3f}) & "
                f"{r.bottom_overlap_with_PRISM_mean:.2f} ({r.bottom_overlap_chance:.2f}) \\\\"
            )
        if panel == "architecture":
            rows.append("\\midrule")
    write("userbound", tabular(
        "Two-way decomposition of BA variance and consistency of participant rankings. Brackets: 95\\% "
        "participant-bootstrap intervals (2{,}000 resamples). Spearman: mean (minimum) pairwise "
        "correlation. Overlap: mean fraction of PRISM's lowest-quartile participants in another "
        "configuration's lowest quartile (chance in parentheses). The alignment panel contains EEGNet and "
        "PRISM with and without CORAL, DANN, and MMD.",
        "tab:supp-userbound", "@{}llccccccc@{}",
        "Panel & Dataset & Participant & Configuration & Residual & ICC(3,1) & ICC(2,1) & Spearman & Overlap",
        rows, "\\scriptsize"))


def alignment(align_panels: dict[str, pd.DataFrame]) -> None:
    rows = []
    for cfg in ALIGN_ORDER:
        backbone = cfg.split("+")[0]
        cells = []
        for ds in DATASETS:
            w = align_panels[ds]
            d = w[cfg] - w[backbone]
            cells.append(
                f"{signed(d.mean())} & {fmt_p(stats.wilcoxon(d).pvalue)} & {stats.spearmanr(w[cfg], w[backbone])[0]:.3f}"
            )
        rows.append(f"{name(cfg)} & " + " & ".join(cells) + " \\\\")
    header = (
        "& \\multicolumn{3}{c}{THU} & \\multicolumn{3}{c}{CAS} & \\multicolumn{3}{c}{THINGS-EEG} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
        "Configuration & $\\Delta$BA & $p$ & $\\rho$ & $\\Delta$BA & $p$ & $\\rho$ & $\\Delta$BA & $p$ & $\\rho$"
    )
    write("alignment", tabular(
        "Effect of source alignment relative to the unaligned backbone. $p$: two-sided paired Wilcoxon "
        "test; $\\rho$: Spearman correlation of participant BA between the aligned and unaligned model.",
        "tab:supp-alignment", "@{}lrrrrrrrrr@{}", header, rows, "\\scriptsize"))


def gains() -> None:
    g = pd.read_csv(SBD / "gain_vs_axes_pooled.csv")
    g = g[g.comparison.str.startswith("PRISM - ")].copy()
    rows = []
    for scale in ("BA", "logit"):
        s = g[g.scale == scale].copy()
        s["q_erp"], s["q_topo"] = np.split(bh(np.r_[s.p_erp.to_numpy(), s.p_topo.to_numpy()]), 2)
        s["other"] = s.comparison.str.replace("PRISM - ", "", regex=False)
        s = s.set_index("other").loc[[m for m in DECODER_ORDER if m != "PRISM"]]
        for i, (m, r) in enumerate(s.iterrows()):
            prefix = f"\\multirow{{10}}{{*}}{{{scale}}}" if i == 0 else ""
            rows.append(
                f"{prefix} & {name(m)} & {r.beta_erp_per_sd:+.4f} & {fmt_p(r.p_erp)} & {fmt_p(r.q_erp)} & "
                f"{r.beta_topo_per_sd:+.4f} & {fmt_p(r.p_topo)} & {fmt_p(r.q_topo)} \\\\"
            )
        if scale == "BA":
            rows.append("\\midrule")
    write("gains", tabular(
        "Association of PRISM's per-participant gain over each decoder with the two descriptors (pooled, "
        "$n=122$, dataset fixed effects). $\\beta$: change in gain per within-dataset standard deviation "
        "of the descriptor. $q$: Benjamini--Hochberg correction over the 20 tests of each scale.",
        "tab:supp-gains", "@{}llrrrrrr@{}",
        "Scale & Comparator & $\\beta_{s_\\mathrm{ERP}}$ & $p$ & $q$ & $\\beta_{r_\\mathrm{topo}}$ & $p$ & $q$",
        rows, "\\scriptsize"))


def regressions() -> None:
    a = pd.read_csv(SBD / "axes_regression_pooled.csv")
    a = a[(a.datasets == "THU+CAS+THINGS") & (a.scale == "BA")].set_index("model")
    order = DECODER_ORDER[::-1] + ALIGN_ORDER
    rows = []
    for m in order:
        r = a.loc[m]
        rows.append(
            f"{name(m)} & {r.R2_LOSO_FE_erp:.3f} & {r.R2_LOSO_FE_erp_topo:.3f} & {r.dR2_LOSO_topo:+.3f} & "
            f"{fmt_p(r.dR2_LOSO_topo_perm_p)} & {r.partial_r_topo:.3f} & {fmt_p(r.partial_r_topo_p)} & "
            f"{r.slope_heterogeneity_p:.2f} \\\\"
        )
        if m == "xDAWN-RG":
            rows.append("\\midrule")
    write("regressions", tabular(
        "Pooled descriptor regressions of held-out BA for all 17 configurations ($n=122$, descriptors "
        "z-scored within dataset, dataset fixed effects). $R^2_\\mathrm{CV}$: leave-one-participant-out "
        "cross-validated $R^2$. Perm. $p$: 2{,}000 within-dataset permutations of $r_\\mathrm{topo}$. "
        "Slope $p$: $F$-test of descriptor-by-dataset interactions.",
        "tab:supp-regressions", "@{}lrrrrrrr@{}",
        "Configuration & $R^2_\\mathrm{CV}$ ($s_\\mathrm{ERP}$) & $R^2_\\mathrm{CV}$ (both) & "
        "$\\Delta R^2_\\mathrm{CV}$ & Perm. $p$ & Partial $r$ & $p$ & Slope $p$", rows, "\\scriptsize"))

    per = pd.read_csv(SBD / "axes_regression_per_model.csv")
    per = per[per.panel == "architecture"].set_index(["dataset", "model"])
    rows = []
    for ds in DATASETS:
        for i, m in enumerate(DECODER_ORDER[::-1]):
            r = per.loc[(ds, m)]
            prefix = f"\\multirow{{11}}{{*}}{{{DISPLAY[ds]}}}" if i == 0 else ""
            rows.append(
                f"{prefix} & {name(m)} & {r['mean']:.4f} & {r.R2_LOSO:.3f} & {r.beta_erp_per_sd:+.4f} & "
                f"{fmt_p(r.p_erp)} & {r.beta_topo_per_sd:+.4f} & {r.partial_r_topo_given_erp:.3f} & {fmt_p(r.partial_r_p)} \\\\"
            )
        if ds != DATASETS[-1]:
            rows.append("\\midrule")
    write("regressions_per_dataset", tabular(
        "Descriptor regressions within each dataset. $R^2_\\mathrm{CV}$ refers to the model with both "
        "descriptors; $\\beta$ is the change in BA per standard deviation.",
        "tab:supp-regressions-dataset", "@{}llrrrrrrr@{}",
        "Dataset & Decoder & BA & $R^2_\\mathrm{CV}$ & $\\beta_{s_\\mathrm{ERP}}$ & $p$ & "
        "$\\beta_{r_\\mathrm{topo}}$ & Partial $r$ & $p$", rows, "\\scriptsize"))


ROBUST_LABEL = {
    ("s_ERP over dataset FE", "full", "full"): "$s_\\mathrm{ERP}$ over dataset effects, same trials",
    ("s_ERP over dataset FE", "first half", "second half"): "$s_\\mathrm{ERP}$, first $\\to$ second half",
    ("s_ERP over dataset FE", "second half", "first half"): "$s_\\mathrm{ERP}$, second $\\to$ first half",
    ("0 reference (same trials)", "full", "full"): "$r_\\mathrm{topo}$, same trials (reference)",
    ("1 disjoint halves", "first half", "second half"): "Disjoint halves, first $\\to$ second",
    ("1 disjoint halves", "second half", "first half"): "Disjoint halves, second $\\to$ first",
    ("1 disjoint halves", "first half", "second half (logit)"): "Disjoint halves, first $\\to$ second, logit BA",
    ("1 disjoint halves, signed r_topo", "first half", "second half"): "Disjoint halves, signed maps",
    ("1 disjoint halves, THU+CAS", "first half", "second half"): "Disjoint halves, THU and CAS only",
    ("2 calibration", "first 10 targets", "second half"): "First 10 targets $\\to$ second half",
    ("2 calibration", "first 20 targets", "second half"): "First 20 targets $\\to$ second half",
    ("2 calibration", "first 40 targets", "second half"): "First 40 targets $\\to$ second half",
    ("2 calibration", "first 80 targets", "second half"): "First 80 targets $\\to$ second half",
    ("3 disattenuated r_topo", "full", "full"): "Disattenuated $r_\\mathrm{topo}$",
    ("3 spline(s_ERP) + spline(map reliability)", "full", "full"): "Spline control of $s_\\mathrm{ERP}$ and reliability",
    ("3 spline controls + disjoint halves", "first half", "second half"): "Spline control, disjoint halves",
    ("4 r_topo residual from SNR-only curve", "full", "full"): "Residual from SNR-only curve",
}


def robustness() -> None:
    t = pd.read_csv(DCR / "topo_increment_summary.csv")
    rows = []
    for i, r in t.iterrows():
        label = ROBUST_LABEL[(r["analysis"], r["axes"], r["BA"])]
        rows.append(
            f"{label} & {r.PRISM_dR2_LOSO:+.3f} & {fmt_p(r.PRISM_dR2_perm_p)} & {r.PRISM_partial_r:.3f} & "
            f"{fmt_p(r.PRISM_partial_r_p)} & {r.median_partial_r:.3f} & {r.min_partial_r:.3f} & "
            f"{int(r['n_models_perm_p_lt_0.05'])}/{int(r.n_models)} \\\\"
        )
        if i == 2:
            rows.append("\\midrule")
    write("robustness", tabular(
        "Robustness of the descriptor associations (pooled, $n=122$; THU and CAS only, $n=78$, where "
        "stated). The first three rows test $s_\\mathrm{ERP}$ over dataset effects; the others test "
        "$r_\\mathrm{topo}$ over $s_\\mathrm{ERP}$. $\\Delta R^2_\\mathrm{CV}$, permutation $p$, and partial "
        "$r$ refer to PRISM; median and minimum partial $r$ and the count of significant increments "
        "(permutation $p<0.05$) span the eleven decoders.",
        "tab:supp-robustness", "@{}lrrrrrrr@{}",
        "Analysis & $\\Delta R^2_\\mathrm{CV}$ & Perm. $p$ & Partial $r$ & $p$ & Median $r$ & Min. $r$ & Decoders",
        rows, "\\scriptsize"))


def serp_calibration() -> None:
    axes = pd.read_csv(DCR / "subject_axes_by_split.csv")
    halves = pd.read_csv(DCR / "subject_ba_by_half.csv")
    d = axes.merge(halves, on=["dataset", "subject_id"])
    models = [c.split("|")[0] for c in halves.columns if c.endswith("|B")]
    rows = []
    for k in (10, 20, 40, 80):
        x = d.groupby("dataset")[f"erp_K{k}"].transform(lambda v: (v - v.mean()) / v.std())
        res = {}
        for m in models:
            y = d[f"{m}|B"] - d.groupby("dataset")[f"{m}|B"].transform("mean")
            res[m] = stats.pearsonr(x, y)
        rs = np.array([v[0] for v in res.values()])
        ps = np.array([v[1] for v in res.values()])
        rows.append(
            f"{k} & {res['PRISM'][0]:.3f} & {fmt_p(res['PRISM'][1])} & {rs.min():.3f}--{rs.max():.3f} & {fmt_p(ps.max())} \\\\"
        )
    write("serp_calibration", tabular(
        "$s_\\mathrm{ERP}$ estimated from the first $K$ targets (and the backgrounds presented in the same "
        "span) versus BA on the second half of the participant's trials (pooled, $n=122$; descriptor "
        "z-scored and BA centered within dataset). Range and maximum $p$ span the eleven decoders.",
        "tab:supp-serp-calibration", "@{}rrrrr@{}",
        "$K$ & PRISM $r$ & $p$ & Range of $r$ & Max. $p$", rows))


def calibration() -> None:
    summary = pd.read_csv(CALIB / "summary.csv")
    label = {"PRISM-FT": "PRISM fine-tuned", "sLDA": "sLDA", "xDAWN-RG": "xDAWN-RG"}
    rows = []
    for ref in ("PRISM-FT", "sLDA", "xDAWN-RG"):
        for cond, cname in (("half", "first half"), ("K40", "first 40 targets")):
            cells = []
            for ds in DATASETS:
                r = summary[(summary.dataset == ds) & (summary.within == f"{ref}|{cond}")].iloc[0]
                cells.append(f"{r.within_BA:.3f} & {signed(r.gain_mean, 3)} & {r.spearman_within_vs_PRISM:.2f} & "
                             f"{r.q1_overlap:.2f} & {r.PRISM_q1_users_within_BA:.3f}")
            rows.append(f"{label[ref]} & {cname} & " + " & ".join(cells) + " \\\\")
        if ref != "xDAWN-RG":
            rows.append("\\midrule")
    free = summary[summary.within == "PRISM-FT|half"].set_index("dataset")
    rows.append("\\midrule")
    rows.append("PRISM calibration-free & --- & " + " & ".join(
        f"{free.loc[ds, 'PRISM_BA']:.3f} & --- & --- & --- & {free.loc[ds, 'PRISM_q1_users_PRISM_BA']:.3f}"
        for ds in DATASETS) + " \\\\")
    header = (
        "& & \\multicolumn{5}{c}{THU} & \\multicolumn{5}{c}{CAS} & \\multicolumn{5}{c}{THINGS-EEG} \\\\\n"
        "\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}\\cmidrule(lr){13-17}\n"
        "Reference & Calibration & " + " & ".join(["BA & $\\Delta$ & $\\rho$ & Ovl. & Q1"] * 3)
    )
    write("calibration", tabular(
        "Calibrated references on the second half of each user's trials (CAS: second session). "
        "$\\Delta$: difference from calibration-free PRISM on the same trials. $\\rho$: Spearman correlation "
        "with calibration-free PRISM. Ovl.: fraction of PRISM's calibration-free lowest quartile still in the "
        "lowest quartile (chance 0.25, 0.29, 0.25). Q1: BA of PRISM's calibration-free lowest-quartile users.",
        "tab:supp-calibration", "@{}ll" + "rrrrr" * 3 + "@{}", header, rows, "\\tiny"))


def reliability() -> None:
    rel = pd.read_csv(DCR / "reliability_summary.csv").set_index("dataset")
    snr = pd.read_csv(DCR / "snr_simulation_summary.csv").set_index("dataset")
    rows = []
    for ds in DATASETS:
        r, s = rel.loc[ds], snr.loc[ds]
        rows.append(
            f"{DISPLAY[ds]} & {int(r.n)} & {r.s_ERP_halfA_vs_halfB_r:.2f} & {r.r_topo_halfA_vs_halfB_r:.2f} & "
            f"{r.map_reliability_SB_median:.2f} & {r.template_reliability_SB_median:.2f} & "
            f"{int(s.n_typical_simulated)} & {s.sim_r_topo_at_obs_s_ERP_min:.2f} & {s.sim_r_topo_lambda0_median:.2f} & "
            f"{int(s.n_low_r_topo_below_snr_band)}/{int(s.n_low_r_topo_q25)} \\\\"
        )
    write("reliability", tabular(
        "Descriptor reliability and SNR simulation. Split-half $r$: correlation across participants "
        "between descriptors from the two temporal halves (for CAS, the two sessions recorded about 23 "
        "days apart). Map and template: median Spearman--Brown reliability of a participant's map and of "
        "the leave-one-out mean map. Simulated: spatially typical participants whose evoked shift was "
        "scaled down. $r_\\mathrm{topo}$ at min.: simulated value at the lowest observed $s_\\mathrm{ERP}$. "
        "Below band: lowest-quartile participants below the simulated SNR band.",
        "tab:supp-reliability", "@{}lrrrrrrrrr@{}",
        "Dataset & $n$ & $s_\\mathrm{ERP}$ & $r_\\mathrm{topo}$ & Map & Template & Simulated & "
        "$r_\\mathrm{topo}$ at min. & $\\lambda=0$ & Below band", rows, "\\scriptsize"))


def ablation(panels: dict[str, pd.DataFrame]) -> None:
    rows = [
        "PRISM & " + " & ".join(f"{panels[ds]['PRISM'].mean():.4f} & --- & ---" for ds in DATASETS) + " \\\\",
    ]
    cells = {ds: {} for ds in DATASETS}
    for ds in DATASETS:
        full = panels[ds]["PRISM"]
        labels = list(ABLATION[ds])
        deltas = [sbd.read_results_csv(ABLATION[ds][k]).loc[full.index] - full for k in labels]
        p = np.array([stats.wilcoxon(d).pvalue for d in deltas])
        for k, d, pv, ph in zip(labels, deltas, p, holm(p)):
            cells[ds][k] = f"{(full + d).mean():.4f} & {signed(d.mean())} & {fmt_p(ph)}"
    for k in ABLATION["THU"]:
        rows.append(f"\\quad {k} & " + " & ".join(cells[ds][k] for ds in DATASETS) + " \\\\")
    header = (
        "& \\multicolumn{3}{c}{THU} & \\multicolumn{3}{c}{CAS} & \\multicolumn{3}{c}{THINGS-EEG} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
        "Configuration & BA & $\\Delta$ & $p_\\mathrm{Holm}$ & BA & $\\Delta$ & $p_\\mathrm{Holm}$ & BA & $\\Delta$ & $p_\\mathrm{Holm}$"
    )
    write("ablation", tabular(
        "PRISM ablations. $\\Delta$: paired mean difference from the full model. $p_\\mathrm{Holm}$: "
        "two-sided paired Wilcoxon test, Holm-corrected over the five ablations within each dataset.",
        "tab:supp-ablation", "@{}lrrrrrrrrr@{}", header, rows, "\\scriptsize"))


GROUPED = {
    "THU": ("PRISM/THU/cross-subject/prism_thu64_d16_bestsofar_prewsdrop_010",
            ["PRISM/THU/cross-subject/prism_thu64_d16_bestsofar_prewsdrop010_regionconcat_randomlmr_m8_s%d" % s
             for s in range(2026, 2031)]),
    "CAS": ("PRISM/CAS/cross-subject/prism_cas14_d8_bestsofar_prewsdrop_010",
            ["PRISM/CAS/cross-subject/prism_cas14_d8_bestsofar_prewsdrop010_regionconcat_randomlmr_m8_s2026"]),
    "THINGS": ("PRISM/THINGS/cross-subject/prism_things_strict44_d16_bestsofar_sb32_prewsdrop_010",
               ["PRISM/THINGS/cross-subject/prism_things_strict44_d16_bestsofar_sb32_prewsdrop010_regionconcat_randomlmr_m8_s2026_montagefix"]),
}


def grouped_controls(panels: dict[str, pd.DataFrame]) -> None:
    """Left-midline-right grouping versus no grouping and parameter-matched random groupings."""
    rows = []
    for ds in DATASETS:
        anatomical = panels[ds]["PRISM"]
        no_group = sbd.read_results_csv(GROUPED[ds][0]).loc[anatomical.index]
        random = pd.concat([sbd.read_results_csv(r) for r in GROUPED[ds][1]], axis=1).mean(axis=1).loc[anatomical.index]
        comps = [("Anatomical $-$ no group", anatomical, no_group),
                 ("Anatomical $-$ random", anatomical, random),
                 ("Random $-$ no group", random, no_group)]
        p = np.array([stats.wilcoxon(a - b).pvalue for _, a, b in comps])
        for i, ((label, a, b), ph) in enumerate(zip(comps, holm(p))):
            d = (a - b).to_numpy()
            draws = np.random.default_rng(20260703).choice(d, size=(20_000, d.size), replace=True).mean(axis=1)
            lo, hi = np.quantile(draws, [0.025, 0.975])
            prefix = f"\\multirow{{3}}{{*}}{{{DISPLAY[ds]}}}" if i == 0 else ""
            rows.append(f"{prefix} & {label} & {signed(d.mean())} & {lo:+.4f} & {hi:+.4f} & {fmt_p(p[i])} & {fmt_p(ph)} \\\\")
        if ds != DATASETS[-1]:
            rows.append("\\midrule")
    write("grouped_control_tests", tabular(
        "Grouped expansion versus no grouping and parameter-matched random groupings of the same sizes "
        "(THU: mean of five random partitions; CAS and THINGS-EEG: one). Two-sided paired Wilcoxon tests, "
        "Holm-corrected over the three comparisons within each dataset; 95\\% intervals from 20{,}000 "
        "participant bootstrap replicates.",
        "tab:supp-grouped-tests", "@{}llrrrrr@{}",
        "Dataset & Comparison & $\\Delta$BA & 95\\% CI low & 95\\% CI high & Wilcoxon $p$ & Holm $p$", rows))


def per_participant(panels: dict[str, pd.DataFrame]) -> None:
    cols = DECODER_ORDER
    head = "Dataset & ID & $s_\\mathrm{ERP}$ & $r_\\mathrm{topo}$ & " + " & ".join(
        f"\\rotatebox{{90}}{{{name(m)}}}" for m in cols)
    lines = []
    for ds in DATASETS:
        axes = sbd.load_axes(ds)
        w = panels[ds][cols].join(axes, how="inner")
        for sid, r in w.iterrows():
            lines.append(
                f"{DISPLAY[ds]} & {sid} & {r.erp_strength:.3f} & {r.topo_corr_LOO:+.2f} & "
                + " & ".join(f"{100 * r[m]:.1f}" for m in cols) + " \\\\"
            )
    spec = "@{}llrr" + "r" * len(cols) + "@{}"
    n = 4 + len(cols)
    text = (
        "\\begingroup\n\\tiny\n\\setlength{\\tabcolsep}{2.6pt}\n"
        f"\\begin{{longtable}}{{{spec}}}\n"
        "\\caption{Descriptors and held-out BA (\\%) of every participant for the eleven decoders. "
        "Participant IDs follow the source datasets.}\\label{tab:supp-participants}\\\\\n"
        f"\\toprule\n{head} \\\\\n\\midrule\n\\endfirsthead\n"
        f"\\multicolumn{{{n}}}{{l}}{{\\tablename\\ \\thetable\\ (continued)}}\\\\\n"
        f"\\toprule\n{head} \\\\\n\\midrule\n\\endhead\n"
        "\\bottomrule\n\\endlastfoot\n" + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n"
    )
    write("per_participant", text)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    panels = {ds: sbd.load_architecture_panel(ds) for ds in DATASETS}
    align_panels = {ds: sbd.load_alignment_panel(ds) for ds in DATASETS}
    benchmark_tests(panels)
    positive_metrics()
    seeds()
    userbound()
    alignment(align_panels)
    gains()
    regressions()
    robustness()
    serp_calibration()
    calibration()
    reliability()
    ablation(panels)
    grouped_controls(panels)
    per_participant(panels)


if __name__ == "__main__":
    main()
