"""Figures of the manuscript.

No model is trained. Outputs PDF and PNG into figures/:
  fig_userbound    participant-by-decoder BA heatmaps with variance shares
  fig_descriptors  THU PRISM BA versus s_ERP and r_topo (no profile labels)
  fig_topography   THU ERP-separation maps by descriptor group (no profile labels)
  fig_snr          SNR simulation trajectories over observed participants

CAS uses PRISM with D = 8 (selected by source-validation BA), as reported in the manuscript.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_subject_bound_difficulty import (  # noqa: E402
    load_architecture_panel,
    load_axes,
    read_results_csv,
    two_way_decomposition,
)

from paths import ANALYSIS  # noqa: E402

OUT = ROOT / "figures"
CIRC = ANALYSIS / "diagnostic_circularity_reliability"
TOPO_BANK = ANALYSIS / "erp_topo_bank_thu.npz"
CAS_PRISM = "PRISM/CAS/cross-subject/prism_cas14_d8_bestsofar_prewsdrop010_regionconcat_lmr_m8"
DATASETS = ("THU", "CAS", "THINGS")
DISPLAY = {"THU": "THU", "CAS": "CAS", "THINGS": "THINGS-EEG"}
DECODER_LABEL = {"DC-tCNN-DG": "DC-tCNN", "EEGTransferNetDG": "EEGTransferNet"}

# Reference data-viz palette (light mode): ink, chrome, categorical slots 1-2, blue ramp.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
RED_RAMP = ["#fbd5d4", "#f2a3a2", "#e66767", "#e34948", "#b8302f"]
NEUTRAL = "#f0efec"
SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("blue_red", BLUE_RAMP[::-1][:5] + [NEUTRAL] + RED_RAMP)

THU_CHANNELS = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "O1", "CB1", "Oz",
    "O2", "CB2",
]
TOPO_MASK = [i for i, name in enumerate(THU_CHANNELS) if name not in {"CB1", "CB2"}]


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", facecolor="white", dpi=400)
    plt.close(fig)


def panel_letter(ax, letter: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK)


def ba_panel(dataset: str) -> pd.DataFrame:
    wide = load_architecture_panel(dataset)
    if dataset == "CAS":
        wide["PRISM"] = read_results_csv(CAS_PRISM)
    return wide


# ----------------------------------------------------------------- figure 2


def fig_userbound() -> None:
    panels = {ds: ba_panel(ds) for ds in DATASETS}
    order = pd.concat([p.mean() for p in panels.values()], axis=1).mean(axis=1).sort_values(ascending=False).index
    vmin = np.floor(min(p.values.min() for p in panels.values()) * 20) / 20
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.4), gridspec_kw={"wspace": 0.28})
    image = None
    for ax, (ds, wide), letter in zip(axes, panels.items(), "abc"):
        wide = wide[order]
        wide = wide.loc[wide.mean(axis=1).sort_values(ascending=False).index]
        dec = two_way_decomposition(wide.to_numpy())
        image = ax.imshow(wide.to_numpy(), aspect="auto", cmap=SEQ, vmin=vmin, vmax=1.0,
                          interpolation="nearest")
        q = int(np.ceil(0.25 * len(wide)))
        ax.axhline(len(wide) - q - 0.5, color=INK, lw=0.8, ls=(0, (3, 2)))
        ax.text(-0.2, len(wide) - q - 0.9, "lowest quartile", ha="left", va="bottom", fontsize=6,
                color=INK, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85})
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([DECODER_LABEL.get(m, m) for m in order], rotation=90)
        ax.set_yticks([0, len(wide) - 1])
        ax.set_yticklabels(["best", "worst"])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{DISPLAY[ds]} ($n$={len(wide)})", color=INK, pad=14)
        ax.text(0.5, 1.012, f"participant {dec['share_subject']:.1%}  |  decoder {dec['share_model']:.1%}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=6.3, color=INK2)
        panel_letter(ax, letter, x=-0.18, y=1.08)
    axes[0].set_ylabel("Held-out participants (sorted by mean BA)")
    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Balanced accuracy", color=INK2)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2, color=MUTED)
    save(fig, "fig_userbound")


# ----------------------------------------------------------------- descriptors


def fig_descriptors() -> None:
    ax_df = load_axes("THU")
    ba = load_architecture_panel("THU")["PRISM"]
    df = ax_df.join(ba.rename("BA"), how="inner")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3), gridspec_kw={"wspace": 0.38})
    for ax, col, label, letter in (
        (axes[0], "erp_strength", r"ERP strength, $s_\mathrm{ERP}$", "a"),
        (axes[1], "topo_corr_LOO", r"Spatial typicality, $r_\mathrm{topo}$", "b"),
    ):
        x, y = df[col].to_numpy(), df["BA"].to_numpy()
        slope, icpt, r, p, _ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, icpt + slope * xs, color=INK, lw=1.2, zorder=2)
        ax.scatter(x, y, s=16, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(0.04, 0.95, f"$r$ = {r:.2f}", transform=ax.transAxes, va="top", color=INK)
        ax.set_xlabel(label)
        ax.set_ylabel("Held-out BA (PRISM)")
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        panel_letter(ax, letter)
    ax = axes[2]
    q25 = df["erp_strength"].quantile(0.25)
    ax.axvline(q25, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.axhline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    sc = ax.scatter(df["erp_strength"], df["topo_corr_LOO"], c=df["BA"], cmap=SEQ, s=18,
                    edgecolor="white", linewidth=0.6, vmin=df["BA"].min(), vmax=df["BA"].max(), zorder=3)
    ax.set_xlabel(r"ERP strength, $s_\mathrm{ERP}$")
    ax.set_ylabel(r"Spatial typicality, $r_\mathrm{topo}$")
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Held-out BA", color=INK2)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2)
    panel_letter(ax, "c")
    save(fig, "fig_descriptors")


# ----------------------------------------------------------------- topography


def fig_topography() -> None:
    diag = load_axes("THU")
    q25 = diag["erp_strength"].quantile(0.25)
    adequate = diag["erp_strength"] >= q25
    groups = {
        "typical": diag.index[adequate & (diag["topo_corr_LOO"] >= 0)].tolist(),
        "atypical": diag.index[adequate & (diag["topo_corr_LOO"] < 0)].tolist(),
    }
    bank = np.load(TOPO_BANK)
    maps = {g: np.stack([bank[str(s)] for s in ids]).mean(axis=0)[TOPO_MASK] for g, ids in groups.items()}
    diff = maps["atypical"] - maps["typical"]

    info = mne.create_info([THU_CHANNELS[i] for i in TOPO_MASK], sfreq=128, ch_types="eeg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False,
                         on_missing="ignore")
    vmax = float(max(m.max() for m in maps.values()))
    dmax = float(np.abs(diff).max())

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.25), gridspec_kw={"wspace": 0.16})
    titles = {
        "typical": f"$r_\\mathrm{{topo}}\\geq0$ ($n$={len(groups['typical'])})",
        "atypical": f"$r_\\mathrm{{topo}}<0$ ($n$={len(groups['atypical'])})",
    }
    image = None
    for ax, g, letter in zip(axes[:2], ("typical", "atypical"), "ab"):
        image, _ = mne.viz.plot_topomap(maps[g], info, axes=ax, show=False, sensors=False, contours=5,
                                        outlines="head", extrapolate="head", cmap=SEQ,
                                        vlim=(0, vmax), sphere=0.095)
        mean_r = diag.loc[groups[g], "topo_corr_LOO"].mean()
        ax.set_title(f"{titles[g]}\nmean $r_\\mathrm{{topo}}$ = {mean_r:.2f}", color=INK, pad=4)
        panel_letter(ax, letter, x=-0.07, y=1.05)
    dimg, _ = mne.viz.plot_topomap(diff, info, axes=axes[2], show=False, sensors=False, contours=5,
                                   outlines="head", extrapolate="head", cmap=DIV,
                                   vlim=(-dmax, dmax), sphere=0.095)
    axes[2].set_title("Difference\n(b) minus (a)", color=INK, pad=4)
    panel_letter(axes[2], "c", x=-0.07, y=1.05)
    for img, axs, lab, shrink in ((image, axes[:2], r"Mean $\delta(c)$, 300--700 ms", 0.72),
                                  (dimg, axes[2], r"$\Delta\,\delta(c)$", 0.8)):
        cb = fig.colorbar(img, ax=axs, orientation="horizontal", fraction=0.055, pad=0.10, shrink=shrink)
        cb.set_label(lab, fontsize=6.5, color=INK2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6, length=2)
    fig.text(0.5, 0.98, r"THU participants with adequate ERP strength ($s_\mathrm{ERP}\geq Q_{25}$)",
             ha="center", fontsize=7.5, color=INK2)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.19, top=0.80)
    save(fig, "fig_topography")


# ----------------------------------------------------------------- SNR


def fig_snr() -> None:
    sim = pd.read_csv(CIRC / "snr_simulation_curves.csv")
    obs = pd.read_csv(CIRC / "snr_residuals.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4), gridspec_kw={"wspace": 0.3})
    for ax, ds, letter in zip(axes, DATASETS, "abc"):
        s, o = sim[sim["dataset"] == ds], obs[obs["dataset"] == ds]
        for _, g in s.groupby("subject_id"):
            g = g.sort_values("lambda")
            ax.plot(g["erp"], g["topo"], color=AXIS, lw=0.6, zorder=1)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(s["erp"], s["topo"])
        xs = np.linspace(s["erp"].min(), max(s["erp"].max(), o["erp_full"].max()), 200)
        ax.plot(xs, iso.predict(xs), color=INK, lw=1.6, zorder=2)
        low = o["topo_full"] <= o["topo_full"].quantile(0.25)
        ax.scatter(o.loc[~low, "erp_full"], o.loc[~low, "topo_full"], s=14, color=BLUE,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter(o.loc[low, "erp_full"], o.loc[low, "topo_full"], s=14, color=ORANGE,
                   edgecolor="white", linewidth=0.6, zorder=4)
        ax.axvline(o["erp_full"].min(), color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=0)
        ax.set_title(f"{DISPLAY[ds]} ($n$={len(o)})", color=INK)
        ax.set_xlabel(r"ERP strength, $s_\mathrm{ERP}$")
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.6, 1.0)
        panel_letter(ax, letter)
    axes[0].set_ylabel(r"Spatial typicality, $r_\mathrm{topo}$")
    handles = [
        Line2D([], [], color=AXIS, lw=0.8, label=r"Simulated typical participant ($\lambda$: 1$\to$0)"),
        Line2D([], [], color=INK, lw=1.6, label="Expected from SNR alone"),
        Line2D([], [], ls="", marker="o", ms=4.5, mfc=BLUE, mec="white", label="Observed participant"),
        Line2D([], [], ls="", marker="o", ms=4.5, mfc=ORANGE, mec="white",
               label=r"Observed, lowest quartile of $r_\mathrm{topo}$"),
        Line2D([], [], color=MUTED, lw=0.7, ls=(0, (3, 2)), label=r"Lowest observed $s_\mathrm{ERP}$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    save(fig, "fig_snr")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    style()
    fig_userbound()
    fig_descriptors()
    fig_topography()
    fig_snr()
    print("Figures written to", OUT)


if __name__ == "__main__":
    main()
