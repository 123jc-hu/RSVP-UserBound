"""Graphical abstract (IEEE specification: 660 x 295 px).

Writes figures/graphical_abstract.{pdf,png} (1980 x 885 px) and
figures/graphical_abstract_660x295.jpg. Inputs: the results archive and, for the
two example ERP waveforms, the preprocessed THU data.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_subject_bound_difficulty import load_architecture_panel  # noqa: E402
from Data.npz_io import ensure_npz_cache  # noqa: E402
from paths import ANALYSIS, DATA_DIRS, EXP  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from plot_figures import AXIS, BLUE, GRID, INK, INK2, ORANGE, SEQ  # noqa: E402

OUT = ROOT / "figures"
CIRC = ANALYSIS / "diagnostic_circularity_reliability"
FINETUNE = EXP / "PRISM/THU/within-subject/prism_thu_finetune_half_lr0.0001_s2026/results.csv"
PZ = 45  # index of Pz in the THU montage
FS = 128
W, H = 6.6, 2.95  # inches; 100 dpi gives 660 x 295 px


def quartile_mask(values: pd.Series) -> pd.Series:
    q = int(np.ceil(0.25 * len(values)))
    return values.index.isin(values.nsmallest(q).index)


def erp_waveforms(subject: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean target and non-target waveform at Pz (z-scored units)."""
    cache = ensure_npz_cache(DATA_DIRS["THU"] / f"sub{subject}.npz", ["x_data.npy", "y_data.npy"])
    x = np.load(cache / "x_data.npy", mmap_mode="r")
    y = np.load(cache / "y_data.npy").ravel()
    pz = np.asarray(x[:, PZ, :] if x.ndim == 3 else x[:, 0, PZ, :], dtype=np.float64)
    return pz[y == 1].mean(axis=0), pz[y == 0].mean(axis=0)


def panel_title(fig, x, number, text):
    fig.text(x, 0.862, number, fontsize=13, fontweight="bold", color=ORANGE, va="top")
    fig.text(x + 0.028, 0.862, text, fontsize=7.2, fontweight="bold", color=INK, va="top", linespacing=1.05)


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(length=2, color=AXIS, labelsize=5.6, labelcolor=INK2, pad=1.5)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.6})
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.945, "Calibration-free RSVP-EEG decoding: difficulty is bound to the user, not the decoder",
             ha="center", va="center", fontsize=9.6, fontweight="bold", color=INK)
    for x0 in (0.012, 0.340, 0.668):
        fig.add_artist(FancyBboxPatch((x0, 0.115), 0.320, 0.775, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, facecolor="#f7f7f5", edgecolor=GRID, lw=0.6,
                                      zorder=-1))

    # (1) same users are hardest for every decoder
    panel_title(fig, 0.022, "1", "The same users are hardest\nfor every decoder")
    wide = load_architecture_panel("THU")
    wide = wide[wide.mean().sort_values(ascending=False).index]
    wide = wide.loc[wide.mean(axis=1).sort_values(ascending=False).index]
    ax = fig.add_axes([0.060, 0.255, 0.205, 0.500])
    image = ax.imshow(wide.to_numpy(), aspect="auto", cmap=SEQ, vmin=0.6, vmax=1.0, interpolation="nearest")
    q = int(np.ceil(0.25 * len(wide)))
    ax.axhline(len(wide) - q - 0.5, color=ORANGE, lw=1.1)
    ax.text(len(wide.columns) - 0.4, len(wide) - q / 2 - 0.5, "lowest\nquartile", ha="left", va="center",
            fontsize=5.6, color=INK, fontweight="bold", linespacing=0.95)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("11 decoders (xDAWN-RG, EEGNet, ..., PRISM)", fontsize=5.6, color=INK2, labelpad=2)
    ax.set_ylabel("64 THU users, sorted", fontsize=5.6, color=INK2, labelpad=2)
    cax = fig.add_axes([0.302, 0.40, 0.007, 0.28])
    cbar = fig.colorbar(image, cax=cax, ticks=[0.6, 0.8, 1.0])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=1.5, labelsize=5, color=AXIS, labelcolor=INK2, pad=1)
    cbar.ax.set_title("BA", fontsize=5.4, color=INK2, pad=2)
    fig.text(0.172, 0.140, "Participants explain 77–94% of BA variance,\ndecoders only 2–11% (3 datasets)",
             ha="center", va="bottom", fontsize=5.8, color=INK, linespacing=1.1)

    # (2) neither better decoders nor calibration rescue them (THU, same second-half trials)
    panel_title(fig, 0.350, "2", "Better decoders and calibration\ndo not rescue them")
    halves = pd.read_csv(CIRC / "subject_ba_by_half.csv").query("dataset == 'THU'").set_index("subject_id")
    ft = pd.read_csv(FINETUNE)
    ft = ft[ft["SUB"].str.match(r"SUB\d+$")]
    ft.index = ft["SUB"].str[3:].astype(int)
    stages = pd.DataFrame({"EEGNet": halves["EEGNet|B"], "PRISM": halves["PRISM|B"], "tuned": ft["BA"]}).dropna()
    low = quartile_mask(stages["EEGNet"])
    ax = fig.add_axes([0.395, 0.275, 0.245, 0.480])
    xs = np.arange(3)
    for i, (_, row) in enumerate(stages.iterrows()):
        hard = low[i]
        ax.plot(xs, row.to_numpy(), color=ORANGE if hard else "#c9c8c1", lw=0.9 if hard else 0.5,
                alpha=1.0 if hard else 0.9, zorder=3 if hard else 2, solid_capstyle="round")
    stay = [int(quartile_mask(stages[c])[low].sum()) for c in stages.columns]
    labels = [f"{int(low.sum())} hardest"] + [f"{n}/{int(low.sum())} stay" for n in stay[1:]]
    for x, lab in zip(xs, labels):
        ax.text(x, 0.585, lab, ha="center", va="bottom", fontsize=5.8, color=ORANGE, fontweight="bold")
    ax.text(1.5, 0.556, "in the lowest quartile", ha="center", va="bottom", fontsize=5.0, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["EEGNet", "PRISM\n(best decoder)", "PRISM +\nown-data calib."], fontsize=5.6)
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylim(0.55, 1.0)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("BA (second half)", fontsize=5.6, color=INK2, labelpad=1.5)
    ax.grid(axis="y", color=GRID, lw=0.4)
    clean(ax)
    fig.text(0.504, 0.140, "Source alignment: no consistent gain.\nOwn-data calibration: only +0.016 BA",
             ha="center", va="bottom", fontsize=5.8, color=INK, linespacing=1.1)

    # (3) weak ERP strength explains the difficulty (disjoint trials, all 122 users)
    panel_title(fig, 0.678, "3", "Weak ERP strength\nexplains why")
    axes_df = pd.read_csv(CIRC / "subject_axes_by_split.csv")
    ba = pd.read_csv(CIRC / "subject_ba_by_half.csv")
    d = axes_df.merge(ba, on=["dataset", "subject_id"])
    d["x"] = d.groupby("dataset")["erp_A"].transform(lambda s: (s - s.mean()) / s.std())
    d["y"] = d.groupby("dataset")["PRISM|B"].transform(lambda s: s - s.mean())
    d["low"] = d.groupby("dataset")["PRISM|B"].transform(lambda s: quartile_mask(s))
    r = np.corrcoef(d["x"], d["y"])[0, 1]
    ax = fig.add_axes([0.712, 0.275, 0.150, 0.480])
    ax.scatter(d.loc[~d["low"], "x"], d.loc[~d["low"], "y"], s=7, color=BLUE, edgecolor="white", lw=0.3, zorder=3)
    ax.scatter(d.loc[d["low"], "x"], d.loc[d["low"], "y"], s=7, color=ORANGE, edgecolor="white", lw=0.3, zorder=4)
    slope, icpt = np.polyfit(d["x"], d["y"], 1)
    xl = np.linspace(d["x"].min(), d["x"].max(), 50)
    ax.plot(xl, icpt + slope * xl, color=INK, lw=0.8, zorder=5)
    ax.text(0.97, 0.04, f"r = {r:.2f}\n122 users\ndisjoint trials", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=5.6, color=INK, linespacing=1.05)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=3, color=ORANGE, mec="white", mew=0.3)],
              labels=["hardest quartile"], loc="lower right", bbox_to_anchor=(1.0, 0.25), fontsize=5.0,
              frameon=False, handletextpad=0.2, borderaxespad=0.2, labelcolor=INK2)
    ax.set_xlabel("ERP strength, 1st half (z)", fontsize=5.6, color=INK2, labelpad=1.5)
    ax.set_ylabel("BA, 2nd half (centered)", fontsize=5.6, color=INK2, labelpad=1.5)
    ax.grid(color=GRID, lw=0.4)
    clean(ax)

    thu = d[d["dataset"] == "THU"].set_index("subject_id")["erp_full"].sort_values()
    groups = {"strong ERP": (thu.index[-6:], BLUE), "weak ERP": (thu.index[:6], ORANGE)}
    t = np.arange(FS) / FS * 1000
    wax = fig.add_axes([0.885, 0.275, 0.098, 0.370])
    wax.axvspan(300, 700, color=GRID, lw=0, zorder=0)
    wax.axhline(0, color=AXIS, lw=0.5, zorder=1)
    for label, (subjects, color) in groups.items():
        diff = np.mean([np.subtract(*erp_waveforms(s)) for s in subjects], axis=0)
        wax.plot(t, diff, color=color, lw=1.1, zorder=3, label=f"{label}")
    wax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=5.0, frameon=False,
               handlelength=1.0, handletextpad=0.3, labelspacing=0.15, borderaxespad=0.1, labelcolor=INK2,
               title="Pz, target − non-target\n(6 THU users each)", title_fontsize=4.6)
    wax.set_xlim(0, 1000)
    wax.set_xticks([0, 500, 1000], ["0", "0.5", "1 s"])
    wax.set_yticks([])
    wax.tick_params(axis="x", labelsize=5.0)
    clean(wax)
    wax.spines["left"].set_visible(False)
    fig.text(0.828, 0.140, "→ Screen new users with\n~40 target trials (2–7 min)",
             ha="center", va="bottom", fontsize=5.8, linespacing=1.1, color=INK, fontweight="bold")

    fig.text(0.5, 0.045, "11 decoders  ·  3 source-alignment objectives  ·  3 public RSVP datasets  ·  "
             "122 participants  ·  no data from the held-out user", ha="center", va="center", fontsize=6.2,
             color=INK2)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "graphical_abstract.pdf", facecolor="white")
    fig.savefig(OUT / "graphical_abstract.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "graphical_abstract_660x295.png", dpi=100, facecolor="white")
    small = Image.open(OUT / "graphical_abstract_660x295.png").convert("RGB")
    for quality in range(92, 40, -4):  # IEEE recommends < 45 KB
        small.save(OUT / "graphical_abstract_660x295.jpg", quality=quality, optimize=True)
        if (OUT / "graphical_abstract_660x295.jpg").stat().st_size < 45_000:
            break
    (OUT / "graphical_abstract_660x295.png").unlink()
    print("r =", round(r, 3), "| stay in Q1:", stay)


if __name__ == "__main__":
    main()
