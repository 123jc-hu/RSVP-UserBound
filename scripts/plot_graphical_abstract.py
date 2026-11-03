"""Graphical abstract (IEEE specification: 660 x 295 px).

Layout: title, then Problem / Methods / Findings, then a conclusion bar. Writes
figures/graphical_abstract.{pdf,png} (1980 x 885 px) and
figures/graphical_abstract_660x295.jpg. Inputs: the results archive.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paths import ANALYSIS, EXP  # noqa: E402
from plot_figures import AXIS, BLUE, GRID, INK, INK2, ORANGE  # noqa: E402

OUT = ROOT / "figures"
CIRC = ANALYSIS / "diagnostic_circularity_reliability"
FINETUNE = EXP / "PRISM/THU/within-subject/prism_thu_finetune_half_lr0.0001_s2026/results.csv"
W, H = 6.6, 2.95  # inches; 100 dpi gives 660 x 295 px
PANEL = "#f7f7f5"
TITLE = ("User-Bound Limits of Calibration-Free RSVP-EEG Decoding:\n"
         "Evidence From Eleven Decoders and 122 Participants")


def quartile_mask(values: pd.Series) -> np.ndarray:
    q = int(np.ceil(0.25 * len(values)))
    return values.index.isin(values.nsmallest(q).index)


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(length=1.5, color=AXIS, labelsize=4.6, labelcolor=INK2, pad=1)


def section(fig, x0, x1, heading):
    fig.add_artist(FancyBboxPatch((x0, 0.170), x1 - x0, 0.665, boxstyle="round,pad=0,rounding_size=0.010",
                                  transform=fig.transFigure, facecolor=PANEL, edgecolor=GRID, lw=0.6, zorder=-1))
    fig.text(x0 + 0.010, 0.805, heading, fontsize=7.2, fontweight="bold", color=INK, va="center")
    fig.add_artist(plt.Line2D([x0 + 0.010, x0 + 0.045], [0.780, 0.780], transform=fig.transFigure,
                              color=ORANGE, lw=1.2))


def person(ax, x, y, s, color):
    ax.add_patch(Circle((x, y + 0.62 * s), 0.22 * s, facecolor=color, edgecolor="none"))
    ax.add_patch(FancyBboxPatch((x - 0.30 * s, y - 0.05 * s), 0.60 * s, 0.50 * s,
                                boxstyle=f"round,pad=0,rounding_size={0.22 * s}", facecolor=color, edgecolor="none"))


def arrow(ax, xy0, xy1):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle="-|>", mutation_scale=5, lw=0.8, color=INK2))


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.5})
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    halves = pd.read_csv(CIRC / "subject_ba_by_half.csv")
    axes_df = pd.read_csv(CIRC / "subject_axes_by_split.csv")

    # title
    fig.text(0.5, 0.920, TITLE, ha="center", va="center", fontsize=8.6, fontweight="bold", color=INK,
             linespacing=1.15)

    # ------------------------------------------------------------------ problem
    section(fig, 0.010, 0.262, "PROBLEM")
    fig.text(0.020, 0.755, "Calibration-free RSVP brain-computer\ninterfaces serve a new user without\n"
             "a recording session, but accuracy\nvaries widely between users.", fontsize=4.9, color=INK,
             va="top", linespacing=1.15)
    fig.text(0.020, 0.600, "Is the difficulty set by the\ndecoder, or by the user?", fontsize=5.4,
             fontweight="bold", color=INK, va="top", linespacing=1.15)
    ba = halves.set_index(["dataset", "subject_id"])["PRISM|full"]
    low = ba.groupby(level=0, group_keys=False).apply(lambda s: pd.Series(quartile_mask(s), index=s.index))
    ax = fig.add_axes([0.040, 0.250, 0.200, 0.210])
    bins = np.linspace(0.60, 1.0, 21)
    ax.hist([ba[~low], ba[low]], bins=bins, stacked=True, color=[BLUE, ORANGE], edgecolor=PANEL, lw=0.4)
    ax.set_xlim(0.60, 1.0)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Balanced accuracy of each user (best decoder)", fontsize=4.5, color=INK2, labelpad=1)
    ax.set_facecolor(PANEL)
    ax.text(0.62, ax.get_ylim()[1] * 0.92, "weak\ntail", fontsize=4.8, color=INK, fontweight="bold", va="top",
            linespacing=1.0)
    clean(ax)

    # ------------------------------------------------------------------ methods
    section(fig, 0.272, 0.540, "METHODS")
    x0, y0, w, h = 0.282, 0.470, 0.250, 0.280
    mx = fig.add_axes([x0, y0, w, h])
    mx.set_xlim(0, w * W)
    mx.set_ylim(0, h * H)
    mx.axis("off")
    for i in range(4):
        for j in range(3):
            person(mx, 0.10 + 0.105 * i, 0.20 + 0.17 * j, 0.13, BLUE if (i + j) % 3 else "#6da7ec")
    mx.text(0.255, 0.07, "121 source users", ha="center", va="center", fontsize=4.6, color=INK)
    arrow(mx, (0.53, 0.38), (0.66, 0.38))
    mx.add_patch(FancyBboxPatch((0.68, 0.17), 0.46, 0.45, boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor="white", edgecolor=AXIS, lw=0.6))
    mx.text(0.91, 0.395, "11 decoders\n+ 3 source-\nalignment\nobjectives", ha="center", va="center",
            fontsize=4.5, color=INK, linespacing=1.05)
    arrow(mx, (1.16, 0.38), (1.29, 0.38))
    mx.add_patch(FancyBboxPatch((1.31, 0.15), 0.30, 0.48, boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor="white", edgecolor=ORANGE, lw=0.7, ls=(0, (2, 1.2))))
    person(mx, 1.46, 0.30, 0.17, ORANGE)
    mx.text(1.46, 0.07, "new user\n(no data used)", ha="center", va="center", fontsize=4.4, color=INK,
            linespacing=1.0)
    bullets = ("3 public RSVP datasets, 122 participants",
               "xDAWN-RG + 10 deep nets; CORAL/MMD/DANN",
               "Strict leave-one-subject-out evaluation",
               "Calibration on each user's own data",
               "ERP descriptors from disjoint trials")
    for k, b in enumerate(bullets):
        fig.text(0.287, 0.435 - 0.046 * k, "•  " + b, fontsize=4.8, color=INK, va="top")

    # ------------------------------------------------------------------ findings
    section(fig, 0.550, 0.990, "FINDINGS")
    thu = halves.query("dataset == 'THU'").set_index("subject_id")
    ft = pd.read_csv(FINETUNE)
    ft = ft[ft["SUB"].str.match(r"SUB\d+$")]
    ft.index = ft["SUB"].str[3:].astype(int)
    stages = pd.DataFrame({"EEGNet": thu["EEGNet|B"], "PRISM": thu["PRISM|B"], "tuned": ft["BA"]}).dropna()
    hard = quartile_mask(stages["EEGNet"])
    ax = fig.add_axes([0.588, 0.420, 0.165, 0.300])
    xs = np.arange(3)
    for k, (_, row) in enumerate(stages.iterrows()):
        ax.plot(xs, row.to_numpy(), color=ORANGE if hard[k] else "#c9c8c1", lw=0.75 if hard[k] else 0.4,
                zorder=3 if hard[k] else 2)
    stay = [int(quartile_mask(stages[c])[hard].sum()) for c in stages.columns]
    for x, lab in zip(xs, [f"{int(hard.sum())}\nhardest", f"{stay[1]}/16\nstay", f"{stay[2]}/16\nstay"]):
        ax.text(x, 0.60, lab, ha="center", va="bottom", fontsize=4.4, color=ORANGE, fontweight="bold",
                linespacing=0.95)
    ax.set_xticks(xs)
    ax.set_xticklabels(["EEGNet", "PRISM", "PRISM +\ncalib."], fontsize=4.5)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(0.58, 1.0)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("BA (THU users)", fontsize=4.5, color=INK2, labelpad=1)
    ax.set_title("2  Same users stay hardest", fontsize=5.0, color=INK, fontweight="bold", loc="left", pad=3)
    ax.grid(axis="y", color=GRID, lw=0.35)
    ax.set_facecolor(PANEL)
    clean(ax)

    d = axes_df.merge(halves, on=["dataset", "subject_id"])
    d["x"] = d.groupby("dataset")["erp_A"].transform(lambda s: (s - s.mean()) / s.std())
    d["y"] = d.groupby("dataset")["PRISM|B"].transform(lambda s: s - s.mean())
    d["low"] = d.groupby("dataset")["PRISM|B"].transform(lambda s: quartile_mask(s))
    r = np.corrcoef(d["x"], d["y"])[0, 1]
    ax = fig.add_axes([0.802, 0.420, 0.170, 0.300])
    ax.scatter(d.loc[~d["low"], "x"], d.loc[~d["low"], "y"], s=3.5, color=BLUE, lw=0, zorder=3)
    ax.scatter(d.loc[d["low"], "x"], d.loc[d["low"], "y"], s=3.5, color=ORANGE, lw=0, zorder=4)
    slope, icpt = np.polyfit(d["x"], d["y"], 1)
    xl = np.linspace(d["x"].min(), d["x"].max(), 50)
    ax.plot(xl, icpt + slope * xl, color=INK, lw=0.7, zorder=5)
    ax.text(0.97, 0.05, f"r = {r:.2f}\n122 users", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=4.6, color=INK, linespacing=1.0)
    ax.set_xlabel("ERP strength (z)", fontsize=4.5, color=INK2, labelpad=1)
    ax.set_ylabel("BA (centered)", fontsize=4.5, color=INK2, labelpad=1)
    ax.set_title("3  ERP strength predicts BA", fontsize=5.0, color=INK, fontweight="bold", loc="left", pad=3)
    ax.grid(color=GRID, lw=0.35)
    ax.set_facecolor(PANEL)
    clean(ax)

    findings = ("1  Participants explain 77–94% of BA variance; decoders only 2–11%",
                "2  Better decoders, alignment, and calibration leave the same users hardest",
                f"3  Weak ERP strength explains the difficulty (r = {r:.2f}, disjoint trials)")
    for k, f in enumerate(findings):
        fig.text(0.562, 0.318 - 0.046 * k, f, fontsize=4.8, color=INK, va="top")
    fig.text(0.978, 0.805, "orange = hardest quartile of users", ha="right", va="center", fontsize=4.6, color=INK2)

    # ------------------------------------------------------------------ conclusion
    fig.add_artist(FancyBboxPatch((0.010, 0.020), 0.980, 0.130, boxstyle="round,pad=0,rounding_size=0.010",
                                  transform=fig.transFigure, facecolor="#fdf0ea", edgecolor=ORANGE, lw=0.6,
                                  zorder=-1))
    fig.text(0.022, 0.085, "CONCLUSION", fontsize=7.2, fontweight="bold", color=INK, va="center")
    fig.text(0.155, 0.085,
             "Difficulty in calibration-free RSVP-EEG decoding is bound to the user: users with weak ERPs stay "
             "hardest whatever the decoder,\nalignment, or calibration. New users should be screened with about "
             "40 target trials (2–7 min), and studies should report user-level accuracy.",
             fontsize=5.3, color=INK, va="center", linespacing=1.25)

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
