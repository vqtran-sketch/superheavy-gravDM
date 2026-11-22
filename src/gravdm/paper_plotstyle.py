"""paper_plotstyle.py — matplotlib style of the paper figures.

Usage
-----
    from gravdm import paper_plotstyle as ps
    ps.apply()                                        # once, before creating figures
    fig, ax = plt.subplots(figsize=(ps.W1, 2.4))      # W1 = one column, W2 = full page width
    ax.plot(x, y, color=ps.COL["c1"], lw=ps.LW["main"], label="model A")
    ax.plot(x, y2, color=ps.COL["c1"], lw=ps.LW["second"], ls="--", label="approximation")
    ax.axvline(x0, color=ps.COL["ref"], lw=ps.LW["ref"], ls=":")
    ps.panel(ax, "(a)", "upper left")
    ps.save(fig, "plots/01_name")                     # writes .pdf and .png

Conventions (fixed across all figures of the paper)
---------------------------------------------------
* Colours carry meaning: each physical case has one colour (c1..c6), reused in every figure.
  Variants of the same case are distinguished by line style, not by a new colour:
  solid = main result, dashed = alternative/approximation, dotted = third variant, dash-dot = comparison model.
* Line widths: LW["total"] for combined/joint/headline curves (black), LW["main"] for curves,
  LW["second"] for dashed/secondary ones, LW["ref"] for reference lines (thresholds, zero lines, grey).
* Shaded bands: COL["band"] with alpha=0.5, lw=0, zorder=0 (behind curves).
* Legends: frameless, 6.5 pt (crowded panels: fontsize=5.8). Panel labels "(a)", "(b)" via panel(), 8 pt.
* No titles inside single-panel figures; the caption carries benchmark parameters.
  Two-panel figure with a suptitle: use fig.subplots_adjust(top=0.91, wspace=0.28) instead of tight_layout.
* Text sizes: 8 pt labels, 7 pt tick labels.
* Always save both .pdf (vector, TrueType fonts) and .png (300 dpi).
"""
import pathlib
import matplotlib.pyplot as plt

# Figure widths in inches (RevTeX: \columnwidth = 3.375 in, \textwidth = 7.0 in)
W1, W2 = 3.375, 7.0

# Colour-blind-safe palette. c1..c6 = semantic slots, each fixed to one physical case.
COL = dict(
    c1="#2a78d6",          # blue
    c2="#eb6834",          # orange
    c3="#1baf7a",          # green
    c4="#8e5bb5",          # purple
    c5="#4477AA",          # blue-grey
    c6="#228833",          # dark green
    data="k",              # data points / headline combined curve
    ref="0.45",            # grey reference lines (thresholds, zero, asymptotes)
    shade_ref="0.5",       # grey shading
    band="#f4c7cf",        # highlighted range (use alpha=0.5)
    seq=("#9ecae1", "#4292c6", "#08519c"),   # sequential blues for a parameter scan (light -> dark = small -> large)
)
LW = dict(total=2.0, main=1.6, second=1.3, ref=0.6, data=0.8)
MS = 3.0


def apply():
    """Set rcParams: serif/Computer Modern math, small journal font sizes, inward ticks on all sides."""
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "legend.frameon": False, "legend.handlelength": 2.0, "legend.borderaxespad": 0.4,
        "axes.linewidth": 0.6,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "lines.linewidth": LW["main"], "lines.markersize": MS,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": 300,
        "pdf.fonttype": 42,                      # embed TrueType (journals reject Type 3)
    })


def panel(ax, s, loc="upper left"):
    """Panel label, e.g. panel(ax, '(a)', 'lower right')."""
    x, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    y, va = (0.95, "top") if "upper" in loc else (0.05, "bottom")
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=8)


def save(fig, stem):
    """Save <stem>.pdf and <stem>.png; creates the parent directory if needed."""
    p = pathlib.Path(stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(p.with_suffix(f".{ext}"))
