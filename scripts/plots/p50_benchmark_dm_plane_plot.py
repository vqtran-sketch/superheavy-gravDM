"""p50: SM Fig. S4, dark-matter curves of the flat-space benchmark (gravity-only Dirac X).

(a) T_reh vs M_X for Omega h^2 = 0.12, (b) present-day GW peak frequency (envelope approximation) along the same curves.
Loads output/50_benchmark_dm_plane.npz (script 50), curves with the yield above Lambda_cut = 0.46 M_P discarded.
Colors: H/beta = 0.1 red, 0.03 purple, 0.01 blue; Dphi = 3 L^{1/4} solid, 2 L^{1/4} dashed. Thick: core (conditions of SM Sec. S5:
collisions >= 1/1.1 of the total, R >= 0.1, M_X >= 10 beta, M_X < Lambda_cut/2); thin: outside the core. Dash-dot black: freeze-in only.
Shaded: M_X > Lambda_cut/2 = 0.23 M_P.
Output: plots/p50_benchmark_dm_plane.{pdf,png}"""
import sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import paper_plotstyle as ps
from gravdm.units import M_P

D = np.load(ROOT / "output/50_benchmark_dm_plane.npz")
CFG = dict(col={0.1: "#d1352b", 0.03: ps.COL["c4"], 0.01: ps.COL["c1"]}, ls={3.0: "-", 2.0: "--"}, xlim=(1e9, M_P), Mmax_frac=0.23,
           ylim_a=(3e10, 1e15), ylim_b=(1e4, 3e7))
ps.apply()
M = D["M"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ps.W2, 2.6))
for hb in (0.1, 0.03, 0.01):
    for nu in (3.0, 2.0):
        k = f"hb{hb:g}_nu{nu:g}_cut"   # yield above Lambda_cut = 0.46 M_P discarded
        T, core, fp = D[f"{k}_T"], D[f"{k}_core"], D[f"{k}_fpeak"]
        lab = rf"$H/\beta={hb:g}$" if nu == 3.0 else None
        ax1.plot(M, T, color=CFG["col"][hb], ls=CFG["ls"][nu], lw=ps.LW["ref"] + 0.2)
        Tc = np.where(core, T, np.nan)
        ax1.plot(M, Tc, color=CFG["col"][hb], ls=CFG["ls"][nu], lw=ps.LW["main"], label=lab)
        ax2.plot(M, np.where(core, fp, np.nan), color=CFG["col"][hb], ls=CFG["ls"][nu], lw=ps.LW["main"],
                 label=(rf"$H/\beta={hb:g}$, $h^2\Omega_{{\rm peak}}\sim10^{{{int(np.round(np.log10(D[f'hb{hb:g}_Omega_gw'])))}}}$"
                        if nu == 3.0 else None))
ax1.plot(M, D["T_FIonly"], color="k", ls="-.", lw=ps.LW["second"], label="freeze-in only")
ax1.plot([], [], color="0.3", ls="-", label=r"$\Delta\phi=3L^{1/4}$"); ax1.plot([], [], color="0.3", ls="--", label=r"$\Delta\phi=2L^{1/4}$")
for ax in (ax1, ax2):
    ax.axvspan(CFG["Mmax_frac"] * M_P, M_P, color=ps.COL["shade_ref"], alpha=0.25, lw=0, zorder=0)
    ax.set(xscale="log", yscale="log", xlim=CFG["xlim"], xlabel=r"$M_X$ [GeV]")
ax1.set(ylim=CFG["ylim_a"], ylabel=r"$T_{\rm reh}$ [GeV]")
ax2.set(ylim=CFG["ylim_b"], ylabel=r"$f_{\rm peak}$ [Hz] (envelope estimate)")
ax1.legend(loc="upper left", fontsize=5.8, ncol=2)
ax2.legend(loc="lower left", fontsize=5.8)
ps.panel(ax1, "(a)", "lower left"); ps.panel(ax2, "(b)", "upper right")
fig.tight_layout(); ps.save(fig, ROOT / "plots/p50_benchmark_dm_plane")
