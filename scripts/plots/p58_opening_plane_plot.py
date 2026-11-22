"""p58: main-text Fig. 3, reheating temperature that gives the observed DM density for Dirac-fermion DM.

Gravity-only Dirac X in the (M_X, T_reh) plane: relic curves (Omega h^2 = 0.12) for thermal gravitational freeze-in alone (dash-dotted)
and for first collisions of runaway bubbles plus freeze-in (solid), H/beta = 0.1, 0.03, 0.01 with Dphi = 3 L^{1/4} (colors as p50:
0.1 red, 0.03 purple, 0.01 blue). Thick: core (conditions of SM Sec. S5); thin: outside it. Pink: overproduced by freeze-in.
Gray: T_reh above the CMB bound (instantaneous reheating) and M_X > Lambda_cut/2. Dotted: M_X = T_reh.
Vertical dash-dotted: M_X = H_inf^max, the largest H_inf allowed by the CMB.
Loads output/58_opening_plane.npz (script 58). Output: plots/p58_opening_plane.{pdf,png}"""
import sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import paper_plotstyle as ps
from gravdm.units import M_P

D = np.load(ROOT / "output/58_opening_plane.npz")
CFG = dict(col={0.1: "#d1352b", 0.03: ps.COL["c4"], 0.01: ps.COL["c1"]}, nu=3.0, xlim=(1e8, M_P), ylim=(1e10, 1e17),
           Mmax_frac=0.23, show_Hinf=True)
ps.apply()
M = D["M"]
fig, ax = plt.subplots(figsize=(ps.W1, 2.7))
# excluded / untrusted regions
ax.axhspan(D["T_cmb_max"], CFG["ylim"][1], color=ps.COL["shade_ref"], alpha=0.25, lw=0, zorder=0)
ax.axvspan(CFG["Mmax_frac"] * M_P, CFG["xlim"][1], color=ps.COL["shade_ref"], alpha=0.25, lw=0, zorder=0)
ax.text(2e8, 1.6 * D["T_cmb_max"], "excluded by CMB tensor modes", fontsize=6, va="bottom")
ax.text(0.5 * M_P, 3e10, "above cutoff", rotation=90, fontsize=6, ha="center", va="bottom")
# overproduction by thermal freeze-in alone (instantaneous reheating; slower reheating, T_max > T_reh, raises freeze-in for
# M_X >~ T_reh, eq. (23) of Garny:2017kha, so the region is conservative there)
TFI = D["T_FI"]; okfi = np.isfinite(TFI) & (TFI < D["T_cmb_max"])
ax.fill_between(M, np.where(okfi, TFI, np.nan), D["T_cmb_max"], where=okfi, color=ps.COL["band"], alpha=0.5, lw=0, zorder=0)
# guide M_X = T_reh
g = np.logspace(10, 17, 10)
ax.plot(g, g, color=ps.COL["ref"], lw=ps.LW["ref"], ls=":")
if CFG["show_Hinf"]:
    ax.axvline(D["H_inf_max"], color=ps.COL["ref"], lw=ps.LW["ref"], ls="-.")
    ax.text(0.8 * D["H_inf_max"], 1.3e10, r"$H_{\rm inf}^{\max}$", fontsize=6, ha="right", va="bottom", color="0.35")
# relic lines
ax.plot(M, D["T_FI"], color="k", ls="-.", lw=ps.LW["second"], label="thermal freeze-in only")
for hb in (0.1, 0.03, 0.01):
    k = f"hb{hb:g}_nu{CFG['nu']:g}_cut"
    T, core = D[f"{k}_T"], D[f"{k}_core"]
    ax.plot(M, T, color=CFG["col"][hb], lw=ps.LW["ref"] + 0.2)
    ax.plot(M, np.where(core, T, np.nan), color=CFG["col"][hb], lw=ps.LW["main"], label=rf"collisions, $H/\beta={hb:g}$")
ax.set(xscale="log", yscale="log", xlim=CFG["xlim"], ylim=CFG["ylim"], xlabel=r"$M_X$ [GeV]", ylabel=r"$T_{\rm reh}$ [GeV]")
ax.legend(loc="upper left", fontsize=5.6, bbox_to_anchor=(0.0, 0.815), handlelength=2.6)
fig.tight_layout()
# label parallel to the M_X = T_reh guide: angle of the line in display coordinates (after layout)
p0, p1 = ax.transData.transform([(1e14, 1e14), (1e15, 1e15)])
ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
ax.text(1.2e11, 1.2e11 * 1.5, r"$M_X=T_{\rm reh}$", rotation=ang, rotation_mode="anchor", fontsize=6, color="0.35",
        ha="center", va="bottom")
# label parallel to the heavy branch of the freeze-in line, just above it
Ma, Mb = 1e15, 1e16
Ta, Tb = np.exp(np.interp(np.log([Ma, Mb]), np.log(M[okfi]), np.log(TFI[okfi])))
q0, q1 = ax.transData.transform([(Ma, Ta), (Mb, Tb)])
ax.text(5.0e15, 1.5 * np.exp(np.interp(np.log(5.0e15), np.log(M[okfi]), np.log(TFI[okfi]))), "overproduced by freeze-in",
        rotation=np.degrees(np.arctan2(q1[1] - q0[1], q1[0] - q0[0])), rotation_mode="anchor", fontsize=6, ha="center", va="bottom")
ps.save(fig, ROOT / "plots/p58_opening_plane")
