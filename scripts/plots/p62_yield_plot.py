"""p62: main-text Fig. 2, yield of Dirac-fermion DM in units of (T_reh/M_P)^3 against M_X/T_reh at T_reh = 1e11 GeV.

Solid: first collisions, main-text Eq. (5), with Dphi = 3 L^{1/4} and xi_phi = 0, for H/beta = 0.1 and 0.01:
  Y/(T/M_P)^3 = coeff (H/beta) R(M/Mbar_coll), coeff = Q_beta K/(960 pi) = 6.81e-3 (script 11), R from output/47 (Dirac, tanh),
  Mbar_coll = 11.16 (H/beta) M_P/nu^2 (main-text Eq. (6), nu = 3).
Dashed: thermal freeze-in with instantaneous reheating (output/46, T_max = T_reh).
Dotted: the yield M_X Y_X = MY_DM that gives the observed DM density, with circles at its crossings with the H/beta = 0.1 curve
  (relic solutions from output/49).
Dash-dotted: M_X = H_inf^max = 4.7e13 GeV, the CMB bound on the inflationary Hubble rate (script 58), at the same T_reh.
Gray: M_X > Lambda_cut/2. A top axis gives M_X in GeV.
Inputs: python scripts/compute/46_freezein_curve.py 1e11; python scripts/compute/49_universal_relic.py 1e11.
Output: plots/p62_yield.{pdf,png}"""
import sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as Gf
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import paper_plotstyle as ps
from gravdm.units import M_P, MY_DM

CFG = dict(Treh=1e11, Mmax_frac=0.23, Hbeta=[0.1, 0.01], colors=["#d1352b", ps.COL["c1"]], nu=3.0, Q_beta=1.0 / 3.0, spin="dirac", omega_label_x=1e4, xmin=3e-2, legend_anchor=(0.0, 0.95), omega_label_fac=3.0,
           gmh_coeff=2 * np.sqrt(3) * (Gf(4.5) / Gf(4.0)) ** 2 * 6 / 7)
CFG["coeff"] = CFG["Q_beta"] * float(np.load(ROOT / "output/11_universal_flat.npz")["K"]) / (960 * np.pi)
H_inf_max = float(np.load(ROOT / "output/58_opening_plane.npz")["H_inf_max"])
ps.apply()
S = np.load(ROOT / "output/47_R_spread.npz"); assert str(S["spin"]) == CFG["spin"] and str(S["profile"]) == "tanh"
R = lambda mu: np.exp(np.interp(np.log(mu), np.log(S["mu"]), np.log(np.maximum(S["R"], 1e-300)), left=0.0))
T = CFG["Treh"]; norm = (T / M_P) ** 3
TAG = "" if T == 1e12 else f"_T{T:.0e}".replace("+", "")          # inputs from scripts 46 and 49 run with argv[1] = T_reh
x = np.logspace(np.log10(CFG["xmin"]), np.log10(M_P / T), 400)
fig, ax = plt.subplots(figsize=(ps.W1, 2.7))
ax.axvspan(CFG["Mmax_frac"] * M_P / T, M_P / T, color=ps.COL["shade_ref"], alpha=0.25, lw=0, zorder=0)
ax.text(np.sqrt(CFG["Mmax_frac"]) * M_P / T, 3e-2, r"$M_X>\Lambda_{\rm cut}/2$", rotation=90, fontsize=6, color="0.3", ha="center", va="center")
for hb, col in zip(CFG["Hbeta"], CFG["colors"]):
    gt = CFG["gmh_coeff"] * hb * M_P / CFG["nu"] ** 2 / T
    e = int(np.floor(np.log10(gt)))
    ax.plot(x, CFG["coeff"] * hb * R(x / gt), color=col, lw=ps.LW["main"],
            label=rf"Bubble coll., $H/\beta={hb}$")
F = np.load(ROOT / f"output/46_freezein_curve{TAG}.npz"); assert float(F["Treh"]) == T and str(F["spin"]) == CFG["spin"]
ax.plot(F["x"], F["Y"], color="k", ls="--", lw=ps.LW["main"], label="thermal freeze-in")
ax.plot(x, MY_DM / (x * T * norm), color=ps.COL["ref"], ls=":", lw=ps.LW["second"])
C = np.load(ROOT / f"output/49_universal_relic{TAG}.npz"); assert float(C["Treh"]) == T and float(C["nu"]) == CFG["nu"]
for hb, col in zip(CFG["Hbeta"], CFG["colors"]):
    r = C[f"hb{hb:g}_roots"]; r = r[r < CFG["Mmax_frac"] * M_P]
    gt = CFG["gmh_coeff"] * hb * M_P / CFG["nu"] ** 2 / T
    core = R(r / T / gt) >= 0.1   # relic solutions inside the core (R >= 0.1) and outside it (profile-dependent tail), both as circles
    ax.plot(r[core] / T, MY_DM / (r[core] * norm), "o", mfc="none", mec=col, ms=4, mew=0.9, zorder=5)
    ax.plot(r[~core] / T, MY_DM / (r[~core] * norm), "o", mfc="none", mec=col, ms=4, mew=0.9, zorder=5)   # same symbol as inside the core
# end of inflation: efficient only below H_inf <= H_inf^max
ax.axvline(H_inf_max / T, color=ps.COL["ref"], lw=ps.LW["ref"], ls="-.")
ax.text(H_inf_max / T * 1.35, 2e-9, r"$M_X=H_{\rm inf}^{\max}$", rotation=90, fontsize=6, color="0.35", va="bottom")
ax.set(xscale="log", yscale="log", xlabel=r"$M_X/T_{\rm reh}$", ylabel=r"$Y_X/(T_{\rm reh}/M_{\rm P})^3$", ylim=(1e-9, 1.0), xlim=(CFG["xmin"], M_P / T))
top = ax.secondary_xaxis("top", functions=(lambda u: u * T, lambda M: M / T))
top.set_xlabel(rf"$M_X$ [GeV] for $T_{{\rm reh}}=10^{{{int(round(np.log10(T)))}}}$ GeV", fontsize=7)
ax.legend(loc="upper left", fontsize=5.6, bbox_to_anchor=CFG["legend_anchor"], handlelength=1.5)   # legend upper left, relic label above the dotted line
fig.tight_layout()
# relic-line label parallel to the dotted line (angle in display coordinates after layout)
yl = lambda u: MY_DM / (u * T * norm)
q0, q1 = ax.transData.transform([(1e2, yl(1e2)), (1e3, yl(1e3))])
ax.text(CFG["omega_label_x"], CFG["omega_label_fac"] * yl(CFG["omega_label_x"]), r"$\Omega_Xh^2=0.12$", color=ps.COL["ref"], fontsize=6.5, ha="center", va="bottom",
        rotation=np.degrees(np.arctan2(q1[1] - q0[1], q1[0] - q0[0])), rotation_mode="anchor")
ps.save(fig, ROOT / "plots/p62_yield")
