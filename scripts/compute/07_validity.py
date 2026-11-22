"""07: validity checks along the An-Li DM curves (minimal scalar): free streaming and wall friction.

SM: Sec. S7, "Conditions on the inflationary era" (friction from radiation produced at the end of inflation gives the weakest
bound; via script 20).

(1) Free streaming [DERIVED estimate]: X is produced with energy E_X ~ gamma m eperp (hard partons) at a_c; its momentum redshifts,
    p0 = E_X (a_c/a_0) with a_c/a_0 = (g_s0/g_*)^{1/3} T0/T_*  (T_* = comoving-entropy temperature at a_c, script 06).
    Today's velocity v0 = p0/M, compared with the warm-DM scale v0 ~ 1e-8 [LIT, order of magnitude; Irsic:2017ixq].
(2) Friction on the walls from the SM radiation produced gravitationally at the end of inflation (gravdm.friction): upper bound on
    H_inf per DM-curve point.
(3) A rough proxy H_inf < M_X/40 for end-of-inflation production (stored as *_Hinfprod; the SM uses scripts 10 and 14).
Output: output/07_validity.npz; printed summary."""
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import window
from gravdm import friction as fr
from gravdm.units import M_P, T0, GS0, G_STAR_SM

CFG = dict(cases=[(0.05, 1.0), (0.05, 3.0), (0.0005, 1.0), (0.0005, 3.0)], spin="scalar", Rcore=0.1)
d = np.load(ROOT / "output/06_dm_plane.npz"); Md = np.load(ROOT / "output/04_mass_dependence.npz")
M = d["M_over_MP"] * M_P
out = {}
for alpha, nu in CFG["cases"]:
    k = f"a{alpha}_nu{nu}_{CFG['spin']}"
    L4 = d[k + "_L4"]; Ts = d[k + "_Tstar"]; R = Md[f"R_a{alpha}_nu{nu}_{CFG['spin']}"]
    gmh = float(d[k + "_gm_med"]) * M_P                              # median effective collision scale (GeV)
    core = np.isfinite(L4) & (R >= CFG["Rcore"]) & (d[k + "_fcoll"] >= window.FCOLL_MIN) & (M < 0.25 * M_P)   # robust core (src/gravdm/window.py)
    EX = np.maximum(gmh, M)                                           # produced-X energy ~ hard parton scale
    v0 = EX * (GS0 / G_STAR_SM) ** (1 / 3) * T0 / Ts / M
    gm_wall = float(Md[f"a{alpha}_nu{nu}_wallgm_wmed"]) * M_P          # wall boost gamma m (no oblique h), yield-weighted median
    Hfr = np.array([fr.H0_max_runaway(l, alpha, gm_wall) if np.isfinite(l) else np.nan for l in L4])
    Hinf = M / 40
    out[k + "_v0"] = v0; out[k + "_Hfric"] = Hfr; out[k + "_Hinfprod"] = Hinf; out[k + "_core"] = core
    i = np.where(core)[0]
    print(f"alpha={alpha} nu={nu}: core M_X {M[i[0]]:.1e}-{M[i[-1]]:.1e} GeV | max v0 = {v0[i].max():.1e} | "
          f"H_inf < friction bound {np.nanmin(Hfr[i]):.1e}-{np.nanmax(Hfr[i]):.1e} GeV | H_inf < M_X/40 = {Hinf[i[0]]:.1e}-{Hinf[i[-1]]:.1e} GeV")
np.savez(ROOT / "output/07_validity.npz", **out)
