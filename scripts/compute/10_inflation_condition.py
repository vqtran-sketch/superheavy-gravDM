"""10: upper bound on H_inf from end-of-inflation gravitational production of a minimal-scalar X along the An-Li DM curves.

SM: Sec. S7, "Conditions on the inflationary era" (via script 20).

Y_inf(M, H0) = (n a^3(M/H0) / C) H0^2 H_c / ((2 pi^2/45) g* T_*^3)       [script 08; DERIVED]
Requirement: Y_inf < 0.1 Y_DM(M) = 0.1 MY_DM/M, solved for H0_max(M) on each point of the core.
n a^3(mu): log-interpolated from script 08 (converged points only: the two y_out values agree within 15%); beyond the last converged
point, exponential extrapolation with the slope of the last two converged points [ASSUMED], or a power law (argument "power").
Points at the numerical floor (0) are treated as unconverged.
Output: output/10_inflation_condition.npz (output/10_inflation_condition_power.npz with "power")"""
import sys, pathlib
import numpy as np
from scipy import optimize
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import window
from gravdm.units import M_P, MY_DM, G_STAR_SM

EXTRAP = sys.argv[1] if len(sys.argv) > 1 else "exp"   # "exp" (default) or "power" (n a^3 ∝ mu^p from the last two points)
CFG = dict(y0s=[0.1, 1.0], conv_tol=0.15, cases=[(0.05, 3.0), (0.05, 1.5), (0.0005, 3.0), (0.0005, 1.0)], Rcore=0.1)


def table(y0):
    d = np.load(ROOT / f"output/08_inflationary_X_y0{y0}.npz")
    Ms = sorted(float(k.split("_M")[1].split("_")[0]) for k in d if k.endswith("_na3"))
    mu, n = [], []
    for M in Ms:
        a, b = d[f"y0{y0}_M{M}_na3"], d[f"y0{y0}_M{M}_na3_yout_lo"]
        if a > 0 and abs(a / b - 1) < CFG["conv_tol"]:
            mu.append(M); n.append(a)
    mu, n = np.array(mu), np.array(n)
    slope = np.log(n[-1] / n[-2]) / (mu[-1] - mu[-2])
    pw = np.log(n[-1] / n[-2]) / np.log(mu[-1] / mu[-2])
    C = float(d[f"y0{y0}_C"])
    def na3(x):
        x = np.asarray(x, float)
        inside = np.exp(np.interp(x, mu, np.log(n)))
        ext = n[-1] * np.exp(slope * (x - mu[-1])) if EXTRAP == "exp" else n[-1] * (np.maximum(x, mu[-1]) / mu[-1]) ** pw
        return np.where(x <= mu[-1], inside, ext)
    return na3, C, mu[-1], slope


d6 = np.load(ROOT / "output/06_dm_plane.npz"); Md = np.load(ROOT / "output/04_mass_dependence.npz")
M = d6["M_over_MP"] * M_P
out = {}
for y0 in CFG["y0s"]:
    na3, C, mu_last, slope = table(y0)
    print(f"y0={y0}: converged up to M/H0 = {mu_last}, extrapolation slope d ln(n a^3)/d mu = {slope:.2f}")
    for alpha, nu in CFG["cases"]:
        k = f"a{alpha}_nu{nu}_scalar"
        L4 = d6[k + "_L4"]; Ts = d6[k + "_Tstar"]; R = Md[f"R_a{alpha}_nu{nu}_scalar"]
        core = np.isfinite(L4) & (R >= CFG["Rcore"]) & (d6[k + "_fcoll"] >= window.FCOLL_MIN) & (M < 0.25 * M_P)   # robust core (src/gravdm/window.py)
        Hmax = np.full_like(M, np.nan)
        for i in np.where(core)[0]:
            Hc = np.sqrt(L4[i] ** 4 / (3 * alpha)) / M_P
            f = lambda lnH: np.log((na3(M[i] / np.exp(lnH)) / C) * np.exp(lnH) ** 2 * Hc /
                                   ((2 * np.pi ** 2 / 45) * G_STAR_SM * Ts[i] ** 3) / (0.1 * MY_DM / M[i]))
            lo, hi = np.log(Hc), np.log(M[i])
            Hmax[i] = np.exp(optimize.brentq(f, lo, hi)) if f(lo) < 0 < f(hi) else (np.nan if f(lo) > 0 else M[i])
        out[f"y0{y0}_{k}_Hmax"] = Hmax
        j = np.where(core)[0]
        r = M[j] / Hmax[j]
        print(f"  alpha={alpha} nu={nu}: H_inf,max = {np.nanmin(Hmax[j]):.1e}-{np.nanmax(Hmax[j]):.1e} GeV; "
              f"M_X/H_inf,max = {np.nanmin(r):.1f}-{np.nanmax(r):.1f}")
np.savez(ROOT / f"output/10_inflation_condition{'' if EXTRAP == 'exp' else '_power'}.npz", **out)
