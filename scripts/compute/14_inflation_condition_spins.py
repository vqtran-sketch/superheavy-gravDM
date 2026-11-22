"""14: upper bound on H_inf from end-of-inflation gravitational production of a Dirac or Proca X along the An-Li DM curves.

SM: Sec. S7, "Conditions on the inflationary era" (M_X/H_inf >~ 70 to 180, abrupt end; 9 to 21, smooth end; via script 20) and
Sec. S5 (abrupt end moves the benchmark threshold to 3e15 to 8e15 GeV). Same method as script 10 for the minimal scalar.

Y_inf(M, H0) = (n a^3(M/H0) / C) H0^2 H_c / ((2 pi^2/45) g* T_*^3)       [scripts 08, 10; DERIVED]
  n a^3 from script 13: Dirac = X + Xbar, both helicities (the DM yield counts two particles per collision event, X and Xbar for a
  Dirac X, so the total particle number is the consistent comparison); Proca = 2 T + L (real field).
Requirement: Y_inf < 0.1 Y_DM(M) = 0.1 MY_DM/M, solved for H0_max on each point of the spin's own DM curve
(06_dm_plane keys a{alpha}_nu{nu}_{spin}_L4/_Tstar/_fcoll; R from 04_mass_dependence).
core = finite L4 & R >= 0.1 & Y_FI <= 0.1 Y_coll (src/gravdm/window.py) & M < M_P/4 (as script 10).
n a^3(mu): log-interpolated over converged points (> 0 and |y_out variants| < conv_tol) on the falling branch: past the production
peak, n a^3 decreases with M, so points from the first increase on are treated as a numerical floor and dropped (this happens for
y0 = 1, Dirac, at M/H0 >= 8, n ~ 1e-10-1e-9). Beyond the last converged point: 'exp' (slope in mu of the last two points) or
'power' (n ∝ mu^p) extrapolation [ASSUMED; both stored].
C = 1 (late kination H a^3 of the analytic background).
Output: output/14_inflation_condition_spins.npz"""
import sys, pathlib, re
import numpy as np
from scipy import optimize
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import window
from gravdm.units import M_P, MY_DM, G_STAR_SM
from gravdm import infprod_spins as sp

CFG = dict(y0s=[0.1, 1.0], spins=["dirac", "vector"], conv_tol=0.15, extraps=["exp", "power"],
           cases=[(0.05, 3.0), (0.05, 1.5), (0.0005, 3.0), (0.0005, 1.0)], Rcore=0.1, Mmax_over_MP=0.25)
d13 = np.load(ROOT / "output/13_inflationary_spins.npz")
# Script 13 stores "Ms" only in its final save; a checkpointed (interrupted) run has the masses only in the key names
# y0{y0}_M{M}_{spin}_na3, so read them from there when "Ms" is absent.
MS = (np.array(d13["Ms"]) if "Ms" in d13 else
      np.array(sorted({float(m.group(1)) for k in d13.files if (m := re.fullmatch(r"y0[0-9.]+_M([0-9.]+)_(?:dirac|vector)_na3", k))})))


def table(y0, spin, extrap):
    mu, n = [], []
    for M in MS:
        kk = f"y0{y0}_M{M}_{spin}_na3"
        if kk not in d13:
            continue
        lo, hi = d13[kk]
        if hi > 0 and lo > 0 and abs(hi / lo - 1) < CFG["conv_tol"]:
            mu.append(M); n.append(hi)
    mu, n = np.array(mu), np.array(n)
    ipk = int(np.argmax(n))
    rise = np.where(np.diff(n[ipk:]) > 0)[0]                 # first increase after the peak -> numerical floor
    if len(rise):
        cut = ipk + rise[0] + 1
        print(f"   [{extrap}] y0={y0} {spin}: dropped floor points M/H0 = {mu[cut:].tolist()} (n = {np.array2string(n[cut:], precision=2)})")
        mu, n = mu[:cut], n[:cut]
    slope = np.log(n[-1] / n[-2]) / (mu[-1] - mu[-2])
    pw = np.log(n[-1] / n[-2]) / np.log(mu[-1] / mu[-2])
    C = sp.AnalyticBackground(y0).late_C()

    def na3(x):
        x = np.asarray(x, float)
        inside = np.exp(np.interp(x, mu, np.log(n)))       # below mu[0]: flat (conservative-low: n grows towards small mu)
        ext = n[-1] * np.exp(slope * (x - mu[-1])) if extrap == "exp" else n[-1] * (np.maximum(x, mu[-1]) / mu[-1]) ** pw
        return np.where(x <= mu[-1], inside, ext)
    return na3, C, mu, slope, pw


if __name__ == "__main__":
    np.seterr(divide="ignore", under="ignore")      # exp tail underflows to 0 far above the bound: log -> -inf is the right sign for brentq
    d6 = np.load(ROOT / "output/06_dm_plane.npz"); Md = np.load(ROOT / "output/04_mass_dependence.npz")
    d10 = {e: np.load(ROOT / f"output/10_inflation_condition{'' if e == 'exp' else '_power'}.npz") for e in CFG["extraps"]}
    M = d6["M_over_MP"] * M_P
    out = {"M": M}
    for extrap in CFG["extraps"]:
        for y0 in CFG["y0s"]:
            for spin in CFG["spins"]:
                na3, C, mu, slope, pw = table(y0, spin, extrap)
                out[f"{extrap}_y0{y0}_{spin}_mu_conv"] = mu
                print(f"[{extrap}] y0={y0} {spin}: converged M/H0 = {mu.min()}..{mu.max()}, tail slope d ln n/d mu = {slope:.2f}, "
                      f"power {pw:.2f}")
                for alpha, nu in CFG["cases"]:
                    k = f"a{alpha}_nu{nu}_{spin}"
                    L4 = d6[k + "_L4"]; Ts = d6[k + "_Tstar"]; R = Md[f"R_a{alpha}_nu{nu}_{spin}"]
                    core = np.isfinite(L4) & (R >= CFG["Rcore"]) & (d6[k + "_fcoll"] >= window.FCOLL_MIN) & (M < CFG["Mmax_over_MP"] * M_P)
                    Hmax = np.full_like(M, np.nan)
                    for i in np.where(core)[0]:
                        Hc = np.sqrt(L4[i] ** 4 / (3 * alpha)) / M_P
                        f = lambda lnH: np.log((na3(M[i] / np.exp(lnH)) / C) * np.exp(lnH) ** 2 * Hc /
                                               ((2 * np.pi ** 2 / 45) * G_STAR_SM * Ts[i] ** 3) / (0.1 * MY_DM / M[i]))
                        lo, hi = np.log(Hc), np.log(M[i])
                        Hmax[i] = np.exp(optimize.brentq(f, lo, hi)) if f(lo) < 0 < f(hi) else (np.nan if f(lo) > 0 else M[i])
                    out[f"{extrap}_y0{y0}_{k}_Hmax"] = Hmax; out[f"{extrap}_y0{y0}_{k}_core"] = core
                    j = np.where(core)[0]
                    if len(j) == 0:
                        print(f"   alpha={alpha} nu={nu}: empty core"); continue
                    r = M[j] / Hmax[j]
                    hs = d10[extrap][f"y0{y0}_a{alpha}_nu{nu}_scalar_Hmax"]
                    print(f"   alpha={alpha} nu={nu}: H_inf,max = {np.nanmin(Hmax[j]):.2e}-{np.nanmax(Hmax[j]):.2e} GeV; "
                          f"M_X/H_inf,max = {np.nanmin(r):.1f}-{np.nanmax(r):.1f}  [scalar: {np.nanmin(hs):.2e}-{np.nanmax(hs):.2e} GeV, "
                          f"M/H = {np.nanmin(M / hs):.1f}-{np.nanmax(M / hs):.1f}]", flush=True)
    np.savez(ROOT / "output/14_inflation_condition_spins.npz", **out)
