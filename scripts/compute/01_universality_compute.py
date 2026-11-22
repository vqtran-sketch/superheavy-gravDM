"""01: mass-suppression function g(mu) of the graviton-mediated pair yield, for each spin of X and each wall profile.

Main text: Eq. (3), "g(0) = 1 for every wall profile", the 10% points at fixed wall energy. SM: Sec. S2 (Eq. (S7), Dirac 10% points
mu = 0.42, 0.53, 0.68, profile comparison), Fig. S1. Input to scripts 04, 21, 24 and 47.

g(mu) = I(mu)/I(0),  I(mu) = int dz K(z) c(mu^2/(4z)),  mu = M_X/(gamma m)                    [SM Eq. (S7)]
so that N_ev/A = 4 c(0) (E1/A)(E2/A)/M_P^4 g(mu)                                                [main-text Eq. (3), SM Eq. (S7)]
The identity g(0) = 1 (int dz K = [int dp W]^2) is checked numerically for every profile and spin (theorem_check_*).
Also stored: the energy-weighted mean parton momentum <x>_E = int x W / int W (units of gamma m) of each profile.
Output: output/01_universality.npz"""
import sys, pathlib
import numpy as np
from scipy import integrate
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import parton as pt
from gravdm import xsec_analytic as xa

CFG = dict(mu=np.logspace(-3, 1.0, 81), ktab=dict(zmin=1e-12, zmax=60.0, n=600))

out = dict(mu=CFG["mu"])
for prof in pt.PROFILES:
    W = pt.PROFILES[prof]["W"]
    xE = integrate.quad(lambda x: x * W(x), 0, 60)[0] / integrate.quad(W, 0, 60)[0]
    out[f"xE_{prof}"] = xE
    zt, Kt = pt.K_table(prof, **CFG["ktab"])
    for case in xa.CASES:
        cf = lambda r, c=case: xa.c_of_r(c, r)
        I0 = pt.yield_integral(cf, 0.0, prof, zt, Kt)
        Wint = integrate.quad(W, 0, 60)[0]
        out[f"theorem_check_{prof}_{case}"] = I0 / (xa.c0(case) * Wint ** 2)   # must be 1
        g = np.array([pt.yield_integral(cf, m, prof, zt, Kt) for m in CFG["mu"]]) / I0
        out[f"g_{prof}_{case}"] = g
        print(prof, case, "theorem ratio", out[f"theorem_check_{prof}_{case}"], "g(1)=", np.interp(1, CFG['mu'], g))
np.savez(ROOT / "output/01_universality.npz", **out)
