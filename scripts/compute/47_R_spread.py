"""47: mass-suppression factor R(M_X/Mbar_coll) of the flat-space benchmark, averaged over the yield-weighted collision energies.

Main text: Eq. (5) (R), "The yield is halved at M_X ≃ Mbar_coll/6 and falls to a tenth at M_X ≃ 0.45 Mbar_coll". SM: Sec. S4 (R = 0.9,
0.5, 0.1 at M_X/Mbar_coll = 0.04, 0.17, 0.45). Input to scripts 49, 50 and p62.

R(mu) = < g(mu / s) >, s = M_coll/Mbar_coll = sqrt(rhat_1 rhat_2) eta/S0, rhat_i ~ Gamma(4), eta ~ 6 eta^5 [DERIVED, SM Sec. S4,
gravdm.rspread]. g: single-collision mass-suppression function of planar walls (profile and spin from CFG), output/01_universality.npz.
Also stores the fixed-scale approximation g(mu) (all collisions at Mbar_coll) for comparison.
Checks: tests/test_rspread.py (normalization, <s> = 1, analytic <1/s>, Monte-Carlo agreement, grid convergence); here h vs h/2.
Output: output/47_R_spread.npz."""
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import rspread as rs

CFG = dict(profile="tanh", spin="dirac", mu_grid=(-10.0, 3.0, 521), h=2e-3)
U = np.load(ROOT / "output/01_universality.npz")
key = f"g_{CFG['profile']}_{CFG['spin']}"
g = lambda mu: np.exp(np.interp(np.log(mu), np.log(U["mu"]), np.log(np.maximum(U[key], 1e-300)), left=0.0))
mu = np.logspace(*CFG["mu_grid"])
R = rs.R_averaged(mu, g, h=CFG["h"])
R2 = rs.R_averaged(mu, g, h=CFG["h"] / 2)
vis = R > 1e-6
print("convergence h vs h/2 (R > 1e-6): max rel diff", np.max(np.abs(R[vis] / R2[vis] - 1)))
np.savez(ROOT / "output/47_R_spread.npz", mu=mu, R=R2, g_fixed=g(mu), profile=CFG["profile"], spin=CFG["spin"])
cross = lambda y, t: np.exp(np.interp(-t, -y, np.log(mu)))
for t in (0.9, 0.5, 0.1, 0.01):
    print(f"R = {t}: averaged mu = {cross(R2, t):.4g}, fixed mu = {cross(g(mu), t):.4g}")
for m in (1e-3, 0.06, 0.2, 1.0, 3.0, 10.0):
    print(f"mu = {m}: R = {np.interp(np.log(m), np.log(mu), R2):.4g}, g = {g(m):.4g}")
