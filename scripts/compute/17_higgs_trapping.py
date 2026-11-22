"""17: parametrized estimate of keeping the Standard Model field in its false vacuum during inflation (helper read by script 20).

Its parametrized bounds are not quoted in the main text or the SM; the false-vacuum retention bounds of SM Sec. S7 come from script 28.

The SM field must stay in its false vacuum during the last N_e e-folds of inflation. Parametrization [ASSUMED]:
barrier height Delta V_b = b L, barrier width ~ Delta phi = nu L^{1/4}; curvature coupling xi_H gives m_eff^2 = 12 xi_H H^2 during inflation.
(i) Stochastic fluctuations [LIT: Starobinsky:1994bd (equilibrium variance)]: for a light field
    <dh^2> = H^2 N_e/(4 pi^2) (massless) or 3 H^4/(8 pi^2 m_eff^2) (equilibrium, massive). Require 3 sqrt(<dh^2>) < Delta phi.
(ii) Hawking-Moss [LIT: Hawking:1981fz]: rate per Hubble volume ~ H^4 exp(-8 pi^2 Delta V_b/(3 H^4)); require that the integrated
    probability over the ~e^{3 N_e} Hubble volumes that become our universe stays below one:
    8 pi^2 Delta V_b/(3 H^4) > 3 N_e + ln N_e  [ESTIMATE].
Output: H_inf upper bounds along the DM curves (output/17_higgs_trapping.npz) for b in {0.1, 1}, xi_H in {0, 0.1}."""
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import window
from gravdm.units import M_P

CFG = dict(Ne=60, bs=[0.1, 1.0], xis=[0.0, 0.1], cases=[(0.05, 2.0), (0.05, 3.0), (0.0005, 2.0), (0.0005, 3.0)], spin="dirac")


def H_fluct(L4, nu, xi, Ne):
    """Largest H with 3 sigma_h < Delta phi. Massless: sigma = H sqrt(Ne)/(2 pi). Massive (12 xi > 0): sigma^2 = min(massless, 3H^2/(96 pi^2 xi))."""
    dphi = nu * L4
    if xi <= 0:
        return 2 * np.pi * dphi / (3 * np.sqrt(Ne))
    s_over_H = np.sqrt(min(Ne / (4 * np.pi ** 2), 3 / (8 * np.pi ** 2 * 12 * xi)))
    return dphi / (3 * s_over_H)


def H_HM(L4, b, Ne):
    return (8 * np.pi ** 2 * b * L4 ** 4 / (3 * (3 * Ne + np.log(Ne)))) ** 0.25


d = np.load(ROOT / "output/06_dm_plane.npz"); Md = np.load(ROOT / "output/04_mass_dependence.npz")
M = d["M_over_MP"] * M_P
out = {}
for alpha, nu in CFG["cases"]:
    k = f"a{alpha}_nu{nu}_{CFG['spin']}"
    L4 = d[k + "_L4"]; R = Md[f"R_a{alpha}_nu{nu}_{CFG['spin']}"]
    core = np.isfinite(L4) & (R >= 0.1) & (d[k + "_fcoll"] >= window.FCOLL_MIN) & (M < 0.25 * M_P)
    i = np.where(core)[0]
    for xi in CFG["xis"]:
        Hf = np.array([H_fluct(l, nu, xi, CFG["Ne"]) for l in L4[i]])
        out[f"{k}_xi{xi}_Hfluct"] = Hf
        for b in CFG["bs"]:
            Hh = np.array([H_HM(l, b, CFG["Ne"]) for l in L4[i]])
            out[f"{k}_b{b}_HHM"] = Hh
            Hmin = np.minimum(Hf, Hh)
            print(f"alpha={alpha} nu={nu} xi_H={xi} b={b}: fluct bound {Hf.min():.1e}-{Hf.max():.1e}, HM bound {Hh.min():.1e}-{Hh.max():.1e};"
                  f" combined H_inf < {Hmin.min():.1e}-{Hmin.max():.1e} GeV (M_X {M[i[0]]:.0e}-{M[i[-1]]:.0e})")
np.savez(ROOT / "output/17_higgs_trapping.npz", **out)
