"""24: An-Li Dirac DM curve and window as a function of the wall-field (Higgs) curvature coupling xi_H = xi_phi.

SM: Sec. S7, "Window" (for the Dirac fermion xi_H matters only near the upper mass edge, which xi_H = 1 raises by a factor of about 1.7).
The flat-space benchmark analogue is script 24b.

The spin-0 part of the massive Dirac cross section, c = v^3 [1 + 8r/3 + (10/3)(1-6xi)^2 r]/(960 pi) (SM Eqs. (S3)-(S4),
xsec_analytic.c_dirac_xi), is independent of s, so after the luminosity fold it adds
[(1-6xi)^2 - 1] mu^2 (6.08 ln^2(1/mu) - 17.97 ln(1/mu) + 16.4) to g(mu).
Steps (same definitions as scripts 01, 04, 06 and src/gravdm/window.py):
  g(mu; xi) = I(mu; xi)/I(0)   tanh walls, I(mu) = int dz K(z) c(mu^2/(4z))           [parton.yield_integral; I(0) is xi-independent]
  R(M; xi)  = <w e1 e2 h^3 g(M/gm)>/<w e1 e2 h^3>  over the stored collisions of script 04 (gm = 6 sqrt(3 alpha) sqrt(e1 e2) h/nu^2)
  DM curve  = total yield (collisions + freeze-in with the same xi_H in the Higgs trace channel) = MY_DM/M   [as script 06]
  window    = R >= 0.1, freeze-in <= 10% of collisions, L^{1/4} < alpha^{1/4} 1.4e16 GeV, M < M_P/4      [src/gravdm/window.py]
Units: masses and L^{1/4} in GeV. Output: output/24_dirac_xi.npz; summary printed (log: output/24_dirac_xi.log)."""
import sys, pathlib
import numpy as np
from scipy import optimize
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import parton as pt, xsec_analytic as xa, freezein as fz, window
from gravdm.units import M_P, MY_DM, G_STAR_SM

CFG = dict(xis=[0.0, 1 / 6, 0.5, 1.0, 2.0, 5.0, -0.5], profile="tanh", mu=np.logspace(-3, 1.0, 81),
           ktab=dict(zmin=1e-12, zmax=60.0, n=600), nus=[1.0, 1.5, 2.0, 3.0], n=6, Vinf4=1.4e16, Mmax_frac=0.25,
           xfi=np.logspace(-2, np.log10(40.0), 160), Tref=1e12)

Mdep = np.load(ROOT / "output/04_mass_dependence.npz")
K = np.load(ROOT / "output/05_kappaY.npz")["rows"]
MoP = Mdep["M_over_MP"]; M = MoP * M_P
mus = CFG["mu"]
zt, Kt = pt.K_table(CFG["profile"], **CFG["ktab"])
I0 = pt.yield_integral(lambda r: xa.c_dirac_xi(r, 0.0), 0.0, CFG["profile"], zt, Kt)
out = dict(M_over_MP=MoP, mu=mus, xis=np.array(CFG["xis"]))
G = {}
for xi in CFG["xis"]:
    G[xi] = np.array([pt.yield_integral(lambda r, x=xi: xa.c_dirac_xi(r, x), m, CFG["profile"], zt, Kt) for m in mus]) / I0
    out[f"g_xi{xi:.4f}"] = G[xi]


def g_of(x, xi):
    return np.exp(np.interp(np.log(x), np.log(mus), np.log(np.maximum(G[xi], 1e-300)), left=0.0, right=-700))


FTAB = {xi: np.array([fz.yield_freezein_X(CFG["Tref"], x * CFG["Tref"], "dirac", xi_H=xi) for x in CFG["xfi"]]) / (CFG["Tref"] / M_P) ** 3
        for xi in CFG["xis"]}                                    # Y_FI = (T/M_P)^3 F(M/T)  [as script 06]


def F_fi(xi, x):
    x = np.asarray(x, float); X = CFG["xfi"]
    return np.where(x < X[-1], np.exp(np.interp(np.log(np.maximum(x, X[0])), np.log(X), np.log(np.maximum(FTAB[xi], 1e-300)))), 0.0)


print("g(mu;xi)/g(mu;0) at mu = 0.05, 0.1, 0.2, 0.4:")
for xi in CFG["xis"][1:]:
    print(f"  xi={xi:+.3f}: " + "  ".join(f"{g_of(m, xi) / g_of(m, 0.0):.3g}" for m in (0.05, 0.1, 0.2, 0.4)))
for row in K[K[:, 1] == CFG["n"]]:
    alpha, kap, chat = row[0], row[2], row[9]
    cT = (30 * chat / (np.pi ** 2 * G_STAR_SM)) ** 0.25
    e1, e2, w, h = (Mdep[f"a{alpha}_{k}"] for k in ("e1", "e2", "w", "h")); W0 = np.sum(w * e1 * e2 * h ** 3)
    for nu in CFG["nus"]:
        gm = 6 * np.sqrt(3 * alpha) * np.sqrt(e1 * e2) / nu ** 2 * h
        for xi in CFG["xis"]:
            R = np.array([np.sum(w * e1 * e2 * h ** 3 * g_of(Mx / gm, xi)) / W0 for Mx in MoP])
            L4 = np.full_like(M, np.nan); fcoll = np.full_like(M, np.nan)
            for i in range(len(M)):
                yc = kap * R[i]                                      # c0hat = 1 (Dirac, any xi)
                tot = lambda l4: (l4 / M_P) ** 3 * (yc + cT ** 3 * F_fi(xi, M[i] / (cT * l4))) * M[i] / MY_DM - 1
                if yc > 0 and tot(1e6) < 0 < tot(1e19):
                    L4[i] = np.exp(optimize.brentq(lambda x: tot(np.exp(x)), np.log(1e6), np.log(1e19), xtol=1e-10))
                    fcoll[i] = yc / (yc + cT ** 3 * F_fi(xi, M[i] / (cT * L4[i])))
            ok = (np.isfinite(L4) & (fcoll >= window.FCOLL_MIN) & (R >= window.R_CORE) & (L4 < alpha ** 0.25 * CFG["Vinf4"])
                  & (M < CFG["Mmax_frac"] * M_P))
            key = f"a{alpha}_nu{nu}_xi{xi:.4f}"
            out[key + "_R"] = R; out[key + "_L4"] = L4; out[key + "_fcoll"] = fcoll; out[key + "_ok"] = ok
            if xi == 0.0:
                assert np.allclose(R, Mdep[f"R_a{alpha}_nu{nu}_dirac"], rtol=2e-2, atol=1e-6)   # reproduces script 04
            print(f"alpha={alpha:g} nu={nu} xi={xi:+.3f}: window M = {M[ok].min():.2e}-{M[ok].max():.2e} GeV, "
                  f"L^1/4 = {np.nanmin(L4[ok]):.2e}-{np.nanmax(L4[ok]):.2e} GeV", flush=True)
np.savez(ROOT / "output/24_dirac_xi.npz", **out)
