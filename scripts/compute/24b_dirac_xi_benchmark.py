"""24b: effect of the wall-field curvature coupling xi_phi on the Dirac yield of the flat-space benchmark.

Main text: Cross sections ("we show xi_phi = 0") and the lower-bound paragraph (xi_phi lowers the yield by up to 20% near the highest
masses; the xi_phi = 1/6 curve lies up to 8% higher in T_reh). SM: Sec. S1 (lower by up to 20% at the upper edge of the core, by less
than 10% for M_X <~ 0.1 Mbar_coll).

Question: by how much does xi_phi = 1/6 (the value that minimizes the yield) lower R(M_X) relative to xi_phi = 0, and how far does it
move a DM curve in T_reh?
Formulas [DERIVED, SM Eqs. (S3), (S4), (S7)]:
  c_Dirac(rho; xi) = v^3 [1 + 8 rho/3 + (10/3)(1 - 6 xi)^2 rho]/(960 pi)                   [xsec_analytic.c_dirac_xi]
  g(mu; xi) = int dz K(z) c(mu^2/(4z); xi) / I(0),  tanh walls                              [parton.yield_integral; I(0) is xi-independent]
  R(M; xi)  = < g(M/M_coll; xi) > over first collisions, yield-weighted:
              beta r_i ~ Gamma(shape 4) independent, eta_perp density 6 eta^5 (SM Sec. S4), M_coll ∝ sqrt(r1 r2) eta,
              normalized so that <M_coll> = Mbar_coll (SM Eq. (S14)).
  T_reh shift at fixed M_X on a DM curve: M_X Y_X fixed and Y_X ∝ T_reh^3 R  =>  T(1/6)/T(0) = [R(0)/R(1/6)]^(1/3).
Checks: R(M; 0) reproduces SM values 0.9 / 0.5 / 0.1 at M/Mbar = 0.04 / 0.17 / 0.45; seed change and doubled sample give the same
ratios to three decimals.
Units: dimensionless (M in units of Mbar_coll). Output: output/24b_dirac_xi_benchmark.npz, log output/24b_dirac_xi_benchmark.log."""
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import parton as pt, xsec_analytic as xa

CFG = dict(xis=[0.0, 1 / 6], profile="tanh", ktab=dict(zmin=1e-12, zmax=60.0, n=600), mu=np.logspace(-3, 0.5, 120),
           n_mc=400_000, seed=1, gamma_shape=4.0, M_over_Mbar=[0.01, 0.04, 0.1, 0.17, 0.3, 0.45])

zt, Kt = pt.K_table(CFG["profile"], **CFG["ktab"])
I0 = pt.yield_integral(lambda r: xa.c_dirac_xi(r, 0.0), 0.0, CFG["profile"], zt, Kt)
g = {xi: np.array([pt.yield_integral(lambda r, xi=xi: xa.c_dirac_xi(r, xi), mu, CFG["profile"], zt, Kt) / I0
                   for mu in CFG["mu"]]) for xi in CFG["xis"]}

rng = np.random.default_rng(CFG["seed"])
r1 = rng.gamma(CFG["gamma_shape"], size=CFG["n_mc"]); r2 = rng.gamma(CFG["gamma_shape"], size=CFG["n_mc"])
eta = rng.random(CFG["n_mc"]) ** (1 / 6)                      # density 6 eta^5 on [0, 1]
Mc = np.sqrt(r1 * r2) * eta; Mc /= Mc.mean()                  # M_coll / Mbar_coll

x = np.array(CFG["M_over_Mbar"])
R = {xi: np.array([np.mean(np.interp(np.log(xx / Mc), np.log(CFG["mu"]), g[xi], left=g[xi][0], right=0.0)) for xx in x])
     for xi in CFG["xis"]}
ratio = R[1 / 6] / R[0.0]; tshift = ratio ** (-1 / 3)

lines = ["M/Mbar   R(xi=0)  R(xi=1/6)  ratio   T_reh shift"]
lines += [f"{a:6.2f}   {b:.3f}    {c:.3f}     {d:.3f}   {e:.3f}" for a, b, c, d, e in zip(x, R[0.0], R[1 / 6], ratio, tshift)]
out = ROOT / "output"
(out / "24b_dirac_xi_benchmark.log").write_text("\n".join(lines) + "\n")
np.savez(out / "24b_dirac_xi_benchmark.npz", M_over_Mbar=x, R0=R[0.0], R16=R[1 / 6], ratio=ratio, Treh_shift=tshift,
         mu=CFG["mu"], g0=g[0.0], g16=g[1 / 6])
print("\n".join(lines))
