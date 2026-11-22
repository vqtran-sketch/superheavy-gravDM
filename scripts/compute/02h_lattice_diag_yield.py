"""02h: differential pair yield along the diagonal p_1 = p_2 = p, parton (curve) and lattice (points), gamma = 80 and 160.

Main text: Fig. 1 (M_X = 1e-3 gamma m) and the lattice paragraph (80% of the diagonal yield above 0.2 gamma m; within 0.3% up to
0.7 gamma m; uncertainty 1 to 2% at gamma m). SM: Fig. S3 (M_X = 0.2 gamma m) and Sec. S3, "Yield". Dirac X, mu in CFG["mus"].

Parton [DERIVED, SM Eqs. (S5)-(S6)]: d^2(N/A)/(dp_1 dp_2) = (dN/A dp_1)(dN/A dp_2) sigma(4 p_1 p_2)
    = 4 c_X(r) Dphi^4 W(x_1) W(x_2) / (pi^2 M_P^4),   x_i = p_i/(gamma m),  r = M_X^2/(4 p_1 p_2) = mu^2/(4 x_1 x_2).
Stored in units of 4 c_X(0) Dphi^4/(pi^2 M_P^4): D(x) = W(x)^2 c_X(r)/c_X(0) on the diagonal, and the per-log-interval form x^2 D(x)
(= p_1 p_2 d^2N/dp_1 dp_2 in units of 4 c_X(0) Dphi^4 gamma^2 m^2/(pi^2 M_P^4)), the quantity plotted in Fig. 1.
Lattice: the same density times f_2 r_2 + (1 - f_2) r_0 (script 02g, imported), with the invariant ratios Richardson-extrapolated
from 20 and 40 points per wall (script 02f): R_40 + (R_40 - R_20)/3. Points with p >= 2m only (below, the potential cross term
~ 2 (m/p)^2 dominates; script 02i).
Numerical uncertainty of each lattice point, added in quadrature:
  sigma_dx  = |R_40 - R_20|/3            (size of the Richardson step)
  sigma_win = |R_20(Tw = 5/m) - R_20|    (Fourier window x1.25 at fixed collision time; runs 02f ..._n20_win)
  sigma_dt  = |R_20(dt/2) - R_20|        (half sampling step; runs 02f ..._n20_dt2)
Energy drift <= 6e-5 is neglected. Output: output/02h_lattice_diag_yield.npz"""
import sys, pathlib, importlib.util
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
spec = importlib.util.spec_from_file_location("g", ROOT / "scripts/compute/02g_lattice_yield_ratio.py")
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)       # reuses W, c_X table, f_2 of script 02g

CFG = dict(gammas=(80, 160), mus=(1e-3, 0.2), p_min=2.0, xfine=np.geomspace(5e-3, 3.0, 400))
c00 = G.cX(np.array(0.0))


def density(x, mu):
    r = mu ** 2 / (4 * x * x)
    return G.W(x) ** 2 * G.cX(r) / c00


if __name__ == "__main__":
    out = dict(xfine=CFG["xfine"], mus=np.array(CFG["mus"]))
    for mu in CFG["mus"]:
        out[f"par_mu{mu}"] = density(CFG["xfine"], mu)
    for g in CFG["gammas"]:
        d40 = np.load(ROOT / f"output/02f_lattice_dense_g{g}.npz"); d20 = np.load(ROOT / f"output/02f_lattice_dense_g{g}_n20.npz")
        n = int(round(np.sqrt(len(d40["a"]))))
        dg = lambda d, k1, k2: np.diag((d[k1] / d[k2]).reshape(n, n))
        ex = lambda k1, k2: dg(d40, k1, k2) + (dg(d40, k1, k2) - dg(d20, k1, k2)) / 3
        r2, r0 = ex("I2l", "I2p"), ex("I0l", "I0p")
        A = np.diag(d40["a"].reshape(n, n)); B = np.diag(d40["b"].reshape(n, n))
        k = (A >= CFG["p_min"]) & (B >= CFG["p_min"])
        x1, x2 = A[k] / g, B[k] / g; x = np.sqrt(x1 * x2)
        for mu in CFG["mus"]:
            r = mu ** 2 / (4 * x1 * x2); f2 = G.f2(r)
            par = G.W(x1) * G.W(x2) * G.cX(r) / c00                      # parton density at the (snapped) lattice momenta
            R = f2 * r2[k] + (1 - f2) * r0[k]
            Rv = lambda d: f2 * dg(d, "I2l", "I2p")[k] + (1 - f2) * dg(d, "I0l", "I0p")[k]
            R20 = Rv(d20); sig = {"dx": np.abs(Rv(d40) - R20) / 3}
            for name in ("win", "dt2"):
                pv = ROOT / f"output/02f_lattice_dense_g{g}_n20_{name}.npz"
                if pv.exists():
                    sig[name] = np.abs(Rv(np.load(pv)) - R20)
                else:
                    print(f"  gamma={g}: {pv.name} missing, sigma_{name} omitted")
            sR = np.sqrt(sum(v ** 2 for v in sig.values()))
            for name, v in sig.items():
                out[f"g{g}_mu{mu}_sig_{name}"] = v
            out[f"g{g}_mu{mu}_sigR"] = sR; out[f"g{g}_mu{mu}_siglat"] = par * sR
            hk = (x >= 0.1) & (x <= 1)
            print(f"  gamma={g} mu={mu}: max sigma in x=[0.1,1]: " + ", ".join(f"{n}={v[hk].max():.1e}" for n, v in sig.items())
                  + f"; total {sR[hk].max():.1e}; max sigma all points {sR.max():.1e} at x={x[sR.argmax()]:.3f}")
            out[f"g{g}_mu{mu}_x"] = x; out[f"g{g}_mu{mu}_par"] = par; out[f"g{g}_mu{mu}_lat"] = par * R; out[f"g{g}_mu{mu}_ratio"] = R
            ok = par > 0
            print(f"gamma={g} mu={mu}: points={ok.sum()}, ratio range (x in [0.1,1]) "
                  f"{R[ok & (x >= 0.1) & (x <= 1)].min():.4f}-{R[ok & (x >= 0.1) & (x <= 1)].max():.4f}")
    np.savez(ROOT / "output/02h_lattice_diag_yield.npz", **out)
