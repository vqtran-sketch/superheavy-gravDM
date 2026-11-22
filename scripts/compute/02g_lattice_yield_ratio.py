"""02g: lattice/parton ratio of the graviton-mediated pair yield of a Dirac X, from the dense grids of script 02f.

Main text: lattice paragraph ("integrated over the simulated momenta ... within 1.3% up to M_X = gamma m"). SM: Sec. S3, "Yield"
(ratio f_2 r_2 + (1 - f_2) r_0; integrated agreement 0.8% at gamma = 80, 0.2% and 1.3% at gamma = 160; up to 13% at gamma = 10).

[DERIVED] dN/A = int d omega dk/(2 pi)^2 [rho_2 I_2 + rho_0 I_0/12]/(M_P^4 q^4), q^2 = 4 p_1 p_2.
Free passage has I_0^par = 6 I_2^par at every (p_1, p_2) (SM Sec. S3), so at given parton momenta
    dN^lat/dN^par = f_2 r_2 + (1 - f_2) r_0,   r_i = I_i^lat/I_i^par,
with f_2(r) = c_X(r; xi_phi = 1/6)/c_X(r; xi_phi = 0) the spin-2 share of the Dirac cross section at xi_phi = 0 (SM Eq. (S3)),
r = M_X^2/s; f_2 = 1 for M_X -> 0 and 2/3 at threshold.
N^par/A ∝ int dx_1 dx_2 W(x_1) W(x_2) c_X(mu^2/(4 x_1 x_2)), x_i = p_i/(gamma m)  (SM Eq. (S6)).
Hence N^lat/N^par(mu) = < dN^lat/dN^par > weighted by x_1 x_2 W W c_X over the grid in (ln x_1, ln x_2) (trapezoid).
Diagnostics: the same ratio for the superposed walls, lattice/superposition, the weight fraction with p_1 or p_2 < 2m (where the
potential cross term matters, script 02i), and the parton weight outside the grid (analytic).
Output: output/02g_lattice_yield_ratio.npz and .log"""
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import gravxsec as gx
from gravdm import parton as pt

CFG = dict(runs={"10": "02f_lattice_dense_g10", "80": "02f_lattice_dense_g80", "160": "02f_lattice_dense_g160",
                 "10_n20": "02f_lattice_dense_g10_n20", "80_n20": "02f_lattice_dense_g80_n20", "160_n20": "02f_lattice_dense_g160_n20"},
           mus=np.geomspace(0.02, 1.5, 41), soft=2.0, nr=241)

W = pt.PROFILES["tanh"]["W"]
rt = np.linspace(0.0, 0.25, CFG["nr"])[:-1]
c0t = np.array([gx.c_hat(0.5, r, xi_phi=0.0) for r in rt]); c6t = np.array([gx.c_hat(0.5, r, xi_phi=1 / 6) for r in rt])
cX = lambda r: np.where(r < 0.25, np.interp(r, rt, c0t, right=0.0), 0.0)
ZT, KT = pt.K_table("tanh")                       # cached luminosity table for the analytic coverage check
f2 = lambda r: np.interp(np.minimum(r, rt[-1]), rt, c6t / c0t)


def ratios(d, mu):
    g = float(d["gamma"]); n = int(round(np.sqrt(len(d["a"]))))
    a = d["a"].reshape(n, n); b = d["b"].reshape(n, n)
    x1 = a / g; x2 = b / g
    r = mu ** 2 / (4 * x1 * x2)
    w = x1 * x2 * W(x1) * W(x2) * cX(r)
    tw = np.outer(np.gradient(np.arange(n)), np.gradient(np.arange(n)))   # uniform ln spacing: trapezoid ends
    tw[0, :] *= 0.5; tw[-1, :] *= 0.5; tw[:, 0] *= 0.5; tw[:, -1] *= 0.5
    w = w * tw
    R = lambda num, den: f2(r) * (num[0] / den[0]).reshape(n, n) + (1 - f2(r)) * (num[1] / den[1]).reshape(n, n)
    lat = R((d["I2l"], d["I0l"]), (d["I2p"], d["I0p"])); free = R((d["I2f"], d["I0f"]), (d["I2p"], d["I0p"]))
    soft = (a < CFG["soft"]) | (b < CFG["soft"])
    out = dict(lat=np.sum(w * lat) / np.sum(w), free=np.sum(w * free) / np.sum(w),
               lat_hard=np.sum((w * lat)[~soft]) / np.sum(w[~soft]) if np.sum(w[~soft]) > 0 else np.nan,
               spin2=np.sum(w * (d["I2l"] / d["I2p"]).reshape(n, n)) / np.sum(w),
               soft_frac=np.sum(w[soft]) / np.sum(w))
    dl = np.log(3.0 * g / 0.5) / (n - 1)                     # nominal ln spacing of the 02f grid (a_min = 0.5 m, a_max = 3 gamma m)
    out["grid_cover"] = np.sum(w) * dl * dl / pt.yield_integral(cX, mu, "tanh", ZT, KT)      # grid weight / analytic parton integral
    return out


if __name__ == "__main__":
    res = {}; log = []
    for key, f in CFG["runs"].items():
        p = ROOT / f"output/{f}.npz"
        if not p.exists():
            log.append(f"{key}: missing {p.name}"); continue
        d = np.load(p)
        rows = [ratios(d, mu) for mu in CFG["mus"]]
        for k in rows[0]:
            res[f"g{key.replace('_n20', '_n20')}_{k}"] = np.array([rw[k] for rw in rows])
        for mu, rw in zip(CFG["mus"], rows):
            if np.any(np.isclose(mu, [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5], rtol=0.03)):
                log.append(f"gamma={key} mu={mu:.3f}: N_lat/N_par={rw['lat']:.4f} free/par={rw['free']:.4f} hard-only={rw['lat_hard']:.4f} "
                           f"spin2-only={rw['spin2']:.4f} soft weight={rw['soft_frac']:.3f} grid/analytic={rw['grid_cover']:.3f}")
    # Richardson extrapolation to zero lattice spacing [DERIVED: leapfrog + 4th-order Laplacian, error O(dx^2) at fixed CFL;
    # checked: the deficit at mu = 0.5-1 shrinks by 3.5-4 from 20 to 40 points per wall]: R_ext = R_40 + (R_40 - R_20)/3
    for g in ("10", "80", "160"):
        if f"g{g}_lat" in res and f"g{g}_n20_lat" in res:
            for k in ("lat", "free"):
                res[f"g{g}_ext_{k}"] = res[f"g{g}_{k}"] + (res[f"g{g}_{k}"] - res[f"g{g}_n20_{k}"]) / 3
            res[f"g{g}_ext_soft_frac"] = res[f"g{g}_soft_frac"]
            for mu, e, r40, r20 in zip(CFG["mus"], res[f"g{g}_ext_lat"], res[f"g{g}_lat"], res[f"g{g}_n20_lat"]):
                if np.any(np.isclose(mu, [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5], rtol=0.03)):
                    log.append(f"gamma={g} mu={mu:.3f}: Richardson N_lat/N_par={e:.4f} (n40 {r40:.4f}, n20 {r20:.4f}, ratio of deficits "
                               f"{(1 - r20) / (1 - r40) if abs(1 - r40) > 1e-4 else np.nan:.2f})")
    print("\n".join(log))
    np.savez(ROOT / "output/02g_lattice_yield_ratio.npz", mus=CFG["mus"], **res)
    (ROOT / "output/02g_lattice_yield_ratio.log").write_text("\n".join(log) + "\n")
