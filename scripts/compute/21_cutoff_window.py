"""21: An-Li DM window with a hard cut on the parton collision energy, sqrt(s) < Lambda_cut.

SM: Table SII (window with the yield above Lambda_cut = 0.46 M_P discarded; column '> Lambda_cut'), Sec. S7 (28% above Lambda_cut
for nu = 1 at alpha = 0.05), Sec. S8 (collision-energy quantiles for the loop and contact-term estimates). Via script 20.

Per collision with effective scale S = sqrt(g1 g2) m eperp (script 04), a parton pair has sqrt(s) = 2 sqrt(z) S (gravdm.parton, z = s/(4 S^2)).
Keeping only sqrt(s) < Lambda:                                                                              [DERIVED]
  G(mu, zc) = int_{mu^2}^{zc} dz K(z) c(mu^2/(4z)) / I(0),   zc = (Lambda/(2S))^2,  G(mu, inf) = g(mu)  (script 01)
  R_cut(M) = <w e1 e2 h^3 G(M/S, (Lambda/2S)^2)> / <w e1 e2 h^3>          (R_cut -> R of script 04 for Lambda -> inf)
DM curve with the cut: the total-yield solver of script 06 with R -> R_cut (collisions + freeze-in).
Window (src/gravdm/window.py): f_coll >= 1/1.1, R >= 0.1 (uncut R: profile robustness), L^{1/4} cut, M < M_P/4; with a cut, also
R_cut >= 0.1 (at least 10% of the uncut massless yield survives the cut).
q50, q90 = yield-weighted quantiles of the parton sqrt(s) (massless X). The SM uses Lambda_cut = 0.46 M_P (Sec. S5); the values
0.1 and 0.2 M_P in CFG['cuts'] are computed as well but not quoted.
Treats the cut as a sharp removal of the yield above Lambda [ASSUMED: no model of the ultraviolet completion].
Output: output/21_cutoff_window.npz"""
import sys, pathlib
import numpy as np
from scipy import optimize
from scipy.interpolate import RegularGridInterpolator
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import parton as pt, xsec_analytic as xa, freezein as fz, window
from gravdm.units import M_P, MY_DM, G_STAR_SM

CFG = dict(cuts=[0.1, 0.2, 0.46], profile="tanh", spins=["scalar", "conformal", "dirac", "vector"], nus=[1.0, 1.5, 2.0, 3.0],
           zgrid=dict(zmin=1e-10, zmax=30.0, n=4000), mu=np.logspace(-4, 1.0, 101), Vinf4=1.4e16, Mmax_frac=0.25,
           q_levels=(0.5, 0.9))
Mdep = np.load(ROOT / "output/04_mass_dependence.npz"); d6 = np.load(ROOT / "output/06_dm_plane.npz")
K5 = np.load(ROOT / "output/05_kappaY.npz")["rows"]
MoP = Mdep["M_over_MP"]; M = MoP * M_P

# ---- G(mu, zc): cumulative z-integral on a fine log grid (trapezoid in ln z), normalised to I(0)
z = np.logspace(np.log10(CFG["zgrid"]["zmin"]), np.log10(CFG["zgrid"]["zmax"]), CFG["zgrid"]["n"]); lz = np.log(z)
Kz = np.array([pt.K(zi, CFG["profile"]) for zi in z])


U01 = np.load(ROOT / "output/01_universality.npz")


def g_of(mu, spin):
    """g(mu) exactly as script 04 interpolates it (log-log, 0 beyond the table), so that R_cut -> R for Lambda -> inf."""
    gtab = U01[f"g_{CFG['profile']}_{spin}"]
    return np.exp(np.interp(np.log(mu), np.log(U01["mu"]), np.log(np.maximum(gtab, 1e-300)), left=0.0, right=-700))


def massless_sqrts_cdf():
    """Fraction of the massless single-collision yield with sqrt(s)/S < y, y = 2 sqrt(z)."""
    f = z * Kz; c = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(lz))])
    return 2 * np.sqrt(z), c / c[-1]


XF = np.logspace(-2, np.log10(40.0), 160)
FTAB = {sp: np.array([fz.yield_freezein_X(1e12, x * 1e12, sp, xi_H=0.0) for x in XF]) / (1e12 / M_P) ** 3 for sp in CFG["spins"]}
def F_fi(sp, x):    # as script 06
    x = np.asarray(x, float)
    return np.where(x < XF[-1], np.exp(np.interp(np.log(np.maximum(x, XF[0])), np.log(XF), np.log(np.maximum(FTAB[sp], 1e-300)))), 0.0)


def dm_line(R, kap, c0hat, chat, sp):
    """Total-yield DM line (script 06 solver): returns L^{1/4} [GeV], f_coll, T* [GeV]."""
    cT = (30 * chat / (np.pi ** 2 * G_STAR_SM)) ** 0.25
    L4 = np.full_like(M, np.nan); fc = np.full_like(M, np.nan)
    for i in range(len(M)):
        yc = kap * c0hat * R[i]
        if yc <= 0:
            continue
        tot = lambda l4: (l4 / M_P) ** 3 * (yc + cT ** 3 * F_fi(sp, M[i] / (cT * l4))) * M[i] / MY_DM - 1
        lo, hi = 1e6, 1e19
        if tot(lo) < 0 < tot(hi):
            L4[i] = np.exp(optimize.brentq(lambda x: tot(np.exp(x)), np.log(lo), np.log(hi), xtol=1e-10))
            fc[i] = yc / (yc + cT ** 3 * F_fi(sp, M[i] / (cT * L4[i])))
    return L4, fc, cT * L4


if __name__ == "__main__":
    out = dict(M_over_MP=MoP, cuts=np.array(CFG["cuts"]))
    ys, cdf = massless_sqrts_cdf()
    out["sqrts_over_S"] = ys; out["sqrts_cdf"] = cdf
    print(f"single collision, massless: sqrt(s)/S median {np.interp(0.5, cdf, ys):.3f}, 90% {np.interp(0.9, cdf, ys):.3f}, "
          f"yield-weighted mean {np.trapezoid(ys, cdf):.3f}")
    Gi = {}
    for sp in CFG["spins"]:
        Gt = pt.cut_yield_table(lambda r, c=sp: xa.c_of_r(c, r), CFG["mu"], z, Kz)
        g01 = np.load(ROOT / "output/01_universality.npz")
        gchk = np.interp(np.log(CFG["mu"]), np.log(g01["mu"]), g01[f"g_{CFG['profile']}_{sp}"])
        m = gchk > 1e-3
        print(f"check {sp}: max |G(mu, inf)/g_01(mu) - 1| = {np.max(np.abs(Gt[m, -1] / gchk[m] - 1)):.1e} (g > 1e-3)")
        Gi[sp] = RegularGridInterpolator((np.log(CFG["mu"]), lz), Gt, bounds_error=False, fill_value=None)
    for row in K5[K5[:, 1] == 6]:
        alpha, kap, chat = row[0], row[2], row[9]
        e1, e2, w, h = (Mdep[f"a{alpha}_{k}"] for k in ("e1", "e2", "w", "h"))
        wt = w * e1 * e2 * h ** 3; W0 = wt.sum()
        for nu in CFG["nus"]:
            S = 6 * np.sqrt(3 * alpha) * np.sqrt(e1 * e2) / nu ** 2 * h          # as script 04 [units of M_P]
            # parton sqrt(s) quantiles: mixture over collisions of the single-collision cdf scaled by S
            grid = np.logspace(-4, 1.5, 2000)
            mix = np.array([np.sum(wt * np.interp(x / S, ys, cdf, left=0.0, right=1.0)) / W0 for x in grid])
            kk = f"a{alpha}_nu{nu}"
            for q in CFG["q_levels"]:
                out[f"{kk}_sqrts_q{int(q * 100)}"] = np.interp(q, mix, grid)
            for lam in CFG["cuts"]:
                out[f"{kk}_cut{lam}_frac_above"] = 1 - np.interp(lam, grid, mix)
            print(f"alpha={alpha:g} nu={nu}: parton sqrt(s) q50 {out[kk + '_sqrts_q50']:.3f}, q90 {out[kk + '_sqrts_q90']:.3f} M_P "
                  f"(wall scale q90 {float(Mdep[kk + '_gm_w90']):.3f}); yield above " +
                  ", ".join(f"{lam}: {out[f'{kk}_cut{lam}_frac_above']:.2f}" for lam in CFG["cuts"]), flush=True)
            for sp in CFG["spins"]:
                c0hat = xa.c0(sp) * 960 * np.pi
                Rfull = Mdep[f"R_{kk}_{sp}"]
                for lam in [np.inf] + CFG["cuts"]:
                    if np.isinf(lam):
                        Rc = Rfull
                    else:
                        Rc = np.zeros_like(MoP)
                        lzc = np.log(np.clip((lam / (2 * S)) ** 2, z[0], z[-1]))
                        below = (lam / (2 * S)) ** 2 < z[0]
                        for j, x in enumerate(MoP):
                            mu = x / S
                            pts = np.column_stack([np.log(np.clip(mu, CFG["mu"][0], CFG["mu"][-1])), lzc])
                            # exact kinematic limit: no pair with sqrt(s) < Lambda if 2 M >= Lambda (mu^2 >= z_c); the bilinear
                            # interpolation across the threshold would otherwise leak a small spurious yield
                            Gv = np.where((mu > CFG["mu"][-1]) | below | (2 * x >= lam), 0.0, np.maximum(Gi[sp](pts), 0.0))
                            # normalise per sample to the uncut g of script 04: the table G(mu, inf) differs from g_01 by ~1%,
                            # so the surviving fraction G(zc)/G(inf) <= 1 is applied to g_01 and R_cut <= R holds exactly.
                            pinf = np.column_stack([pts[:, 0], np.full(len(mu), lz[-1])])
                            Ginf = np.where(mu > CFG["mu"][-1], 0.0, np.maximum(Gi[sp](pinf), 0.0))
                            frac = np.where(Ginf > 0, np.minimum(Gv / np.where(Ginf > 0, Ginf, 1.0), 1.0), 0.0)
                            Rc[j] = np.sum(wt * frac * g_of(mu, sp)) / W0
                    L4, fc, Ts = dm_line(Rc, kap, c0hat, chat, sp)
                    core = (np.isfinite(L4) & (Rfull >= window.R_CORE) & (Rc >= window.R_CORE) & (fc >= window.FCOLL_MIN)
                            & (L4 < alpha ** 0.25 * CFG["Vinf4"]) & (M < CFG["Mmax_frac"] * M_P))
                    tag = "nocut" if np.isinf(lam) else f"cut{lam}"
                    key = f"{kk}_{sp}_{tag}"
                    out[key + "_R"] = Rc; out[key + "_L4"] = L4; out[key + "_fcoll"] = fc; out[key + "_Tstar"] = Ts; out[key + "_core"] = core
                    if np.isinf(lam):    # cross-check against script 06 (same solver, same R)
                        ref = d6[f"{kk}_{sp}_L4"]; mm = np.isfinite(ref) & np.isfinite(L4)
                        print(f"   {sp}: no-cut vs 06 max |dL4/L4| = {np.max(np.abs(L4[mm] / ref[mm] - 1)):.1e}")
                    if sp == "dirac":
                        s = (f"M {M[core].min():.2e}-{M[core].max():.2e}, L4 {np.nanmin(L4[core]):.2e}-{np.nanmax(L4[core]):.2e}"
                             if core.any() else "empty")
                        print(f"   dirac {tag:8s}: R_cut(0) = {Rc[0]:.3f}; core {s}")
    np.savez(ROOT / "output/21_cutoff_window.npz", **out)
