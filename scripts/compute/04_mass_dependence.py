"""04: mass dependence R(M_X) of the collision yield on the cosmological first-collision surfaces of the An-Li scenario.

SM: Sec. S7 (mass dependence computed on the cosmological first-collision surfaces; gamma m/M_P = 6 sqrt(3 alpha) e_c/nu^2),
Fig. S5, Table SII. Input to scripts 06, 07, 10, 14, 17, 20, 21, 24, 26-28 and p53.

Folds g(mu) (script 01) over the Monte-Carlo distribution of colliding-wall energies.
For a collision of walls with gamma_1 m and gamma_2 m:  N/A ∝ e1 e2 eperp^3 g( M / (sqrt(gamma_1 gamma_2) m eperp) )   [SM Eq. (S10)]
tanh wall: gamma_i m = 6 (E_i/A)/Dphi^2 = 6 (L/H_c) e_i / Dphi^2; with L/H_c = sqrt(3 alpha L) M_P and Dphi = nu L^{1/4}:
   gamma_i m / M_P = 6 sqrt(3 alpha) e_i / nu^2                                                    [SM Sec. S7]
R(M) = <w e1 e2 h^3 g(M/(sqrt(g1 g2) m h))> / <w e1 e2 h^3>   (h = eperp, w = Monte-Carlo surface weight)
   -> Y(M) = kappa_Y chat (L^{1/4}/M_P)^3 R(M).
The samples of the seeds in CFG['seeds'] are pooled; the per-seed R is stored as Rseeds_*.
Output: output/04_mass_dependence.npz with R(M/M_P) per (alpha, nu, spin)."""
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import anli, bubbles_mc as mc

CFG = dict(alphas=[0.05, 5e-3, 5e-4], n=6, Lbox=6.0, seeds=(21, 31, 41, 51, 71), npts=1_000_000, eps=0.01,
           nus=[1.0, 1.5, 2.0, 3.0], M_over_MP=np.logspace(-8, 0.5, 150), spins=["scalar", "conformal", "dirac", "vector"],
           profile="tanh")
U = np.load(ROOT / "output/01_universality.npz")
mu_tab = U["mu"]


def g_of(mu, spin):
    gtab = U[f"g_{CFG['profile']}_{spin}"]
    lg = np.interp(np.log(mu), np.log(mu_tab), np.log(np.maximum(gtab, 1e-300)), left=0.0, right=-700)
    return np.exp(lg)


out = dict(M_over_MP=CFG["M_over_MP"])
for alpha in CFG["alphas"]:
    sol = anli.solve(alpha, n=CFG["n"])
    bg = mc.Background(sol)
    # pooled over independent nucleation and sample-point seeds. The surface weight w is not normalised per run, so with equal npts
    # concatenation is the equal-weight average.
    parts, Qs, sid = [], [], []
    for sd in CFG["seeds"]:
        tn, xn = mc.nucleate(bg, CFG["Lbox"], np.random.default_rng(sd))
        q = mc.surface_Q(bg, tn, xn, CFG["Lbox"], np.random.default_rng(sd + 1), npts=CFG["npts"], eps=CFG["eps"])
        parts.append(q["samples"][:, 3:7]); Qs.append(q["Q"]); sid.append(np.full(len(q["samples"]), sd))
    S = np.concatenate(parts); sid = np.concatenate(sid)
    e1, e2, w, h = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
    out[f"a{alpha}_e1"] = e1; out[f"a{alpha}_e2"] = e2; out[f"a{alpha}_w"] = w; out[f"a{alpha}_h"] = h; out[f"a{alpha}_Q"] = np.mean(Qs)
    out[f"a{alpha}_seed"] = sid; out[f"a{alpha}_Q_seeds"] = np.array(Qs)
    q = dict(Q=np.mean(Qs))
    W0 = np.sum(w * e1 * e2 * h ** 3)
    for nu in CFG["nus"]:
        gm = 6 * np.sqrt(3 * alpha) * np.sqrt(e1 * e2) / nu ** 2 * h      # effective head-on scale sqrt(g1 g2) m h / M_P (oblique: s = 4 p1 p2 h^2)
        out[f"a{alpha}_nu{nu}_gm_median"] = np.median(gm)
        # yield-weighted quantiles of the effective collision scale (weights w e1 e2 h^3: the M << gamma m yield) and the wall boost
        # gamma m (without h) for the friction condition
        wt = w * e1 * e2 * h ** 3; o = np.argsort(gm); cw = np.cumsum(wt[o]) / wt.sum()
        out[f"a{alpha}_nu{nu}_gm_wmed"] = gm[o][np.searchsorted(cw, 0.5)]
        out[f"a{alpha}_nu{nu}_gm_w90"] = gm[o][np.searchsorted(cw, 0.9)]
        out[f"a{alpha}_nu{nu}_frac_above_half"] = wt[gm > 0.5].sum() / wt.sum()   # wall scale > M_P/2 (wall scale, not the parton sqrt(s) of script 21)
        gw = gm / h; o2 = np.argsort(gw); cw2 = np.cumsum(wt[o2]) / wt.sum()
        out[f"a{alpha}_nu{nu}_wallgm_wmed"] = gw[o2][np.searchsorted(cw2, 0.5)]
        sidx = np.searchsorted(np.array(CFG["seeds"]), sid); Wsd = np.bincount(sidx, weights=wt)
        for sp in CFG["spins"]:
            R = np.empty(len(CFG["M_over_MP"])); Rsd = np.empty((len(CFG["seeds"]), len(CFG["M_over_MP"])))
            for j, Mx in enumerate(CFG["M_over_MP"]):
                v = wt * g_of(Mx / gm, sp)
                R[j] = v.sum() / W0
                Rsd[:, j] = np.bincount(sidx, weights=v, minlength=len(CFG["seeds"])) / Wsd
            out[f"R_a{alpha}_nu{nu}_{sp}"] = R
            out[f"Rseeds_a{alpha}_nu{nu}_{sp}"] = Rsd     # per-seed R: sampling spread of the mass dependence (not used downstream)
        print(f"alpha={alpha} nu={nu}: median sqrt(g1g2) m h / M_P = {np.median(gm):.3f}; yield-weighted median "
              f"{out[f'a{alpha}_nu{nu}_gm_wmed']:.3f}, 90% {out[f'a{alpha}_nu{nu}_gm_w90']:.3f}, yield frac above 0.5 M_P "
              f"{out[f'a{alpha}_nu{nu}_frac_above_half']:.2f}; wall gamma m (w-med) {out[f'a{alpha}_nu{nu}_wallgm_wmed']:.3f}; Q={q['Q']:.3f}", flush=True)
np.savez(ROOT / "output/04_mass_dependence.npz", **out)
