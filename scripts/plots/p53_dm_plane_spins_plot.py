"""p53: SM Fig. S5, latent heat vs DM mass for Omega h^2 = 0.12 in the An-Li scenario, for the minimal scalar, the Dirac fermion and the Proca vector.

(a) x = M_X, y = L^{1/4}; (b) present-day GW peak frequency along the same curves (bulk-flow model of An:2026ghw). nu = 3, n_DS = 6,
xi_phi = 0; alpha = 0.05 (solid) and 5e-4 (dashed).
Curves: total yield (collisions + thermal freeze-in), shown where freeze-in < 10% of the collision yield (gravdm.window),
L^{1/4} < alpha^{1/4} V_inf^{1/4} [V_inf^{1/4} = 1.4e16 GeV, LIT: CMB tensor bound, BICEP:2021xfz, Planck:2018jri] and M_X < M_P/4.
Thick: R >= 0.1 (M_X below the knee); thin: R < 0.1 (exponentially sensitive to the wall profile and to the tail of the
wall-energy distribution).
Faint: the low-mass part of the minimal-scalar curve at alpha = 0.05, where freeze-in stays below 10% of the collision yield for all
M_X << T_* (FI/coll = 0.086), so that freeze-in is not under control there; drawn below the Dirac lower edge at the same alpha.
The CFG entries Smax and xi_band are not used in this figure.
Reads output/06_dm_plane.npz (script 06) and output/04_mass_dependence.npz. Writes plots/p53_dm_plane_spins.{pdf,png}."""
import sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import window
from gravdm import paper_plotstyle as ps
from gravdm.units import M_P

CFG = dict(Vinf4=1.4e16, Mmax_frac=0.25, Rcore=0.1, FImax=0.1,
           cases=[(0.05, 3.0, "-"), (5e-4, 3.0, "--")], Smax={0.05: 4.3, 5e-4: 5.1}, xi_band=(1 / 6, 1.0),
           spins=[("scalar", "c1", r"scalar $\xi_X=0$"), ("dirac", "c2", "Dirac"), ("vector", "c3", "Proca")])
ps.apply()
d = np.load(ROOT / "output/06_dm_plane.npz")
Mdep = np.load(ROOT / "output/04_mass_dependence.npz")
M = d["M_over_MP"] * M_P


def ok_mask(k, alpha, nu, sp):
    R = Mdep[f"R_a{alpha}_nu{nu}_{sp}"]
    ok = np.isfinite(d[k + "_L4"]) & (d[k + "_fcoll"] >= window.FCOLL_MIN) & (d[k + "_L4"] < alpha ** 0.25 * CFG["Vinf4"]) & (M < CFG["Mmax_frac"] * M_P)
    return ok, R


fig, (a, b) = plt.subplots(1, 2, figsize=(ps.W2, 2.8))
for alpha, nu, ls in CFG["cases"]:
    for sp, col, lab in CFG["spins"]:
        k = f"a{alpha}_nu{nu}_{sp}"
        L4 = d[k + "_L4"]; f0 = d[k + "_f0"]
        ok, R = ok_mask(k, alpha, nu, sp)
        core = ok & (R >= CFG["Rcore"]); tail = ok & (R < CFG["Rcore"])
        okD, RD = ok_mask(f"a{alpha}_nu{nu}_dirac", alpha, nu, "dirac")
        marg = core & (M < M[okD & (RD >= CFG["Rcore"])].min()) if core[0] else np.zeros_like(core)
        core = core & ~marg
        lbl = f"{lab}" if alpha == 0.05 else None
        a.plot(M, np.where(core, L4, np.nan), color=ps.COL[col], ls=ls, lw=ps.LW["main"], label=lbl)
        mj = marg | (core & np.roll(marg, 1))       # include the first core point so the two segments join
        a.plot(M, np.where(mj, L4, np.nan), color=ps.COL[col], ls=ls, lw=ps.LW["second"], alpha=0.4)
        b.plot(np.where(mj, M, np.nan), f0, color=ps.COL[col], ls=ls, lw=ps.LW["second"], alpha=0.4)
        jn = tail | np.roll(core, 1) | np.roll(core, -1)
        a.plot(M, np.where(jn, L4, np.nan), color=ps.COL[col], ls=ls, lw=ps.LW["ref"] + 0.2)
        b.plot(np.where(core, M, np.nan), f0, color=ps.COL[col], ls=ls, lw=ps.LW["main"])
        b.plot(np.where(jn, M, np.nan), f0, color=ps.COL[col], ls=ls, lw=ps.LW["ref"] + 0.2)
a.set(xscale="log", yscale="log", ylabel=r"$L^{1/4}$ [GeV]  ($\Omega_X h^2=0.12$)", xlabel=r"$M_X$ [GeV]",
      ylim=(5e10, 1e14), xlim=(1e12, 1e18))
a.plot([], [], color="k", ls="-", label=r"$\alpha=0.05$"); a.plot([], [], color="k", ls="--", label=r"$\alpha=5\times10^{-4}$")
a.legend(loc="lower left", fontsize=5.8, ncol=2); ps.panel(a, "(a)", "upper left")
b.set(xscale="log", yscale="log", xlabel=r"$M_X$ [GeV]", ylabel=r"$f_{\rm peak}$ today [Hz]", xlim=(1e12, 1e18))
b.axhspan(1e1, 1e4, color=ps.COL["band"], alpha=0.5, lw=0, zorder=0)   # frequency range of ground-based interferometers
b.text(2e12, 2e2, "ground-based\ninterferometers", fontsize=6, color="0.35")
ps.panel(b, "(b)", "upper left")
fig.subplots_adjust(wspace=0.3)
ps.save(fig, ROOT / "plots/p53_dm_plane_spins")
