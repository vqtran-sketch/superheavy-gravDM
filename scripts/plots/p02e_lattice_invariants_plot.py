"""p02e: SM Fig. S2, ratio of lattice to parton results for the Fourier-transformed stress tensor in a kink-antikink collision.

(a) Along p_1 = p_2, for gamma = 40, 80, 160 (sequential blues): I_2 (spin-2, solid with circles) and I_0 (trace, dashed with squares);
thin dotted: I_2 of the two walls superposed without interaction (with their potential energy). Band +-10%. Curves start at
p_1 = p_2 = 2m (below, the potential cross term ~ 2 (m/p)^2 dominates; script 02i).
(b) I_2^lattice / I_2^parton over the (p_1, p_2) plane at gamma = 160, diverging colors centered on 1, clipped at 1 +- CFG["clip"];
contour at 0.99; region p_1, p_2 between 2m and 1.5 gamma m, drawn on the grid of script 02f.
All lattice values Richardson-extrapolated from 20 and 40 points per wall (as script 02g): R_40 + (R_40 - R_20)/3.
matplotlib places the "0.99" contour label automatically; its position can differ between matplotlib versions.
Reads output/02f_lattice_dense_g{gamma}[_n20].npz. Writes plots/p02e_lattice_invariants.{pdf,png}."""
import sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import paper_plotstyle as ps

CFG = dict(gammas=(40, 80, 160), band=0.10, ylim=(0.6, 1.6), every=2, p_min=2.0, xlim=(5e-3, 2.0),
           heat_gamma=160, x_max=1.5, clip=0.03, a_min=0.5, a_max_over_gamma=3.0, n=28, contours=(0.01,), cmap="RdBu_r")


def ratios(g):
    """Richardson-extrapolated lattice/parton ratios on the 28 x 28 grid: a, b, r2, r0, f2 (free superposition I_2)."""
    d40 = np.load(ROOT / f"output/02f_lattice_dense_g{g}.npz"); d20 = np.load(ROOT / f"output/02f_lattice_dense_g{g}_n20.npz")
    n = int(round(np.sqrt(len(d40["a"]))))
    ex = lambda k1, k2: (lambda r40, r20: (r40 + (r40 - r20) / 3).reshape(n, n))(d40[k1] / d40[k2], d20[k1] / d20[k2])
    return d40["a"].reshape(n, n), d40["b"].reshape(n, n), ex("I2l", "I2p"), ex("I0l", "I0p"), ex("I2f", "I2p")


ps.apply()
fig, (a, h) = plt.subplots(1, 2, figsize=(ps.W2, 2.6), gridspec_kw=dict(width_ratios=[1, 1.12], wspace=0.32))
for g, c in zip(CFG["gammas"], ps.COL["seq"]):
    A, B, r2, r0, f2 = ratios(g)
    p = np.diag(A); k = p >= CFG["p_min"]; x = p[k] / g
    a.plot(x, np.diag(r2)[k], color=c, marker="o", ms=ps.MS, markevery=CFG["every"], label=rf"$\gamma={g}$, $I_2$")
    a.plot(x, np.diag(r0)[k], color=c, ls="--", marker="s", ms=ps.MS, markevery=CFG["every"], label=rf"$\gamma={g}$, $I_0$")
    a.plot(x, np.diag(f2)[k], color=c, ls=":", lw=ps.LW["ref"])
a.axhspan(1 - CFG["band"], 1 + CFG["band"], color=ps.COL["band"], alpha=0.5, lw=0, zorder=0)
a.axhline(1, color=ps.COL["ref"], lw=ps.LW["ref"])
a.set(xscale="log", xlabel=r"$p_1/(\gamma m)=p_2/(\gamma m)$", ylabel="lattice / parton", ylim=CFG["ylim"], xlim=CFG["xlim"])
a.legend(loc="upper right", fontsize=5.4, ncol=2)
ps.panel(a, "(a)", loc="lower right")

g = CFG["heat_gamma"]; A, B, r2, _, _ = ratios(g)
xn = np.geomspace(CFG["a_min"], CFG["a_max_over_gamma"] * g, CFG["n"]) / g       # nominal 02f grid (k-snapping shifts a, b slightly)
keep = np.outer(xn * g >= CFG["p_min"], xn * g >= CFG["p_min"]) & np.outer(xn <= CFG["x_max"], xn <= CFG["x_max"])
i = np.where((xn * g >= CFG["p_min"]) & (xn <= CFG["x_max"]))[0]
xc = xn[i]; e = np.sqrt(xc[1:] * xc[:-1]); edges = np.concatenate([[xc[0] ** 2 / e[0]], e, [xc[-1] ** 2 / e[-1]]])
Z = r2[np.ix_(i, i)].T                                                            # rows: p_2 (y), columns: p_1 (x)
im = h.pcolormesh(edges, edges, Z, cmap=CFG["cmap"], norm=TwoSlopeNorm(1.0, 1 - CFG["clip"], 1 + CFG["clip"]), rasterized=True)
lv = sorted([1 - c for c in CFG["contours"]] + [1 + c for c in CFG["contours"]])
cs = h.contour(xc, xc, Z, levels=lv, colors="k", linewidths=ps.LW["ref"])
h.clabel(cs, fmt={v: f"{v:g}" for v in lv}, fontsize=5)
h.set(xlim=(edges[0], edges[-1]), ylim=(edges[0], edges[-1]))
h.set(xscale="log", yscale="log", xlabel=r"$p_1/(\gamma m)$", ylabel=r"$p_2/(\gamma m)$")
cb = fig.colorbar(im, ax=h, extend="both"); cb.set_label(r"$I_2^{\rm lattice}/I_2^{\rm parton}$, $\gamma=160$")
ps.panel(h, "(b)", loc="lower right")
ps.save(fig, ROOT / "plots/p02e_lattice_invariants")
