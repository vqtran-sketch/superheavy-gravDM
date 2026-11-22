"""p02h: main-text Fig. 1 (M_X = 1e-3 gamma m) and SM Fig. S3 (M_X = 0.2 gamma m): differential pair yield of a Dirac X along p_1 = p_2.

Upper panel: p_1 p_2 d^2(N/A)/dp_1 dp_2 in units of 4 c_X(0) Dphi^4 gamma^2 m^2/(pi^2 M_P^4), linear y (the area is the yield per
ln p_1 ln p_2). Curve: parton prediction. Points: lattice at gamma = 80 (circles) and 160 (squares), extrapolated to zero lattice
spacing, for p <= 2 gamma m (parton weight beyond < 1e-5 of the peak). Lower panel: ratio of lattice to parton. Error bars:
numerical uncertainty of the lattice (script 02h: lattice spacing, Fourier window, sampling step).
Reads output/02h_lattice_diag_yield.npz. Writes plots/p02h_lattice_diag_yield_perlog[_mu0p2].{pdf,png}
(no suffix: mu = 1e-3, main-text Fig. 1; _mu0p2: SM Fig. S3)."""
import sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import paper_plotstyle as ps

CFG = dict(gammas=(80, 160), marker={80: "o", 160: "s"}, col={80: "#d62728", 160: ps.COL["c1"]},
           ls={1e-3: "-"}, hspace=0.16, xlim={1e-3: (1e-2, 2.0), 0.2: (0.1, 2.0)}, x_max=2.0, rlim=(0.9, 1.1), forms=("perlog",))

ps.apply()
d = np.load(ROOT / "output/02h_lattice_diag_yield.npz"); xf = d["xfine"]
for mu in d["mus"]:
  for form in CFG["forms"]:
    fig, (a, b) = plt.subplots(2, 1, figsize=(ps.W1, 3.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.2], hspace=CFG["hspace"]))
    wf = (lambda x: x ** 2) if form == "perlog" else (lambda x: 1.0)
    y = d[f"par_mu{mu}"] * wf(xf); m = y > 0
    a.plot(xf[m], y[m], color=ps.COL["data"], ls="-", lw=ps.LW["second"], label="parton")
    for g in CFG["gammas"]:
        x = d[f"g{g}_mu{mu}_x"]; lat = d[f"g{g}_mu{mu}_lat"]; R = d[f"g{g}_mu{mu}_ratio"]; m = (d[f"g{g}_mu{mu}_par"] > 0) & (x <= CFG["x_max"])
        sR = d[f"g{g}_mu{mu}_sigR"] if f"g{g}_mu{mu}_sigR" in d.files else np.zeros_like(R)
        kw = dict(ls="none", marker=CFG["marker"][g], ms=ps.MS + 0.5, mfc=CFG["col"][g], mec=CFG["col"][g], mew=0.8, ecolor=CFG["col"][g],
                  elinewidth=0.8, capsize=1.5)
        a.errorbar(x[m], lat[m] * wf(x[m]), yerr=(d[f"g{g}_mu{mu}_par"] * sR)[m] * wf(x[m]), label=rf"lattice, $\gamma={g}$", **kw)
        b.errorbar(x[m], R[m], yerr=sR[m], **kw)
    b.axhline(1, color=ps.COL["ref"], lw=ps.LW["ref"])
    if form == "dens":
        a.set(yscale="log", ylim=(1e-4, 2), ylabel=r"$d^2(N_{\rm ev}/A)/dp_1dp_2$ [norm.]")
    else:
        a.set(ylim=(0, None), ylabel=r"$p_1p_2\,d^2(N_{\rm ev}/A)/dp_1dp_2$ [norm.]")
    a.set(xscale="log", xlim=[v for k, v in CFG["xlim"].items() if np.isclose(k, mu)][0])
    mlab = r"10^{-3}" if np.isclose(mu, 1e-3) else f"{mu:g}"
    a.legend(loc="lower left" if form == "dens" else "upper left", fontsize=5.6, title=rf"Dirac $X$, $M_X={mlab}\,\gamma m$", title_fontsize=5.6)
    b.set(xlabel=r"$p_1/(\gamma m)=p_2/(\gamma m)$", ylabel="lat./parton", ylim=CFG["rlim"])
    suffix = "" if np.isclose(mu, 1e-3) else "_mu" + f"{mu:g}".replace(".", "p")
    fig.align_ylabels([a, b]); ps.save(fig, ROOT / f"plots/p02h_lattice_diag_yield_{form}{suffix}"); plt.close(fig)
