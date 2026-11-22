"""p01b: SM Fig. S1, mass-suppression function g(mu), mu = M_X/(gamma m), for two colliding planar walls (xi_phi = 0).

(a) g(mu) for the minimally coupled scalar, the conformally coupled scalar and the Dirac fermion, tanh profile;
(b) Dirac g(mu) for the tanh, sine-Gordon and error-function profiles.
Dirac 10% points (log interpolation): mu = 0.42 (tanh), 0.53 (sine-Gordon), 0.68 (erf), as quoted in SM Sec. S2.
Reads output/01_universality.npz (script 01). Writes plots/p01b_universality.{pdf,png}."""
import sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import paper_plotstyle as ps

CFG = dict(ylim=(1e-6, 2), profiles=(("tanh", "-", r"tanh ($\phi^4$)"), ("sG", "--", "sine-Gordon"), ("erf", ":", "erf")))

ps.apply()
d = np.load(ROOT / "output/01_universality.npz")
mu = d["mu"]
fig, (a, b) = plt.subplots(1, 2, figsize=(ps.W2, 2.6))
spec = [("scalar", "c1", "-", r"scalar, $\xi_X=0$ (= Proca shape)"), ("conformal", "c4", "--", r"scalar, $\xi_X=1/6$"),
        ("dirac", "c2", "-", "Dirac")]
for case, col, ls, lab in spec:
    a.plot(mu, d[f"g_tanh_{case}"], color=ps.COL[col], ls=ls, label=lab)
a.set(xscale="log", yscale="log", xlabel=r"$\mu = M_X/(\gamma m)$", ylabel=r"$g(\mu)$", ylim=CFG["ylim"])
a.legend(loc="lower left", title="tanh wall", title_fontsize=6.5); ps.panel(a, "(a)", "upper right")
for prof, ls, lab in CFG["profiles"]:
    b.plot(mu, d[f"g_{prof}_dirac"], color=ps.COL["c2"], ls=ls, label=lab)
b.set(xscale="log", yscale="log", xlabel=r"$\mu = M_X/(\gamma m)$", ylabel=r"$g(\mu)$", ylim=CFG["ylim"])
b.legend(loc="lower left", title="Dirac fermion, wall profile", title_fontsize=6.5); ps.panel(b, "(b)", "upper right")
fig.subplots_adjust(wspace=0.28)
ps.save(fig, ROOT / "plots/p01b_universality")
