"""03: first-collision Monte Carlo on the self-consistent background of the An-Li scenario -> yield coefficient kappa_Y.

SM: Sec. S7, "Yield" (the Monte-Carlo coefficient, known to about 1%, combined by script 05). Long run (hours).

Y_X = kappa_Y chat (L^{1/4}/M_P)^3 R(M_X)
  kappa_Y = 8 Q sqrt(3 alpha) / [ (2 pi^2/45) g* (30 c_hat/(pi^2 g*))^{3/4} ] / (960 pi)
  c_hat = lim a^4 rho_rad / L: comoving released energy in units of L; entropy is assumed conserved after thermalization [ASSUMED].
Q includes the oblique-collision factor eperp^3 (SM Eq. (S10)); Q_head is the head-on value (eperp = 1).
Settings (points, box sizes, seeds, eps, npts) can be passed as a JSON argument; the stored runs are output/03_yield_mcc1-c4.npz.
Output: output/03_yield_mc<tag>.npz"""
import sys, pathlib, time, itertools
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import anli, bubbles_mc as mc
from gravdm.units import G_STAR_SM

CFG = dict(points=[(0.05, 6), (5e-3, 6), (5e-4, 6), (0.05, 5)],
           boxes=[4.0, 6.0], seeds=[11, 12, 13], eps=[0.01, 0.02], npts=1_000_000, Nbg=6000)

import json
if len(sys.argv) > 1:                       # e.g. '{"points": [[0.05, 6]], "boxes": [6.0], "seeds": [14,15], "eps": [0.01], "tag": "b"}'
    CFG.update(json.loads(sys.argv[1]))
TAG = CFG.get("tag", "")
rows = []
for (alpha, n) in CFG["points"]:
    sol = anli.solve(alpha, n=n, N=CFG["Nbg"])
    bg = mc.Background(sol)
    c_hat = (sol["a"][-1] ** 4 * sol["rho_rad"][-1] + sol["a"][-1] ** 4 * sol["rho_V"][-1]) / sol["L"]
    for Lb, sd in itertools.product(CFG["boxes"], CFG["seeds"]):
        t0 = time.time()
        rng = np.random.default_rng(sd)
        tn, xn = mc.nucleate(bg, Lb, rng)
        for eps in CFG["eps"]:
            q = mc.surface_Q(bg, tn, xn, Lb, np.random.default_rng(sd + 1000), npts=CFG["npts"], eps=eps,
                             chunk=CFG.get("chunk", 100_000))      # smaller chunk = less memory (~25 kB x N_b per point)
            s = q["samples"]
            kY = 8 * q["Q"] * np.sqrt(3 * alpha) / ((2 * np.pi ** 2 / 45) * G_STAR_SM * (30 * c_hat / (np.pi ** 2 * G_STAR_SM)) ** 0.75) / (960 * np.pi)
            rows.append([alpha, n, Lb, sd, eps, len(tn), q["Q"], q["AV"], c_hat, kY,
                         np.mean(s[:, 0]), np.mean(s[:, 3]), np.percentile(s[:, 3], 90), q["Q_head"]])
            print(f"alpha={alpha:g} n={n} L={Lb} seed={sd} eps={eps}: bubbles={len(tn)} Q={q['Q']:.4f} (head-on {q['Q_head']:.4f}) A/V={q['AV']:.3f} "
                  f"c_hat={c_hat:.3f} kappa_Y={kY:.3e} <tau*>={np.mean(s[:,0]):.3f} <e>={np.mean(s[:,3]):.3f} ({time.time()-t0:.0f}s)", flush=True)
np.savez(ROOT / f"output/03_yield_mc{TAG}.npz", rows=np.array(rows),
         header="alpha n Lbox seed eps Nbubbles Q AV c_hat kappa_Y(c0=1/960pi) <tau*> <e_hat> e_hat_90pct Q_headon")
