"""19: first-collision Monte Carlo (gravdm.bubbles_mc) for alpha = 5e-3, n_DS = 6, with paired eps on the same nucleation and sample points.

SM: Sec. S7 (Monte-Carlo coefficient for alpha = 5e-3, and the box check; combined by script 05). Long run.

Usage (from the repository root):
  python scripts/compute/19_kappa_recheck.py L seeds eps npts chunk > output/19_kappa_recheck_L<L>.jsonl
  e.g. python scripts/compute/19_kappa_recheck.py 6 201,...,224 0.01,0.005 2000000 20000
Analysis (script 05): control variate on N_b/V with its exact mean int a^4 f_FV dtau, and linear eps -> 0 extrapolation."""
import sys, json, time
import numpy as np
from gravdm import anli, bubbles_mc as mc
from gravdm.units import G_STAR_SM
Lb = float(sys.argv[1]); seeds = [int(s) for s in sys.argv[2].split(",")]; epss = [float(e) for e in sys.argv[3].split(",")]
npts = int(sys.argv[4]); chunk = int(sys.argv[5])
alpha = 5e-3
sol = anli.solve(alpha, n=6, N=6000); bg = mc.Background(sol)
c_hat = (sol["a"][-1] ** 4 * sol["rho_rad"][-1] + sol["a"][-1] ** 4 * sol["rho_V"][-1]) / sol["L"]
pref = 8 * np.sqrt(3 * alpha) / ((2 * np.pi ** 2 / 45) * G_STAR_SM * (30 * c_hat / (np.pi ** 2 * G_STAR_SM)) ** 0.75) / (960 * np.pi)
for sd in seeds:
    tn, xn = mc.nucleate(bg, Lb, np.random.default_rng(sd))
    for eps in epss:
        t0 = time.time()
        q = mc.surface_Q(bg, tn, xn, Lb, np.random.default_rng(sd + 1000), npts=npts, eps=eps, chunk=chunk)
        print(json.dumps(dict(L=Lb, seed=sd, eps=eps, N=len(tn), nb=len(tn) / Lb**3, Q=q["Q"], Qh=q["Q_head"], AV=q["AV"],
                              kappa=pref * q["Q"], c_hat=c_hat, sec=round(time.time() - t0))), flush=True)
