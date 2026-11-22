"""05: combine the Monte-Carlo runs of scripts 03 and 19 into kappa_Y(alpha, n_DS) with statistical and systematic errors.

SM: Sec. S7, "Yield" (the Monte-Carlo coefficient used for the DM window, known to about 1%). Input to scripts 06, 21 and 24.

Data: box L = 6/(a_c H_c), 24 seeds per point, paired eps = 0.01 and 0.005 on the same nucleation and sample points
      (output/03_yield_mcc1-c3.npz; alpha = 5e-3 from script 19 with the same settings). Box check: L = 8 (03_yield_mcc4, script 19).
  1. eps -> 0 per seed: kappa_0 = 2 kappa(0.005) - kappa(0.01). The surface estimator has an O(eps) deficit from points near triple
     lines (tested on a Poisson-Voronoi tessellation in tests/test_cosmo_mc.py).
  2. control variate on the bubble number: kappa_cv = kappa_0 - b (N_b/V - E[N_b/V]), b = cov/var over seeds,
     E[N_b/V] = int a^4 f_FV dtau (exact for the Poisson nucleation of gravdm.bubbles_mc.nucleate).
  stat = std(kappa_cv)/sqrt(N_seeds);  sys_eps = |kappa_0 - kappa(0.005)|/2 (half the extrapolation shift);
  sys_box = |kappa(L=8) - kappa(L=6)| (measured for alpha = 0.05 and 5e-3; the larger of the two is used for the other points).
Also fits kappa_Y ∝ alpha^p over the n_DS = 6 points (expected p ≈ 1/2).
Output: output/05_kappaY.npz and a printed table."""
import sys, pathlib, json
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import anli

CFG = dict(L6={(0.05, 6): "c1", (5e-4, 6): "c2", (0.05, 5): "c3", (5e-3, 6): "19_L6"},
           L8={(0.05, 6): "c4", (5e-3, 6): "19_L8"}, eps=(0.01, 0.005), Nbg=6000)


def load(tag):
    """-> dict eps -> arrays (seed, N_b/V, kappa, Q, AV, c_hat), sorted by seed."""
    if tag.startswith("19_"):
        R = [json.loads(l) for l in open(ROOT / f"output/19_kappa_recheck_{tag[3:]}.jsonl")]
        rows = np.array([[r["seed"], r["eps"], r["nb"], r["kappa"], r["Q"], r["AV"], r["c_hat"]] for r in R])
    else:
        d = np.load(ROOT / f"output/03_yield_mc{tag}.npz")["rows"]   # alpha n Lbox seed eps Nb Q AV c_hat kappa ...
        rows = np.c_[d[:, 3], d[:, 4], d[:, 5] / d[:, 2] ** 3, d[:, 9], d[:, 6], d[:, 7], d[:, 8]]
    out = {}
    for e in CFG["eps"]:
        r = rows[np.isclose(rows[:, 1], e)]
        out[e] = r[np.argsort(r[:, 0])]
    assert np.array_equal(out[0.01][:, 0], out[0.005][:, 0]), f"{tag}: unpaired seeds"
    return out


def estimate(tag, nb_exact):
    D = load(tag)
    k1, k2 = D[0.01][:, 3], D[0.005][:, 3]
    k0 = 2 * k2 - k1
    nb = D[0.01][:, 2]
    b = np.cov(k0, nb)[0, 1] / np.var(nb, ddof=1)
    kcv = k0 - b * (nb - nb_exact)
    q0 = 2 * D[0.005][:, 4] - D[0.01][:, 4]; av0 = 2 * D[0.005][:, 5] - D[0.01][:, 5]
    return dict(kappa=kcv.mean(), stat=kcv.std(ddof=1) / np.sqrt(len(kcv)), raw005=k2.mean(), raw0=k0.mean(), n=len(kcv),
                eps_shift=np.mean(k2 / k1 - 1), Q=kcv.mean() * q0.mean() / k0.mean(), AV=av0.mean(), c_hat=D[0.01][0, 6], nb=nb.mean())   # Q gets the same CV (kappa ∝ Q per run)


res, box = {}, {}
for key, tag in CFG["L6"].items():
    sol = anli.solve(key[0], n=key[1], N=CFG["Nbg"])
    nb_exact = np.trapezoid(sol["a"] ** 4 * sol["f"], sol["tau"])
    res[key] = estimate(tag, nb_exact) | dict(nb_exact=nb_exact)
    if key in CFG["L8"]:
        r8 = estimate(CFG["L8"][key], nb_exact)
        box[key] = abs(r8["kappa"] - res[key]["kappa"]) / res[key]["kappa"]
        print(f"box check {key}: L=8 {r8['kappa']:.4e} ± {r8['stat']:.1e} ({r8['n']} seeds) vs L=6 {res[key]['kappa']:.4e}"
              f"  -> |diff| = {100*box[key]:.2f}%")
box_default = max(box.values())

rows = []
for (alpha, n) in sorted(res, key=lambda x: (x[1], -x[0])):
    r = res[(alpha, n)]
    k = r["kappa"]
    sys_eps = abs(r["raw0"] - r["raw005"]) / 2      # half the eps extrapolation (same seeds, before the control variate)
    sys_box = box.get((alpha, n), box_default) * k
    tot = np.sqrt(r["stat"] ** 2 + sys_box ** 2 + sys_eps ** 2)
    rows.append([alpha, n, k, r["stat"], sys_box, sys_eps, tot, r["Q"], r["AV"], r["c_hat"]])
    print(f"alpha={alpha:g} n={int(n)}: kappa_Y = {k:.3e} ± {r['stat']:.1e}(stat) ± {sys_box:.1e}(box) ± {sys_eps:.1e}(eps)"
          f"  [tot {100*tot/k:.1f}%]  seeds={r['n']}  eps-shift(0.01->0.005)={100*r['eps_shift']:+.2f}%  N_b/V={r['nb']:.3f}"
          f" (exact {r['nb_exact']:.3f})  Q={r['Q']:.3f} A/V={r['AV']:.3f} c_hat={r['c_hat']:.3f}")
rows = np.array(rows)
m6 = rows[:, 1] == 6
p = np.polyfit(np.log(rows[m6, 0]), np.log(rows[m6, 2]), 1)
print(f"fit (n=6): kappa_Y ∝ alpha^{p[0]:.3f}  (expected 0.5);  kappa_Y/sqrt(alpha/0.05) = "
      + ", ".join(f"{k/np.sqrt(a/0.05):.3e}" for a, k in zip(rows[m6, 0], rows[m6, 2])))
np.savez(ROOT / "output/05_kappaY.npz", rows=rows, slope=p[0],
         header="alpha n kappa_Y stat sys_box sys_eps tot Q AV c_hat  (kappa_Y for c0 = 1/(960 pi), minimal scalar; extrapolated to eps->0)")
