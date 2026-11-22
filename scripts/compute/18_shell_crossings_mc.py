"""18: later shell crossings with the exact free-passage geometry.

Main text: lower-bound paragraph ("raises the yield by up to a factor of five if the hard quanta survive"). SM: Sec. S8 and Table SIII
(S_cr ≈ 4.3 for alpha = 0.05 and 5.1 for alpha = 5e-4). Long run (~3 h, ~0.5 GB).

All pairs of free-passage shells arriving simultaneously at a point contribute (gravdm.bubbles_mc.crossing_Q) with the first-collision
weight; a shell gains energy only until it leaves the false vacuum (e_hat_capped). S_cr = Q_all/Q_first.
Degradation of the hard component per crossing f = exp(-rank/k_th), k_th in {inf, 100, 10, 3} [ASSUMED model].
Truncations: shells are trusted to r < rmax = Lbox/2 (periodic images); generations up to G. Convergence: two boxes; the generation tail
is fitted with a power law to estimate the contribution of g > G.
Output: output/18_shell_crossings_mc.npz"""
import sys, pathlib, time, json
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import anli, bubbles_mc as mc

CFG = dict(alphas=[0.05, 5e-4], boxes=[6.0, 8.0], seed=11, npts=200_000, eps=0.02, G=30)
if len(sys.argv) > 1:
    CFG.update(json.loads(sys.argv[1]))
out = {}
for alpha in CFG["alphas"]:
    bg = mc.Background(anli.solve(alpha, n=6))
    for Lb in CFG["boxes"]:
        t0 = time.time()
        tn, xn = mc.nucleate(bg, Lb, np.random.default_rng(CFG["seed"]))
        q = mc.crossing_Q(bg, tn, xn, Lb, np.random.default_rng(CFG["seed"] + 7), npts=CFG["npts"], eps=CFG["eps"], chunk=5000, G=CFG["G"])
        Qg = q["Q_by_gen"]; g = np.arange(len(Qg))
        # power-law fit of the last 10 generations -> tail estimate beyond G
        sl = slice(CFG["G"] - 9, CFG["G"] + 1)
        p = np.polyfit(np.log(g[sl]), np.log(np.maximum(Qg[sl], 1e-300)), 1)
        tail = np.sum(np.exp(np.polyval(p, np.log(np.arange(CFG["G"] + 1, 5000))))) if p[0] < -1 else np.inf
        S = Qg[1:].sum() / Qg[1]
        key = f"a{alpha}_L{Lb}"
        out[key + "_Qg"] = Qg; out[key + "_S"] = S; out[key + "_tail"] = tail / Qg[1]; out[key + "_pfit"] = p[0]
        for kt, v in q["Q_tot"].items():
            out[key + f"_S_kth{kt}"] = v / Qg[1]
        print(f"alpha={alpha} L={Lb}: bubbles={len(tn)} Q1={Qg[1]:.4f} S(g<={CFG['G']})={S:.3f}, tail fit slope {p[0]:.2f} -> +{tail/Qg[1]:.3f};"
              f" S(k_th=100,10,3)={[round(q['Q_tot'][k]/Qg[1],3) for k in (100.0, 10.0, 3.0)]} ({time.time()-t0:.0f}s)", flush=True)
        np.savez(ROOT / "output/18_shell_crossings_mc.npz", **out)
