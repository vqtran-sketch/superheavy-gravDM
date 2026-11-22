"""02: 1+1-dimensional kink-antikink collision in the phi^4 potential at constant gamma.

Lattice machinery of the main text and SM: its function one() is reused by script 02f (main-text Fig. 1, SM Sec. S3, Figs. S2, S3). The
stored coarse-grid output of this script itself is not quoted in the main text or the SM.

Three tensors per run: the full nonlinear lattice evolution ("lat"), the exact superposition of the two walls passed through the same
pipeline ("free"), and the closed-form parton cross term ("par", SM Eq. (S8)). Invariants I2 (spin-2) and I0 (trace) of the
Gaussian-windowed Fourier transform of T^{mu nu} (SM Sec. S3) at light-cone momenta (a, b) = (p_1, p_2), on a logarithmic 7x7 grid
a, b in [2m, 0.8 gamma m]. Convergence variants at gamma = 20: twice the points per wall, and a window 1.5 times wider.
Units: m = 1. Output: output/02_lattice.npz"""
import sys, pathlib, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import lattice1d as L

CFG = dict(
    gammas=[10.0, 20.0, 40.0],
    base=dict(Tw=4.0, pts_per_wall=20, tc_over_Tw=6.0, sample_dt=0.01),
    conv_gamma=20.0,
    conv=dict(fine=dict(pts_per_wall=40), wide=dict(Tw=6.0)),
    n_ab=7,
)


def targets(gamma):
    ab = np.geomspace(2.0, 0.8 * gamma, CFG["n_ab"])
    return [(a, b) for a in ab for b in ab]


def one(gamma, **over):
    p = dict(CFG["base"]); p.update(over)
    tg = targets(gamma)
    rl = L.run(gamma, targets=tg, free_passage=False, **p)
    rf = L.run(gamma, targets=tg, free_passage=True, **p)
    par = L.parton_tensor(rl["a"], rl["b"], gamma)
    I2l, I0l = L.invariants(rl["T"]); I2f, I0f = L.invariants(rf["T"]); I2p, I0p = L.invariants(par)
    # conservation diagnostic k_mu T~^{mu nu} / |T~| (window breaks it at O(1/(omega Tw)))
    kv = np.stack([rl["omega"], rl["k"], 0 * rl["k"], 0 * rl["k"]], 1)
    kl = kv @ L.ETA4
    cons = np.linalg.norm(np.einsum("nm,nmv->nv", kl, rl["T"]), axis=1) / (np.linalg.norm(rl["T"], axis=(1, 2)) * np.linalg.norm(kv, axis=1))
    E = rl["Elog"][:, 1]
    return dict(a=rl["a"], b=rl["b"], I2l=I2l, I0l=I0l, I2f=I2f, I0f=I0f, I2p=I2p, I0p=I0p, cons=cons,
                Edrift=np.max(np.abs(E / E[0] - 1)), N=rl["N"], dx=rl["dx"])


if __name__ == "__main__":
    out = {}
    for g in CFG["gammas"]:
        t0 = time.time(); r = one(g)
        for k, v in r.items():
            out[f"g{int(g)}_{k}"] = v
        h = (r["a"] > 4) & (r["b"] > 4)
        print(f"gamma={g}: N={r['N']} Edrift={r['Edrift']:.1e} t={time.time()-t0:.0f}s | hard (a,b>4m): "
              f"I2 lat/par in [{(r['I2l']/r['I2p'])[h].min():.3f},{(r['I2l']/r['I2p'])[h].max():.3f}], "
              f"I0 lat/par in [{(r['I0l']/r['I0p'])[h].min():.3f},{(r['I0l']/r['I0p'])[h].max():.3f}], "
              f"max cons {r['cons'][h].max():.1e}", flush=True)
    for name, over in CFG["conv"].items():
        r = one(CFG["conv_gamma"], **over)
        for k, v in r.items():
            out[f"conv_{name}_{k}"] = v
        base_I2 = out[f"g{int(CFG['conv_gamma'])}_I2l"]
        print(f"convergence {name}: max |I2/I2_base - 1| = {np.max(np.abs(r['I2l']/base_I2-1)):.2e}, "
              f"I0: {np.max(np.abs(r['I0l']/out[f'g{int(CFG['conv_gamma'])}_I0l']-1)):.2e}", flush=True)
    np.savez(ROOT / "output/02_lattice.npz", **out, cfg=str(CFG))
