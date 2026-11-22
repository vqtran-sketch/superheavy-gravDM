"""13: gravitational production of a Dirac fermion and a Proca vector at the transition from inflation to kination.

SM: Sec. S7, "Conditions on the inflationary era" (M_X/H_inf >~ 70 to 180 for an abrupt end, 9 to 21 for a smooth one, via scripts
14 and 20); Sec. S5 (the abrupt-end range used for the benchmark). Long run.

n a^3 (M/H0) [units H0^3, a(y=0) = 1] on the inflation-kination background of Chun:2009yu (tanh transition of width y0),
y0 in {0.1 (abrupt), 1.0 (smooth)}, with gravdm.infprod_spins:
  dirac:  n_{X+Xbar} a^3 = (2/pi^2) int dk k^2 |beta|^2  (two helicities, particles + antiparticles)
  vector: n a^3 = (1/(2 pi^2)) int dk k^2 (2|beta_T|^2 + |beta_L|^2)  (real field); T and L stored separately
Integration: trapezoid in ln k of k^3 |beta|^2 on a log grid [kmin, kmax(M, y0)]; in- and out-states first-order adiabatic; two y_out.
Convergence checks (prefix conv_): y_out (two values always), k-grid x2 (nk), kmax x1.5 with kmin/4, rtol/10.
Extra checks: chk_dirac_smallM_* (small-mass scaling of a light fermion, n_X/(M^2 H0) = A ln(H0/M) + B with A = 1/(24 pi));
  chk_scalar_kmax_*: minimal scalar at M = 1, 4 with the k-grid of script 08 vs the wider grid used here.
Resumable: results stored per key in output/13_inflationary_spins.npz; one main process + CFG['workers'] pool workers.
Usage: nice -n 15 env OMP_NUM_THREADS=1 python scripts/compute/13_inflationary_spins.py"""
import sys, pathlib, time, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import infprod_spins as sp

CFG = dict(y0s=[0.1, 1.0], Ms=[0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0], spins=["dirac", "vector"],
           Ms_extra={0.1: [24.0, 32.0]},     # abrupt only: tests the tail extrapolation that dominates the H_inf bound
           nk=32, kmin=2e-3, kmax_a={0.1: 30.0, 1.0: 8.0}, kmax_b=3.0,          # kmax = kmax_a(y0) + kmax_b M
           y_outs={0.1: (2.0, 2.5), 1.0: (2.5, 3.0)}, rtol=1e-10,   # smooth: later y_out (tiny n a^3, out-state floor ~ eps^4)
           workers=2,
           conv_points=[(0.1, 4.0), (1.0, 2.0)],                              # (y0, M) for the k-grid / kmax / rtol checks
           smallM=[1e-3, 4e-3, 1.6e-2], smallM_y0=0.1, smallM_kmin=1e-4, smallM_kmax=40.0, smallM_nk=48, smallM_yout=(3.5, 4.0),
           scalar_kmax_M=[1.0, 4.0], scalar08_nk=28)
OUT = ROOT / "output/13_inflationary_spins.npz"


def grid(y0, M, nk=None, kfac=1.0, kminfac=1.0):
    return np.geomspace(CFG["kmin"] * kminfac, kfac * (CFG["kmax_a"][y0] + CFG["kmax_b"] * M), nk or CFG["nk"])


def lnk_int(ks, f):
    """int dk k^2 g = int d ln k k^3 g (trapezoid in ln k); f = k^3 g on the grid, shape (nk, ...)."""
    return np.trapezoid(f, np.log(ks), axis=0)


def run(task):
    kind, y0, M, var = task
    bg = sp.AnalyticBackground(y0)
    t0 = time.time()
    rtol = CFG["rtol"] / (10 if var == "rtol" else 1)
    if kind in ("dirac", "vector"):
        ks = grid(y0, M, nk=2 * CFG["nk"] if var == "nk2" else None, kfac=1.5 if var == "kmax" else 1.0,
                  kminfac=0.25 if var == "kmax" else 1.0)
        if kind == "dirac":
            b2 = np.array([sp.beta2_dirac(bg, k, M, y_outs=CFG["y_outs"][y0], rtol=rtol) for k in ks])
            res = {"na3": 2 / np.pi ** 2 * lnk_int(ks, ks[:, None] ** 3 * b2), "ks": ks, "b2": b2}
        else:
            bT = np.array([sp.beta2_scalar_type(bg, k, M, "T", y_outs=CFG["y_outs"][y0], rtol=rtol) for k in ks])
            bL = np.array([sp.beta2_scalar_type(bg, k, M, "L", y_outs=CFG["y_outs"][y0], rtol=rtol) for k in ks])
            nT = 1 / (2 * np.pi ** 2) * lnk_int(ks, ks[:, None] ** 3 * 2 * bT)
            nL = 1 / (2 * np.pi ** 2) * lnk_int(ks, ks[:, None] ** 3 * bL)
            res = {"na3": nT + nL, "na3T": nT, "na3L": nL, "ks": ks, "b2T": bT, "b2L": bL}
    elif kind == "smallM":
        ks = np.geomspace(CFG["smallM_kmin"], CFG["smallM_kmax"], CFG["smallM_nk"])
        b2 = np.array([sp.beta2_dirac(bg, k, M, y_outs=CFG["smallM_yout"], rtol=rtol) for k in ks])
        res = {"nX": 1 / np.pi ** 2 * lnk_int(ks, ks[:, None] ** 3 * b2), "ks": ks, "b2": b2}   # particles only, 2 helicities
    elif kind == "scalar":
        ks8 = np.geomspace(0.02, 3.0 + 2.5 * M, CFG["scalar08_nk"])                            # script 08 grid
        ks = grid(y0, M)
        n8 = np.trapezoid(ks8 ** 2 * np.array([sp.beta2_scalar_type(bg, k, M, "S", y_outs=(2.5,))[0] for k in ks8]), ks8) / (2 * np.pi ** 2)
        b2 = np.array([sp.beta2_scalar_type(bg, k, M, "S", y_outs=(2.5,))[0] for k in ks])
        res = {"na3_08grid": np.array(n8), "na3_wide": np.array(lnk_int(ks, ks ** 3 * b2) / (2 * np.pi ** 2)), "ks": ks, "b2": b2}
    res["time"] = np.array(time.time() - t0)
    return task, res


def key(task):
    kind, y0, M, var = task
    pre = {"dirac": "", "vector": "", "smallM": "chk_dirac_smallM_", "scalar": "chk_scalar_kmax_"}[kind]
    base = f"{pre}y0{y0}_M{M}" + (f"_{kind}" if kind in ("dirac", "vector") else "")
    return base if var == "main" else f"conv_{var}_{base}"


def tasks():
    T = [(s, y0, M, "main") for M in CFG["Ms"] for y0 in CFG["y0s"] for s in CFG["spins"]]
    T += [(s, y0, M, "main") for y0, Ms in CFG["Ms_extra"].items() for M in Ms for s in CFG["spins"]]
    T += [(s, y0, M, v) for (y0, M) in CFG["conv_points"] for s in CFG["spins"] for v in ("nk2", "kmax", "rtol")]
    T += [("smallM", CFG["smallM_y0"], M, "main") for M in CFG["smallM"]]
    T += [("scalar", 0.1, M, "main") for M in CFG["scalar_kmax_M"]]
    return T


if __name__ == "__main__":
    res = dict(np.load(OUT)) if OUT.exists() else {}
    todo = [t for t in tasks() if key(t) + "_time" not in res]
    print(f"{len(todo)} tasks to do ({len(tasks()) - len(todo)} done)", flush=True)
    with Pool(CFG["workers"]) as pool:
        for task, r in pool.imap_unordered(run, todo):
            kk = key(task)
            for name, v in r.items():
                res[f"{kk}_{name}"] = v
            main = r.get("na3", r.get("nX", r.get("na3_wide")))
            print(f"{kk}: {np.array2string(np.atleast_1d(main), precision=4)} ({float(r['time']):.0f}s)", flush=True)
            np.savez(OUT, **res)
    for y0 in CFG["y0s"]:
        res[f"y0{y0}_y_outs"] = np.array(CFG["y_outs"][y0])
    res["Ms"] = np.array(sorted(set(CFG["Ms"]) | {M for Ms in CFG["Ms_extra"].values() for M in Ms}))
    np.savez(OUT, **res)
