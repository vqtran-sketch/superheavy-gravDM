"""02f: lattice test of free passage on a dense (p_1, p_2) grid, the data of main-text Fig. 1 and SM Figs. S2, S3.

Main text: Fig. 1 and the lattice paragraph (via scripts 02g, 02h). SM: Sec. S3 (grid, 20 and 40 points per wall, invariants at
gamma = 40, 80, 160, lattice/superposition ratios, gamma = 10), Fig. S2.

Kink-antikink collision in the phi^4 potential at constant gamma, with the machinery of script 02 (function one(); its momentum targets
are replaced). Grid: a, b in geomspace(a_min m, a_max_over_gamma gamma m, n), 28 x 28, from m/2 to 3 gamma m, so that the parton weight
W(a/gamma m) W(b/gamma m) is covered (W(3) ~ 2e-6). Resolution pts_per_wall = 40 (or 20 for the Richardson pair), sample_dt = 0.0025/m.
Stored per run: lattice (l), superposed walls with their potential (f) and parton (p) invariants I_2, I_0, energy drift.
Usage: python 02f_lattice_dense.py GAMMA [PTS_PER_WALL] [JSON_OVERRIDES TAG]
Uncertainty runs: '{"Tw": 5.0, "tc_over_Tw": 4.8}' win (Fourier window x1.25 at fixed collision time 24/m),
'{"sample_dt": 0.00125}' dt2 (half sampling step).
Output: output/02f_lattice_dense_g{GAMMA}[_n{PTS}][_{TAG}].npz and .log"""
import sys, pathlib, time, importlib.util, json
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
spec = importlib.util.spec_from_file_location("s02", ROOT / "scripts/compute/02_lattice_compute.py")
s02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s02)

CFG = dict(a_min=0.5, a_max_over_gamma=3.0, n=28, pts_per_wall=40, sample_dt=0.0025)

if __name__ == "__main__":
    g = float(sys.argv[1]); pts = int(sys.argv[2]) if len(sys.argv) > 2 else CFG["pts_per_wall"]
    ab = np.geomspace(CFG["a_min"], CFG["a_max_over_gamma"] * g, CFG["n"])
    s02.targets = lambda gamma: [(a, b) for a in ab for b in ab]
    over = dict(sample_dt=CFG["sample_dt"]); over.update(json.loads(sys.argv[3]) if len(sys.argv) > 3 else {})
    t0 = time.time(); r = s02.one(g, pts_per_wall=pts, **over)
    tag = ("" if pts == CFG["pts_per_wall"] else f"_n{pts}") + (f"_{sys.argv[4]}" if len(sys.argv) > 4 else "")
    np.savez(ROOT / f"output/02f_lattice_dense_g{int(g)}{tag}.npz", **r, gamma=g, pts_per_wall=pts, cfg=str(CFG), over=json.dumps(over))
    hh = (r["a"] >= 8) & (r["b"] >= 8) & (r["a"] <= 0.8 * g) & (r["b"] <= 0.8 * g)
    print(f"gamma={g} pts={pts} done t={time.time()-t0:.0f}s N={r['N']} Edrift={r['Edrift']:.1e} | 8m<=a,b<=0.8gm: "
          f"max|lat/free-1| I2={np.abs(r['I2l']/r['I2f']-1)[hh].max() if hh.any() else np.nan:.4f} "
          f"max|free/par-1| I2={np.abs(r['I2f']/r['I2p']-1)[hh].max() if hh.any() else np.nan:.4f}", flush=True)
