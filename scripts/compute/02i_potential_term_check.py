"""02i: origin of the soft-momentum deviation of the lattice from the parton prediction.

Main text: lattice paragraph ("the potential energy of the overlapping walls ... lowers the lattice result by about 2(m/p)^2").
SM: Sec. S3 (superposition without potential energy reproduces the parton prediction to 5e-5; the deviations come from the
potential cross term).

The superposition phi_1 + phi_2 of the two uniformly moving walls is passed through the same pipeline as the lattice (window, sampling,
Fourier sum), once with the full potential and once with V = 0. With V = 0 its stress tensor at timelike q is the gradient cross term
phi_1' phi_2' (single walls contribute only at spacelike q), which is exactly the parton prediction (SM Eq. (S8)).
Result [DERIVED]: superposition(V = 0)/parton = 1 at all p (I_2 and I_0), so the pipeline is exact; with V the I_2 deficit is
~ 2 (m/p)^2, from the potential cross term V(phi_1 + phi_2) - V(phi_1) - V(phi_2), which the parton prediction drops.
gamma = 80, pts_per_wall = 20, diagonal p_1 = p_2. Output: output/02i_potential_term_check.npz and .log"""
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import lattice1d as L

CFG = dict(gamma=80.0, idx=(4, 16), n=28, a_min=0.5, a_max_over_gamma=3.0, run=dict(Tw=4.0, tc_over_Tw=6.0, pts_per_wall=20, sample_dt=0.0025))

if __name__ == "__main__":
    g = CFG["gamma"]; ab = np.geomspace(CFG["a_min"], CFG["a_max_over_gamma"] * g, CFG["n"])[slice(*CFG["idx"])]
    tg = [(a, a) for a in ab]
    I2p, I0p = L.invariants(L.parton_tensor(ab, ab, g))
    base = L.run(g, targets=tg, free_passage=True, **CFG["run"])
    V0 = L.V; L.V = lambda phi, **k: 0.0 * phi
    nov = L.run(g, targets=tg, free_passage=True, **CFG["run"]); L.V = V0
    I2b, I0b = L.invariants(base["T"]); I2n, I0n = L.invariants(nov["T"])
    p = base["a"]
    log = [f"p/m: {np.array2string(p, precision=3)}",
           f"free/par I2: {np.array2string(I2b / I2p, precision=4)}", f"free/par I0: {np.array2string(I0b / I0p, precision=4)}",
           f"free(V=0)/par: max|I2-1| = {np.abs(I2n / I2p - 1).max():.2e}, max|I0-1| = {np.abs(I0n / I0p - 1).max():.2e}",
           f"(1 - free/par I2)(p/m)^2: {np.array2string((1 - I2b / I2p) * p ** 2, precision=3)}"]
    print("\n".join(log))
    np.savez(ROOT / "output/02i_potential_term_check.npz", p=p, I2p=I2p, I0p=I0p, I2b=I2b, I0b=I0b, I2n=I2n, I0n=I0n, cfg=str(CFG))
    (ROOT / "output/02i_potential_term_check.log").write_text("\n".join(log) + "\n")
