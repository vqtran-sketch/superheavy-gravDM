"""16: lattice runs with accelerating walls in the phi^4 potential with a linear tilt (helper).

Not quoted in the main text or the SM. Imported by script 25 (thin-wall kinematics of accelerating walls), which scripts 39 and 56 use.

Potential: phi^4 + linear tilt (eps/2)(1 - phi/v); the region between the walls is false vacuum (energy density eps).
Wall kinematics [DERIVED, thin wall, 1+1D]: the momentum per area p = sigma gamma u obeys dp/dt = eps (sigma = 2/3 for m = v = 1); each
wall starts at distance x0 from the center with gamma0. Collision time t_c from int_0^{t_c} u dt = x0; gamma_c = sqrt(1 + (p(t_c)/sigma)^2).
The lattice invariants at hard (a, b) are compared with the parton prediction at gamma_c, and with an eps = 0 run at gamma = gamma_c.
[ESTIMATE] The trace source of the false-vacuum strip is ~ 4 eps/q^2, suppressed by ~ m^4/q^4 relative to the parton term.
Output: output/16_lattice_accelerating.npz (not part of the stored outputs)"""
import sys, pathlib
import numpy as np
from scipy import integrate, optimize
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import lattice1d as L

CFG = dict(gamma0s=[10.0, 20.0], eps=[0.04, 0.08], Tw=3.0, tc_over_Tw=6.0, pts_per_wall=20, sigma=2 / 3,
           targets_frac=[(0.4, 0.4), (0.6, 0.6), (0.8, 0.8), (0.4, 0.8)])


def kinematics(gamma0, eps, x0, sigma):
    p0 = sigma * np.sqrt(gamma0 ** 2 - 1)
    u = lambda t: (p0 + eps * t) / np.sqrt((p0 + eps * t) ** 2 + sigma ** 2)
    dist = lambda t: integrate.quad(u, 0, t)[0] - x0
    tcol = optimize.brentq(dist, 0.1, 2 * x0)
    pc = p0 + eps * tcol
    return tcol, np.sqrt(1 + (pc / sigma) ** 2)


if __name__ == "__main__":
    rows = []
    for g0, eps in [(g, e) for g in CFG["gamma0s"] for e in CFG["eps"]]:
      u0 = np.sqrt(1 - 1 / g0 ** 2); x0 = u0 * CFG["tc_over_Tw"] * CFG["Tw"]
      if True:
        tcol, gc = kinematics(g0, eps, x0, CFG["sigma"])
        tg = [(fa * gc, fb * gc) for fa, fb in CFG["targets_frac"]]
        ra = L.run(g0, Tw=CFG["Tw"], tc_over_Tw=CFG["tc_over_Tw"], pts_per_wall=CFG["pts_per_wall"], targets=tg, eps=eps, window_center=tcol)
        r0 = L.run(gc, Tw=CFG["Tw"], tc_over_Tw=CFG["tc_over_Tw"], pts_per_wall=CFG["pts_per_wall"], targets=tg)
        P2, P0 = L.invariants(L.parton_tensor(ra["a"], ra["b"], gc))
        A2, A0 = L.invariants(ra["T"]); Z2, Z0 = L.invariants(r0["T"])
        for j, (fa, fb) in enumerate(CFG["targets_frac"]):
            rows.append([g0, eps, gc, tcol, fa, fb, A2[j] / P2[j], A0[j] / P0[j], Z2[j] / P2[j], Z0[j] / P0[j]])
            print(f"eps={eps}: t_col={tcol:.2f} gamma_c={gc:.2f} | (a,b)=({fa},{fb}) gamma_c m: accel/parton I2 {A2[j]/P2[j]:.3f} I0 {A0[j]/P0[j]:.3f}"
                  f" | const-gamma lattice/parton I2 {Z2[j]/P2[j]:.3f} I0 {Z0[j]/P0[j]:.3f}", flush=True)
    np.savez(ROOT / "output/16_lattice_accelerating.npz", rows=np.array(rows),
             header="gamma0 eps gamma_c t_col a/gm b/gm I2acc/par I0acc/par I2const/par I0const/par")
