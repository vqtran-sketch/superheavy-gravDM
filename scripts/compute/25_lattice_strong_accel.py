"""25: lattice check of free passage with strongly accelerating walls in a linearly tilted quartic potential (helper).

Not quoted in the main text or the SM. Scripts 39 and 56 import `measured_F` (spectra of the walls measured on the lattice just before the
collision) and, through `s16`, the thin-wall kinematics of script 16. SM Sec. S3 ("Other potentials") uses the cubic-tilt runs of script 39.

Model (as script 16): V = (lam/4)(phi^2 - v^2)^2 + (eps/2)(1 - phi/v), lam = 1/2, m = v = 1, sigma = 2/3 (eps/(sigma m) = 1.5 eps).
[DERIVED] The false vacuum exists only for eps < eps_sp = 4 lam/(3 sqrt 3) = 0.385, i.e. eps/(sigma m) < 0.577; this linear-tilt model
therefore cannot reach eps/(sigma m) ~ 1, which is why script 39 uses a cubic tilt. Here eps/(sigma m) = 0.30 and 0.45.
Setup at large eps:
  - the initial walls interpolate between the actual minima phi_f (false) and phi_t (true) of the tilted potential (affine map of the
    symmetric kinks);
  - the resolution scales with the boost at the collision, pts_per_wall(gamma0) = PTS * gamma_c/gamma0, so the colliding walls keep PTS points;
  - the parton prediction is evaluated at gamma_c with Dphi = phi_t - phi_f (I2 ∝ Dphi^4); the ratio to Dphi = 2 is also stored.
Reference: eps = 0 lattice at gamma = gamma_c. gamma_c follows from thin-wall kinematics with constant sigma [approximation].
Measured-spectrum test (separates free passage from the wall profile): F_1(a), F_2(b) are measured on the same run from phi'(x) in each
half box at t_col - (1.5, 1.0, 0.5), |F| is extrapolated linearly in t to t_col and inserted in the parton cross term
(lattice1d.parton_tensor_F). The same procedure on the eps = 0 reference run validates it.
Usage: python 25_lattice_strong_accel.py [one]   ('one': a single timed point)
Output: output/25_lattice_strong_accel.npz (not included in the stored outputs)"""
import sys, time, pathlib, importlib.util
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import lattice1d as L

CFG = dict(gamma0s=[10.0, 20.0], eps=[0.2, 0.3], Tw=3.0, tc_over_Tw=6.0, PTS=20, sigma=2 / 3, lam=0.5,
           targets_frac=[(0.4, 0.4), (0.6, 0.6), (0.8, 0.8), (0.4, 0.8)])
_spec = importlib.util.spec_from_file_location("s16", ROOT / "scripts/compute/16_lattice_accelerating.py")
s16 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s16)
_two_walls_sym = L.two_walls


def minima(eps, lam=CFG["lam"]):
    """False and true minima of the tilted potential (v = 1): roots of lam phi^3 - lam phi - eps/2 = 0."""
    r = np.sort(np.roots([lam, 0, -lam, -eps / 2]).real)
    return r[0], r[2]


SNAP_DT = (1.5, 1.0, 0.5)


def measured_F(run, tcol, p_right, p_left):
    """|F_R(p)| of the right-moving wall (x < 0) and |F_L(p)| of the left-moving wall (x > 0), F = int dx phi' e^{i p x},
    from snapshots before the collision, extrapolated linearly in t to t_col. Dimensionless (phi in units of v)."""
    x = run["x"]; dx = run["dx"]; ts, FR, FL = [], [], []
    for t, ph in run["snaps"]:
        dph = L.grad4(ph, dx)
        ts.append(t)
        FR.append(np.abs(np.exp(1j * np.outer(p_right, x[x < 0])) @ dph[x < 0] * dx))
        FL.append(np.abs(np.exp(1j * np.outer(p_left, x[x > 0])) @ dph[x > 0] * dx))
    ts = np.array(ts); FR = np.array(FR); FL = np.array(FL)
    ext = lambda F: np.array([np.polyval(np.polyfit(ts, F[:, j], 1), tcol) for j in range(F.shape[1])])
    return ext(FR), ext(FL), FR[-1], FL[-1]


def run_point(g0, eps):
    phf, pht = minima(eps)
    u0 = np.sqrt(1 - 1 / g0 ** 2); x0 = u0 * CFG["tc_over_Tw"] * CFG["Tw"]
    tcol, gc = s16.kinematics(g0, eps, x0, CFG["sigma"])
    tg = [(fa * gc, fb * gc) for fa, fb in CFG["targets_frac"]]
    pts = int(np.ceil(CFG["PTS"] * gc / g0))
    # affine map of the symmetric two-wall configuration: -1 -> phi_f, +1 -> phi_t
    L.two_walls = lambda x, t, x0, u, v=1.0: (lambda p, d: (phf + (pht - phf) * (p + 1) / 2, (pht - phf) * d / 2))(*_two_walls_sym(x, t, x0, u, v))
    try:
        ra = L.run(g0, Tw=CFG["Tw"], tc_over_Tw=CFG["tc_over_Tw"], pts_per_wall=pts, targets=tg, eps=eps, window_center=tcol,
                   snap_ts=[tcol - d for d in SNAP_DT])
    finally:
        L.two_walls = _two_walls_sym
    tc0 = CFG["tc_over_Tw"] * CFG["Tw"]
    r0 = L.run(gc, Tw=CFG["Tw"], tc_over_Tw=CFG["tc_over_Tw"], pts_per_wall=CFG["PTS"], targets=tg, snap_ts=[tc0 - d for d in SNAP_DT])
    FR, FL, FR1, FL1 = measured_F(ra, tcol, ra["a"], ra["b"])
    M2, M0 = L.invariants(L.parton_tensor_F(ra["a"], ra["b"], FR, FL))
    N2, _ = L.invariants(L.parton_tensor_F(ra["a"], ra["b"], FR1, FL1))         # last snapshot only (no extrapolation)
    GR, GL, _, _ = measured_F(r0, tc0, r0["a"], r0["b"])
    V2, V0 = L.invariants(L.parton_tensor_F(r0["a"], r0["b"], GR, GL))
    dphi = pht - phf
    P2, P0 = L.invariants(L.parton_tensor(ra["a"], ra["b"], gc, dphi=dphi))
    Q2, Q0 = L.invariants(L.parton_tensor(ra["a"], ra["b"], gc))            # Dphi = 2
    A2, A0 = L.invariants(ra["T"]); Z2, Z0 = L.invariants(r0["T"])
    E = ra["Elog"][:, 1]
    rows = []
    for j, (fa, fb) in enumerate(CFG["targets_frac"]):
        rows.append([g0, eps, gc, tcol, fa, fb, A2[j] / P2[j], A0[j] / P0[j], A2[j] / Q2[j], Z2[j] / Q2[j], Z0[j] / Q0[j], dphi, pts,
                     A2[j] / M2[j], A0[j] / M0[j], A2[j] / N2[j], Z2[j] / V2[j], Z0[j] / V0[j]])
        print(f"g0={g0} eps={eps} (eps/sigma m={eps / CFG['sigma']:.2f}) gamma_c={gc:.1f} Dphi={dphi:.3f} pts={pts} | (a,b)=({fa},{fb}) gamma_c m: "
              f"accel/parton I2 {A2[j] / P2[j]:.3f} I0 {A0[j] / P0[j]:.3f} (Dphi=2: I2 {A2[j] / Q2[j]:.3f}) | const-gamma I2 {Z2[j] / Q2[j]:.3f} I0 {Z0[j] / Q0[j]:.3f}",
              flush=True)
        print(f"      measured-spectrum parton: accel I2 {A2[j] / M2[j]:.3f} I0 {A0[j] / M0[j]:.3f} (no extrapolation: I2 {A2[j] / N2[j]:.3f}) | "
              f"const-gamma (method check) I2 {Z2[j] / V2[j]:.3f} I0 {Z0[j] / V0[j]:.3f}", flush=True)
    print(f"   energy drift (accelerating run includes vacuum-energy conversion): E_end/E_0 = {E[-1] / E[0]:.4f}, N = {ra['N']}", flush=True)
    return rows


if __name__ == "__main__":
    pts_list = [(CFG["gamma0s"][0], CFG["eps"][0])] if "one" in sys.argv[1:] else [(g, e) for g in CFG["gamma0s"] for e in CFG["eps"]]
    rows = []
    for g0, eps in pts_list:
        t0 = time.time(); rows += run_point(g0, eps); print(f"   wall time {time.time() - t0:.0f} s", flush=True)
    if "one" not in sys.argv[1:]:
        np.savez(ROOT / "output/25_lattice_strong_accel.npz", rows=np.array(rows),
                 header="gamma0 eps gamma_c t_col a/gm b/gm I2acc/par(Dphi) I0acc/par(Dphi) I2acc/par(Dphi=2) I2const/par I0const/par Dphi pts "
                        "I2acc/par(measured F) I0acc/par(measured F) I2acc/par(measured F, last snapshot) I2const/par(measured F) I0const/par(measured F)")
