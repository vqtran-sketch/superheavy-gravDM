"""39: lattice test of free passage with accelerating walls in a quartic potential with a cubic tilt.

SM: Sec. S3, "Other potentials" (with the spectra measured just before the collision, the parton cross term reproduces the lattice
within 6% (I_2) and 13% (I_0) for walls that start at gamma_0 = 20; for walls that start almost at rest the test is inconclusive).
Runtime: hours for the full set.

V = (lam/4)(phi^2 - v^2)^2 + (Delta V/4)(2 - 3x + x^3), x = phi/v (lattice1d tilt="cubic", equal to bounce.U with
eps_lattice = lam v^4 eps_bounce), at bounce-consistent points: R = Delta V/(sigma m) = 0.75 eps_bounce for S_req = 39.2 / 61.6 and
nu = 1.5, 2, 3 (R = 0.578/0.489, 0.788/0.712, 0.932/0.918). The false-vacuum mass is m_f^2 = m^2 (1 - R), 0.26 m at nu = 3, so the
survival of the false vacuum is monitored (max |phi + v| between the walls at t_col/2, t_col - 2, t_col - 1).
Method: initial boosted tanh walls between the exact vacua -+v; resolution scaled by gamma_c/gamma0; single-wall spectra measured at
t_col - (1.5, 1.0, 0.5), extrapolated to t_col and inserted in the parton cross term (script 25, measured_F); constant-gamma method check
at gamma_c. gamma_c from the thin-wall equation of motion (script 16).
Usage: python 39_lattice_anli_cubic.py [one | conv | var snaps | var long | var rest]
       (one: nu = 3, S = 39.2, gamma0 = 10; conv: resolution x2 at nu = 3; var: spectra closer to the collision, longer acceleration,
       walls starting almost at rest)
Output: output/39_lattice_anli_cubic.npz (conv: _conv.npz; var: _var_snaps*.npz, _var_long.npz, _var_rest.npz)"""
import sys, time, pathlib, importlib.util
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import lattice1d as L

CFG = dict(gamma0s=[10.0, 20.0], eps_bounce={"S39.2_nu1.5": 0.7703, "S61.6_nu1.5": 0.6524, "S39.2_nu2": 1.0507, "S61.6_nu2": 0.9496,
                                             "S39.2_nu3": 1.2430, "S61.6_nu3": 1.2235},
           Tw=3.0, tc_over_Tw=6.0, PTS=20, sigma=2 / 3, lam=0.5,
           targets_frac=[(0.4, 0.4), (0.6, 0.6), (0.8, 0.8), (0.4, 0.8)], snap_dt=(1.5, 1.0, 0.5), monitor_before=(None, 2.0, 1.0))
_s = importlib.util.spec_from_file_location("s25", ROOT / "scripts/compute/25_lattice_strong_accel.py")
s25 = importlib.util.module_from_spec(_s); _s.loader.exec_module(s25)


def run_point(g0, name):
    eps = CFG["lam"] * CFG["eps_bounce"][name]; R = eps / CFG["sigma"]
    u0 = np.sqrt(1 - 1 / g0 ** 2); x0 = u0 * CFG["tc_over_Tw"] * CFG["Tw"]
    tcol, gc = s25.s16.kinematics(g0, eps, x0, CFG["sigma"])
    tg = [(fa * gc, fb * gc) for fa, fb in CFG["targets_frac"]]
    pts = int(np.ceil(CFG["PTS"] * gc / g0))
    mon = [tcol / 2 if b is None else tcol - b for b in CFG["monitor_before"]]
    ra = L.run(g0, Tw=CFG["Tw"], tc_over_Tw=CFG["tc_over_Tw"], pts_per_wall=pts, targets=tg, eps=eps, tilt="cubic",
               window_center=tcol, snap_ts=mon + [tcol - d for d in CFG["snap_dt"]])
    x = ra["x"]; dev = []
    for t, ph in ra["snaps"][:len(mon)]:
        cross = np.where((x[:-1] < 0) & (np.sign(ph[:-1]) != np.sign(ph[1:])))[0]
        if len(cross) == 0:                       # no wall left of centre: the region between the walls has left the false vacuum
            print(f"   t={t:.1f}: no zero crossing for x<0; phi in |x|<5: min {ph[np.abs(x) < 5].min():.3f} max {ph[np.abs(x) < 5].max():.3f}", flush=True)
            dev.append(np.inf); continue
        xw = x[cross[0]]
        inner = np.abs(x) < abs(xw) - 0.5
        dev.append(np.max(np.abs(ph[inner] + 1.0)) if inner.any() else np.nan)
    raF = dict(ra); raF["snaps"] = ra["snaps"][len(mon):]
    tc0 = CFG["tc_over_Tw"] * CFG["Tw"]
    r0 = L.run(gc, Tw=CFG["Tw"], tc_over_Tw=CFG["tc_over_Tw"], pts_per_wall=CFG["PTS"], targets=tg, snap_ts=[tc0 - d for d in CFG["snap_dt"]])
    FR, FL, FR1, FL1 = s25.measured_F(raF, tcol, ra["a"], ra["b"])
    M2, M0 = L.invariants(L.parton_tensor_F(ra["a"], ra["b"], FR, FL)); N2, _ = L.invariants(L.parton_tensor_F(ra["a"], ra["b"], FR1, FL1))
    GR, GL, _, _ = s25.measured_F(r0, tc0, r0["a"], r0["b"])
    V2, V0 = L.invariants(L.parton_tensor_F(r0["a"], r0["b"], GR, GL)); Z2, Z0 = L.invariants(r0["T"])
    P2, P0 = L.invariants(L.parton_tensor(ra["a"], ra["b"], gc))
    A2, A0 = L.invariants(ra["T"]); E = ra["Elog"][:, 1]
    rows = []
    for j, (fa, fb) in enumerate(CFG["targets_frac"]):
        rows.append([g0, R, eps, gc, tcol, fa, fb, A2[j] / M2[j], A0[j] / M0[j], A2[j] / N2[j], A2[j] / P2[j], A0[j] / P0[j],
                     Z2[j] / V2[j], Z0[j] / V0[j], pts] + dev + [E[-1] / E[0]])
        print(f"{name} g0={g0:g} R={R:.3f} gamma_c={gc:.1f} | (a,b)=({fa},{fb}) gamma_c m: measured-F I2 {A2[j] / M2[j]:.3f} I0 {A0[j] / M0[j]:.3f} "
              f"(last snapshot I2 {A2[j] / N2[j]:.3f}) | tanh@gamma_c I2 {A2[j] / P2[j]:.3f} | method check I2 {Z2[j] / V2[j]:.3f} I0 {Z0[j] / V0[j]:.3f}",
              flush=True)
    print(f"   false vacuum max|phi+v| between walls at t_col/2, t_col-2, t_col-1: {np.round(dev, 4)}; E_end/E_0 = {E[-1] / E[0]:.6f}; N = {ra['N']}",
          flush=True)
    return rows


if __name__ == "__main__":
    one = "one" in sys.argv[1:]; conv = "conv" in sys.argv[1:]; var = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "var" else None
    if conv:
        CFG["PTS"] = 40
    if var == "snaps":                # spectra measured closer to the collision
        CFG["snap_dt"] = (0.8, 0.6, 0.4)
    if var == "long":                 # walls accelerate 2x longer (t_c = 36/m)
        CFG["tc_over_Tw"] = 12.0
    if var == "rest":                 # walls starting almost at rest (gamma0 = 1.5), x0 = 28/m
        CFG["tc_over_Tw"] = 28.0 / np.sqrt(1 - 1 / 1.5 ** 2) / CFG["Tw"]
        rows = []
        for n in ("S39.2_nu2", "S39.2_nu3"):
            rows += run_point(1.5, n)
        np.savez(ROOT / "output/39_lattice_anli_cubic_var_rest.npz", rows=np.array(rows))
        sys.exit(0)
    if var:
        rows = []
        for n in ("S39.2_nu1.5", "S39.2_nu3"):
            rows += run_point(10.0, n)
        np.savez(ROOT / f"output/39_lattice_anli_cubic_var_{var}.npz", rows=np.array(rows))
        sys.exit(0)
    pts_list = ([(10.0, "S39.2_nu3")] if one else [(10.0, "S39.2_nu3"), (20.0, "S39.2_nu3")] if conv
                else [(g, n) for g in CFG["gamma0s"] for n in CFG["eps_bounce"]])
    rows = []
    for g0, n in pts_list:
        t0 = time.time(); rows += run_point(g0, n); print(f"   wall time {time.time() - t0:.0f} s", flush=True)
    if not one:
        np.savez(ROOT / ("output/39_lattice_anli_cubic" + ("_conv" if conv else "") + ".npz"), rows=np.array(rows),
                 header="gamma0 R eps gamma_c t_col a/gc b/gc I2/par(measF) I0/par(measF) I2/par(last snapshot) I2/par(tanh@gc) "
                        "I0/par(tanh@gc) I2 method I0 method pts dev(t_col/2) dev(t_col-2) dev(t_col-1) E_end/E_0")
