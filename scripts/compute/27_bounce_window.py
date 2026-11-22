"""27: O(4) bounce of the tilted quartic toy model and the bounce-action window of the An-Li scenario.

Not quoted directly in the main text or the SM; its tables are read by script 28 (false-vacuum retention bounds of SM Sec. S7).

[DERIVED] Gamma/V = A e^{-S}, H_c^4 = Gamma/V, H_c = sqrt(L/3 alpha)/M_P:
    S_req = 4 ln(A^{1/4}/H_c);  A = L:  S_req = 4 ln(sqrt(3 alpha) M_P / L^{1/4})
    A = (S/2 pi)^2 m1^4 (Callan-Coleman form, determinant ratio set to 1 [ESTIMATE]): S_req + 4 ln(m1/L^{1/4}) + 2 ln(S/2 pi).
Toy family (src/gravdm/bounce.py): S = S_hat(eps)/lambda, Delta phi = Dx v, L = lambda v^4 DU (all hats dimensionless):
    lambda = S_hat / S_req,  nu^4 = Dx^4/(lambda DU),  b_eff = [U(x_top) - U(x_false)]/DU  (barrier above v1 in units of L),
    m1^2/L^{1/2} = lambda^{1/2} U''(x_false)/DU^{1/2}.
Both a cubic tilt (used for the retention bounds) and a linear tilt (comparison) are tabulated.
Inputs: output/06_dm_plane.npz, output/04_mass_dependence.npz (Dirac core L^{1/4} ranges). Output: output/27_bounce_window.{npz,log}."""
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import bounce as B
from gravdm import window
from gravdm.units import M_P

CFG = dict(b_cubic=np.concatenate([np.geomspace(0.04, 0.186, 24), np.geomspace(0.18752, 0.1889, 12), np.geomspace(0.189, 3.0, 40)]),
           eps_linear=np.concatenate([np.linspace(0.05, 0.76, 40), np.linspace(0.762, 0.7697, 12)]), alphas=[0.05, 5e-4], nus=[2.0, 3.0], lam_max=[1.0, 4 * np.pi])
out = {}; log = []
P = lambda s: (print(s), log.append(s))

# --- S_req along the Dirac core lines
d = np.load(ROOT / "output/06_dm_plane.npz"); Md = np.load(ROOT / "output/04_mass_dependence.npz")
M = d["M_over_MP"] * M_P
Sreq_all = []
for a in CFG["alphas"]:
    for nu in CFG["nus"]:
        k = f"a{a}_nu{nu}_dirac"; L4 = d[k + "_L4"]; R = Md["R_" + k]
        core = np.isfinite(L4) & (R >= 0.1) & (d[k + "_fcoll"] >= window.FCOLL_MIN) & (M < 0.25 * M_P)
        S = 4 * np.log(np.sqrt(3 * a) * M_P / L4[core])
        Sreq_all += [S.min(), S.max()]
        P(f"S_req (A = L) along Dirac core alpha={a:g} nu={nu}: L4 = {L4[core].min():.2e}-{L4[core].max():.2e} GeV -> S_req = {S.min():.1f}-{S.max():.1f}")
Smax_all = [4 * np.log(np.sqrt(3) * M_P / x) for x in (1.5e11, 7e12)]
P(f"S_max (alpha = 1) for L4 = 1.5e11-7e12: {Smax_all[0]:.1f}-{Smax_all[1]:.1f}")
out["Sreq_range"] = np.array([min(Sreq_all), max(Sreq_all)])

# --- toy families
for tilt, grid in (("cubic", [1 / (4 * b) for b in CFG["b_cubic"]]), ("linear", CFG["eps_linear"])):
    rows = []
    for eps in grid:
        b = 1 / (4 * eps)
        r = B.bounce(b, tilt=tilt)
        if not r["ok"]:
            P(f"{tilt} eps={eps:.4f}: no bounce ({r.get('reason', 'no false vacuum')})"); continue
        DU = r["dU"]; Dx = r["xT"] - r["xf"]
        beff = float((B.U(np.array(r["xt"]), eps, tilt) - B.U(np.array(r["xf"]), eps, tilt)) / DU)
        U2 = float(B.d2U(np.array(r["xf"]), eps, tilt))
        rows.append([eps, r["S_hat"], r["S_hat_virial"], DU, Dx, beff, U2, r["R"], (r["x0"] - r["xf"]) / Dx])
    rows = np.array(rows); out[tilt] = rows
    P(f"\n{tilt} tilt: eps, S_hat, S_hat_virial, DU, Dx, b_eff, U''(xf), R, centre (x0-xf)/Dx")
    for row in rows:
        P("  " + "  ".join(f"{v:.4g}" for v in row))
    for Sreq in out["Sreq_range"]:
        lam = rows[:, 1] / Sreq
        nu = (rows[:, 4] ** 4 / (lam * rows[:, 3])) ** 0.25
        m1 = np.sqrt(np.sqrt(lam) * rows[:, 6] / np.sqrt(rows[:, 3]))
        out[f"{tilt}_S{Sreq:.0f}_lam"] = lam; out[f"{tilt}_S{Sreq:.0f}_nu"] = nu; out[f"{tilt}_S{Sreq:.0f}_m1"] = m1
        # S_hat -> 0 at the degenerate point (cubic: b = 3/16; linear: eps -> eps_spinodal); S_hat is monotonic on each side,
        # so lambda <= lam_max is one interval in eps per side. Edges by log-interpolation in eps.
        jmin = np.argmin(rows[:, 1])
        for side, sl in (("far", slice(0, jmin + 1)), ("near", slice(jmin, None))):
            e, S_, be, nu_, m1_ = rows[sl, 0], rows[sl, 1], rows[sl, 5], nu[sl], m1[sl]
            if len(e) < 2:
                continue
            o = np.argsort(S_)
            for lm in CFG["lam_max"]:
                if S_.max() < lm * Sreq:
                    P(f"  S_req={Sreq:.1f} side {side}: lambda<={lm:.3g} holds over the whole scanned range"); continue
                ip = lambda y: float(np.interp(np.log(lm * Sreq), np.log(S_[o]), y[o]))
                P(f"  S_req={Sreq:.1f} side {side} ({'b<3/16' if tilt == 'cubic' and side == 'far' else 'b>3/16' if tilt == 'cubic' else ''}): "
                  f"edge lambda={lm:.3g} at eps={ip(e):.4f}: b_eff={np.exp(ip(np.log(be))):.3g}, nu={ip(nu_):.2f}, m1/L4={ip(m1_):.2f}")
        for nu_t in CFG["nus"]:                      # near side (cubic b > 3/16; linear: only side), where nu is monotonic
            sl = slice(jmin, None) if tilt == "cubic" else slice(0, jmin + 1)
            nn, oo = nu[sl], np.argsort(nu[sl])
            g = lambda y: float(np.interp(nu_t, nn[oo], y[sl][oo]))
            P(f"  S_req={Sreq:.1f}, nu={nu_t}: lambda={g(lam):.3f}, b_eff={np.exp(g(np.log(rows[:, 5]))):.3g}, m1/L4={g(m1):.2f}, "
              f"bubble centre at {g(rows[:, 8]):.2f} of the way to the true vacuum")
        i = np.argmin(np.abs(lam - 1))
        P(f"  prefactor A=(S/2pi)^2 m1^4 near lambda=1 (m1/L4={m1[i]:.2f}): S_req shifts by {4 * np.log(m1[i]) + 2 * np.log(Sreq / (2 * np.pi)):+.1f}")
np.savez(ROOT / "output/27_bounce_window.npz", **out)
(ROOT / "output/27_bounce_window.log").write_text("\n".join(log) + "\n")
