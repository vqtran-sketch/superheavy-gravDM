"""28: false-vacuum retention and v1-kination bounds on H_inf along the An-Li Dirac DM curves.

SM: Sec. S7, "Conditions on the inflationary era" (keeping the Higgs in its false vacuum during 60 e-folds requires
H_inf <~ (0.14-0.19) L^{1/4} for nu = 2 and (0.05-0.07) L^{1/4} for nu = 3). Main text: "A realization" (H_inf bound, via script 20).

For each end of a Dirac DM curve and each nu, the tilted-quartic toy potential is fixed by (nu, S_req) with
S_req = 4 ln(sqrt(3 alpha) M_P / L^{1/4}) [DERIVED]:
    nu^4 = Dx^4 S_req / (S_hat DU)  -> eps  -> b_eff = barrier above v1 / L, m1 / L^{1/4}, dphi_top = distance v1 -> barrier top (/ L^{1/4}).
Conditions (N_e = 60) [ESTIMATE]:
  Hawking-Moss, no transition in our patch (the criterion used; labelled "draft" in the printed output):
      8 pi^2 b_eff L / (3 H^4) > 3 N_e + ln N_e
  Hawking-Moss, volume fraction f_max of space converted in N_e e-folds:  8 pi^2 b_eff L/(3 H^4) > ln[(4 pi/3) N_e / f_max]
  stochastic fluctuations: 3 sigma_h < dphi_top, sigma_h^2 = min(H^2 N_e/4 pi^2, 3 H^4/(8 pi^2 m1^2)) [Starobinsky:1994bd]
  v1 curvature during kination: 6 xi_H H^2 < m1^2.
O(4) (CdL) tunnelling during inflation: rate per Hubble volume per Hubble time ~ (H_c/H_inf)^4 (flat-space Gamma; de Sitter corrections
  neglected [ESTIMATE]); bubbles in our patch ~ (4 pi/9)(H_c/H_inf)^4 e^{3 N_e}; converted fraction ~ (4 pi/3) N_e (H_c/H_inf)^4.
Prefactor variants: Gamma/V = L e^{-S} ("L", the one used) or (S/2 pi)^2 m1^4 e^{-S} ("CC", Callan-Coleman form, determinant ratio 1).
Inputs: output/06_dm_plane.npz, output/04_mass_dependence.npz, output/27_bounce_window.npz. Output: output/28_trapping_bounce.{npz,log}."""
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import bounce as B
from gravdm import window
from gravdm.units import M_P

CFG = dict(Ne=60, f_max=0.01, cases=[(0.05, 2.0), (0.05, 3.0), (5e-4, 2.0), (5e-4, 3.0)], tilts=["cubic", "linear"], xis=[1.0, 5.0],
           prefactor=["L", "CC"])     # Gamma/V = L e^{-S}  or  (S/2pi)^2 m1^4 e^{-S} (Callan-Coleman form, determinant ratio 1)
out = {}; log = []
P = lambda s: (print(s), log.append(s))
T = np.load(ROOT / "output/27_bounce_window.npz")


def table(tilt):
    rows = T[tilt]; j = np.argmin(rows[:, 1])
    r = rows[j + 1:] if tilt == "cubic" else rows[:j]          # cubic: b > 3/16 branch; linear: single branch
    eps, Sh, DU, Dx, beff, U2 = r[:, 0], r[:, 1], r[:, 3], r[:, 4], r[:, 5], r[:, 6]
    dtop = np.array([(lambda e: e[1] - e[0])(B.extrema(e_, tilt)) for e_ in eps]) / Dx     # in units of Delta phi
    g = Dx ** 4 / (Sh * DU)                                                                  # nu^4 / S_req
    o = np.argsort(g)
    return dict(lg=np.log(g[o]), eps=eps[o], lb=np.log(beff[o]), U2=U2[o], DU=DU[o], Sh=Sh[o], dtop=dtop[o])


TAB = {t: table(t) for t in CFG["tilts"]}


def toy(nu, Sreq, tilt):
    """Bounce-consistent toy quantities for given nu and S_req (interpolated in ln g = ln(nu^4/S_req))."""
    t = TAB[tilt]; x = np.log(nu ** 4 / Sreq)
    ip = lambda y: np.interp(x, t["lg"], y)
    lam = ip(t["Sh"]) / Sreq
    m1 = np.sqrt(np.sqrt(lam) * ip(t["U2"]) / np.sqrt(ip(t["DU"])))
    return dict(eps=ip(t["eps"]), beff=np.exp(ip(t["lb"])), m1=m1, dtop=ip(t["dtop"]) * nu, lam=lam)


# --- cross-check of the interpolation against a direct bounce at one point
c = toy(3.0, 50.0, "cubic"); r = B.bounce(1 / (4 * c["eps"]))
lam_direct = r["S_hat"] / 50.0; nu_direct = (16 / (lam_direct * r["eps"])) ** 0.25
P(f"interpolation check (nu=3, S=50, cubic): eps={c['eps']:.4f}, direct bounce gives nu={nu_direct:.3f} (target 3), "
  f"lambda {lam_direct:.3f} vs {c['lam']:.3f}")

d = np.load(ROOT / "output/06_dm_plane.npz"); Md = np.load(ROOT / "output/04_mass_dependence.npz")
M = d["M_over_MP"] * M_P
lnHM_draft = 3 * CFG["Ne"] + np.log(CFG["Ne"]); lnHM_vol = np.log(4 * np.pi / 3 * CFG["Ne"] / CFG["f_max"])
P(f"HM thresholds: draft {lnHM_draft:.1f}, volume fraction f<{CFG['f_max']}: {lnHM_vol:.1f}")
for alpha, nu in CFG["cases"]:
    k = f"a{alpha}_nu{nu}_dirac"; L4 = d[k + "_L4"]; R = Md["R_" + k]
    core = np.isfinite(L4) & (R >= 0.1) & (d[k + "_fcoll"] >= window.FCOLL_MIN) & (M < 0.25 * M_P)
    i = np.where(core)[0]; ends = [i[np.argmax(M[i])], i[np.argmin(M[i])]]       # high-mass end, low-mass end
    for tilt in CFG["tilts"]:
        for pf in CFG["prefactor"]:
            res = []
            for j in ends:
                l4 = L4[j]; Hc = np.sqrt(l4 ** 4 / (3 * alpha)) / M_P
                Sreq = 4 * np.log(np.sqrt(3 * alpha) * M_P / l4)
                q = toy(nu, Sreq, tilt)
                if pf == "CC":                                    # S = 4 ln(A^{1/4}/H_c), A = (S/2pi)^2 m1^4; iterate
                    for _ in range(20):
                        Sreq = 4 * np.log(q["m1"] * l4 / Hc) + 2 * np.log(Sreq / (2 * np.pi)); q = toy(nu, Sreq, tilt)
                HMd = (8 * np.pi ** 2 * q["beff"] / (3 * lnHM_draft)) ** 0.25 * l4
                HMv = (8 * np.pi ** 2 * q["beff"] / (3 * lnHM_vol)) ** 0.25 * l4
                m1 = q["m1"] * l4; dtop = q["dtop"] * l4
                Hf0 = 2 * np.pi * dtop / (3 * np.sqrt(CFG["Ne"]))
                Hfeq = np.sqrt(dtop * m1 / (3 * np.sqrt(3 / (8 * np.pi ** 2))))
                Hfl = max(Hf0, Hfeq)                                   # sigma = min(massless, equilibrium) -> weaker bound
                Hk = {xi: m1 / np.sqrt(6 * xi) for xi in CFG["xis"]}
                res.append((l4, M[j], Sreq, q, HMd, HMv, Hfl, Hk, Hc))
            for (l4, Mj, Sreq, q, HMd, HMv, Hfl, Hk, Hc), end in zip(res, ("high-M", "low-M")):
                P(f"a={alpha:g} nu={nu} {tilt:6s} A={pf:2s} {end:6s} L4={l4:.2e} M={Mj:.1e} S_req={Sreq:.1f} lam={q['lam']:.2f} b_eff={q['beff']:.2e} "
                  f"m1/L4={q['m1']:.2f} dtop/L4={q['dtop']:.3f} | HM(draft)<{HMd:.1e} HM(vol)<{HMv:.1e} fluct<{Hfl:.1e} "
                  f"kin(xi=1)<{Hk[1.0]:.1e} kin(xi=5)<{Hk[5.0]:.1e} | m1/HMd={q['m1'] * l4 / HMd:.1f} "
                  f"S_HM(at HMd)={lnHM_draft:.0f} vs S_CdL~{Sreq:.0f}; CdL bubbles in patch at H=HMd: {np.log10(4 * np.pi / 9) + 4 * np.log10(Hc / HMd) + 3 * CFG['Ne'] * np.log10(np.e):.0f} (log10)")
                out[f"a{alpha}_nu{nu}_{tilt}_{pf}_{end}"] = np.array([l4, Mj, Sreq, q["beff"], q["m1"], HMd, HMv, Hfl, Hk[1.0], Hk[5.0]])
np.savez(ROOT / "output/28_trapping_bounce.npz", **out)
(ROOT / "output/28_trapping_bounce.log").write_text("\n".join(log) + "\n")
