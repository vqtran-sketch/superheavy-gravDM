"""57: order-of-magnitude estimate of X production during thermalization of the hard collision products ("pre-thermal" production),
relative to the first-collision yield, along the DM curves. ESTIMATE, not a calculation.

Main text: lower-bound paragraph ("We estimate that scatterings of hard quanta on the forming plasma there add, for M_X <~ 1e15 GeV, up to
20% for alpha = 0.05 and 80% for alpha = 5e-4"). SM: Sec. S8 and Table SIII (an estimate uncertain by a factor of a few).

Physics [LIT + estimate]:
- Thermalization of a burst of hard quanta of energy E_h into a bath of final temperature T (bottom-up, LPM-suppressed splittings):
  t_th = kappa alpha_s^-2 T^-1 (E_h/T)^{1/2}  [form: Kurkela:2011ti Eq. (2.75), Harigaya:2013vwa Eqs. (4.29)-(4.32); coefficient kappa ~ 0.2
  (0.1-0.4) is an ESTIMATE (leading-log splitting kernel), not given in the literature].
- For sigma = c s/M_P^4 and light X the production rate per volume depends only on the energy densities, gamma = (8c/3) rho_1 rho_2/M_P^4
  (isotropic distributions). With Y_FI = gamma_th/(3 H s) at the final temperature, the yield produced during t_th is
  Y_pre/Y_FI(M_X << T) ~ 3 H t_th in total; the hard-soft part, which is new relative to collisions of hard quanta (later crossings), is
  f_hs H t_th with f_hs = 0.3-0.8 [estimate], dominated by the late stage T ~ T_final; its mass reach is M_X ~ 2.3 (E_h T)^{1/2}.
- Relative to the first-collision yield: multiply by Y_FI/Y_coll at M_X << T: benchmark 1/(142 H/beta) [SM Eq. (S16), Dirac];
  An-Li 4.8e-5 / [6.8e-3 (Q/Q_beta) sqrt(alpha)/eps^{3/2}] [SM Sec. S7 vs Eq. (S15), Dirac, T -> T_*].
- E_h ~ collision energy scale: benchmark Mbar_coll = 11.16 (H/beta)/nu^2 M_P [main-text Eq. (6)]; An-Li gamma m = 6 sqrt(3 alpha) <e>_A/nu^2 M_P [SM Sec. S7].
- T ranges: benchmark T_reh along the core (output/50_benchmark_dm_plane.npz); An-Li T_* = (eps L/A_g)^{1/4} with the L^{1/4}
  range of SM Table SII (rounded values, CFG) and eps, Q of the Monte Carlo (script 05).
- H: radiation value at T (freeze-in convention). alpha_s(T): one-loop SM running from alpha_s(M_Z) = 0.118 (nf = 5 to m_t, nf = 6 above).
Units: GeV. Output: output/57_prethermal_estimate.log (and .npz)."""
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm.units import M_P

CFG = dict(gstar=106.75, kappa=(0.1, 0.2, 0.4), f_hs=(0.3, 0.8), alpha_s_MZ=0.118, MZ=91.19, mt=172.5, e_hat=0.27,
           bench_HB=(0.1, 0.03, 0.01), bench_nu=(2, 3), ratioFI=142.0, Mbar_coef=11.16,
           anli={0.05: dict(Q=0.341, eps=6.25, L4={2: (2e11, 3e12), 3: (3e11, 3e12)}),
                 5e-3: dict(Q=0.316, eps=5.54, L4={2: (4e11, 4e12), 3: (5e11, 4e12)}),
                 5e-4: dict(Q=0.317, eps=5.47, L4={2: (9e11, 5e12), 3: (1e12, 5e12)})},
           Y_FI0=4.8e-5, Ycoef=6.8e-3, Q_beta=1 / 3)
A_g = np.pi ** 2 * CFG["gstar"] / 30


def alpha_s(mu):
    """One-loop running: b0 = (33 - 2 nf)/(12 pi); nf = 5 below m_t, 6 above."""
    inv = 1 / CFG["alpha_s_MZ"] + (33 - 10) / (6 * np.pi) * np.log(CFG["mt"] / CFG["MZ"])
    return 1 / (inv + (33 - 12) / (6 * np.pi) * np.log(np.maximum(mu, CFG["mt"]) / CFG["mt"]))


def H_rad(T):
    return np.sqrt(np.pi ** 2 * CFG["gstar"] / 90) * T ** 2 / M_P


def Ht_th(T, E, kappa):
    return H_rad(T) * kappa / (alpha_s(T) ** 2 * T) * np.sqrt(E / T)


def ranges(T_lo, T_hi, E, fi_over_coll):
    """min/max over T in [T_lo, T_hi], kappa range and f_hs range: (H t_th, hard-soft/coll, total/coll, reach)."""
    Ts = np.array([T_lo, T_hi])
    hts = [Ht_th(Ts, E, k) for k in CFG["kappa"]]
    ht_c = Ht_th(Ts, E, CFG["kappa"][1])
    hs = [f * h * fi_over_coll for f in CFG["f_hs"] for h in hts]
    tot = [3 * h * fi_over_coll for h in hts]
    return (np.min(hts), np.max(hts), ht_c.min(), ht_c.max(), np.min(hs), np.max(hs), np.min(tot), np.max(tot),
            2.3 * np.sqrt(E * T_lo), 2.3 * np.sqrt(E * T_hi))


if __name__ == "__main__":
    lines = [f"alpha_s(1e11, 1e12, 1e13 GeV) = {alpha_s(1e11):.4f}, {alpha_s(1e12):.4f}, {alpha_s(1e13):.4f}; kappa = {CFG['kappa']}, "
             f"f_hs = {CFG['f_hs']}"]
    out = {}
    d = np.load(ROOT / "output/50_benchmark_dm_plane.npz")
    for hb in CFG["bench_HB"]:
        for nu in CFG["bench_nu"]:
            core = d[f"hb{hb}_nu{nu}_nocut_core"]; T = d[f"hb{hb}_nu{nu}_nocut_T"]
            Tc = T[core & np.isfinite(T)]
            E = CFG["Mbar_coef"] * hb / nu ** 2 * M_P
            r = ranges(Tc.min(), Tc.max(), E, 1 / (CFG["ratioFI"] * hb))
            out[f"bench_hb{hb}_nu{nu}"] = np.array(r)
            lines.append(f"benchmark H/beta={hb} nu={nu}: T_reh {Tc.min():.1e}-{Tc.max():.1e}, E_h {E:.1e}: H t_th {r[0]:.3f}-{r[1]:.3f} "
                         f"(kappa=0.2: {r[2]:.3f}-{r[3]:.3f}); hard-soft/coll {r[4]:.3f}-{r[5]:.3f}; total/coll {r[6]:.3f}-{r[7]:.3f}; "
                         f"reach {r[8]:.1e}-{r[9]:.1e} GeV")
    for a, p in CFG["anli"].items():
        ycoll = CFG["Ycoef"] * (p["Q"] / CFG["Q_beta"]) * np.sqrt(a) / p["eps"] ** 1.5
        for nu, (l4lo, l4hi) in p["L4"].items():
            Tlo, Thi = [(p["eps"] / A_g) ** 0.25 * l for l in (l4lo, l4hi)]
            E = 6 * np.sqrt(3 * a) * CFG["e_hat"] / nu ** 2 * M_P
            r = ranges(Tlo, Thi, E, CFG["Y_FI0"] / ycoll)
            out[f"anli_a{a}_nu{nu}"] = np.array(r)
            lines.append(f"An-Li alpha={a} nu={nu}: T_* {Tlo:.1e}-{Thi:.1e}, E_h {E:.1e}, FI/coll {CFG['Y_FI0']/ycoll:.2f}: H t_th {r[0]:.3f}-{r[1]:.3f} "
                         f"(kappa=0.2: {r[2]:.3f}-{r[3]:.3f}); hard-soft/coll {r[4]:.3f}-{r[5]:.3f}; total/coll {r[6]:.3f}-{r[7]:.3f}; "
                         f"reach {r[8]:.1e}-{r[9]:.1e} GeV")
    txt = "\n".join(lines); print(txt)
    (ROOT / "output/57_prethermal_estimate.log").write_text(txt + "\n")
    np.savez(ROOT / "output/57_prethermal_estimate.npz", **out)
