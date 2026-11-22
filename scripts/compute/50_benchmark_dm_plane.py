"""50: dark-matter curves of the flat-space benchmark (runaway vacuum transition, exponential nucleation, beta >> H, instantaneous
reheating) for a gravity-only Dirac X.

Main text: Fig. 3 (via script 58), Dark-matter paragraph (cores from 1e10 to 3e17 GeV for 0.01 <= H/beta <= 0.1, nu = 2, 3).
SM: Sec. S5, Fig. S4 (cores, lowest T_reh, Lambda_cut, envelope GW frequencies 1e4 to 1e7 Hz), Sec. S4 (Mbar_coll = 0.12 M_P and
0.012 M_P for nu = 3; factor 1.5 relative to Ghoshal:2026pew). Input to scripts 57, 58, 64 and p50.

Relic condition [DERIVED]: M_X [Y_coll + Y_FI] = MY_DM (units.py, Omega h^2 = 0.12; SM Eq. (S17)), with
  Y_coll = coeff (H/beta) R(M_X/Mbar) (T_reh/M_P)^3,  coeff = K Q_beta/(960 pi) = 6.81e-3 (script 11, main-text Eq. (5)),
  Y_FI   = F(M_X/T_reh) (T_reh/M_P)^3 (script 46; independent of T_reh in these units),
  Mbar = 2 sqrt(3) S0 (H/beta) M_P/nu^2 (Dphi = nu L^{1/4}; main-text Eq. (6)), independent of T_reh.
  Solved for T_reh(M_X) at fixed (H/beta, nu); also the freeze-in-only curve (Y_coll = 0) for reference.
EFT cut: yield from parton sqrt(s) > Lambda_cut = 0.46 M_P discarded; Mmax_frac = Lambda_cut/(2 M_P) = 0.23 (pair threshold):
  R_cut(M) = int du p(u) G(M/(Mbar e^u), (Lambda/(2 Mbar e^u))^2), p(u) = density of ln(M_coll/Mbar) (gravdm.rspread),
  G = parton.cut_yield_table (zero if 2 M_X >= Lambda). Curves are stored with ("cut", the result used) and without ("nocut") the cut.
Core (gravdm.window; conditions (i)-(iv) of SM Sec. S5): collisions >= 1/1.1 of the total, R >= 0.1 (and R_cut >= 0.1 with the
cut), M_X < Mmax_frac M_P, and M_X >= beta/beta_frac = 10 beta (beta = (beta/H) H(T_reh)), where Eq. (5) applies.
GW [LIT, estimate]: envelope approximation, Eqs. (7)-(10) in Caprini:2015zlo (arXiv:1512.06239), vacuum transition
  (alpha -> inf, kappa = 1, v_w = 1): f*/beta = 0.62/2.7, f_peak = 1.65e-5 Hz (f*/beta)(beta/H)(T_reh/100 GeV)(g*/100)^{1/6},
  h^2 Omega_peak = 1.67e-5 (H/beta)^2 (100/g*)^{1/3} 0.11/1.42; H at T_reh = vacuum H (instantaneous reheating, A_g T_reh^4 = L).
Cross-check with Eq. (51) of Ghoshal:2026pew: n = 3N/(R* A), R* = (8pi)^{1/3}/beta, E/A = L R*/3 for both walls, no oblique factor,
  vs ours n = 8c L^2 Q_beta/(beta M_P^4) at the same event yield per area, N_ev/A = 4c (E1/A)(E2/A)/M_P^4 (N/A in that reference counts
  events, its Eq. (23); the 2 in its Eq. (51) is the pair factor): ratio = (8pi)^{1/3}/(6 Q_beta) = 1.46.
Closed forms (Mbar, envelope GW, comparison ratio) are in gravdm.benchdm (tests/test_benchdm.py).
Output: output/50_benchmark_dm_plane.npz"""
import sys, pathlib
import numpy as np
from scipy import optimize
from scipy.interpolate import RegularGridInterpolator
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import rspread as rs, parton as pt, xsec_analytic as xa, window, benchdm as bd
from gravdm.units import M_P, MY_DM, G_STAR_SM

CFG = dict(Hbeta=(0.1, 0.03, 0.01), nus=(2.0, 3.0), spin="dirac", profile="tanh", Q_beta=1.0 / 3.0, cut=0.46, Mmax_frac=0.23,
           M_grid=(8.0, np.log10(0.25 * M_P), 260), mu_tab=np.logspace(-4, 1.0, 101), zgrid=(-10.0, np.log10(30.0), 4000),
           Tbracket=(1e6, 1e19), beta_frac=0.1, fstar_over_beta=0.62 / 2.7, f_redshift=1.65e-5, Omega_env=1.67e-5 * 0.11 / 1.42)
coeff = CFG["Q_beta"] * float(np.load(ROOT / "output/11_universal_flat.npz")["K"]) / (960 * np.pi)
S = np.load(ROOT / "output/47_R_spread.npz"); assert str(S["spin"]) == CFG["spin"] and str(S["profile"]) == CFG["profile"]
Rf = lambda m: np.exp(np.interp(np.log(m), np.log(S["mu"]), np.log(np.maximum(S["R"], 1e-300)), left=0.0, right=-700.0))
F = np.load(ROOT / "output/46_freezein_curve.npz"); assert str(F["spin"]) == CFG["spin"]
Fi = lambda x: np.where(x < F["x"][-1], np.exp(np.interp(np.log(np.maximum(x, F["x"][0])), np.log(F["x"]),
                                                         np.log(np.maximum(F["Y"], 1e-300)))), 0.0)

# ---- cut-yield table G(mu, z_c) (Dirac, tanh)
z = np.logspace(*CFG["zgrid"]); lz = np.log(z)
Kz = np.array([pt.K(zi, CFG["profile"]) for zi in z])
Gt = pt.cut_yield_table(lambda r: xa.c_of_r(CFG["spin"], r), CFG["mu_tab"], z, Kz)
Gi = RegularGridInterpolator((np.log(CFG["mu_tab"]), lz), Gt, bounds_error=False, fill_value=None)
u, p = rs.lns_density(h=4e-3)
keep = p > 1e-12 * p.max(); u, p = u[keep], p[keep]


def R_cut(M, Mb, lam):
    """Cut-averaged suppression factor (dimensionless); M, Mb, lam in GeV."""
    if 2 * M >= lam:
        return 0.0
    s = Mb * np.exp(u); mu = M / s; zc = (lam / (2 * s)) ** 2
    pts = np.column_stack([np.log(np.clip(mu, CFG["mu_tab"][0], CFG["mu_tab"][-1])), np.log(np.clip(zc, z[0], z[-1]))])
    Gv = np.where((mu > CFG["mu_tab"][-1]) | (zc < z[0]), 0.0, np.maximum(Gi(pts), 0.0))
    # normalise to the uncut R of script 47: R_cut = R x [int p G(zc)] / [int p G(inf)] <= R
    pinf = np.column_stack([pts[:, 0], np.full(len(mu), lz[-1])])
    Ginf = np.where(mu > CFG["mu_tab"][-1], 0.0, np.maximum(Gi(pinf), 0.0))
    den = float(np.trapezoid(p * Ginf, u))
    return min(float(np.trapezoid(p * Gv, u)) / den, 1.0) * Rf(M / Mb) if den > 0 else 0.0


def T_relic(M, ycoll):
    """T_reh [GeV] with M (ycoll + F(M/T)) (T/M_P)^3 = MY_DM; ycoll = coeff (H/beta) R (dimensionless). nan if none."""
    f = lambda lT: np.log(M * (ycoll + Fi(M / np.exp(lT))) * (np.exp(lT) / M_P) ** 3 / MY_DM)
    lo, hi = np.log(CFG["Tbracket"][0]), np.log(CFG["Tbracket"][1])
    if not (f(lo) < 0 < f(hi)):
        return np.nan
    return float(np.exp(optimize.brentq(f, lo, hi, xtol=1e-12)))


M = np.logspace(*CFG["M_grid"])
out = dict(M=M, Hbeta=np.array(CFG["Hbeta"]), nus=np.array(CFG["nus"]), cut=CFG["cut"])
out["T_FIonly"] = np.array([T_relic(m, 0.0) for m in M])
Ag = np.pi ** 2 * G_STAR_SM / 30
for hb in CFG["Hbeta"]:
    for nu in CFG["nus"]:
        Mb = bd.mbar_coll(hb, nu)
        k = f"hb{hb:g}_nu{nu:g}"
        for tag, Rv in (("nocut", np.array([Rf(m / Mb) for m in M])),
                        ("cut", np.array([R_cut(m, Mb, CFG["cut"] * M_P) for m in M]))):
            T = np.array([T_relic(m, coeff * hb * r) if r > 0 else np.nan for m, r in zip(M, Rv)])
            yc = coeff * hb * Rv
            fcoll = yc / (yc + Fi(M / T))
            Rfull = np.array([Rf(m / Mb) for m in M])
            H = np.sqrt(Ag) * T ** 2 / (np.sqrt(3) * M_P)
            core = np.isfinite(T) & (fcoll >= window.FCOLL_MIN) & (Rfull >= window.R_CORE) & (Rv >= window.R_CORE) \
                & (M < CFG["Mmax_frac"] * M_P) & (H / hb / M <= CFG["beta_frac"])
            fpk = bd.fpeak_envelope(T, hb)
            out |= {f"{k}_{tag}_R": Rv, f"{k}_{tag}_T": T, f"{k}_{tag}_fcoll": fcoll, f"{k}_{tag}_core": core,
                    f"{k}_{tag}_beta_over_M": H / hb / M, f"{k}_{tag}_fpeak": fpk}
            if core.any():
                i = np.where(core)[0]
                lo, hi = window.interp_edges(M, core, fcoll, np.minimum(Rfull, Rv))
                jmin = i[np.nanargmin(T[i])]
                by_beta = bool(i[0] > 0 and (H / hb / M)[i[0] - 1] > CFG["beta_frac"])   # lower edge set by M_X = 10 beta
                if by_beta:
                    lo = np.exp(np.interp(np.log(CFG["beta_frac"]), np.log((H / hb / M)[i[0]:i[0] - 2:-1]), np.log(M[i[0]:i[0] - 2:-1])))
                out[f"{k}_{tag}_edges"] = np.array([lo if lo else M[i[0]], hi if hi else M[i[-1]]])
                out[f"{k}_{tag}_lowedge_by_beta"] = by_beta
                print(f"H/beta={hb:5g} nu={nu:g} {tag:5s}: Mbar = {Mb:.2e} GeV ({Mb / M_P:.3f} M_P); window M_X "
                      f"{out[f'{k}_{tag}_edges'][0]:.2e}-{out[f'{k}_{tag}_edges'][1]:.2e} GeV, T_reh {np.nanmin(T[core]):.2e}-"
                      f"{np.nanmax(T[core]):.2e} GeV (min at M_X = {M[jmin]:.2e}); f_peak {np.nanmin(fpk[core]):.1e}-{np.nanmax(fpk[core]):.1e} Hz;"
                      f" max beta/M_X = {np.nanmax((H / hb / M)[core]):.1e}; low edge by beta: {out[f'{k}_{tag}_lowedge_by_beta']}")
            else:
                print(f"H/beta={hb:5g} nu={nu:g} {tag:5s}: empty window")
        out[f"hb{hb:g}_Omega_gw"] = bd.omega_peak_envelope(hb)
        out[f"hb{hb:g}_nu{nu:g}_Mbar"] = Mb
    print(f"   h^2 Omega_peak(envelope) = {out[f'hb{hb:g}_Omega_gw']:.2e} for H/beta = {hb}")
# analytic plateau check: T^3 = MY_DM M_P^3/(coeff hb M) where R = 1 and FI negligible
m0 = 1e13; hb = 0.1
print(f"plateau check H/beta=0.1, M=1e13: numeric {np.interp(np.log(m0), np.log(M), out['hb0.1_nu3_nocut_T']):.4e} vs analytic "
      f"{(MY_DM * M_P ** 3 / (coeff * hb * m0)) ** (1 / 3):.4e}")
print(f"GPS Eq. (51) comparison: n_GPS/n_ours = {bd.gps_sphere_ratio(CFG['Q_beta']):.3f}")
np.savez(ROOT / "output/50_benchmark_dm_plane.npz", **out)
