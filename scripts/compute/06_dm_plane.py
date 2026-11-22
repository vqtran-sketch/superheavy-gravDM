"""06: An-Li dark-matter curves: latent-heat scale L^{1/4} vs M_X for Omega_X h^2 = 0.12, per spin, alpha and nu, with the GW peak frequency.

SM: Sec. S7 (window, GW peak frequencies), Fig. S5, Table SII (via script 20). Input to scripts 07, 10, 14, 17, 20, 21, 23, 24, 26-28 and p53.

Y(M) = kappa_Y(alpha) chat_spin (L^{1/4}/M_P)^3 R_spin(M/M_P; alpha, nu)                  [kappa_Y from script 05; R from script 04]
  collisions only: L^{1/4} = M_P [ MY_DM / (M kappa_Y chat R(M)) ]^{1/3}   (R depends on M/M_P only, not on L)
The stored DM curve solves the relic condition with the total yield, collisions + thermal freeze-in at T_max = T_*   [SM Eq. (S17)]
(gravdm.freezein, xi_H = 0); f_coll = Y_coll/(Y_coll + Y_FI) at the solution.
GW peak [LIT: Eqs. (10), (13) and Table I in An:2026ghw, the bulk-flow model]: f0 = C_k H_c/(2 pi) a_c/a_0 with C_k = 4.6, and
  h^2 Omega_peak = 5e-5 C_p alpha with C_p = 0.010 (n_DS = 6, alpha = 0.05; extrapolated in alpha)
  a_c/a_0 from comoving SM entropy: (a_c T_*)^4 = 30 c_hat L/(pi^2 g*) with a_c = 1.
Output: output/06_dm_plane.npz"""
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import xsec_analytic as xa, freezein as fz
from gravdm.units import M_P, MY_DM, G_STAR_SM, T0, GS0, GEV_TO_HZ

CFG = dict(Ck=4.6, Cp=0.010, f_calib=1.0,   # f_calib = 1: standard redshift with our entropy bookkeeping (no fit to Fig. 3 of An:2026ghw)
          Lam_sp=0.46,  # printout only: species scale M_Pl/sqrt(N), N ~ 118, non-reduced M_Pl, in units of M_P (SM Sec. S5); the cut is applied in script 21
             # Cp: Table I of An:2026ghw, n_DS = 6, alpha = 0.05 (extrapolated for other alpha)
           spins=["scalar", "conformal", "dirac", "vector"], fi_check=True)
K = np.load(ROOT / "output/05_kappaY.npz")["rows"]          # alpha n kappa stat ... c_hat
Mdep = np.load(ROOT / "output/04_mass_dependence.npz")
MoP = Mdep["M_over_MP"]

from scipy import optimize
XF = np.logspace(-2, np.log10(40.0), 160)
FTAB = {sp: np.array([fz.yield_freezein_X(1e12, x * 1e12, sp, xi_H=0.0) for x in XF]) / (1e12 / M_P) ** 3
        for sp in CFG["spins"]}   # Y_FI = (T/M_P)^3 F(M/T)  (scale-free: rate ∝ T^8/M_P^4 f(M/T)) [DERIVED]
def F_fi(sp, x):
    x = np.asarray(x, float)
    return np.where(x < XF[-1], np.exp(np.interp(np.log(np.maximum(x, XF[0])), np.log(XF), np.log(np.maximum(FTAB[sp], 1e-300)))), 0.0)
out = dict(M_over_MP=MoP)
for row in K[K[:, 1] == 6]:
    alpha, kap, ktot, chat = row[0], row[2], row[6], row[9]
    for nu in (1.0, 1.5, 2.0, 3.0):
        gm_med = float(Mdep[f"a{alpha}_nu{nu}_gm_wmed"])     # yield-weighted median of the collision scale
        frac_np = float(Mdep[f"a{alpha}_nu{nu}_frac_above_half"])
        for sp in CFG["spins"]:
            c0hat = xa.c0(sp) * 960 * np.pi
            R = Mdep[f"R_a{alpha}_nu{nu}_{sp}"]
            M = MoP * M_P
            ok = R > 1e-30
            cT = (30 * chat / (np.pi ** 2 * G_STAR_SM)) ** 0.25          # T* = cT L^{1/4}
            L4c = np.where(ok, M_P * (MY_DM / (M * kap * c0hat * np.maximum(R, 1e-300))) ** (1 / 3), np.nan)   # collisions only
            L4 = np.full_like(M, np.nan); fcoll = np.full_like(M, np.nan)
            for i in range(len(M)):            # solve the relic condition with the total yield, collisions + freeze-in
                yc = kap * c0hat * R[i]
                tot = lambda l4: (l4 / M_P) ** 3 * (yc + cT ** 3 * F_fi(sp, M[i] / (cT * l4))) * M[i] / MY_DM - 1
                lo, hi = 1e6, 1e19
                if tot(lo) < 0 < tot(hi):
                    L4[i] = optimize.brentq(lambda x: tot(np.exp(x)), np.log(lo), np.log(hi), xtol=1e-10)
                    L4[i] = np.exp(L4[i]); fcoll[i] = yc / (yc + cT ** 3 * F_fi(sp, M[i] / (cT * L4[i])))
            L4_lo = M_P * (MY_DM / (M * (kap + ktot) * c0hat * np.maximum(R, 1e-300))) ** (1 / 3)
            L = L4 ** 4
            Tstar = (30 * chat * L / (np.pi ** 2 * G_STAR_SM)) ** 0.25          # comoving-entropy temperature at a_c
            Hc = np.sqrt(L / (3 * alpha)) / M_P
            # Standard redshift of the peak, f0 = Ck Hc/(2 pi) a_c/a_0 with our entropy bookkeeping (f_calib = 1).
            # The frequencies read from Fig. 3 of An:2026ghw are higher by a factor 2-5; no fit to that figure.
            f_naive = CFG["Ck"] * Hc * GEV_TO_HZ * (GS0 / G_STAR_SM) ** (1 / 3) * T0 / Tstar
            f0 = f_naive * CFG["f_calib"]
            # freeze-in comparison (prompt thermalisation at T*): ratio Y_FI / Y_coll
            rfi = (1 - fcoll) / np.maximum(fcoll, 1e-300)           # Y_FI / Y_coll at the solution (kept for downstream scripts)
            key = f"a{alpha}_nu{nu}_{sp}"
            out[key + "_L4"] = L4; out[key + "_L4_lo"] = L4_lo; out[key + "_f0"] = f0; out[key + "_Tstar"] = Tstar
            out[key + "_FIratio"] = rfi; out[key + "_fcoll"] = fcoll; out[key + "_L4coll"] = L4c; out[key + "_wallscale_lt_half"] = np.full_like(L4, frac_np < 0.1, dtype=bool)
            # diagnostic: fraction of the massless yield from collisions whose effective WALL scale sqrt(g1 g2) m h exceeds M_P/2
            # (not the parton sqrt(s) = 2 sqrt(z) x that scale, which script 21 computes). Not used downstream.
            out[key + "_gm_med"] = gm_med
            out[key + "_gm_w90"] = float(Mdep[f"a{alpha}_nu{nu}_gm_w90"])
            out[f"a{alpha}_Omega_peak"] = 5e-5 * CFG["Cp"] * alpha
        print(f"alpha={alpha:g} nu={nu}: yield-weighted median scale/M_P = {gm_med:.3f}, yield fraction with wall scale > M_P/2 = {frac_np:.2f} "
              f"(legacy; parton sqrt(s) fractions: script 21); 90% quantile {float(Mdep[f'a{alpha}_nu{nu}_gm_w90']):.3f} vs species cutoff {CFG['Lam_sp']}")
np.savez(ROOT / "output/06_dm_plane.npz", **out)
