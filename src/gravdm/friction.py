"""Pre-existing SM radiation from gravitational reheating at the inflation->kination transition, and the friction it exerts on
the An-Li walls (runaway-wall condition, section 5.6 and appendix D.5 of the paper).

Inputs [LIT: Chun, Scopel, Zaballa, JCAP 07 (2009) 022, arXiv:0904.0675]:
  per minimally coupled real scalar d.o.f.:  a^4 rho = I H0^4  (eq. 4.5; a = 1 at the end of inflation, eta = -1/H0),
      0.03 <~ I <~ 0.19 for inflation->kination (H0 Dt >~ 1e-3);
  kination after the transition: H = b1 H0 / (2 a^3)  (eq. 3.11), b1 ~= 2 + 3 x0 (abrupt: b1 -> 2);
  their T_* (kination-radiation equality), eq. (5.7): T_* = 2 30^{1/4}/sqrt(3 pi) g_I/sqrt(g_*) I^{3/4}/b1 H0^2/M_P.
Conformal SM fields (fermions, gauge bosons) are not produced at tree level [ASSUMED: anomaly-induced production neglected];
g_I = 4 (Higgs doublet, minimally coupled, light during inflation) [ASSUMED].

Friction [LIT: Bodeker-Moore 1703.08215, as used in Sec. 5.1 of Ghoshal:2026pew (2607.15279)]: transition-radiation pressure P ~ gamma g^2 M_V T^3
for a wall that gives mass M_V = g v to gauge bosons. Runaway to gamma_need survives only if
  P(gamma_need) < L   <=>   T_c^3 < L / (g^2 M_V gamma_need)                                     (FR)
Order-one factors in P are uncertain [ESTIMATE]; we keep an explicit fudge factor kappa_BM (default 1).
Units: GeV; M_P = reduced Planck mass.
"""
import numpy as np
from .units import M_P, G_STAR_SM


def rho_ratio_at(Hc, H0, I=0.1, b1=2.0, gI=4, gstar=G_STAR_SM, thermalisation="energy"):
    """rho_r/rho_kin when the kination Hubble rate has dropped to Hc (radiation not yet dominant) [DERIVED]:
    produced: a^4 rho = gI I H0^4; rho_K = 3 M_P^2 H^2 with H = b1 H0/(2 a^3) => a^3 = b1 H0/(2 Hc);
    ratio = 4 gI I f H0^2 a^2 / (3 b1^2 M_P^2), with
      f = 1             thermalisation="energy" (DEFAULT, physical: the gI produced d.o.f. share their energy with all g* SM d.o.f.;
                        this is an irreversible process that conserves energy, not entropy),
      f = (gI/g*)^{1/3} thermalisation="entropy" (Chun et al. eqs. 5.3-5.5 bookkeeping: a gI-d.o.f. bath at T_I, then iso-entropic
                        expansion with g*; used only to reproduce their eq. 5.7 in the tests).
    The "entropy" bookkeeping would raise the H_inf bounds by a factor ~1.5."""
    a = (b1 * H0 / (2 * Hc)) ** (1 / 3)
    f = 1.0 if thermalisation == "energy" else (gI / gstar) ** (1 / 3)
    return 4 * gI * I * f * H0 ** 2 * a ** 2 / (3 * b1 ** 2 * M_P ** 2)


def T_equality(H0, I=0.1, b1=2.0, gI=4, gstar=G_STAR_SM):
    """Kination-radiation equality temperature, derived here from rho_ratio_at = 1 plus entropy conservation from
    rho_r = (pi^2/30) gI T_I^4 at a_I [DERIVED]. Must equal Chun et al. eq. (5.7) (tested)."""
    a_star = np.sqrt(3 * b1 ** 2 * M_P ** 2 / (4 * gI * I * (gI / gstar) ** (1 / 3) * H0 ** 2))   # Chun bookkeeping (entropy)
    aT_I = (30 * I / (np.pi ** 2)) ** 0.25 * H0                  # a_I T_I from gI I H0^4 = (pi^2/30) gI (a T)^4
    aT_star = (gI / gstar) ** (1 / 3) * aT_I                       # entropy: gI (aT_I)^3 = g* (aT_*)^3
    return aT_star / a_star


def T_equality_chun(H0, I=0.1, b1=2.0, gI=4, gstar=G_STAR_SM):
    """[LIT] Chun et al. eq. (5.7)."""
    return 2 * 30 ** 0.25 / np.sqrt(3 * np.pi) * gI / np.sqrt(gstar) * I ** 0.75 / b1 * H0 ** 2 / M_P


def T_at(Hc, H0, I=0.1, b1=2.0, gI=4, gstar=G_STAR_SM, thermalisation="energy"):
    """Temperature of the (assumed thermalised) SM radiation when the DS Hubble rate is Hc, during kination
    (rho_tot ~= rho_K = 3 M_P^2 Hc^2). [ASSUMED: thermalised with g* = 106.75; if not, the particles are harder and the
    friction estimate changes by O(1)]."""
    rho_r = rho_ratio_at(Hc, H0, I, b1, gI, gstar, thermalisation) * 3 * M_P ** 2 * Hc ** 2
    return (30 * rho_r / (np.pi ** 2 * gstar)) ** 0.25


def H0_max_runaway(L4, alpha, gamma_m, g=0.65, kappa_BM=1.0, I=0.19, b1=2.0, gI=4, thermalisation="energy"):
    """Largest inflationary Hubble rate H0 for which pre-existing radiation does not stop the runaway before
    gamma reaches gamma_m/m [condition (FR)]. M_V = g L^{1/4} [ASSUMED: true-vacuum vev ~ L^{1/4}], m ~ L^{1/4}.
    Uses the conservative upper end I = 0.19 by default. Returns H0_max [GeV]."""
    L = L4 ** 4
    Hc = np.sqrt(L / (3 * alpha)) / M_P
    MV = g * L4
    gamma_need = gamma_m / L4
    Tmax = (L / (kappa_BM * g ** 2 * MV * gamma_need)) ** (1 / 3)
    # T_at ∝ H0^{2/3}: rho_r ∝ H0^2 a^2 ∝ H0^{8/3} -> T ∝ H0^{2/3}
    T1 = T_at(Hc, 1.0, I, b1, gI, thermalisation=thermalisation)
    return (Tmax / T1) ** 1.5
