"""Closed-form pieces of the flat-space benchmark DM plane (section 4.4 and figure 4 of the paper; used by script 50).

Benchmark: runaway vacuum transition, Gamma ∝ e^{beta t}, beta >> H, vacuum domination, prompt reheating A_g T_reh^4 = L,
Dphi = nu L^{1/4}, gravity-only X. Units: GeV unless stated; frequencies in Hz.
"""
import numpy as np
from .units import M_P, MY_DM, G_STAR_SM
from .rspread import S0


def mbar_coll(hbeta, nu):
    """Yield-weighted mean collision scale Mbar_coll = 2 sqrt(3) S0 (H/beta) M_P / nu^2 [GeV] (eqs. (4.11), (4.12)) [DERIVED].
    Independent of T_reh for Dphi = nu L^{1/4}."""
    return 2 * np.sqrt(3) * S0 * hbeta * M_P / nu ** 2


def treh_plateau(M, hbeta, coeff):
    """T_reh [GeV] on the plateau (R = 1, freeze-in neglected): M coeff (H/beta) (T/M_P)^3 = MY_DM  (eqs. (4.10), (4.17))."""
    return (MY_DM * M_P ** 3 / (coeff * hbeta * np.asarray(M, float))) ** (1 / 3)


def treh_min(hbeta, nu, coeff, muR_max):
    """Minimum T_reh [GeV] for which collisions alone give Omega h^2 = 0.12: max_M [M R(M/Mbar)] = muR_max Mbar
    (muR_max = max_mu mu R(mu), dimensionless, from script 47). Scales as (H/beta)^{-2/3} nu^{2/3} [DERIVED]."""
    return (MY_DM * M_P ** 3 / (coeff * hbeta * mbar_coll(hbeta, nu) * muR_max)) ** (1 / 3)


def gps_sphere_ratio(Q_beta=1.0 / 3.0):
    """n_X of the spherical-bubble estimate of Ghoshal:2026pew Eq. (5.1) (JHEP 11 (2026) 142; radius R* = (8 pi)^{1/3}/beta, E/A = L R*/3, no oblique
    factor) divided by ours, n_X = 8 c L^2 Q_beta/(beta M_P^4), for the same EVENT yield per unit wall area [DERIVED].
    Eq. (51): n = (2/2)(4 pi R*^2)/(4 pi R*^3/3) N/A, with N/A the number of hard collisions per area (their Eq. (23)); the numerator 2 is
    the pair factor, the denominator 2 the sharing of each wall area by two bubbles (their Eq. (69) uses the surface density 3/(2R*)).
    Hence n_sphere = (3/R*) 4c (L R*/3)^2/M_P^4 and the ratio (8 pi)^{1/3}/(6 Q_beta) = 1.46 (the factor ~1.5 of section 4.4)."""
    return (8 * np.pi) ** (1 / 3) / (6 * Q_beta)


# Envelope approximation, Eqs. (7)-(10) in Caprini:2015zlo [LIT], vacuum transition: alpha -> inf, kappa = 1, v_w = 1
FSTAR_OVER_BETA = 0.62 / (1.8 - 0.1 + 1.0)
OMEGA_ENV = 1.67e-5 * 0.11 / (0.42 + 1.0)


def fpeak_envelope(treh, hbeta, gstar=G_STAR_SM):
    """Present-day peak frequency [Hz]: 16.5e-6 Hz (f*/beta)(beta/H)(T/100 GeV)(g*/100)^{1/6}; H at T_reh (prompt reheating)."""
    return 1.65e-5 * FSTAR_OVER_BETA / hbeta * (np.asarray(treh, float) / 100.0) * (gstar / 100.0) ** (1 / 6)


def omega_peak_envelope(hbeta, gstar=G_STAR_SM):
    """h^2 Omega_peak = 1.67e-5 (H/beta)^2 (100/g*)^{1/3} 0.11 v^3/(0.42 + v^2) at v = 1, kappa alpha/(1 + alpha) -> 1."""
    return OMEGA_ENV * hbeta ** 2 * (100.0 / gstar) ** (1 / 3)
