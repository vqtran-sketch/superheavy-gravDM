"""Closed-form sigma(ss -> XX) via s-channel graviton (eqs. (2.3), (2.4), (A.20), (A.21) of the paper) [DERIVED twice:
route A = gravxsec.py (explicit spinors and polarisation vectors), route B = trace technology; agree to 1e-12].

Returns c(r) = sigma * M_P^4 / s with r = M^2/s (dimensionless). M_P = reduced Planck mass.
Identical-particle factor 1/2 included for real scalar and Proca vector, not for Dirac.   [eq. (2.3)]
  scalar (xi):  c = beta [ (1 - 10 xi + 30 xi^2) + 2 r (1 - 10 xi) + 6 r^2 ] / (960 pi)
  Dirac:        c = beta^3 (1 + 6 r) / (960 pi)
  Proca:        c = beta (1 + 2 r + 6 r^2) / (320 pi)
  beta = sqrt(1 - 4 r); zero for r >= 1/4.
Validity: tree level, s << M_P^2 (strong gravity / unitarity beyond); massless initial scalars (wall quanta, s >> m_s^2).
"""
import numpy as np

CASES = {  # name: (spin, xi)
    "scalar": (0, 0.0), "conformal": (0, 1 / 6), "dirac": (0.5, 0.0), "vector": (1, 0.0),
}


def c_of_r(case, r):
    r = np.asarray(r, float)
    b = np.sqrt(np.clip(1 - 4 * r, 0, None))
    spin, xi = CASES[case]
    if spin == 0:
        c = b * ((1 - 10 * xi + 30 * xi ** 2) + 2 * r * (1 - 10 * xi) + 6 * r ** 2) / (960 * np.pi)
    elif spin == 0.5:
        c = b ** 3 * (1 + 6 * r) / (960 * np.pi)
    else:
        c = b * (1 + 2 * r + 6 * r ** 2) / (320 * np.pi)
    return np.where(r < 0.25, c, 0.0)


def c0(case):
    return float(c_of_r(case, 0.0))


def c_dirac_xi(r, xi_phi=0.0):
    """Dirac c(r) = sigma M_P^4/s for a wall field with curvature coupling xi_phi [DERIVED;
    checked against gravxsec.c_hat(0.5, r, xi_phi=...) in tests/test_dirac_xi.py]:
      c = beta^3 [1 + (8/3) r + (10/3)(1 - 6 xi_phi)^2 r] / (960 pi)       (xi_phi = 0: beta^3 (1 + 6 r)/(960 pi))
    The (10/3)(1-6xi)^2 r part is the spin-0 (trace) channel: sigma_0 = (1-6xi)^2 M^2 beta^3/(288 pi M_P^4) is independent of s,
    so after folding with the wall luminosity it adds Delta Y/Y = [(1-6xi)^2-1] mu^2 [c2 L^2 - 17.97 L + 16.42 + ...], L = ln(1/mu),
    with c2 = (5/3)/(int_0^inf W dx)^2 = (5/3)(6/pi)^2 = 60/pi^2 = 6.08 for tanh walls [DERIVED; eq. (A.26);
    subleading terms fitted, tests/test_dirac_xi.py].
    Dimensionless; r = M^2/s; zero for r >= 1/4."""
    r = np.asarray(r, float)
    b = np.sqrt(np.clip(1 - 4 * r, 0, None))
    c = b ** 3 * (1 + 8 / 3 * r + 10 / 3 * (1 - 6 * xi_phi) ** 2 * r) / (960 * np.pi)
    return np.where(r < 0.25, c, 0.0)


def c0_general(case, xi_phi=0.0, xi_X=None):
    """Massless-limit coefficient chat0 = 960 pi sigma M_P^4/s including the wall-field curvature coupling xi_phi [DERIVED; eq. (2.4), table 1;
    checked against gravxsec.c_hat with xi_phi in tests]. Only trace(T_wall) couples to xi_phi:
      scalar(xi_X): 1/6 + (5/6)(1 - 6 xi_phi)^2 (1 - 6 xi_X)^2   (xi_X = 0 for 'scalar', 1/6 for 'conformal')
      Dirac:        1                                            (traceless at M = 0: independent of xi_phi)
      Proca:        13/6 + (5/6)(1 - 6 xi_phi)^2                 (2 transverse + longitudinal, which behaves as a minimal scalar)
    The xi_phi = 0 values (1, 1/6, 1, 3) are the ones used throughout the analysis."""
    tr = (1 - 6 * xi_phi) ** 2
    if case in ("scalar", "conformal"):
        xX = CASES[case][1] if xi_X is None else xi_X
        return 1 / 6 + 5 / 6 * tr * (1 - 6 * xX) ** 2
    if case == "dirac":
        return 1.0
    if case == "vector":
        return 13 / 6 + 5 / 6 * tr
    raise ValueError(case)
