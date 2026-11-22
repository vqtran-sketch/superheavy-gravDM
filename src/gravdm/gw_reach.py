"""GW spectrum template and detector comparison for the An--Li signal (section 5.5 and appendix D.6 of the paper).

Units: frequencies in Hz today; Omega means h^2 Omega_GW (dimensionless); h_c dimensionless.
"""
import numpy as np

H100 = 3.2408e-18   # H_0/h [1/s] = 100 km/s/Mpc
A_OMEGA = 2 * np.pi**2 / (3 * H100**2)   # h^2 Omega = A_OMEGA f^2 h_c^2 [s^2]; Guo et al. 2501.18146 below their Eq. (5) [LIT]
X_TRANS = 1 / 15    # k/k_peak where the k^1.5 region ends: An--Li Fig. 2 and text, k/H_c ~ 0.1 with peak at k/H_c ~ 1.5 [LIT, read by eye]


def omega_of_hc(f, hc):
    """h^2 Omega from characteristic strain. Valid for any stochastic background."""
    return A_OMEGA * np.asarray(f, float)**2 * np.asarray(hc, float)**2


def hc_of_omega(f, omega):
    """Characteristic strain from h^2 Omega; inverse of omega_of_hc."""
    return np.sqrt(np.asarray(omega, float) / A_OMEGA) / np.asarray(f, float)


def template(f, f_peak, omega_peak, deep_ir_log=True):
    """Piecewise spectrum, h^2 Omega(f), continuous at x = f/f_peak = 1 and X_TRANS [An--Li Eqs. (13), (15), Fig. 2; n_DS = 6].

    x >= 1: x^-2.  X_TRANS <= x < 1: x^1.5.  x < X_TRANS: x^3 ln^2 x (deep_ir_log, alpha = 0.05, radiation domination soon after
    the transition) or x^2 (alpha = 5e-4, long kination).  An estimate: the normalisation of the tail is good to a factor ~2 against
    An--Li Fig. 2 at k/H_c = 0.01.  Valid for free-passage walls; the shape is theirs, not computed here.
    """
    x = np.asarray(f, float) / f_peak
    xt = X_TRANS
    low = xt**1.5 * (x / xt)**3 * (np.log(x) / np.log(xt))**2 if deep_ir_log else xt**1.5 * (x / xt)**2
    return omega_peak * np.where(x >= 1, x**-2.0, np.where(x >= xt, x**1.5, low))


def snr(f, omega_noise, omega_signal, n_det, t_obs):
    """rho = [n_det t_obs int df (Omega_signal/Omega_noise)^2]^(1/2) [Schmitz 2002.04615 Eq. (1.1)]. t_obs in s; arrays on the grid f."""
    return np.sqrt(n_det * t_obs * np.trapezoid((omega_signal / omega_noise)**2, f))


def integrated_omega(f_peak, omega_peak, deep_ir_log=True):
    """int dln f h^2 Omega, for the Delta N_eff bound."""
    f = f_peak * np.logspace(-8, 8, 8001)
    return np.trapezoid(template(f, f_peak, omega_peak, deep_ir_log), np.log(f))
