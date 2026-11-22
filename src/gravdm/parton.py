"""Wall parton densities, luminosity and graviton-mediated pair yield (sections 3.1-3.3 of the paper).

Planar wall moving along +x with rest-frame profile phi(z), field jump Dphi, mass scale m, Lorentz factor gamma.
  F(k)    = int dz phi'(z) e^{i k z}                     (rest frame; F(0) = Dphi)            [eq. (3.3)]
  dN/(A dp) = |F(p/gamma)|^2 / (pi p)                     (lab frame, quasi-real quanta)       [eq. (3.5)]   [LIT: Ghoshal:2026pew, eq. (19)]
  E/A     = int p dN = gamma int_0^inf dk |F|^2/pi = gamma sigma_wall                            [eq. (3.4)]
Using x = p/(gamma m) and W(x) = |F(m x)|^2/Dphi^2 (W(0) = 1):
  s dL/ds = (Dphi^4/pi^2) K(z),  K(z) = int_0^inf dx/x W(x) W(z/x),  z = s/(4 gamma^2 m^2)      [eq. (3.7)]
  N_X/A   = int ds (dL/ds) sigma(s) = (4 gamma^2 m^2 Dphi^4/(pi^2 M_P^4)) int_0^inf dz K(z) c(M^2/(4 gamma^2 m^2 z))
                                                                                                 [eq. (3.12)]
Theorem (M -> 0, c -> c0):  N_X/A = 4 c0 (E_1/A)(E_2/A)/M_P^4   (any profiles)                    [eq. (3.13)]
Units: momenta and masses in GeV; M_P in GeV; N/A in GeV^2.
"""
import numpy as np
from scipy import integrate
from scipy.interpolate import CubicSpline

# ---------------------------------------------------------------- profiles: W(x) = |F(m x)|^2 / Dphi^2
PROFILES = {
    # phi = (Dphi/2)(1 + tanh(m z/2))  (phi^4 kink, wall thickness 2/m)       F = Dphi (pi k/m)/sinh(pi k/m)
    "tanh": dict(W=lambda x: np.where(np.abs(x) < 1e-8, 1.0, (np.pi * x / np.sinh(np.minimum(np.pi * np.abs(x), 700.0))) ** 2),
                 dphi_prime=lambda z: 0.25 * (1 / np.cosh(z / 2)) ** 2),
    # erf wall: phi' = Dphi m exp(-m^2 z^2/2)/sqrt(2 pi)                          F = Dphi exp(-k^2/(2 m^2))
    "erf": dict(W=lambda x: np.exp(-x ** 2),
                dphi_prime=lambda z: np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)),
    # sine-Gordon kink phi = 4 arctan e^{m z} (Dphi = 2 pi): phi' = 2 m sech(m z)  F = Dphi sech(pi k/(2m))
    "sG": dict(W=lambda x: (1 / np.cosh(np.minimum(np.pi * x / 2, 700.0))) ** 2,
               dphi_prime=lambda z: (1 / np.cosh(z)) / np.pi),
}
# dphi_prime: phi'(z)/(Dphi m) at m = 1 (normalised so that int dz = 1)


def sigma_over_dphi2_m(profile):
    """Wall tension / (Dphi^2 m) two ways: int dz phi'^2 (direct) and int dk |F|^2/pi (Parseval).  [eqs. (3.2), (3.4)]"""
    f = PROFILES[profile]["dphi_prime"]; W = PROFILES[profile]["W"]
    direct = integrate.quad(lambda z: f(z) ** 2, -60, 60, limit=400)[0]
    fourier = integrate.quad(lambda x: W(x), 0, 60, limit=400)[0] / np.pi
    return direct, fourier


def K(z, profile, xmin=1e-12, xmax=60.0):
    """K(z) = int dx/x W(x) W(z/x)  [eq. (3.7)]; computed in ln x."""
    W = PROFILES[profile]["W"]
    lo = max(np.log(xmin), np.log(z / xmax)); hi = min(np.log(xmax), np.log(z / xmin))
    if hi <= lo:
        return 0.0
    f = lambda t: W(np.exp(t)) * W(z * np.exp(-t))
    # integrand is peaked; split at the symmetric point
    tc = 0.5 * np.log(z)
    a = integrate.quad(f, lo, tc, limit=400, epsabs=0, epsrel=1e-9)[0]
    b = integrate.quad(f, tc, hi, limit=400, epsabs=0, epsrel=1e-9)[0]
    return a + b


def K_table(profile, zmin=1e-10, zmax=30.0, n=400):
    z = np.logspace(np.log10(zmin), np.log10(zmax), n)
    return z, np.array([K(zi, profile) for zi in z])


def yield_integral(cfun, mu, profile, ztab=None, Ktab=None):
    """I(mu) = int_0^inf dz K(z) c(mu^2/(4 z)),  mu = M/(gamma m).  [eqs. (3.12), (3.14)]
    cfun(r) = sigma M_P^4/s at r = M^2/s (zero for r >= 1/4).  Massless limit: I = c0 (int W)^2."""
    if ztab is None:
        ztab, Ktab = K_table(profile)
    lz = np.log(ztab)
    zthr = mu ** 2                      # r = mu^2/(4z) < 1/4  <=> z > mu^2
    # cubic spline of ln K in ln z (K > 0, smooth); linear interpolation converged only at 1e-3
    spl = CubicSpline(lz, np.log(np.maximum(Ktab, 1e-300)))
    g = lambda t: np.exp(t) * np.exp(spl(t)) * cfun(mu ** 2 / (4 * np.exp(t)))
    lo = max(lz[0], np.log(zthr) if zthr > 0 else lz[0]); hi = lz[-1]
    if hi <= lo:
        return 0.0
    return integrate.quad(g, lo, hi, limit=400, epsrel=1e-8)[0]


def cut_yield_table(cfun, mus, z, Kz):
    """Yield with a hard cut on the parton energy, normalised to the massless uncut yield (script 21; cutoff Lambda_cut, section 6):
    G(mu, z_c) = int_{mu^2}^{z_c} dz K(z) c(mu^2/(4z)) / I(0),   sqrt(s) = 2 sqrt(z) gamma m  ->  z_c = (Lambda/(2 gamma m))^2.
    Returned on the grid z (cumulative trapezoid in ln z), shape (len(mus), len(z)); G(mu, z[-1]) = g(mu) of script 01.
    I(0) is taken from the first row, so mus[0] must be << 1 (threshold effect ~ mus[0]^2).  Dimensionless."""
    lz = np.log(z); rows = []
    for mu in mus:
        r = mu ** 2 / (4 * z)
        f = z * Kz * np.where(r < 0.25, cfun(np.minimum(r, 0.2499999)), 0.0)
        rows.append(np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(lz))]))
    rows = np.array(rows)
    return rows / rows[0, -1]


def NA_prefactor(gamma, m, dphi, MP):
    """4 gamma^2 m^2 Dphi^4 / (pi^2 M_P^4)  [GeV^2 * GeV^4/GeV^4 = GeV^2... with dphi in GeV: GeV^6/GeV^4 = GeV^2]."""
    return 4 * gamma ** 2 * m ** 2 * dphi ** 4 / (np.pi ** 2 * MP ** 4)
