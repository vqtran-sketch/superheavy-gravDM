"""Mass-suppression factor R(M_X) of the bubble-collision yield averaged over the collision scale (eqs. (4.7), (4.11), (4.12); appendix C.1).

R(M) = < g(M / M_coll) >_yield,   M_coll = (2L/(beta Dphi^2)) sqrt(rhat_1 rhat_2) eta_perp   [DERIVED, eq. (4.11)]
with the yield-weighted distributions (weight e1 e2 eta^3 on the first-collision surfaces, appendix C.1):
  rhat_i = beta r_i ~ Gamma(4, 1) iid,   eta_perp ~ 6 eta^5 on [0, 1]   [DERIVED, appendix C.1].
Written as s = M_coll / Mbar_coll with Mbar_coll = <M_coll>_yield, so <s> = 1 and R(M) = < g(mu s^-1) >, mu = M / Mbar_coll.
ln s = (ln rhat_1 + ln rhat_2)/2 + ln eta - ln S0, S0 = (Gamma(9/2)/Gamma(4))^2 (6/7); -ln eta ~ Exp(6).
The density of ln s is built on a uniform grid by numerical convolution; R is then a 1D quadrature in ln s (deterministic, no MC).
Units: all dimensionless. g: callable mass-suppression function of a single collision (planar walls, mu = M/(gamma m)).
Validity: same as eq. (4.10) (runaway walls, exponential nucleation, first collisions).
"""
import numpy as np
from scipy.special import gammaln

S0 = np.exp(2 * (gammaln(4.5) - gammaln(4.0))) * 6 / 7          # <sqrt(rhat_1 rhat_2) eta>_yield = 3.2214


def lns_density(h=2e-3, wmin=-12.0, wmax=4.0):
    """Grid u (uniform, step h) and density p(u) of u = ln s, normalised to int p du = 1."""
    w = np.arange(wmin, wmax + h / 2, h)                          # w = ln rhat
    pw = np.exp(4 * w - np.exp(w) - gammaln(4.0))                 # density of ln of a Gamma(4) variable
    p2 = np.convolve(pw, pw) * h                                  # density of w1 + w2 on grid 2*wmin + k h
    v = (2 * wmin + h * np.arange(p2.size)) / 2                   # v = (w1 + w2)/2, density 2 p2(2v), step h/2
    pv = 2 * p2
    t = np.arange(0, 6.0, h / 2)                                  # t = -ln eta, density 6 e^{-6t} (cut at e^{-36})
    pt = 6 * np.exp(-6 * t)
    pt[0] *= 0.5                                                  # trapezoid end weight at the jump t = 0 (else O(h) shift of ln s)
    p = np.convolve(pv, pt[::-1]) * (h / 2)                       # density of v - t: kernel on the grid -t, reversed to ascending
    u = v[0] - (t.size - 1) * h / 2 + (h / 2) * np.arange(p.size) - np.log(S0)
    p /= np.trapezoid(p, u)
    return u, p


def R_averaged(mu, g, h=2e-3):
    """R(mu) = int du p(u) g(mu e^{-u}), mu = M_X / Mbar_coll (array). Returns array like mu."""
    u, p = lns_density(h)
    keep = p > 1e-300
    u, p = u[keep], p[keep]
    mu = np.atleast_1d(mu)
    return np.array([np.trapezoid(p * g(m * np.exp(-u)), u) for m in mu])
