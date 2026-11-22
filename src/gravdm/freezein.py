"""Gravitational freeze-in of X from the SM thermal bath (section 4.3 and appendix B of the paper).

[LIT: Redi-Tesi, JHEP 11 (2026) 097 (arXiv:2606.28297), eq. (4.6)] for massless conformal sectors:
  gamma(T) = (2/M_P^4) (2 c_J/(7680 pi)) (2 c_SM/(7680 pi)) int ds s^{5/2} T K1(sqrt(s)/T) / (8 pi^3)
           = 3 c_J c_SM T^8 / (1280 pi^5 M_P^4)
yield_freezein_X (used for the paper, eq. (B.2)): spin-2 channel of the conformal SM (c_SM: 4 real scalars (4/3 each),
45 Weyl fermions (4), 12 vectors (16); N_T = 3 c_SM/4 = 283) with the X cross section at xi_phi = 1/6, plus the Higgs trace (spin-0)
channel for xi_H != 1/6. Massive X enters through the full threshold dependence c_X(M^2/s) of each channel.
For a minimal scalar the trace channel (s-wave) dominates above T_reh: 82% of the total at M = 10 T_reh for xi_H = 0.
yield_freezein (not used for the paper): c_X(0) at xi_phi = 0 for the whole SM, i.e. it includes a trace channel of X that a
conformal source cannot excite; it overestimates minimal-scalar freeze-in by x5.6 in the massless limit (see yield_freezein_X).
Quantum statistics neglected (Redi-Tesi: ~10% effect).
Yield: Y = int_{0}^{T_max} dT gamma/(T H s) in radiation domination; gamma is already the PARTICLE production rate (see gamma_rate):
gamma_rate equals int f1 f2 sigma v_Moller d3p1 d3p2/(2pi)^6 (Maxwell-Boltzmann), i.e. (1/2 for identical initial particles)
x (2 particles per event), so the yield is not multiplied by 2 again.
Units: GeV; M_P = reduced Planck mass.
"""
import numpy as np
from scipy import integrate, special
from .units import M_P, G_STAR_SM

C_SM = 4 * (4 / 3) + 45 * 4 + 12 * 16       # = 377.3


def gamma_rate(T, M, cfun, c0, c_sm=C_SM):
    """PARTICLE production rate per unit volume and time [GeV^4] of X (X + Xbar for Dirac): RT eq. (58) equals
    int f1 f2 sigma v_Moller d3p1 d3p2/(2pi)^6 for a Maxwell-Boltzmann bath = 2 x (event rate, with 1/2 for identical initial
    particles) [checked: tests/test_freezein.py::test_gamma_rate_is_particle_rate]. c0 only normalises (cancels between cJ and cfun/c0).
    cfun(r) = sigma M_P^4/s of X at r = M^2/s; c0 = cfun(0); c_sm = spin-2 weight of the source sector (default: whole SM)."""
    cJ = 7680 * np.pi * c0
    # factor 180 = Pi_{mnrs} Pi^{mnrs}/q^4 (RT line above eq. 58); omitted in RT's displayed integrand but present in
    # their closed form (checked: int x^6 K1 = 2^5 Gamma(4) Gamma(3) reproduces 3 c_J c_SM T^8/(1280 pi^5 M_P^4) only with it)
    pref = 180 * (2 / M_P ** 4) * (2 * cJ / (7680 * np.pi)) * (2 * c_sm / (7680 * np.pi)) / (8 * np.pi ** 3)
    smin = max(4 * M * M, 1e-30)
    f = lambda x: (x * T) ** 5 * T * special.k1(x) * cfun(M * M / (x * T) ** 2) / c0 * 2 * x * T * T   # s = (xT)^2, ds = 2 x T^2 dx
    lo = np.sqrt(smin) / T
    val = integrate.quad(f, lo, max(lo, 0) + 60, limit=400)[0]
    return pref * val


def H_rad(T):
    return np.sqrt(np.pi ** 2 * G_STAR_SM / 90) * T * T / M_P


def s_ent(T):
    return 2 * np.pi ** 2 / 45 * G_STAR_SM * T ** 3


def _yield_integral(Tmax, M, cfun, c0, c_sm, epsrel=1e-6):
    """int_0^Tmax dT gamma/(T H s) (particles) by adaptive quadrature in ln T, split at T = M/40 and M/3 so that the
    Boltzmann-suppressed region near T_max (M >> T) is resolved (a fixed 60-point log grid would overestimate the yield at M/T = 10 by 59%)."""
    f = lambda u: gamma_rate(np.exp(u), M, cfun, c0, c_sm=c_sm) / (H_rad(np.exp(u)) * s_ent(np.exp(u)))     # dT/T = du
    lo, hi = np.log(Tmax * 1e-4), np.log(Tmax)
    pts = sorted({x for x in ((np.log(M / 40), np.log(M / 3)) if M > 0 else ()) if lo < x < hi})
    edges = [lo] + pts + [hi]
    return sum(integrate.quad(f, a, b, limit=200, epsrel=epsrel, epsabs=0)[0] for a, b in zip(edges[:-1], edges[1:]))


def yield_freezein(Tmax, M, cfun, c0, n=None):
    """Y = int_0^Tmax dT gamma/(T H s) (particles). n: unused (kept for call compatibility)."""
    return _yield_integral(Tmax, M, cfun, c0, C_SM)


# ---------------------------------------------------------------------------------------------------------------------
# Freeze-in by channel (eqs. (4.15), (B.2)). The function yield_freezein above uses c_X(0) at xi_phi = 0,
# i.e. it includes the trace-trace channel of X, which a CONFORMAL source cannot excite: for a minimal scalar this
# overestimates freeze-in by x5.6, for Proca by x1.38 (Dirac: unchanged in the massless limit). Treatment used here
# [DERIVED, eqs. (2.4), (A.20); spin-2 and trace channels do not interfere]:
#   (a) spin-2 channel, whole SM as a CFT (weight C_SM): X cross section with a TRACELESS source, c_2(r) = c_X(r; xi_phi = 1/6);
#   (b) trace channel, only the 4 real Higgs scalars (weight C_H = 4 * 4/3, curvature coupling xi_H):
#       c_tr(r) = c_X(r; xi_H) - c_X(r; 1/6)   (zero for Dirac and transverse vectors in the massless limit).
# The rate of each part is the RT expression with (c_J, c_SM) -> (7680 pi c_part(0), weight); same s-kernel since sigma ∝ s.
# c_X(r; xi_phi, xi_X) from the explicit amplitudes (gravxsec.c_hat), tabulated in r and interpolated as c / v^p.
C_H = 4 * (4 / 3)
_SPIN = {"scalar": (0, 0.0), "conformal": (0, 1 / 6), "dirac": (0.5, 0.0), "vector": (1, 0.0)}
_VPOW = {0: 1, 0.5: 3, 1: 1}
_TAB = {}


def c_table(spin, xi_phi, nr=121):
    """sigma M_P^4/s of X (case name as in xsec_analytic) for a source of curvature coupling xi_phi, as a callable in r."""
    key = (spin, round(xi_phi, 12))
    if key not in _TAB:
        from . import gravxsec as gx
        sp, xX = _SPIN[spin]
        r = 0.25 * (1 - np.cos(np.linspace(0, np.pi / 2, nr)))                 # denser near r = 1/4
        r[0] = 1e-7                                                             # Proca L-polarisation is singular at M = 0
        r[-1] = 0.25 * (1 - 1e-9)                                               # below r[0]: interp clamps to the massless value
        v = np.sqrt(1 - 4 * r)
        c = np.array([gx.c_hat(sp, x, xi=xX, xi_phi=xi_phi) for x in r])
        g = c / v ** _VPOW[sp]
        _TAB[key] = lambda rr, r=r, g=g, p=_VPOW[sp]: np.where(np.asarray(rr) < 0.25,
                                                               np.interp(rr, r, g) * np.sqrt(np.clip(1 - 4 * np.asarray(rr), 0, None)) ** p, 0.0)
    return _TAB[key]


def channels(spin, xi_H=0.0):
    """[(cfun, c0, weight)] for the spin-2 (whole SM) and trace (Higgs) channels.
    The trace channel is kept whenever it is non-zero at ANY r (a massive Dirac or conformal-scalar X has a trace channel
    although it vanishes at r = 0). Its 'c0' is only a normalisation (cancels in gamma_rate), so the
    maximum over r is used when c_tr(0) = 0."""
    c2 = c_table(spin, 1 / 6)
    out = [(c2, float(c2(0.0)), C_SM)]
    cf = c_table(spin, xi_H)
    ctr = lambda r, cf=cf, c2=c2: np.maximum(cf(r) - c2(r), 0.0)
    rr = np.linspace(0.0, 0.2499, 400)
    cmax = float(np.max(ctr(rr)))
    if cmax > 1e-9 * out[0][1]:
        norm = max(float(ctr(0.0)), cmax)
        out.append((lambda r, f=ctr: f(r), norm, C_H))
    return out


def yield_freezein_X(Tmax, M, spin, xi_H=0.0, n=None):
    """Freeze-in yield (particles) of X = 'scalar' (minimal), 'conformal', 'dirac', 'vector' from an SM bath
    in radiation domination with T_max: spin-2 channel of the conformal SM + trace channel of the Higgs (coupling xi_H).
    n: unused (kept for call compatibility; the T integral is adaptive)."""
    return sum(_yield_integral(Tmax, M, cfun, c0, w) for cfun, c0, w in channels(spin, xi_H))
