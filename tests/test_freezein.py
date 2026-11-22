"""Tests for freezein.py (section 4.3 and appendix B)."""
import sys, pathlib
import numpy as np
from gravdm import freezein as fz
from gravdm import xsec_analytic as xa
from gravdm.units import M_P


def test_rate_reproduces_redi_tesi_eq58():
    """[LIT: RT 2606.28297 eq. (58)] massless conformal: gamma = 3 c_J c_SM T^8 / (1280 pi^5 M_P^4)."""
    cf = lambda r: xa.c_of_r("conformal", r); c0 = xa.c0("conformal"); T = 1e12
    assert np.isclose(fz.gamma_rate(T, 0.0, cf, c0), 3 * (4 / 3) * fz.C_SM * T ** 8 / (1280 * np.pi ** 5 * M_P ** 4), rtol=1e-8)


def test_yield_matches_redi_tesi_eq30():
    """[LIT: RT eq. (30)] Y_GFI ~ 5e-6 c_J (T_R/M_Pl)^3 (DM yield, i.e. particles); our integral gives 6.0e-6 (RT quote is approximate;
    appendix B.1). gamma_rate is already a particle rate, see test_gamma_rate_is_particle_rate."""
    cf = lambda r: xa.c_of_r("conformal", r); c0 = xa.c0("conformal"); T = 1e12
    Y = fz.yield_freezein(T, 0.0, cf, c0)
    assert 4e-6 < Y / ((4 / 3) * (T / M_P) ** 3) < 7e-6


def test_scaling_T3_and_boltzmann():
    cf = lambda r: xa.c_of_r("scalar", r); c0 = xa.c0("scalar")
    y1 = fz.yield_freezein(1e11, 0.0, cf, c0); y2 = fz.yield_freezein(1e12, 0.0, cf, c0)
    assert np.isclose(y2 / y1, 1e3, rtol=1e-6)
    assert fz.yield_freezein(1e12, 3e13, cf, c0) < 1e-10 * y2           # M = 30 T: exponentially suppressed


def test_corrected_channels_massless():
    """Spin-2 coefficients with a traceless source: chat = 1/6, 1, 13/6, 1/6 (minimal scalar, Dirac, Proca, conformal);
    Higgs trace channel (xi_H = 0): chat = 5/6 for minimal scalar and Proca, 0 for Dirac and conformal [DERIVED, eqs. (2.4), (4.15)]."""
    exp2 = {"scalar": 1 / 6, "dirac": 1.0, "vector": 13 / 6, "conformal": 1 / 6}
    exptr = {"scalar": 5 / 6, "dirac": 0.0, "vector": 5 / 6, "conformal": 0.0}
    for sp in exp2:
        ch = fz.channels(sp, 0.0)
        assert np.isclose(ch[0][1] * 960 * np.pi, exp2[sp], rtol=1e-6) and ch[0][2] == fz.C_SM
        tr = float(ch[1][0](0.0)) * 960 * np.pi if len(ch) > 1 else 0.0
        assert np.isclose(tr, exptr[sp], atol=1e-6)


def test_corrected_vs_old_freezein_massless():
    """Ratio of yield_freezein_X to yield_freezein (massless) = [C_SM chat_2 + C_H chat_tr] / [C_SM chat(xi_phi=0)]: 0.1784 (scalar), 1 (Dirac), 0.7262 (Proca)."""
    T = 1e12
    for sp, ratio in (("scalar", 1 / 6 + fz.C_H * (5 / 6) / fz.C_SM), ("dirac", 1.0), ("vector", (13 / 6 + fz.C_H * (5 / 6) / fz.C_SM) / 3)):
        new = fz.yield_freezein_X(T, 0.0, sp, n=40)
        old = fz.yield_freezein(T, 0.0, lambda r, c=sp: xa.c_of_r(c, r), xa.c0(sp), n=40)
        assert np.isclose(new / old, ratio, rtol=1e-4)


def test_corrected_massive_threshold_shape():
    """Dirac with a traceless source falls faster with mass (c/c0 = 0.59 at r = 0.1, section 2) than with xi_phi = 0 (0.74)."""
    c2 = fz.c_table("dirac", 1 / 6); c0 = fz.c_table("dirac", 0.0)
    assert np.isclose(c2(0.1) / c2(0.0), 0.589, atol=2e-3) and np.isclose(c0(0.1) / c0(0.0), xa.c_of_r("dirac", 0.1) / xa.c0("dirac"), rtol=1e-4)


def test_gamma_rate_is_particle_rate():
    """gamma_rate (RT eq. 58) equals the direct Maxwell-Boltzmann collision integral
    int f1 f2 sigma v_Moller d3p1 d3p2/(2 pi)^6 = (8/3) c0 rho^2/M_P^4, rho = 3 T^4/pi^2 (one real massless scalar bath, c_SM = 4/3),
    which is the PARTICLE production rate: (1/2 for identical initial particles) x (2 X per event) [cf. Clery et al. 2203.02004 eq. (21) and
    footnote 6]. Hence the yield must not multiply it by 2."""
    T = 1e12; c2 = fz.c_table("dirac", 1 / 6); c0 = float(c2(0.0))
    rho = 3 * T ** 4 / np.pi ** 2
    assert np.isclose(fz.gamma_rate(T, 0.0, c2, c0, c_sm=4 / 3), 8 / 3 * c0 * rho ** 2 / M_P ** 4, rtol=1e-8)


def test_freezein_quadrature_converged():
    """Adaptive ln T quadrature vs a 4000-point log-grid trapezoid at M/T_max = 3 and 10 (a fixed 60-point grid would be 59% high at 10)."""
    T = 1e12
    for x in (3.0, 10.0):
        M = x * T; cf, c0, w = fz.channels("dirac", 0.0)[0]
        Ts = np.geomspace(T * 1e-4, T, 4000)
        ref = np.trapezoid([fz.gamma_rate(t, M, cf, c0, c_sm=w) / (t * fz.H_rad(t) * fz.s_ent(t)) for t in Ts], Ts)
        assert np.isclose(fz._yield_integral(T, M, cf, c0, w), ref, rtol=2e-3)


def test_trace_channel_kept_for_massive_dirac_and_conformal():
    """The Higgs trace channel vanishes at r = 0 for Dirac and conformal X but not at finite mass; it must be kept.
    At r = 0.1: 960 pi (c(xi_phi = 0) - c(xi_phi = 1/6)) = 0.154919 (Dirac), 0.0258223 (conformal), from the explicit amplitudes."""
    for sp, val in (("dirac", 0.154919), ("conformal", 0.0258223)):
        ch = fz.channels(sp, 0.0)
        assert len(ch) == 2 and np.isclose(float(ch[1][0](0.1)) * 960 * np.pi, val, rtol=1e-4)
