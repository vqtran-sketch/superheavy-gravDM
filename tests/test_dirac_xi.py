"""Tests for xsec_analytic.c_dirac_xi and the xi_phi dependence of the Dirac mass suppression (appendix A.3, eq. (A.26); script 24)."""
import sys, pathlib
import numpy as np
import pytest
from gravdm import xsec_analytic as xa
from gravdm import gravxsec as gx
from gravdm import parton as pt


@pytest.mark.parametrize("xi", [0.0, 1 / 6, 1.0, 5.0])
@pytest.mark.parametrize("r", [0.01, 0.1, 0.2])
def test_formula_vs_spinors(xi, r):
    """Closed form = explicit-spinor amplitude (two routes)."""
    assert np.isclose(xa.c_dirac_xi(r, xi), gx.c_hat(0.5, r, xi_phi=xi), rtol=1e-9)


def test_xi0_and_massless_limits():
    """xi = 0 reproduces c_of_r('dirac'); the massless coefficient is independent of xi."""
    r = np.linspace(0, 0.249, 50)
    assert np.allclose(xa.c_dirac_xi(r, 0.0), xa.c_of_r("dirac", r), rtol=1e-12)
    for xi in (1 / 6, 1.0, 5.0):
        assert np.isclose(xa.c_dirac_xi(0.0, xi), 1 / (960 * np.pi), rtol=1e-12)


@pytest.fixture(scope="module")
def ktab():
    return pt.K_table("tanh", zmin=1e-12, zmax=60.0, n=600)


def test_mu2_log2_law():
    """[DERIVED; eq. (A.26)] g(mu;xi) - g(mu;0) = [(1-6xi)^2-1] mu^2 [c2 L^2 + b L + c], L = ln(1/mu), for mu -> 0.
    Leading coefficient (two routes): analytic c2 = (5/3)/(int W)^2 with int W = pi/6 (tanh, from sigma = m Dphi^2/6), and the numerical
    fold; b = -17.97, c = 16.42 from a fit over mu = 1e-6..1e-2 (residual 4e-5)."""
    from scipy import integrate
    Wint = integrate.quad(pt.PROFILES["tanh"]["W"], 0, 60, limit=400)[0]
    assert np.isclose(Wint, np.pi / 6, rtol=1e-8)
    c2 = 5 / 3 / Wint ** 2
    zt, Kt = pt.K_table("tanh", zmin=1e-16, zmax=60.0, n=1200)
    I = lambda xi, mu: pt.yield_integral(lambda r: xa.c_dirac_xi(r, xi), mu, "tanh", zt, Kt)
    I0 = I(0.0, 0.0)
    for mu in (1e-5, 1e-3):
        L = np.log(1 / mu)
        D = (I(1.0, mu) - I(0.0, mu)) / I0 / 24 / mu ** 2
        assert np.isclose(D, c2 * L ** 2 - 17.97 * L + 16.42, rtol=2e-3), (mu, D)


def test_conformal_point_reduces_yield(ktab):
    """xi = 1/6 removes the spin-0 term: g(mu;1/6) < g(mu;0); xi = 1 doubles g at mu = 0.05 (ratio 2.1)."""
    zt, Kt = ktab
    I = lambda xi, mu: pt.yield_integral(lambda r: xa.c_dirac_xi(r, xi), mu, "tanh", zt, Kt)
    assert I(1 / 6, 0.1) < I(0.0, 0.1)
    assert np.isclose(I(1.0, 0.05) / I(0.0, 0.05), 2.13, rtol=0.02)
