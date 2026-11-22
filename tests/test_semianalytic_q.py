"""Tests for semianalytic_q.py (eq. (4.9), appendix C.1)."""
import sys, pathlib
import numpy as np
from gravdm import semianalytic_q as sq

RATE = sq.exponential_rate()                                       # Gamma = e^t, beta = 1
T0 = np.log(1 / (8 * np.pi))                                       # I(T0) = 1
TR = (T0 - 40, T0 + 5)
E_HAT = lambda r: r / 3


def test_false_vacuum_exponent_exponential():
    """I(t) = 8 pi Gamma0 e^t for Gamma = Gamma0 e^t (int_0^inf r^3 e^{-r} dr = 6) [semianalytic_q eq. (I)]."""
    t = np.array([-3.0, 0.0, 1.0])
    assert np.allclose(sq.false_vacuum_exponent(t, sq.exponential_rate(0.7)), 8 * np.pi * 0.7 * np.exp(t), rtol=1e-10)


def test_bubble_density_exponential():
    """n_b = beta^3/(8 pi), independent of Gamma0 [semianalytic_q eq. (N)]."""
    for g0 in (1.0, 1e-3):
        tr = (np.log(1 / (8 * np.pi * g0)) - 40, np.log(1 / (8 * np.pi * g0)) + 5)
        assert np.isclose(sq.bubble_density(sq.exponential_rate(g0), tr, n_panels=16), 1 / (8 * np.pi), rtol=1e-10)


def test_angular_factor():
    """int dmu |n1-n2| h^k = 8/(k+3) (quadrature), and the same from random unit vectors (independent of mu = 1 - 2h^2)."""
    for k in (0, 3):
        assert np.isclose(sq.angular_factor(k), 8 * np.pi ** 2 * 8 / (k + 3), rtol=1e-12)
    rng = np.random.default_rng(3)
    v = rng.normal(size=(2, 400_000, 3)); v /= np.linalg.norm(v, axis=2, keepdims=True)
    h = np.linalg.norm(v[0] - v[1], axis=1) / 2
    for k in (0, 3):
        mc = 2 * np.mean(2 * h * h ** k)                           # int dmu f = 2 <f> for mu uniform on [-1, 1]
        assert abs(mc / (8 / (k + 3)) - 1) < 5e-3


def test_closed_forms_and_quadrature_agree():
    """A/V = 2/3, Q_head = 2/3, Q_beta = 1/3: closed form (eq. C) vs direct quadrature of eq. (M)."""
    assert np.isclose(sq.closed_form_exponential(0, 0), 2 / 3)
    assert np.isclose(sq.closed_form_exponential(1, 0, 1 / 3), 2 / 3)
    assert np.isclose(sq.closed_form_exponential(1, 3, 1 / 3), 1 / 3)
    assert np.isclose(sq.surface_moment(RATE, TR, n_panels=16), 2 / 3, rtol=1e-10)
    assert np.isclose(sq.surface_moment(RATE, TR, weight=E_HAT, n_panels=16), 2 / 3, rtol=1e-10)
    assert np.isclose(sq.surface_moment(RATE, TR, weight=E_HAT, k=3, n_panels=16), 1 / 3, rtol=1e-10)


def test_independent_of_gamma0():
    """Q is invariant under a shift of the time origin (Gamma0 -> c Gamma0)."""
    g0 = 1e-4; t0 = np.log(1 / (8 * np.pi * g0))
    q = sq.surface_moment(sq.exponential_rate(g0), (t0 - 40, t0 + 5), weight=E_HAT, k=3, n_panels=16)
    assert np.isclose(q, 1 / 3, rtol=1e-10)


def test_convergence_doubling():
    """Grid x2: Q_beta changes by < 1e-8 between 16 and 32 panels (relative 1e-6 for a non-exponential rate)."""
    a, b = (sq.surface_moment(RATE, TR, weight=E_HAT, k=3, n_panels=n) for n in (16, 32))
    assert abs(a - b) < 1e-8
    soft = lambda t: np.logaddexp(0, 2 * np.asarray(t, float))
    a, b = (sq.surface_moment(soft, (-30, 10), weight=E_HAT, k=3, n_panels=n, r_max=40) for n in (16, 32))
    assert abs(a / b - 1) < 1e-6


def test_1d_point_density_equals_bubble_density():
    """Limit d = 1: every real bubble has exactly two first-collision points, each shared by two bubbles, so the point
    density equals n_b for ANY rate that fills space. Tests the empty-light-cone conditioning of eq. (M)."""
    soft = lambda t: np.logaddexp(0, 2 * np.asarray(t, float))
    for rate, tr, rmax in ((RATE, (-45, 5), 60.0), (soft, (-30, 10), 40.0)):
        nb = sq.bubble_density(rate, tr, dim=1, r_max=rmax, n_panels=16)
        npts = sq.surface_moment(rate, tr, dim=1, r_max=rmax, n_panels=16)
        assert np.isclose(nb, npts, rtol=1e-9)
    assert np.isclose(sq.bubble_density(RATE, (-45, 5), dim=1, n_panels=16), 0.5, rtol=1e-10)


def test_1d_direct_monte_carlo():
    """Direct 1D simulation (nucleation only in false vacuum) vs eq. (M) in d = 1 with Gamma = e^t:
    sum r1 r2 / L = 1/2, mean r = 1 (r ~ Exp(1)), n_b = 1/2."""
    L = 20000.0
    n, a, b = sq.mc_1d_collisions(np.random.default_rng(1), L)
    assert abs(n / L / 0.5 - 1) < 0.03
    assert abs((a * b).sum() / L / sq.surface_moment(RATE, (-45, 5), weight=lambda r: r, dim=1, n_panels=16) - 1) < 0.04
    assert abs(np.r_[a, b].mean() - 1) < 0.03
