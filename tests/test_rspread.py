"""Tests for rspread.py (eqs. (4.7), (4.11), (4.12)): density of ln s = ln(M_coll/Mbar_coll) and the averaged suppression R."""
import sys, pathlib
import numpy as np
from scipy.special import gamma as G
from gravdm.rspread import lns_density, R_averaged, S0


def test_S0_value():
    assert abs(S0 - (G(4.5) / G(4)) ** 2 * 6 / 7) < 1e-12 and abs(S0 - 3.2214) < 1e-4


def test_density_normalisation_and_mean():
    u, p = lns_density()
    assert abs(np.trapezoid(p, u) - 1) < 1e-10
    assert abs(np.trapezoid(p * np.exp(u), u) - 1) < 1e-4            # <s> = 1 by construction of Mbar_coll


def test_inverse_moment_analytic():
    # <1/s> = S0 <(r1 r2)^{-1/2}> <1/eta> = S0 (Gamma(3.5)/Gamma(4))^2 (6/5)
    u, p = lns_density()
    exact = S0 * (G(3.5) / G(4)) ** 2 * 6 / 5
    assert abs(np.trapezoid(p * np.exp(-u), u) / exact - 1) < 1e-4


def test_R_vs_monte_carlo_and_limits():
    g = lambda mu: np.exp(-mu ** 2)
    rng = np.random.default_rng(0)
    s = np.sqrt(rng.gamma(4, size=10 ** 6) * rng.gamma(4, size=10 ** 6)) * rng.random(10 ** 6) ** (1 / 6) / S0
    for mu in (0.1, 0.5, 1.0, 2.0):
        mc = np.mean(g(mu / s))
        assert abs(R_averaged(mu, g)[0] - mc) < 5e-3 * max(mc, 1e-2)
    assert abs(R_averaged(1e-6, g)[0] - 1) < 1e-9


def test_grid_convergence():
    g = lambda mu: np.exp(-mu ** 2)
    mu = np.array([0.05, 0.2, 1.0, 3.0])
    a, b = R_averaged(mu, g, h=4e-3), R_averaged(mu, g, h=2e-3)
    assert np.all(np.abs(a / b - 1) < 1e-4)
