"""Tests for benchdm.py (flat-space benchmark DM plane, section 4.4)."""
import sys, pathlib
import numpy as np
from scipy import optimize
from gravdm import benchdm as bd
from gravdm.units import M_P, MY_DM, G_STAR_SM

COEFF = 6.81e-3     # K Q_beta/(960 pi), eq. (4.10)


def test_mbar_coll_value_and_scaling():
    # 11.16 (H/beta) M_P / nu^2 (eq. (4.12))
    assert np.isclose(bd.mbar_coll(0.1, 3.0) / (0.1 * M_P / 9), 11.16, rtol=2e-3)
    assert np.isclose(bd.mbar_coll(0.1, 2.0) / bd.mbar_coll(0.1, 3.0), 9 / 4)


def test_plateau_matches_root_of_relic_condition():
    for M in (1e12, 1e14):
        T = optimize.brentq(lambda T: M * COEFF * 0.1 * (T / M_P) ** 3 - MY_DM, 1e6, 1e18, xtol=1e-6, rtol=1e-12)
        assert np.isclose(bd.treh_plateau(M, 0.1, COEFF), T, rtol=1e-9)
    # T ∝ (M H/beta)^{-1/3}
    assert np.isclose(bd.treh_plateau(8e13, 0.1, COEFF) / bd.treh_plateau(1e13, 0.1, COEFF), 0.5)


def test_treh_min_scaling_and_script49_value():
    r = bd.treh_min(0.01, 3.0, COEFF, 0.0826) / bd.treh_min(0.1, 3.0, COEFF, 0.0826)
    assert np.isclose(r, 100 ** (1 / 3))
    assert np.isclose(bd.treh_min(0.1, 3.0, COEFF, 0.0826), 7.19e10, rtol=0.01)      # script 49


def test_gps_ratio():
    assert np.isclose(bd.gps_sphere_ratio(), (8 * np.pi) ** (1 / 3) / 2, rtol=1e-12)
    assert np.isclose(bd.gps_sphere_ratio(), 1.46, atol=0.01)


def test_envelope_numbers():
    assert np.isclose(bd.FSTAR_OVER_BETA, 0.2296, atol=1e-4)
    assert np.isclose(bd.omega_peak_envelope(0.1), 1.27e-8, rtol=0.01)
    # f_peak ∝ (beta/H) T_reh; dimension check via the H* redshift factor 16.5e-6 Hz at T = 100 GeV, g* = 100, f*/H* = 1
    assert np.isclose(bd.fpeak_envelope(100.0, 1 / (1 / bd.FSTAR_OVER_BETA), gstar=100.0), 1.65e-5)
