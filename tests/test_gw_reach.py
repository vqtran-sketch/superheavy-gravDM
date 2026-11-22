"""Tests for gw_reach.py (section 5.5, appendix D.6)."""
import sys, pathlib
import numpy as np
from gravdm import gw_reach as gr


def test_hc_conversion_standard_coefficient():
    """h_c = 1.26e-18 (1 Hz/f) sqrt(h^2 Omega), the standard coefficient, and the two directions invert each other."""
    assert np.isclose(gr.hc_of_omega(1.0, 1.0), 1.26e-18, rtol=0.01)
    assert np.isclose(gr.omega_of_hc(3e4, gr.hc_of_omega(3e4, 2e-9)), 2e-9, rtol=1e-12)


def test_template_continuous_with_slopes():
    fp, Op = 1e5, 2.5e-8
    assert np.isclose(gr.template(fp, fp, Op), Op)
    assert np.isclose(gr.template(10 * fp, fp, Op), Op / 100)
    assert np.isclose(gr.template(fp / 4, fp, Op), Op / 8)
    ft = fp * gr.X_TRANS
    for log in (True, False):
        assert np.isclose(gr.template(ft * 0.9999, fp, Op, log), gr.template(ft * 1.0001, fp, Op, log), rtol=1e-2)
    ratio = gr.template(ft / 200, fp, Op, False) / gr.template(ft / 2000, fp, Op, False)
    assert np.isclose(ratio, 100)


def test_integrated_omega_analytic():
    """Above X_TRANS the integral is 1/2 + (1 - x_t^1.5)/1.5; the tail below adds x_t^1.5/2 for the x^2 case."""
    xt = gr.X_TRANS
    exact = 0.5 + (1 - xt**1.5) / 1.5 + xt**1.5 / 2
    assert np.isclose(gr.integrated_omega(1e5, 1.0, deep_ir_log=False), exact, rtol=1e-3)


def test_snr_constant_ratio():
    """Omega_s/Omega_n = r on a band df: rho = r sqrt(n_det T df)."""
    f = np.linspace(10, 110, 1001)
    assert np.isclose(gr.snr(f, 2 * np.ones_like(f), 2e-3 * np.ones_like(f), 2, 1e7), 1e-3 * np.sqrt(2 * 1e7 * 100))
