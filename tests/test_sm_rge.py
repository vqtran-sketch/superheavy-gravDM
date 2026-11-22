"""Tests for sm_rge: SM running (literature reproduction) and the kination barrier condition (two routes)."""
import sys, pathlib
import numpy as np
import pytest
from gravdm import sm_rge as S

BUTTAZZO = dict(Mt=173.34, Mh=125.15, alphas=0.1184, MW=80.384)


@pytest.fixture(scope="module")
def run3():
    return S.run(mu_max=1.22e19, loops=3, **BUTTAZZO)


def test_one_loop_lambda_beta_matches_LR():
    """d lambda/d ln mu at one loop with gauge couplings off = (24 l^2 + 12 y^2 l - 6 y^4)/16 pi^2 [Laverda-Rubio Eq. (3.2)]."""
    l, y = 0.05, 0.6
    c = np.array([1e-9, 1e-9, 1.0, y, l])
    assert np.isclose(S.beta(c, loops=1)[4], (24 * l ** 2 + 12 * y ** 2 * l - 6 * y ** 4) / (16 * np.pi ** 2), rtol=1e-6)


def test_planck_scale_couplings_reproduce_buttazzo(run3):
    """[LIT] Buttazzo et al. 1307.3536 Eq. (61) at M_Pl = 1.22e19 GeV (3-loop RGE, central inputs). y_b, y_tau neglected here."""
    mu, C = run3
    ref = [0.6154, 0.5055, 0.4873, 0.3825, -0.0143]
    assert np.allclose(C[-1, :4], ref[:4], atol=1e-3)
    assert abs(C[-1, 4] - ref[4]) < 5e-4


def test_instability_scale_matches_buttazzo(run3):
    """[LIT] Buttazzo et al. Eq. (67) and text: Lambda_lambda ~ 2 Lambda_V, log10 Lambda_V = 9.5 at central inputs -> ~6e9 GeV (+-30%)."""
    muL, _ = S.mu_Lambda(*run3)
    assert 0.7 < muL / (2 * 10 ** 9.5) < 1.3


def test_mt_dependence_slope():
    """Buttazzo Eq. (67): d log10 Lambda/d M_t = -1.0 per GeV (scale of lambda = 0 follows within ~30%)."""
    m1, _ = S.mu_Lambda(*S.run(loops=3, **{**BUTTAZZO, "Mt": 172.84}))
    m2, _ = S.mu_Lambda(*S.run(loops=3, **{**BUTTAZZO, "Mt": 173.84}))
    assert 0.7 < -np.log10(m2 / m1) < 1.4


def test_convergence_rtol():
    """Changing the integrator tolerance by 1e3 and the grid by x2 leaves mu_Lambda stable to 1e-3."""
    a, _ = S.mu_Lambda(*S.run(loops=3, rtol=1e-10, n=600))
    b, _ = S.mu_Lambda(*S.run(loops=3, rtol=1e-7, n=1200))
    assert np.isclose(a, b, rtol=1e-3)


def test_barrier_condition_reduces_to_LR_eq36():
    """[DERIVED] With lambda = -(3 y^4/16 pi^2) ln(h^2/mu_L^2) (constant slope), max h^2 (lambda + beta/4)/(6 H^2) equals
    Laverda-Rubio Eq. (3.6) exactly (maximum at h^2 = mu_L^2 e^{-3/2})."""
    y, muL, H = 0.52, 3e10, 1e8
    b = 3 * y ** 4 / (16 * np.pi ** 2)
    h = muL * np.exp(np.linspace(-6, 1, 200001))
    lam = -b * np.log(h ** 2 / muL ** 2)
    assert np.isclose(S.xi_max_from_lambda(H, h, lam, -2 * b * np.ones_like(h)), S.xi_max_LR(H, muL, y), rtol=1e-6)


def test_barrier_numeric_vs_LR_order_of_magnitude():
    """Full running vs the constant-slope approximation: same order (the slope of lambda flattens toward its minimum, so the
    numerical xi_max is smaller by a factor ~1.5-2); and xi_max scales as H^-2."""
    mu, C = S.run(loops=3, Mt=172.57)
    muL, y = S.mu_Lambda(mu, C)
    r = S.xi_max_barrier(1e8, mu, C) / S.xi_max_LR(1e8, muL, y)
    assert 0.3 < r < 1.0
    assert np.isclose(S.xi_max_barrier(1e7, mu, C) / S.xi_max_barrier(1e8, mu, C), 100.0)


def test_alphas_dependence_slope():
    """Buttazzo Eq. (67): d log10 Lambda/d[(alpha_s - 0.1184)/0.0007] = +0.3; the scale of lambda = 0 follows within ~30%."""
    m1, _ = S.mu_Lambda(*S.run(loops=3, **{**BUTTAZZO, "alphas": 0.1177}))
    m2, _ = S.mu_Lambda(*S.run(loops=3, **{**BUTTAZZO, "alphas": 0.1191}))
    assert 0.2 < np.log10(m2 / m1) / 2 < 0.45
