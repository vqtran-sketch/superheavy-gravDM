"""Tests for infprod_spins.py (end-of-inflation production, appendix D.4)."""
import sys, pathlib, math
import numpy as np
from gravdm import infprod as ip
from gravdm import infprod_spins as sp


def test_analytic_background_matches_infprod():
    """Same CSZ background as the scalar code (CSZ eqs. (3.5)-(3.9)): H and a''/a agree with infprod.Background."""
    for y0 in (0.1, 1.0):
        b = ip.Background(y0=y0, y_in=-12, y_out=3.0, n=4001); a = sp.AnalyticBackground(y0)
        for j in range(0, len(b.y), 97):
            assert math.isclose(a.H(b.y[j]), b.H[j], rel_tol=1e-12)
            assert math.isclose(a.U(b.y[j]), b.U[j], rel_tol=1e-10, abs_tol=1e-14)
        assert math.isclose(a.late_C(), 1.0, rel_tol=1e-6)


def test_scalar_mode_reproduces_infprod_beta2():
    """mode 'S' (zeroth-order out-state) reproduces infprod.beta2; first-order out-state is y_out independent."""
    b = ip.Background(y0=0.1, y_out=3.0); a = sp.AnalyticBackground(0.1)
    k, M = 0.3, 1.0
    ref = ip.beta2(b, k, M, y_start=max(min(np.log(k) - 7, -5.0), -12.0), y_end=3.0)
    new, raw = sp.beta2_scalar_type(a, k, M, "S", y_outs=(2.0, 3.0), return_raw=True)
    assert math.isclose(raw[-1], ref, rel_tol=2e-4)
    assert math.isclose(new[0], new[1], rel_tol=1e-5)          # zeroth order differs by 1e-4


def test_massless_dirac_and_transverse_not_produced():
    """Conformal invariance at M = 0: no production of Dirac fermions nor of transverse vectors (eqs. S2, S7)."""
    for y0 in (0.1, 1.0):
        a = sp.AnalyticBackground(y0)
        assert sp.beta2_dirac(a, 1.0, 0.0, y_outs=(2.0,))[0] < 1e-18
        assert sp.beta2_scalar_type(a, 1.0, 0.0, "T", y_outs=(2.0,))[0] < 1e-18


def test_dirac_born_approximation():
    """M << k: |beta|^2 = (M/2k)^2 |int a'' e^{-2ik eta} d eta|^2/(4k^2) (eq. S13), numeric vs Born < 1%; and ∝ M^2."""
    a = sp.AnalyticBackground(0.1)
    k = 1.0
    b1 = sp.beta2_dirac(a, k, 1e-3, y_outs=(2.5,))[0]
    b2 = sp.beta2_dirac(a, k, 2e-3, y_outs=(2.5,))[0]
    assert abs(b1 / sp.born_dirac(a, k, 1e-3) - 1) < 1e-2
    assert abs(b2 / b1 - 4) < 4e-2


def test_dirac_pauli_and_normalisation():
    """0 <= |beta|^2 <= 1 and the first-order out-state agrees with zeroth order up to the adiabatic floor."""
    a = sp.AnalyticBackground(0.1)
    new, raw = sp.beta2_dirac(a, 2.0, 3.0, y_outs=(2.0, 2.5), return_raw=True)
    assert np.all((new >= 0) & (new <= 1))
    assert abs(new[0] / new[1] - 1) < 1e-5
    assert abs(raw[1] / new[1] - 1) < 1e-2


def test_longitudinal_goldstone_limit():
    """k >> aM: longitudinal Proca = minimally coupled scalar of the same mass (eq. S10 -> a''/a)."""
    for y0 in (0.1, 1.0):
        a = sp.AnalyticBackground(y0)
        for k in (1.0, 3.0):
            L = sp.beta2_scalar_type(a, k, 1e-2, "L", y_outs=(2.0,))[0]
            S = sp.beta2_scalar_type(a, k, 1e-2, "S", y_outs=(2.0,))[0]
            assert abs(L / S - 1) < 1e-3


def test_longitudinal_ff_formula():
    """f''/f of eq. S10 vs finite differences of f(eta) = aM/sqrt(k^2+a^2M^2) in conformal time."""
    a = sp.AnalyticBackground(1.0); k, M = 0.7, 1.3
    om2 = sp.omega2_fn(a, k, M, "L")
    f = lambda y: math.exp(y) * M / math.sqrt(k * k + (math.exp(y) * M) ** 2)
    for y in (-0.8, 0.0, 0.6):
        h = 1e-4; cH = a.calH
        fp = lambda yy: cH(yy) * (f(yy + h) - f(yy - h)) / (2 * h)           # df/deta
        fpp = cH(y) * (fp(y + h) - fp(y - h)) / (2 * h)
        ff_num = fpp / f(y)
        ff_code = k * k + (math.exp(y) * M) ** 2 - om2(y)
        assert math.isclose(ff_num, ff_code, rel_tol=1e-5, abs_tol=1e-8)


def test_longitudinal_heavy_limit_equals_transverse():
    """k << aM at the transition and M >> H0 (no de Sitter-era production when aM crosses k): f -> 1, L -> T.
    (At M = 3 H0 the L/T ratio is 1.18 at k = 0.01: genuine inflationary-era longitudinal production, not a numerical artefact.)"""
    a = sp.AnalyticBackground(0.1)
    L = sp.beta2_scalar_type(a, 0.05, 8.0, "L", y_outs=(2.0,))[0]
    T = sp.beta2_scalar_type(a, 0.05, 8.0, "T", y_outs=(2.0,))[0]
    assert abs(L / T - 1) < 1e-2


def test_number_density_prefactors(monkeypatch):
    """Counting conventions (eqs. S6, S12) exercised through the production functions ; with |beta_k|^2 = 1 on k in [0, 1], na3_dirac (X + Xbar, 2 helicities) = 2/(3 pi^2), particles only
    = 1/(3 pi^2); na3_proca: n_T (2 transverse) = 1/(3 pi^2), n_L = 1/(6 pi^2)."""
    from gravdm import infprod_spins as ips
    monkeypatch.setattr(ips, "beta2_dirac", lambda bg, k, M, y_outs=(3.0,), **kw: np.ones(len(y_outs)))
    monkeypatch.setattr(ips, "beta2_scalar_type", lambda bg, k, M, mode, y_outs=(3.0,), **kw: np.ones(len(y_outs)))
    ks = np.linspace(0, 1, 2001)
    assert math.isclose(ips.na3_dirac(None, 1.0, ks)[0][0], 2 / (3 * np.pi ** 2), rel_tol=1e-6)
    assert math.isclose(ips.na3_dirac(None, 1.0, ks, total=False)[0][0], 1 / (3 * np.pi ** 2), rel_tol=1e-6)
    nT, nL, _, _ = ips.na3_proca(None, 1.0, ks)
    assert math.isclose(nT[0], 1 / (3 * np.pi ** 2), rel_tol=1e-6) and math.isclose(nL[0], 1 / (6 * np.pi ** 2), rel_tol=1e-6)


def test_first_order_instate_removes_floor():
    """First-order adiabatic in-state: |beta|^2 independent of the start time where the zeroth-order start had a floor
    (y0 = 1, M = 6, k = 3: zeroth order from ln k - 7 gave 2.9x the converged value)."""
    a = sp.AnalyticBackground(1.0); k, M = 3.0, 6.0
    b1 = sp.beta2_dirac(a, k, M, y_outs=(2.0,))[0]
    b2 = sp.beta2_dirac(a, k, M, y_outs=(2.0,), y_start=math.log(k) - 10)[0]
    b0 = sp.beta2_dirac(a, k, M, y_outs=(2.0,), y_start=math.log(k) - 7, in_order=0)[0]
    assert abs(b1 / b2 - 1) < 1e-2
    assert b0 / b1 > 2
