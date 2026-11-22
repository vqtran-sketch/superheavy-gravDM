"""Tests for bounce.py: O(4) bounce of the tilted quartic toy model (virial, thin-wall limit, spinodal limit, convergence)."""
import sys, pathlib
import numpy as np
from scipy import integrate
from gravdm import bounce as B


def test_virial_two_routes():
    """[DERIVED] Derrick scaling in d = 4: S = T + V and 2T + 4V = 0 => S = T/2. Both routes agree to < 0.5%."""
    for b in (0.2, 0.25, 0.5, 1.0):
        r = B.bounce(b)
        assert np.isclose(r["S_hat"], r["S_hat_virial"], rtol=5e-3)


def test_thin_wall_limit():
    """[LIT] Coleman (1977) Eq. (4.20): S -> 27 pi^2 sigma^4/(2 eps^3), R -> 3 sigma/eps as eps -> 0 (b = 3: eps = 1/12)."""
    r = B.bounce(3.0)
    assert np.isclose(r["S_hat"] / B.S_hat_thin(3.0), 1.0, atol=0.01)
    assert np.isclose(r["R"] / (3 * (2 * np.sqrt(2) / 3) / r["eps"]), 1.0, atol=0.02)
    assert B.bounce(1.0)["S_hat"] / B.S_hat_thin(1.0) < B.bounce(3.0)["S_hat"] / B.S_hat_thin(3.0)   # approach from below


def test_spinodal():
    """Cubic tilt: at b = 3/16 (U''(-1) = 2 - 3 eps/2 = 0) the false minimum and the barrier top merge; for b < 3/16 they swap
    (false minimum at x = -3 eps/4 < -1, top at -1). The barrier above the false vacuum and S_hat -> 0 from both sides."""
    for b in (0.18, 0.19):
        eps = 1 / (4 * b); xf, xt, xT = B.extrema(eps)
        exp_f, exp_t = (-1.0, -0.75 * eps) if b > 3 / 16 else (-0.75 * eps, -1.0)
        assert np.isclose(xf, exp_f, atol=1e-9) and np.isclose(xt, exp_t, atol=1e-9)
    assert B.bounce(0.188)["S_hat"] < B.bounce(0.2)["S_hat"] < B.bounce(0.25)["S_hat"]
    assert B.bounce(0.187)["S_hat"] < B.bounce(0.18)["S_hat"] < B.bounce(0.15)["S_hat"]


def _cubic_bounce():
    """Independent bounce of V = y^2/2 - y^3/3 (false vacuum y = 0), simple undershoot/overshoot in y(0)."""
    def shoot(y0):
        f = lambda r, y: [y[1], y[0] - y[0] ** 2 - 3 * y[1] / r]
        over = lambda r, y: y[0]; over.terminal = True; over.direction = -1
        und = lambda r, y: y[1]; und.terminal = True; und.direction = 1
        r0 = 1e-6
        s = integrate.solve_ivp(f, (r0, 60), [y0, 0.0], events=[over, und], rtol=1e-12, atol=1e-14, dense_output=True)
        return (1 if s.t_events[0].size else -1), s
    lo, hi = 1.5, 30.0                         # y0 just above the top (y = 1): undershoot; large: overshoot
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if shoot(mid)[0] == 1: hi = mid
        else: lo = mid
    s = shoot(hi)[1]
    r = np.geomspace(1e-6, s.t[-1], 40000); y, v = s.sol(r)
    return 0.5 * np.pi ** 2 * integrate.trapezoid(r ** 3 * v ** 2, r)


def test_spinodal_limit_matches_cubic_bounce():
    """[DERIVED] Near b = 3/16 the toy potential around x = -1 is U ~ U''/2 d^2 + U'''/6 d^3, so S_hat -> S_cub U''(-1)/(U'''(-1)/2)^2
    with S_cub the bounce of y^2/2 - y^3/3 (independent solver). Agreement improves toward the spinodal (< 5% at b = 0.19)."""
    Sc = _cubic_bounce()
    b = 0.19; eps = 1 / (4 * b)
    m2 = 2 - 1.5 * eps; d = abs(-6 + 1.5 * eps) / 2
    assert np.isclose(B.bounce(b)["S_hat"], Sc * m2 / d ** 2, rtol=0.05)


def test_convergence():
    """Doubling the integration range changes S_hat by < 1e-3."""
    a = B.bounce(0.3)["S_hat"]; c = B.bounce(0.3, rho_max=2 * (40 + 12 * 4 * 0.3))["S_hat"]
    assert np.isclose(a, c, rtol=1e-3)
