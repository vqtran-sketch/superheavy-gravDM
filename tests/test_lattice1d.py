"""Tests for lattice1d.py (section 3.4, appendix E). Kept fast (coarse grids)."""
import sys, pathlib
import numpy as np
from gravdm import lattice1d as L


def test_kink_is_static_solution():
    """phi = v tanh(m x/2) solves phi'' = V'(phi) for V = (lam/4)(phi^2 - v^2)^2, m = sqrt(2 lam) v = 1 (section 3.4)."""
    x = np.linspace(-20, 20, 4001); dx = x[1] - x[0]
    phi, _ = L.kink(x, 0.0, 0.0, 0.0, +1)
    res = L.lap4(phi, dx) - L.dV(phi)
    assert np.max(np.abs(res[10:-10])) < 1e-8


def test_boosted_kink_energy_is_gamma_sigma():
    """E/A of a boosted tanh wall = gamma * sigma, sigma = (2/3) m v^2 (Dphi = 2v)."""
    g = 5.0; u = np.sqrt(1 - 1 / g ** 2)
    x = np.linspace(-10, 10, 200001); dx = x[1] - x[0]
    phi, pd = L.kink(x, 0.0, 0.0, u, +1)
    # a single kink is not periodic: drop the edge points where the periodic stencil sees the -v/+v jump
    E = L.stress(phi, pd, dx)["T00"][10:-10].sum() * dx
    assert np.isclose(E, g * 2 / 3, rtol=1e-6)


def test_parton_tensor_conserved_and_invariants():
    """k_mu T~^{mu nu} = 0 for the parton cross term; closed forms I2 = (2/3)|FF|^2, I0 = 4|FF|^2 (ratio 6) (section 3.4)."""
    a, b, g = np.array([3.0, 7.0]), np.array([5.0, 2.0]), 20.0
    T = L.parton_tensor(a, b, g)
    for i in range(2):
        k = np.array([a[i] + b[i], a[i] - b[i], 0, 0])
        assert np.allclose((L.ETA4 @ k) @ T[i], 0, atol=1e-12)
    I2, I0 = L.invariants(T)
    assert np.allclose(I0 / I2, 6.0, rtol=1e-12)


def test_free_passage_pipeline_quick():
    """Windowed FT of the analytic free-passage field reproduces the parton cross term at hard (a,b) (coarse grid, 5%)."""
    r = L.run(10.0, targets=[(8.0, 8.0)], free_passage=True, pts_per_wall=12, Tw=3.0)
    I2, I0 = L.invariants(r["T"]); P2, P0 = L.invariants(L.parton_tensor(r["a"], r["b"], 10.0))
    assert abs(I2[0] / P2[0] - 1) < 0.05 and abs(I0[0] / P0[0] - 1) < 0.1


def test_step_tilt_vacua_and_derivative():
    """tilt="step": V(-v) - V(+v) = eps, V'(+-v) = 0, V''(+-v) = m^2 = 1 for any eps; dV matches a numerical derivative."""
    for eps in (0.3, 0.667, 1.0):
        assert np.isclose(L.V(-1.0, eps=eps, tilt="step") - L.V(1.0, eps=eps, tilt="step"), eps, rtol=1e-12)
        for x0 in (-1.0, 1.0):
            assert abs(L.dV(x0, eps=eps, tilt="step")) < 1e-12
            h = 1e-4
            d2 = (L.dV(x0 + h, eps=eps, tilt="step") - L.dV(x0 - h, eps=eps, tilt="step")) / (2 * h)
            assert np.isclose(d2, 1.0, rtol=1e-3)   # S''' jumps at +-v: central difference has O(h) error
        x = np.linspace(-1.2, 1.2, 241); h = 1e-6
        num = (L.V(x + h, eps=eps, tilt="step") - L.V(x - h, eps=eps, tilt="step")) / (2 * h)
        assert np.allclose(num, L.dV(x, eps=eps, tilt="step"), atol=1e-8)


def test_linear_tilt_default_unchanged():
    """The default tilt is the linear one used by scripts 16 and 25."""
    x = np.linspace(-1.5, 1.5, 7)
    assert np.allclose(L.V(x, eps=0.2), 0.125 * (x * x - 1) ** 2 + 0.1 * (1 - x))
    assert np.allclose(L.dV(x, eps=0.2), 0.5 * x * (x * x - 1) - 0.1)


def test_step_tilt_bounded_outside_vacua():
    """The step is constant outside [-v, v]: V = quartic + eps (phi < -v) or quartic (phi > v); bounded below at large |phi|."""
    x = np.array([-3.0, -1.5, 1.5, 3.0])
    q = 0.125 * (x * x - 1) ** 2
    assert np.allclose(L.V(x, eps=1.0, tilt="step"), q + np.where(x < 0, 1.0, 0.0))
    assert np.allclose(L.dV(x, eps=1.0, tilt="step"), 0.5 * x * (x * x - 1))


def test_cubic_tilt_matches_bounce_potential_and_false_mass():
    """tilt="cubic" = lam v^4 * bounce.U(x, eps_b, "cubic") with eps = lam v^4 eps_b (lam = 1/2, v = 1); V''(-v) = m^2 (1 - DV/(sigma m))."""
    from gravdm import bounce
    x = np.linspace(-1.6, 1.6, 33)
    for eps_b in (0.4, 0.95, 1.24):
        assert np.allclose(L.V(x, eps=0.5 * eps_b, tilt="cubic"), 0.5 * bounce.U(x, eps_b, "cubic"), atol=1e-14)
        assert np.allclose(L.dV(x, eps=0.5 * eps_b, tilt="cubic"), 0.5 * bounce.dU(x, eps_b, "cubic"), atol=1e-14)
        eps = 0.5 * eps_b; h = 1e-4
        d2 = (L.dV(-1 + h, eps=eps, tilt="cubic") - L.dV(-1 - h, eps=eps, tilt="cubic")) / (2 * h)
        assert np.isclose(d2, 1.0 - eps / (2 / 3), rtol=1e-6)
        assert abs(L.dV(1.0, eps=eps, tilt="cubic")) < 1e-14 and abs(L.dV(-1.0, eps=eps, tilt="cubic")) < 1e-14
