"""Tests for gravxsec (section 2, appendix A). Normalisation, conservation, Lorentz invariance, Newtonian limit, limits."""
import sys, pathlib
import numpy as np
import pytest
from gravdm import gravxsec as gx

RNG = np.random.default_rng(1)


def onshell(m, p3):
    return np.concatenate([[np.sqrt(m * m + p3 @ p3)], p3])


def boost(v, b):
    """Boost 4-vector v by velocity vector b (|b|<1)."""
    bb = b @ b; g = 1 / np.sqrt(1 - bb)
    t = v[0]; x = v[1:]
    bx = b @ x
    return np.concatenate([[g * (t - bx)], x + ((g - 1) * bx / bb - g * t) * b])


# --- forward matrix element: <k|T|k> = 2 k^mu k^nu for every spin --------------------------------------------
@pytest.mark.parametrize("m", [0.0, 0.7])
def test_forward_scalar(m):
    k = onshell(m, RNG.normal(size=3))
    T = gx.T_scalar(k, k, m * m, 0.0, +1, -1)
    assert np.allclose(T, 2 * np.outer(k, k), atol=1e-12)
    # improvement term vanishes in the forward limit (q = 0)
    assert np.allclose(gx.T_scalar(k, k, m * m, 0.3, +1, -1), 2 * np.outer(k, k), atol=1e-12)


def test_forward_dirac():
    m = 0.6; k = onshell(m, RNG.normal(size=3))
    for s in range(2):
        u = gx.u_spinors(k, m)[s]; ub = gx.bar(u)
        G = np.array([ub @ gx.GAM[mu] @ u for mu in range(4)])
        T = 0.25 * (np.outer(G, 2 * k) + np.outer(2 * k, G))
        assert np.allclose(T, 2 * np.outer(k, k), atol=1e-12)


def test_forward_vector():
    m = 0.8; k = onshell(m, RNG.normal(size=3))
    for e in gx.pol_vectors(k, m):
        T = gx.T_vector_me(e, k, +1, e, k, -1, m * m)
        assert np.allclose(T, 2 * np.outer(k, k), atol=1e-10)


# --- spinor / polarisation sanity ----------------------------------------------------------------
def test_spinor_normalisation():
    m = 0.5; k = onshell(m, RNG.normal(size=3))
    for u in gx.u_spinors(k, m):
        assert np.isclose(gx.bar(u) @ u, 2 * m)
        assert np.allclose((gx.slash(k) - m * np.eye(4)) @ u, 0, atol=1e-12)
    for v in gx.v_spinors(k, m):
        assert np.isclose(gx.bar(v) @ v, -2 * m)
        assert np.allclose((gx.slash(k) + m * np.eye(4)) @ v, 0, atol=1e-12)


def test_polarisations():
    m = 0.9; k = onshell(m, RNG.normal(size=3))
    P = sum(np.outer(e, e.conj()) for e in gx.pol_vectors(k, m))
    target = -np.linalg.inv(gx.ETA) + np.outer(k, k) / m ** 2       # -g^{mn} + k^m k^n / M^2
    assert np.allclose(P, target, atol=1e-12)


# --- conservation q_mu T^{mu nu} = 0 for pair production ---------------------------------------------
def test_conservation_all_spins():
    s, M = 1.0, 0.3
    p1, p2, k1, k2 = gx.cm_kinematics(s, M, 0.37)
    q = k1 + k2; ql = gx.ETA @ q
    for xi in (0.0, 1 / 6, 0.4):
        assert np.allclose(ql @ gx.T_scalar(k1, k2, M * M, xi), 0, atol=1e-12)
    for a in range(2):
        for b in range(2):
            assert np.allclose(ql @ gx.T_dirac_pair_out(k1, k2, M, a, b), 0, atol=1e-12)
    for ea in gx.pol_vectors(k1, M):
        for eb in gx.pol_vectors(k2, M):
            assert np.allclose(ql @ gx.T_vector_me(ea, k1, +1, eb, k2, +1, M * M), 0, atol=1e-10)
    assert np.allclose((gx.ETA @ (p1 + p2)) @ gx.T_initial(p1, p2), 0, atol=1e-12)


# --- Lorentz invariance of spin-summed |M|^2 --------------------------------------------------------
@pytest.mark.parametrize("spin", [0, 0.5, 1])
def test_lorentz_invariance(spin):
    s, M = 1.0, 0.27
    vecs = gx.cm_kinematics(s, M, -0.41, 1.1)
    a0 = gx.amp2_sum(spin, *vecs, M, xi=0.2)
    b = np.array([0.3, -0.5, 0.6])
    a1 = gx.amp2_sum(spin, *[boost(v, b) for v in vecs], M, xi=0.2)
    assert np.isclose(a0, a1, rtol=1e-9)


# --- Newtonian limit: t-channel graviton between two heavy scalars gives V = -G m1 m2 / r --------------
def test_newton_limit():
    m1, m2 = 3.0, 5.0
    q3 = np.array([1e-3, 0, 0])
    k1 = onshell(m1, np.zeros(3)); k1p = onshell(m1, q3)
    k2 = onshell(m2, np.zeros(3)); k2p = onshell(m2, -q3)
    T1 = gx.T_scalar(k1p, k1, m1 * m1, 0.0, +1, -1)
    T2 = gx.T_scalar(k2p, k2, m2 * m2, 0.0, +1, -1)
    t = gx.mdot(k1p - k1, k1p - k1)
    Mrel = gx.contract(T1, T2) / t                     # in units M_P = 1
    Mnr = Mrel / (2 * m1 * 2 * m2)
    # Born: M_nr = -V~(q), V~ = -4 pi G m1 m2 / q^2 with G = 1/(8 pi M_P^2)  => M_nr = m1 m2 /(2 q^2)
    assert np.isclose(abs(Mnr), m1 * m2 / (2 * (q3 @ q3)), rtol=1e-5)


# --- known limits of the cross section ---------------------------------------------------------------
def test_massless_minimal_scalar_amplitude_tu_over_s():
    s = 1.0
    for c in (-0.7, 0.1, 0.55):
        p1, p2, k1, k2 = gx.cm_kinematics(s, 0.0, c)
        t = gx.mdot(p1 - k1, p1 - k1); u = gx.mdot(p1 - k2, p1 - k2)
        assert np.isclose(np.sqrt(gx.amp2_sum(0, p1, p2, k1, k2, 0.0)), abs(t * u / s), rtol=1e-10)


def test_c_massless_scalar():
    assert np.isclose(gx.c_hat(0, 0.0), 1 / (960 * np.pi), rtol=1e-8)


@pytest.mark.parametrize("spin,xi,n", [(0, 0.0, 1), (0, 1 / 6, 1), (0.5, 0.0, 3), (1, 0.0, 1)])
def test_threshold_power(spin, xi, n):
    """c ~ beta^n near threshold: scalar & vector s-wave (trace / longitudinal coupling survives), Dirac p-wave."""
    b1, b2 = 1e-2, 2e-2
    c1 = gx.c_hat(spin, (1 - b1 ** 2) / 4, xi); c2 = gx.c_hat(spin, (1 - b2 ** 2) / 4, xi)
    assert abs(np.log(c2 / c1) / np.log(2) - n) < 0.01


def test_massless_ratios_conformal_anomaly():
    """[LIT] spin-2 (traceless) parts ~ conformal-anomaly c: real conformal scalar : Dirac : Maxwell = 1 : 6 : 12.
    Units of 1/(960 pi): conformal scalar 1/6, Dirac 1, vector transverse 2 (+ longitudinal 1 = minimal scalar)."""
    u = 960 * np.pi
    assert np.isclose(gx.c_hat(0, 0.0, 1 / 6) * u, 1 / 6, rtol=1e-8)
    assert np.isclose(gx.c_hat(0.5, 0.0) * u, 1.0, rtol=1e-8)
    assert np.isclose(gx.c_hat(1, 1e-8) * u, 3.0, rtol=1e-6)


# --- closed form (route B) vs explicit spinors (route A) ---------------------------------------------
from gravdm import xsec_analytic as xa


@pytest.mark.parametrize("case", list(xa.CASES))
@pytest.mark.parametrize("r", [1e-3, 0.08, 0.2, 0.2495])
def test_closed_form_vs_explicit(case, r):
    spin, xi = xa.CASES[case]
    assert np.isclose(xa.c_of_r(case, r), gx.c_hat(spin, r, xi), rtol=1e-12)


@pytest.mark.parametrize("xi", [0.25, 0.5])
@pytest.mark.parametrize("r", [1e-3, 0.08, 0.2, 0.2495])
def test_closed_form_general_xi(xi, r):
    """General-xi closed form (route B) vs explicit route A; xi = 1/4 has a vanishing s-wave threshold coefficient."""
    b = np.sqrt(1 - 4 * r)
    closed = b * ((1 - 10 * xi + 30 * xi ** 2) + 2 * r * (1 - 10 * xi) + 6 * r ** 2) / (960 * np.pi)
    assert np.isclose(closed, gx.c_hat(0, r, xi), rtol=1e-9)


@pytest.mark.parametrize("case,cJ,frac", [("conformal", 4 / 3, 1.0), ("dirac", 8.0, 1.0), ("vector", 16.0, 2 / 3)])
def test_redi_tesi_master_formula(case, cJ, frac):
    """[LIT: Redi-Tesi 2606.28297 eqs. (3),(38),(53)] on a 2-particle massless initial state the CFT production
    gives sigma = c_J s/(7680 pi M_P^4), c_0 = 4/3 (conformal real scalar), Weyl 4 (Dirac = 8), vector 16.
    For the Proca vector compare the transverse part (2/3 of the massless total; longitudinal = minimal scalar)."""
    assert np.isclose(cJ / (7680 * np.pi), frac * xa.c0(case), rtol=1e-10)


@pytest.mark.parametrize("xi_phi", [0.0, 1 / 6, 0.5, -1.0])
def test_wall_curvature_coupling(xi_phi):
    """Massless coefficients with wall-field coupling xi_phi (eq. (2.4), table 1); explicit route vs closed form."""
    u = 960 * np.pi
    assert np.isclose(gx.c_hat(0, 0.0, 0.0, xi_phi=xi_phi) * u, xa.c0_general("scalar", xi_phi), rtol=1e-8)
    assert np.isclose(gx.c_hat(0, 0.0, 1 / 6, xi_phi=xi_phi) * u, xa.c0_general("conformal", xi_phi), rtol=1e-8)
    assert np.isclose(gx.c_hat(0.5, 1e-10, xi_phi=xi_phi) * u, 1.0, rtol=1e-6)
    assert np.isclose(gx.c_hat(1, 1e-10, xi_phi=xi_phi) * u, xa.c0_general("vector", xi_phi), rtol=1e-6)


@pytest.mark.parametrize("xi_phi", [0.0, 1 / 6, 0.4, 1.0])
@pytest.mark.parametrize("r", [0.03, 0.12, 0.22])
def test_dirac_general_xi_phi(xi_phi, r):
    """[section 2, appendix A.2] c_Dirac(r; xi_phi) = v^3 [1 + 8r/3 + (10/3)(1-6 xi_phi)^2 r]/(960 pi):
    spin-2 part v^3(1 + 8r/3) (xi_phi-independent), trace part prop. to r (Dirac trace ~ M psibar psi). Explicit spinors vs closed form."""
    v = np.sqrt(1 - 4 * r)
    closed = v ** 3 * (1 + 8 * r / 3 + 10 / 3 * (1 - 6 * xi_phi) ** 2 * r) / (960 * np.pi)
    assert np.isclose(gx.c_hat(0.5, r, xi_phi=xi_phi), closed, rtol=1e-8)
