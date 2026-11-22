"""Structure of the s-channel amplitude [eq. (2.2); appendix A.2, eqs. (A.17), (A.18)].

Checks of the statements of section 2 and appendix A:
  * traces of the matrix elements: tr T_phi = -(1-6 xi_phi) s, tr T_scalar = -[(1-6 xi_X) s + 2 M^2],
    tr T_Dirac = M ubar(k1) v(k2), tr T_Proca = -2 M^2 eps1*.eps2*;
  * spin decomposition for conserved sources: T_a P T_b = [T_a.T_b - tr T_a tr T_b/3] - tr T_a tr T_b/6,
    the spin-2 bracket being independent of the improvement terms (xi_phi, xi_X);
  * sign of eq. (2.2), M = -T_phi P T_X/(M_P^2 q^2) from L_int = -h T/M_P: attractive Newtonian limit.
Units: M_P = 1, momenta in arbitrary units (all statements are homogeneous in the momenta).
"""
import sys, pathlib
import numpy as np
import pytest
from gravdm import gravxsec as gx

ETA = gx.ETA


def tr(T):
    return np.trace(ETA @ T)


def dot(Ta, Tb):
    return np.sum((ETA @ Ta @ ETA) * Tb)


KIN = [(1.0, 0.0, 0.3), (1.0, 0.31, -0.6), (2.5, 0.7, 0.85)]      # (s, M, cos theta)


@pytest.mark.parametrize("s,M,c", KIN)
@pytest.mark.parametrize("xi", [0.0, 1 / 6, 0.4, -1.0])
def test_scalar_traces(s, M, c, xi):
    p1, p2, k1, k2 = gx.cm_kinematics(s, M, c)
    Tphi = gx.T_initial(p1, p2, xi)
    TX = gx.T_scalar(k1, k2, M * M, xi, +1, +1)
    # matrix elements carry an overall phase -1 relative to the paper's expressions (i*k per derivative)
    assert np.isclose(-tr(Tphi), -(1 - 6 * xi) * s, atol=1e-12)
    assert np.isclose(-tr(TX), -((1 - 6 * xi) * s + 2 * M * M), atol=1e-12)


@pytest.mark.parametrize("s,M,c", KIN[1:])
def test_dirac_trace(s, M, c):
    _, _, k1, k2 = gx.cm_kinematics(s, M, c)
    for a in range(2):
        for b in range(2):
            T = gx.T_dirac_pair_out(k1, k2, M, a, b)
            ubv = gx.bar(gx.u_spinors(k1, M)[a]) @ gx.v_spinors(k2, M)[b]
            assert np.isclose(tr(T), M * ubv, atol=1e-12)


@pytest.mark.parametrize("s,M,c", KIN[1:])
def test_proca_trace(s, M, c):
    _, _, k1, k2 = gx.cm_kinematics(s, M, c)
    for ea in gx.pol_vectors(k1, M):
        for eb in gx.pol_vectors(k2, M):
            T = gx.T_vector_me(ea, k1, +1, eb, k2, +1, M * M)
            assert np.isclose(tr(T), -2 * M * M * gx.mdot(ea.conj(), eb.conj()), atol=1e-12)


@pytest.mark.parametrize("s,M,c", KIN)
@pytest.mark.parametrize("xi_phi,xi_X", [(0.0, 0.0), (1 / 6, 0.0), (0.4, -1.0), (1.0, 1 / 6)])
def test_spin_decomposition_scalar(s, M, c, xi_phi, xi_X):
    p1, p2, k1, k2 = gx.cm_kinematics(s, M, c)
    Ta = gx.T_initial(p1, p2, xi_phi)
    Tb = gx.T_scalar(k1, k2, M * M, xi_X, +1, +1)
    spin2 = dot(Ta, Tb) - tr(Ta) * tr(Tb) / 3
    spin0 = -tr(Ta) * tr(Tb) / 6
    assert np.isclose(gx.contract(Ta, Tb), spin2 + spin0, atol=1e-12)
    # spin-2 part independent of both improvement terms
    Ta0 = gx.T_initial(p1, p2, 0.0); Tb0 = gx.T_scalar(k1, k2, M * M, 0.0, +1, +1)
    assert np.isclose(spin2, dot(Ta0, Tb0) - tr(Ta0) * tr(Tb0) / 3, atol=1e-12)


@pytest.mark.parametrize("xi_phi", [0.0, 1 / 6, 0.4])
def test_spin_decomposition_dirac_proca(xi_phi):
    s, M, c = 1.0, 0.31, -0.6
    p1, p2, k1, k2 = gx.cm_kinematics(s, M, c)
    Ta = gx.T_initial(p1, p2, xi_phi); Ta0 = gx.T_initial(p1, p2, 0.0)
    outs = [gx.T_dirac_pair_out(k1, k2, M, a, b) for a in range(2) for b in range(2)]
    outs += [gx.T_vector_me(ea, k1, +1, eb, k2, +1, M * M) for ea in gx.pol_vectors(k1, M) for eb in gx.pol_vectors(k2, M)]
    for Tb in outs:
        spin2 = dot(Ta, Tb) - tr(Ta) * tr(Tb) / 3
        assert np.isclose(gx.contract(Ta, Tb), spin2 - tr(Ta) * tr(Tb) / 6, atol=1e-12)
        assert np.isclose(spin2, dot(Ta0, Tb) - tr(Ta0) * tr(Tb) / 3, atol=1e-12)


def test_newton_limit_sign():
    """M = -T1 P T2/(M_P^2 t) (eq. (2.2) with L_int = -h T/M_P) gives M_nr = +m1 m2/(2 q^2), i.e. V = -G m1 m2/r (attractive)."""
    m1, m2 = 3.0, 5.0
    q3 = np.array([1e-3, 0, 0])
    on = lambda m, p: np.concatenate([[np.sqrt(m * m + p @ p)], p])
    k1, k1p = on(m1, np.zeros(3)), on(m1, q3)
    k2, k2p = on(m2, np.zeros(3)), on(m2, -q3)
    T1 = gx.T_scalar(k1p, k1, m1 * m1, 0.0, +1, -1)
    T2 = gx.T_scalar(k2p, k2, m2 * m2, 0.0, +1, -1)
    t = gx.mdot(k1p - k1, k1p - k1)
    Mnr = (-gx.contract(T1, T2) / t).real / (4 * m1 * m2)
    assert np.isclose(Mnr, m1 * m2 / (2 * (q3 @ q3)), rtol=1e-5)


def _outs(spin, k1, k2, M, xi_X):
    if spin == 0:
        return [gx.T_scalar(k1, k2, M * M, xi_X, +1, +1)]
    if spin == 0.5:
        return [gx.T_dirac_pair_out(k1, k2, M, a, b) for a in range(2) for b in range(2)]
    return [gx.T_vector_me(ea, k1, +1, eb, k2, +1, M * M) for ea in gx.pol_vectors(k1, M) for eb in gx.pol_vectors(k2, M)]


@pytest.mark.parametrize("spin,xi_X", [(0, 0.0), (0, -1.0), (0.5, 0.0), (1, 0.0)])
@pytest.mark.parametrize("r,xi_phi", [(0.0, 0.0), (0.1, 0.4), (0.2, 1 / 6)])
def test_no_spin2_spin0_interference(spin, xi_X, r, xi_phi):
    """[Appendix A.2] After the angular integration the spin-2 and spin-0 parts of eq. (2.2) do not interfere (J = 2 vs J = 0),
    so c = c_spin2 + c_spin0 (the basis of eq. (2.4): 1/6 + 5/6 (1-6 xi_phi)^2 (1-6 xi_X)^2 for the massless scalar)."""
    s = 1.0; M = np.sqrt(max(r, 1e-8 if spin == 1 else 0.0) * s)     # Proca polarisations need M > 0
    x, w = np.polynomial.legendre.leggauss(24)
    full = s2 = s0 = 0.0
    for xi_, wi in zip(x, w):
        p1, p2, k1, k2 = gx.cm_kinematics(s, M, xi_)
        Ta = gx.T_initial(p1, p2, xi_phi)
        for Tb in _outs(spin, k1, k2, M, xi_X):
            a2 = dot(Ta, Tb) - tr(Ta) * tr(Tb) / 3
            a0 = -tr(Ta) * tr(Tb) / 6
            full += wi * abs(a2 + a0) ** 2; s2 += wi * abs(a2) ** 2; s0 += wi * abs(a0) ** 2
    assert np.isclose(full, s2 + s0, rtol=1e-10)
    if r == 0.0 and spin == 0:
        assert np.isclose(s0 / full, 5 * (1 - 6 * xi_phi) ** 2 * (1 - 6 * xi_X) ** 2 / (1 + 5 * (1 - 6 * xi_phi) ** 2 * (1 - 6 * xi_X) ** 2))


# --- section 2: curvature coupling enters only through the traces; size of the Dirac spin-0 amplitude ---

@pytest.mark.parametrize("s,M,c", KIN)
@pytest.mark.parametrize("spin", [0, 0.5, 1])
def test_improvement_term_enters_only_through_trace(s, M, c, spin):
    """theta_{mu nu} T_X^{mu nu} = tr T_X for conserved T_X (theta = eta - q q/q^2), so the improvement term 2 xi q^2 theta of T_phi
    changes T_phi P T_X by -xi s tr T_X only, i.e. -(1/6) (Delta tr T_phi) tr T_X: the spin-2 bracket is untouched."""
    if spin == 1 and M == 0.0:
        pytest.skip("Proca needs M > 0")
    p1, p2, k1, k2 = gx.cm_kinematics(s, M, c)
    q = p1 + p2
    theta = ETA - np.outer(q, q) / gx.mdot(q, q)          # upper indices
    for Tb in _outs(spin, k1, k2, M, 0.3):
        assert np.isclose(dot(theta, Tb), tr(Tb), atol=1e-12)
        for xi in (0.4, -1.0):
            dA = gx.contract(gx.T_initial(p1, p2, xi), Tb) - gx.contract(gx.T_initial(p1, p2, 0.0), Tb)
            dtr = tr(gx.T_initial(p1, p2, xi)) - tr(gx.T_initial(p1, p2, 0.0))
            assert np.isclose(abs(dtr), 6 * abs(xi) * s, atol=1e-12)
            assert np.isclose(dA, -dtr * tr(Tb) / 6, atol=1e-12)


@pytest.mark.parametrize("r", [1e-6, 1e-3, 0.05, 0.2])
def test_dirac_trace_size(r):
    """sum over spins |T_Dirac|^2 = M^2 Tr[(k1+M)(k2-M)] = 2 M^2 s (1-4r): the spin-0 amplitude of the Dirac channel is O(sqrt r)
    relative to the spin-2 one and vanishes for M -> 0."""
    s = 1.0; M = np.sqrt(r * s)
    _, _, k1, k2 = gx.cm_kinematics(s, M, 0.37)
    tot = sum(abs(tr(T)) ** 2 for T in _outs(0.5, k1, k2, M, 0.0))
    assert np.isclose(tot, 2 * M * M * s * (1 - 4 * r), rtol=1e-10)
