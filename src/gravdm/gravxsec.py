"""Tree-level s-channel graviton exchange  s s -> X X  (route A: explicit spinors / polarisation vectors).

Conventions [section 2 and appendix A of the paper]:
  metric (+,-,-,-); g = eta + kappa h, kappa = 2/M_P (M_P reduced Planck mass); L_int = -(kappa/2) h_{mu nu} T^{mu nu};
  de Donder propagator i P/(q^2), P_{mu nu rho sigma} = (eta_mr eta_ns + eta_ms eta_nr - eta_mn eta_rs)/2.
  => amplitude  M = T_in^{mu nu} P_{mu nu rho sigma} T_out^{rho sigma} / (M_P^2 q^2)      (cf. eq. (2.2))
Matrix elements of T^{mu nu} are normalised so that <k|T^{mu nu}|k> = 2 k^mu k^nu (relativistic states)
for every spin; tested in tests/test_gravxsec.py, together with the Newtonian limit.

All momenta in units where M_P = 1 unless stated; sigma returned as sigma*M_P^4 (dimension GeV^-2 * GeV^4 -> GeV^2 ... see
sigma_hat: we return the dimensionless c = sigma * M_P^4 / s).
"""
import numpy as np

ETA = np.diag([1.0, -1.0, -1.0, -1.0])

# ---------------------------------------------------------------- Dirac algebra (Dirac representation)
_I2 = np.eye(2); _Z2 = np.zeros((2, 2))
_SIG = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]], complex), np.array([[1, 0], [0, -1]], complex)]
GAM = [np.block([[_I2, _Z2], [_Z2, -_I2]]).astype(complex)] + [np.block([[_Z2, s], [-s, _Z2]]) for s in _SIG]


def slash(p):
    """p-slash = gamma^mu p_mu  (p given with upper index)."""
    pl = ETA @ p
    return sum(GAM[m] * pl[m] for m in range(4))


def _helicity_2spinors(p3):
    """Two orthonormal 2-spinors (eigenstates of sigma.n, n = p3/|p3|; arbitrary basis if p3 = 0)."""
    n = np.linalg.norm(p3)
    if n < 1e-14:
        return [np.array([1, 0], complex), np.array([0, 1], complex)]
    nx, ny, nz = p3 / n
    th = np.arccos(np.clip(nz, -1, 1)); ph = np.arctan2(ny, nx)
    up = np.array([np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)])
    dn = np.array([-np.exp(-1j * ph) * np.sin(th / 2), np.cos(th / 2)])
    return [up, dn]


def u_spinors(p, m):
    """Dirac u(p,s), normalised ubar u = 2m (and u^dagger u = 2E)."""
    E = p[0]; p3 = p[1:]
    out = []
    for chi in _helicity_2spinors(p3):
        sp = sum(_SIG[i] * p3[i] for i in range(3))
        out.append(np.concatenate([np.sqrt(E + m) * chi, (sp @ chi) / np.sqrt(E + m)]))
    return out


def v_spinors(p, m):
    """Dirac v(p,s) = C ubar^T-type construction via the standard formula, normalised vbar v = -2m."""
    E = p[0]; p3 = p[1:]
    out = []
    for eta_ in _helicity_2spinors(p3):
        sp = sum(_SIG[i] * p3[i] for i in range(3))
        out.append(np.concatenate([(sp @ eta_) / np.sqrt(E + m), np.sqrt(E + m) * eta_]))
    return out


def bar(sp):
    return sp.conj() @ GAM[0]


def pol_vectors(k, M):
    """Three polarisation vectors (upper index) of a massive vector with momentum k; eps.k = 0, eps.eps* = -1."""
    E = k[0]; k3 = k[1:]; kk = np.linalg.norm(k3)
    n = k3 / kk if kk > 1e-14 else np.array([0, 0, 1.0])
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    t1 = np.concatenate([[0], e1]).astype(complex); t2 = np.concatenate([[0], e2]).astype(complex)
    L = np.concatenate([[kk / M], E / M * n]).astype(complex)
    return [t1, t2, L]


def lower(v):
    return ETA @ v


def mdot(a, b):
    return a @ ETA @ b


# ---------------------------------------------------------------- stress-tensor matrix elements (appendix A.2)
def T_scalar(ka, kb, M2, xi, sa=+1, sb=+1):
    """<..|T^{mu nu}|..> for a real scalar with external momenta ka, kb.
    s = +1: outgoing (wavefunction e^{+ikx}), s = -1: incoming (e^{-ikx}).
    T = d^m X d^n X - eta(1/2 (dX)^2 - 1/2 M^2 X^2) + xi (eta box - d^m d^n) X^2.
    Each field derivative brings i*s*k. Two contractions -> symmetrisation."""
    A = 1j * sa * ka; B = 1j * sb * kb
    q = A + B                                   # = i (sa ka + sb kb)
    AB = mdot(A, B)
    T = np.outer(A, B) + np.outer(B, A) - np.linalg.inv(ETA) * (AB - M2)
    # improvement: X^2 -> 2 e^{...}; d^m d^n -> q^m q^n ; box -> q.q
    T += 2 * xi * (np.linalg.inv(ETA) * mdot(q, q) - np.outer(q, q))
    return T


def T_dirac_pair_out(k1, k2, M, s1, s2):
    """<f(k1,s1) fbar(k2,s2)| T^{mu nu} |0> = (1/4) ubar(k1) [g^m (k1-k2)^n + g^n (k1-k2)^m] v(k2)."""
    ub = bar(u_spinors(k1, M)[s1]); v = v_spinors(k2, M)[s2]
    d = k1 - k2
    G = np.array([ub @ GAM[m] @ v for m in range(4)])     # ubar gamma^m v (upper index)
    return 0.25 * (np.outer(G, d) + np.outer(d, G))


def T_vector(A, ka, B, kb, M2):
    """Bilinear T^{mu nu}[A,B] for Proca field, with A = (eps_a e^{i ka x}) style plane waves given as
    (polarisation, momentum*i*sign) pairs; symmetrised over the two contractions by the caller.
    T = F^{m a} F_a^{ n} + 1/4 eta F_{ab}F^{ab} + M^2 (A^m A^n - 1/2 eta A.A)   [then symmetrise A<->B]."""
    Fa = np.outer(ka, A) - np.outer(A, ka)     # F^{mu nu} = d^mu A^nu - d^nu A^mu with d -> ka (already i*s*k)
    Fb = np.outer(kb, B) - np.outer(B, kb)
    Fa_l = ETA @ Fa @ ETA; Fb_l = ETA @ Fb @ ETA
    etainv = np.linalg.inv(ETA)
    FF = np.sum(Fa * Fb_l)                     # F_a^{ab} F_b{ab}
    T = Fa @ ETA @ Fb.T                        # F_a^{m a} eta_{a b} F_b^{n b} ... = F^{m a} F^{n}{}_{a}
    T = -T                                     # F^{m a} F_a^{ n} = -F^{m a} F^{n}_a
    T = T + 0.25 * etainv * FF + M2 * (np.outer(A, B) - 0.5 * etainv * mdot(A, B))
    return T


def T_vector_me(epsa, ka, sa, epsb, kb, sb, M2):
    """Matrix element with both contractions: wavefunctions eps* e^{+ikx} (out) or eps e^{-ikx} (in)."""
    ea = epsa.conj() if sa > 0 else epsa
    eb = epsb.conj() if sb > 0 else epsb
    da = 1j * sa * ka; db = 1j * sb * kb
    return T_vector(ea, da, eb, db, M2) + T_vector(eb, db, ea, da, M2)


# ---------------------------------------------------------------- amplitudes
def contract(Ta, Tb):
    """Ta^{mn} P_{mnrs} Tb^{rs} = Ta_{mn} Tb^{mn} - 1/2 Ta^m_m Tb^r_r  (P symmetric part; Ta, Tb symmetric)."""
    Tal = ETA @ Ta @ ETA
    return np.sum(Tal * Tb) - 0.5 * np.trace(ETA @ Ta) * np.trace(ETA @ Tb)


def T_initial(p1, p2, xi_phi=0.0):
    """Incoming massless real scalar pair (wall quanta) with curvature coupling xi_phi (wall field = SM Higgs in the An-Li
    scenario; xi_phi is a free parameter of the theory, section 2). Only the trace couples to xi_phi."""
    return T_scalar(p1, p2, 0.0, xi_phi, -1, -1)


def amp2_sum(spin, p1, p2, k1, k2, M, xi=0.0, xi_phi=0.0):
    """Spin-summed |M|^2 * M_P^4 for s(p1)s(p2) -> X(k1)X(k2)."""
    q = p1 + p2; q2 = mdot(q, q)
    TS = T_initial(p1, p2, xi_phi)
    if spin == 0:
        return abs(contract(TS, T_scalar(k1, k2, M * M, xi, +1, +1)) / q2) ** 2
    if spin == 0.5:
        return sum(abs(contract(TS, T_dirac_pair_out(k1, k2, M, a, b)) / q2) ** 2 for a in range(2) for b in range(2))
    if spin == 1:
        e1s = pol_vectors(k1, M); e2s = pol_vectors(k2, M)
        return sum(abs(contract(TS, T_vector_me(ea, k1, +1, eb, k2, +1, M * M)) / q2) ** 2 for ea in e1s for eb in e2s)
    raise ValueError(spin)


def cm_kinematics(s, M, cth, phi=0.3):
    E = np.sqrt(s) / 2
    p1 = np.array([E, 0, 0, E]); p2 = np.array([E, 0, 0, -E])
    k = np.sqrt(max(E * E - M * M, 0.0)); sth = np.sqrt(1 - cth ** 2)
    n = np.array([sth * np.cos(phi), sth * np.sin(phi), cth])
    return p1, p2, np.concatenate([[E], k * n]), np.concatenate([[E], -k * n])


SYMM = {0: 0.5, 0.5: 1.0, 1: 0.5}     # identical-particle factor: real scalar / real (Proca) vector: 1/2; Dirac: 1


def c_hat(spin, M2_over_s, xi=0.0, n=161, xi_phi=0.0):
    """c = sigma * M_P^4 / s  (dimensionless) for s s -> X X, s-channel graviton.
    sigma = 1/(2s) * S * int |M|^2 dPhi_2,  dPhi_2 = beta/(16 pi) dcos(theta)  (azimuth integrated).
    Validity: tree level, s << M_P^2 (s in units of M_P^2 enters only through the overall s/M_P^4)."""
    s = 1.0; M = np.sqrt(M2_over_s * s)
    if 4 * M * M >= s:
        return 0.0
    beta = np.sqrt(1 - 4 * M * M / s)
    x, w = np.polynomial.legendre.leggauss(n)
    vals = np.array([amp2_sum(spin, *cm_kinematics(s, M, c), M, xi, xi_phi) for c in x])
    return SYMM[spin] / (2 * s) * beta / (16 * np.pi) * np.sum(w * vals) / s
