"""Literature reproduction of sigma(ss -> XX): checks of our cross sections (section 2, appendix A) against published amplitudes
and thermal rates.

[LIT] Garny, Palessandro, Sandora, Sloth, arXiv:1709.09688v2 (JCAP 02 (2018) 027), eqs. (30), (38), (42), (47):
      G = 1/m_p^2 (non-reduced), stress tensors with an extra 1/2 per real-scalar T (their text below eq. 29),
      non-minimal L = +(1/2) xi phi^2 R  =>  xi_Garny = -xi_ours (conformal at -1/6).
[LIT] Clery, Mambrini, Olive, Shkerin, Verner, arXiv:2203.02004v1 (PRD 105 (2022) 095042), App. B, eqs. (74), (83)-(89):
      M_P reduced, S = -(M_P^2/2)(1 + xi S^2/M_P^2) R  =>  xi_Clery = -xi_ours; rate R = beta1 T^8 + beta2 m^2 T^6 + beta3 m^4 T^4
      (/M_P^4) for N_h = 4 massless thermal Higgs scalars, Bose-Einstein.
Rate <-> sigma [DERIVED] (R): for sigma*s = c0 s^2 + c1 m^2 s + c2 m^4 (M_P = 1),
      beta1 = N_h c0 2 pi^4/675,  beta2 = N_h c1 zeta(3)^2/pi^4,  beta3 = N_h c2/288.
Appendix B.1 of the paper quotes the agreement of the massless coefficient with beta1 of Clery et al. and with eq. (47) of Garny et al.;
the finite-mass and non-minimal terms of Clery et al. differ from ours (tests below identify the sources of the differences).
"""
import sys, pathlib
import numpy as np
import pytest
from scipy.special import zeta

from gravdm import gravxsec as gx
from gravdm import xsec_analytic as xa

G = 1 / (8 * np.pi)          # 1/m_p^2 in units M_P = 1
NH = 4
Z3 = zeta(3)


def garny(ch, s, t, m, xp=0.0, xX=0.0):
    """[LIT] 1709.09688 |M|^2 (their normalisation); xp, xX in their sign convention."""
    if ch == "0->0":
        return 4 * G**2 * np.pi**2 * (m**2 - t) ** 2 * (m**2 - s - t) ** 2 / s**2                                  # (30)
    if ch == "0->1/2":
        return 8 * G**2 * np.pi**2 * (m**2 - t) * (-m**2 + s + t) * (-2 * m**2 + s + 2 * t) ** 2 / s**2           # (38)
    if ch == "0->1":
        return 12 * np.pi**2 * G**2 * (m**2 - t) ** 2 * (-m**2 + s + t) ** 2 / s**2                                 # (42)
    if ch == "xi":
        return 4 * G**2 * np.pi**2 * ((m**4 + s * t + t**2 - m**2 * (s + 2 * t + 2 * s * xp)
                                       - s**2 * (xX + xp + 6 * xX * xp)) / s) ** 2                                  # (47)
    raise ValueError(ch)


# ratio ours/Garny = (stress-tensor normalisation)^-2: 1/2 on each real-scalar T -> 16; Dirac DM side has no 1/2 -> 4
@pytest.mark.parametrize("ch,spin,ratio", [("0->0", 0, 16.0), ("0->1/2", 0.5, 4.0), ("0->1", 1, 16.0), ("xi", 0, 16.0)])
def test_garny_pointwise(ch, spin, ratio):
    """Our explicit-amplitude |M|^2 equals Garny et al.'s up to a CONSTANT (normalisation), incl. masses and both xi."""
    rng = np.random.default_rng(7)
    for _ in range(60):
        s = rng.uniform(0.5, 3.0); m = np.sqrt(rng.uniform(0.0, 0.249) * s); c = rng.uniform(-1, 1)
        xp, xX = (rng.uniform(-2, 2), rng.uniform(-2, 2)) if ch == "xi" else (0.0, 0.0)
        p1, p2, k1, k2 = gx.cm_kinematics(s, m, c)
        t = gx.mdot(p1 - k1, p1 - k1)
        ours = gx.amp2_sum(spin, p1, p2, k1, k2, m, xi=-xX, xi_phi=-xp)
        assert ours == pytest.approx(ratio * garny(ch, s, t, m, xp, xX), rel=1e-10)


def _c_from_amp2(amp2, r, with_v=True, n=64):
    """c = sigma M_P^4/s for identical final scalars from |M|^2(s=1, cos theta) (physical normalisation).
    with_v=False drops the final-state velocity (Clery eq.-(21) phase space)."""
    v = np.sqrt(1 - 4 * r)
    x, w = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (v * x - 1) + r                                   # s = 1, m^2 = r
    jac = 0.5 * v if with_v else 0.5
    return 0.5 / (16 * np.pi) * jac * np.sum(w * amp2(t, r))


def _series(cfun):
    """c0, c1, c2 of c(r) = c0 + c1 r + c2 r^2 + ... by a polynomial fit at small r."""
    r = 0.02 * (1 - np.cos(np.linspace(0, np.pi, 25))) / 2
    co = np.polynomial.polynomial.polyfit(r, [cfun(x) for x in r], 7)
    return co[:3]


def _betas(c):
    return NH * c[0] * 2 * np.pi**4 / 675, NH * c[1] * Z3**2 / np.pi**4, NH * c[2] / 288


def test_clery_beta1_minimal_exact():
    """[LIT] Clery eq. (84): beta1 = pi^3/81000 from our c0 = 1/(960 pi) (absolute normalisation check)."""
    b1 = NH * xa.c0("scalar") * 2 * np.pi**4 / 675
    assert b1 == pytest.approx(np.pi**3 / 81000, rel=1e-13)


def test_our_mass_corrections_vanish_to_r2():
    """Ours: c(r) 960 pi = v (1 + 2r + 6r^2) = 1 + O(r^3)  =>  beta2 = beta3 = 0 [DERIVED]."""
    c = _series(lambda r: float(xa.c_of_r("scalar", r)))
    assert c[0] == pytest.approx(1 / (960 * np.pi), rel=1e-10)
    assert abs(c[1]) < 1e-8 * c[0] and abs(c[2]) < 1e-6 * c[0]


def test_clery_beta23_follow_from_eq83_without_velocity_factor():
    """[LIT] Clery (85), (86) are reproduced only by their eq. (83) |M|^2 AND the phase space without v_X.
    Their (83) bracket t(s+t) - 2m^2 t + m^4 differs from Garny (30) / ours (m^2-t)(m^2-u) by + m^2 s."""
    clery83 = lambda t, r: 4 * (t * (1 + t) - 2 * r * t + r**2) ** 2 / 4          # 4 x |M|^2_Clery -> physical
    _, b2, b3 = _betas(_series(lambda r: _c_from_amp2(clery83, r, with_v=False)))
    assert b2 == pytest.approx(-Z3**2 / (30 * np.pi**5), rel=1e-6)
    assert b3 == pytest.approx(1 / (4320 * np.pi), rel=1e-5)
    ours = lambda t, r: ((r - t) * (r - (2 * r - 1 - t))) ** 2
    c = _series(lambda r: _c_from_amp2(ours, r, with_v=True))
    assert c[0] == pytest.approx(xa.c0("scalar"), rel=1e-10)            # the quadrature route reproduces our closed form
    assert abs(c[1]) < 1e-7 * c[0]


@pytest.mark.parametrize("a,b", [(0.3, 0.0), (0.0, -0.7), (0.5, 0.25), (-1.2, 2.0)])
def test_clery_beta1_xi_differs_by_unsymmetrised_contact(a, b):
    """[LIT] Clery (87) vs ours (xi -> -xi): equal up to + 360 xi_h^2 xi_X^2 [x pi^3/81000], which is exactly the
    square of the t<->u-odd part of their contact coupling (74), 12 xi_h xi_X (m^2 - t) -> symmetrised 6 xi_h xi_X s."""
    B87 = 30 * a**2 * (12 * b * (4 * b + 1) + 1) + 10 * a * (6 * b + 1) ** 2 + 10 * b * (3 * b + 1) + 1
    ours = xa.c0_general("scalar", xi_phi=-a, xi_X=-b)
    assert B87 - ours == pytest.approx(360 * a**2 * b**2, abs=1e-10)
    x, w = np.polynomial.legendre.leggauss(16)
    K = a + b + 6 * a * b
    sym = 15 * np.sum(w * ((1 - x**2) / 4 + K) ** 2)                  # symmetrised contact = Garny (47) massless
    unsym = 15 * np.sum(w * ((1 - x**2) / 4 + K - 6 * a * b * x) ** 2)
    assert sym == pytest.approx(ours, rel=1e-12) and unsym == pytest.approx(B87, rel=1e-12)


@pytest.mark.parametrize("a,b", [(0.3, 0.0), (0.0, -0.7), (0.5, 0.25), (-1.2, 2.0)])
def test_corrected_beta23_xi(a, b):
    """[DERIVED] Clery-convention beta2^xi, beta3^xi with the differences removed, from our explicit amplitudes
    (Garny 47 incl. v_X) via the rate map (R); a = xi_h, b = xi_X in their sign (xi = -xi_ours)."""
    c = _series(lambda r: gx.c_hat(0, r, xi=-b, xi_phi=-a))
    _, b2, b3 = _betas(c)
    B2 = -Z3**2 * (108 * a**2 * b**2 - 3 * a**2 + 36 * a * b**2 - a + 3 * b**2) / (12 * np.pi**5)
    B3 = -(36 * a**2 * b**2 + 36 * a**2 * b + 3 * a**2 + 12 * a * b**2 + 12 * a * b + a + b**2 + b) / (1152 * np.pi)
    assert b2 == pytest.approx(B2, rel=1e-6) and b3 == pytest.approx(B3, rel=1e-5)


@pytest.mark.parametrize("xp,xX", [(0.0, 0.0), (1 / 6, 0.0), (0.0, 1 / 6), (0.4, -0.3), (-1.0, 0.7)])
def test_frieman_spectral_formula(xp, xX):
    """[LIT] Frieman, PRD 39 (1989) 389 (SLAC-PUB-3855 scan), eq. (4): pair creation by a classical weak field,
    ∝ v [ |R|^2 {60 a^2 - 40 r (a + s r/6)} + |C|^2 (1-4r)^2 ],  a = xi_X - 1/6, r = M^2/q^2 (xi_X in OUR sign: conformal at 1/6).
    For our wall-quanta source R = -T^mu_mu/M_P^2 with T^mu_mu = -s (1 - 6 xi_phi), and C^2 = 2 (T.T - T^2/3)/M_P^4 = s^2/(3 M_P^4),
    independent of xi_phi. With s = -1 this reproduces our chat(r; xi_phi, xi_X) exactly; the scan prints s = +1, which is not a perfect
    square in the spin-0 channel (|<XX|T^mu_mu|0>|^2 ∝ (a - r/3)^2) and goes negative at xi_X = 1/6: a sign typo (PRD version not checked)."""
    for r in (0.0, 0.05, 0.15, 0.24):
        a = xX - 1 / 6
        fr = np.sqrt(1 - 4 * r) * ((1 - 6 * xp) ** 2 * (60 * a**2 - 40 * r * (a - r / 6)) + (1 - 4 * r) ** 2 / 3) / 2
        assert 960 * np.pi * gx.c_hat(0, r, xi=xX, xi_phi=xp) == pytest.approx(fr, rel=1e-8)
