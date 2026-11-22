"""Tests for parton.py (sections 3.1-3.3)."""
import sys, pathlib
import numpy as np
import pytest
from gravdm import parton as pt


@pytest.mark.parametrize("profile", list(pt.PROFILES))
def test_parseval_tension(profile):
    """[eqs. (3.2), (3.4)] sigma from int phi'^2 equals int |F|^2/pi (two routes)."""
    d, f = pt.sigma_over_dphi2_m(profile)
    assert np.isclose(d, f, rtol=1e-8)


def test_known_tensions():
    """tanh kink: sigma = Dphi^2 m/6 (= (2/3) m_s w^2 for Dphi = 2w, Ghoshal:2026pew below eq. (3.15)) [LIT]."""
    assert np.isclose(pt.sigma_over_dphi2_m("tanh")[1], 1 / 6, rtol=1e-10)
    assert np.isclose(pt.sigma_over_dphi2_m("erf")[1], 1 / (2 * np.sqrt(np.pi)), rtol=1e-10)


@pytest.mark.parametrize("z", [1e-8, 1e-6])
def test_gps_eq30_luminosity(z):
    """[LIT: Ghoshal:2026pew (JHEP 11 (2026) 142) eq. (3.19)] s dL/ds = 32 w^4/pi^2 ln(gamma e m/(pi sqrt s)) for sqrt s << gamma m, Dphi = 2w
    <=> K(z) = 2 ln(e/(2 pi sqrt z))."""
    assert np.isclose(pt.K(z, "tanh"), 2 * np.log(np.e / (2 * np.pi * np.sqrt(z))), rtol=1e-8)


@pytest.mark.parametrize("x", [4.0, 10.0, 20.0])
def test_gps_eq30_high_branch(x):
    """[LIT: Ghoshal:2026pew eq. (3.19), sqrt s >> gamma m branch] K = pi^4 x^{7/2} e^{-2 pi x}, x = sqrt(s)/(gamma m) = 2 sqrt z:
    agrees to 0.5% at x = 4 and better above."""
    assert np.isclose(pt.K(x ** 2 / 4, "tanh"), np.pi ** 4 * x ** 3.5 * np.exp(-2 * np.pi * x), rtol=6e-3)


@pytest.mark.parametrize("profile", list(pt.PROFILES))
def test_theorem_massless(profile):
    """[eq. (3.13)] for c = const: int dz K(z) c = c (int W)^2  <=> N/A = 4 c0 E1 E2 / M_P^4, any profile."""
    c0 = 1.0
    zt, Kt = pt.K_table(profile, zmin=1e-12, zmax=60.0, n=600)
    I = pt.yield_integral(lambda r: c0, 0.0, profile, zt, Kt)
    from scipy import integrate
    Wint = integrate.quad(pt.PROFILES[profile]["W"], 0, 60, limit=400)[0]
    assert np.isclose(I, c0 * Wint ** 2, rtol=1e-7)


def test_K_table_convergence():
    """Doubling table density changes the yield integral at mu = 0.3 by < 1e-3."""
    cf = lambda r: np.where(r < 0.25, np.sqrt(np.clip(1 - 4 * r, 0, None)), 0.0)
    z1, K1 = pt.K_table("tanh", n=300); z2, K2 = pt.K_table("tanh", n=600)
    a = pt.yield_integral(cf, 0.3, "tanh", z1, K1); b = pt.yield_integral(cf, 0.3, "tanh", z2, K2)
    assert np.isclose(a, b, rtol=1e-3)


def test_cut_yield_table():
    """[script 21] Hard-cut yield G(mu, z_c): (i) z_c -> inf reproduces yield_integral(mu)/I(0); (ii) at finite z_c it equals
    a direct quad of int_{mu^2}^{z_c} K c; (iii) monotone in z_c.  Two routes: cumulative trapezoid vs adaptive quad."""
    from gravdm import xsec_analytic as xa
    from scipy import integrate
    z = np.logspace(-10, np.log10(30.0), 3000); Kz = np.array([pt.K(zi, "tanh") for zi in z])
    cf = lambda r: xa.c_of_r("dirac", r)
    mus = np.array([1e-4, 0.1, 0.5])
    G = pt.cut_yield_table(cf, mus, z, Kz)
    zt, Kt = pt.K_table("tanh", zmin=1e-10, zmax=30.0, n=600)
    I0 = pt.yield_integral(cf, 0.0, "tanh", zt, Kt)
    for i, mu in enumerate(mus):
        assert np.isclose(G[i, -1], pt.yield_integral(cf, mu, "tanh", zt, Kt) / I0, rtol=2e-3)
        assert np.all(np.diff(G[i]) >= -1e-15)
    zc = 0.05; j = np.searchsorted(z, zc); zc = z[j]
    direct = integrate.quad(lambda t: np.exp(t) * pt.K(np.exp(t), "tanh") * cf(0.1 ** 2 / (4 * np.exp(t))), np.log(0.1 ** 2), np.log(zc),
                            limit=200, epsrel=1e-8)[0] / I0
    assert np.isclose(G[1, j], direct, rtol=2e-3)


# --- section 3.1: statements made explicit in the paper text ---

@pytest.mark.parametrize("gamma", [2.0, 10.0, 50.0])
def test_boosted_kink_energy(gamma):
    """phi^4 kink phi0 = (Dphi/2)(1+tanh(m z/2)) in V = (lambda/4)(phi^2-v^2)^2 (shifted), boosted: T^00 = (1/2)(phi_t^2+phi_x^2)+V.
    Exact E/A = gamma sigma; the potential share is 1/(2 gamma^2), and sigma = int phi0'^2 = Dphi^2 m/6 (virial)."""
    dphi, m = 2.0, 1.0
    v = np.sqrt(1 - 1 / gamma ** 2)
    phi0p = lambda z: dphi * m / 4 / np.cosh(m * z / 2) ** 2          # phi0'(z)
    phi0 = lambda z: dphi / 2 * (1 + np.tanh(m * z / 2))
    lam = m ** 2 / (2 * (dphi / 2) ** 2)                                # m^2 = 2 lambda w^2, w = Dphi/2
    V = lambda ph: lam / 4 * ((ph - dphi / 2) ** 2 - (dphi / 2) ** 2) ** 2
    from scipy import integrate
    # at t = 0: phi(x) = phi0(gamma x), phi_t = -v gamma phi0'(gamma x), phi_x = gamma phi0'(gamma x)
    grad = integrate.quad(lambda x: 0.5 * (1 + v * v) * (gamma * phi0p(gamma * x)) ** 2, -80 / gamma, 80 / gamma, limit=400)[0]
    pot = integrate.quad(lambda x: V(phi0(gamma * x)), -80 / gamma, 80 / gamma, limit=400)[0]
    sigma = dphi ** 2 * m / 6
    assert np.isclose(grad + pot, gamma * sigma, rtol=1e-9)
    assert np.isclose(pot / (grad + pot), 1 / (2 * gamma ** 2), rtol=1e-9)
    # ultra-relativistic form used in the paper: int du (dphi_R/du)^2 = gamma sigma
    lc = integrate.quad(lambda u: (gamma * phi0p(gamma * u)) ** 2, -80 / gamma, 80 / gamma, limit=400)[0]
    assert np.isclose(lc, gamma * sigma, rtol=1e-9)


@pytest.mark.parametrize("profile", list(pt.PROFILES))
def test_parton_density_energy(profile):
    """[eq. (3.5)] int dp p dN/(A dp) = gamma sigma_wall: the partons carry the wall energy (Dphi = m = 1)."""
    from scipy import integrate
    gamma = 30.0; W = pt.PROFILES[profile]["W"]
    n = lambda p: W(p / gamma) / (np.pi * p)
    E = integrate.quad(lambda p: p * n(p), 0, 60 * gamma, limit=400)[0]
    assert np.isclose(E, gamma * pt.sigma_over_dphi2_m(profile)[1], rtol=1e-8)


@pytest.mark.parametrize("profile", ["tanh", "erf"])
@pytest.mark.parametrize("z", [1e-3, 0.3, 2.0])
def test_luminosity_definition(profile, z):
    """[eqs. (3.6), (3.7)] s dL/ds with dL/ds = int dp_R dp_L n(p_R) n(p_L) delta(s - 4 p_R p_L) equals (Dphi^4/pi^2) K(z)."""
    from scipy import integrate
    gamma, m = 7.0, 1.0; W = pt.PROFILES[profile]["W"]
    n = lambda p: W(p / (gamma * m)) / (np.pi * p)
    s = 4 * gamma ** 2 * m ** 2 * z
    # delta(s - 4 pR pL) -> pL = s/(4 pR), Jacobian 1/(4 pR); integrate in ln pR
    f = lambda t: np.exp(t) * n(np.exp(t)) * n(s / (4 * np.exp(t))) / (4 * np.exp(t))
    tc = 0.5 * np.log(s / 4)
    I = sum(integrate.quad(f, a, b, limit=400, epsabs=0, epsrel=1e-10)[0] for a, b in [(tc - 40, tc), (tc, tc + 40)])
    assert np.isclose(s * I, pt.K(z, profile) / np.pi ** 2, rtol=1e-7)


# closed forms of a_W = lim_{s->0}[M_W(s) - 1/s] from the Mellin transforms of W (section 3.1)
_GLAISHER = 1.2824271291006226
A_W_CLOSED = {"tanh": 1 - np.log(2 * np.pi),                 # M_W = (2 pi)^{-s} Gamma(s+2) zeta(s+1)
              "erf": -np.euler_gamma / 2,                    # M_W = Gamma(s/2)/2
              "sG": 12 * np.log(_GLAISHER) - 1 - np.euler_gamma - np.log(np.pi) - 4 * np.log(2) / 3}   # M_W = (pi)^{-s} 4 Gamma(s) eta(s-1)


@pytest.mark.parametrize("profile", list(pt.PROFILES))
def test_aW_mellin_closed_form(profile):
    """[DERIVED] a_W = int dx/x [W - theta(1-x)] (quadrature) equals the Mellin closed form, and K(zeta) -> -ln zeta + 2 a_W
    with corrections O(zeta^2 ln zeta) (two routes: direct quadrature of K at small zeta)."""
    from scipy import integrate
    W = pt.PROFILES[profile]["W"]
    f = lambda t: W(np.exp(t)) - (t < 0)
    aq = integrate.quad(f, -40, 0, limit=400, epsabs=1e-13)[0] + integrate.quad(f, 0, 6, limit=400, epsabs=1e-13)[0]
    assert np.isclose(aq, A_W_CLOSED[profile], atol=1e-10)
    for z in (1e-4, 1e-6):
        assert abs(pt.K(z, profile) + np.log(z) - 2 * A_W_CLOSED[profile]) < 20 * z ** 2 * np.log(1 / z) + 1e-8   # 1e-8: quadrature floor


@pytest.mark.parametrize("zeta", [1e-6, 1e-3, 0.25, 1.0, 4.0])
def test_gps_eq30_full_J_form(zeta):
    """[LIT: Ghoshal:2026pew eq. (3.19), exact form] s dL/ds = pi^2 w^4 s^2/(gamma^4 m^4) J(pi sqrt s/(2 gamma m)),
    J(z) = int dr csch^2(z e^-r) csch^2(z e^r)  <=>  K(zeta) = pi^4 zeta^2 J(pi sqrt zeta) for the tanh wall (Dphi = 2w)."""
    from scipy import integrate
    y = np.pi * np.sqrt(zeta)
    g = lambda r: (1 / np.sinh(min(y * np.exp(-r), 700))) ** 2 * (1 / np.sinh(min(y * np.exp(r), 700))) ** 2
    J = integrate.quad(g, -60, 60, limit=800, epsabs=0, epsrel=1e-11)[0]
    assert np.isclose(np.pi ** 4 * zeta ** 2 * J, pt.K(zeta, "tanh"), rtol=1e-8)


def test_profile_definitions_draft():
    """[DERIVED] Profiles as defined in section 3.1 (eq. (3.1)): phi^4 kink in V = m^2 phi^2 (phi-Dphi)^2/(2 Dphi^2), sine-Gordon kink
    in V = (m Dphi/2pi)^2 [1 - cos(2 pi phi/Dphi)], and the erf test profile. Checks: Dphi, F(k) (quoted closed forms) and W = |F(mx)|^2/Dphi^2
    against PROFILES, tensions mDphi^2/6, 2mDphi^2/pi^2, mDphi^2/(2 sqrt pi) (direct integral and Parseval), first integral phi'^2/2 = V, V'' = m^2."""
    from scipy import special, integrate
    m, D = 1.3, 0.7
    z = np.linspace(-40 / m, 40 / m, 400001)
    prof = {
        "tanh": (D / 2 * (1 + np.tanh(m * z / 2)), lambda p: m ** 2 / (2 * D ** 2) * p ** 2 * (p - D) ** 2,
                 lambda k: D * (np.pi * k / m) / np.sinh(np.pi * k / m), m * D ** 2 / 6),
        "sG": (2 * D / np.pi * np.arctan(np.exp(m * z)), lambda p: (m * D / (2 * np.pi)) ** 2 * (1 - np.cos(2 * np.pi * p / D)),
               lambda k: D / np.cosh(np.pi * k / (2 * m)), 2 * m * D ** 2 / np.pi ** 2),
        "erf": (D / 2 * (1 + special.erf(m * z / np.sqrt(2))), None, lambda k: D * np.exp(-k ** 2 / (2 * m ** 2)), m * D ** 2 / (2 * np.sqrt(np.pi))),
    }
    for name, (phi, V, F, sig) in prof.items():
        dp = np.gradient(phi, z)
        assert np.isclose(phi[-1] - phi[0], D, rtol=1e-9)
        assert np.isclose(integrate.trapezoid(dp ** 2, z), sig, rtol=1e-6)
        assert np.isclose(m * D ** 2 / np.pi * integrate.quad(pt.PROFILES[name]["W"], 0, np.inf)[0], sig, rtol=1e-8)
        for k in (0.3 * m, m, 2.5 * m):                                      # profiles even about the centre: F real up to a phase
            assert np.isclose(integrate.trapezoid(dp * np.cos(k * z), z), F(k), rtol=1e-6)
        x = np.array([0.2, 0.7, 1.5])
        assert np.allclose(F(m * x) ** 2 / D ** 2, pt.PROFILES[name]["W"](x), rtol=1e-12)
        if V is not None:
            assert np.max(np.abs(dp ** 2 / 2 - V(phi))) < 1e-6 * np.max(V(phi))
            h = 1e-4 * D
            for v0 in (0.0, D):
                assert np.isclose((V(v0 + h) - 2 * V(v0) + V(v0 - h)) / h ** 2, m ** 2, rtol=1e-6)


def test_classical_source_cross_term_and_eq_NA():
    """[DERIVED, section 3.2, eqs. (3.10), (3.12)] Classical-source route to the parton yield.
    (i) T_phi(phi_L(w)+phi_R(u)) - T(phi_L) - T(phi_R) (potential dropped) equals
        phi_L' phi_R' [n nb + nb n - 2 eta] + 2 xi (eta Box - d d)(phi_L phi_R), exactly;
    (ii) for plane waves it equals -<0|T_phi(0)|p1 p2> with p1 = a n, p2 = b nb, incl. the xi term (conserved);
    (iii) Jacobian d^4q = 2 da db d^2q_perp and the factor chain
        2/(2pi)^2 * |F_R F_L/(2ab)|^2 * (flux 2 s) sigma = |F_R|^2/(pi a) |F_L|^2/(pi b) sigma, s = 4ab;
    (iv) lab-frame transform of a boosted wall gradient, int du e^{iau} gamma phi0'(gamma u) = F(a/gamma) (tanh, quadrature)."""
    import sympy as sp
    from scipy import integrate
    t, x, y, z, a, b, xi = sp.symbols('t x y z a b xi', real=True)
    X = [t, x, y, z]; eta = sp.diag(1, -1, -1, -1)
    u, w = t - x, t + x
    up = lambda f, m: sum(eta[m, k] * sp.diff(f, X[k]) for k in range(4))
    box = lambda f: sum(eta[m, m] * sp.diff(f, X[m], 2) for m in range(4))

    def T(phi):
        d = [up(phi, m) for m in range(4)]
        dphi2 = sum(eta[m, m] * d[m] ** 2 for m in range(4))
        return sp.Matrix(4, 4, lambda m, k: d[m] * d[k] - eta[m, k] * dphi2 / 2
                         + xi * (eta[m, k] * box(phi ** 2) - up(up(phi ** 2, k), m)))

    n_ = sp.Matrix([1, 1, 0, 0]); nb = sp.Matrix([1, -1, 0, 0])
    # (i) general profiles
    L, R = sp.Function('fL')(w), sp.Function('fR')(u)
    Tx = T(L + R) - T(L) - T(R)
    dL, dR = sp.diff(L, t), sp.diff(R, t)
    form = sp.Matrix(4, 4, lambda m, k: dL * dR * (n_[m] * nb[k] + nb[m] * n_[k] - 2 * eta[m, k])
                     + 2 * xi * (eta[m, k] * box(L * R) - up(up(L * R, k), m)))
    assert sp.simplify(Tx - form) == sp.zeros(4, 4)
    # (ii) plane waves
    PL, PR = sp.exp(-sp.I * b * w), sp.exp(-sp.I * a * u)
    Tpw = sp.simplify((T(PL + PR) - T(PL) - T(PR)) * sp.exp(sp.I * (a * u + b * w)))
    p1, p2 = a * n_, b * nb; q = p1 + p2
    q2 = (q.T * eta * q)[0]; p12 = (p1.T * eta * p2)[0]
    ME = sp.Matrix(4, 4, lambda m, k: p1[m] * p2[k] + p2[m] * p1[k] - eta[m, k] * p12
                   + 2 * xi * (eta[m, k] * q2 - q[m] * q[k]))
    assert sp.simplify(Tpw + ME) == sp.zeros(4, 4)
    assert sp.simplify(q2 - 4 * a * b) == 0
    assert sp.simplify((eta * q).T * ME) == sp.zeros(1, 4)
    # (iii) Jacobian and factor chain
    ap, bp, FR, FL, sg = sp.symbols('a b F_R F_L sigma', positive=True)
    J = sp.Matrix([[1, 1], [1, -1]]).det()
    chain = abs(J) / (2 * sp.pi) ** 4 * (2 * sp.pi) ** 2 * (FR * FL / (2 * ap * bp)) ** 2 * 2 * (4 * ap * bp) * sg
    assert abs(J) == 2
    assert sp.simplify(chain / (FR ** 2 / (sp.pi * ap) * FL ** 2 / (sp.pi * bp) * sg)) == 1
    # (iv) boosted wall: tanh, phi0' = (m/4) sech^2(m z/2) (Dphi = 1), F(k) = (pi k/m)/sinh(pi k/m)
    m, gam = 1.0, 20.0
    dphi0 = lambda zz: (m / 4) / np.cosh(m * zz / 2) ** 2
    for av in [0.5 * gam * m, 1.0 * gam * m, 2.0 * gam * m]:
        num = integrate.quad(lambda uu: gam * dphi0(gam * uu) * np.cos(av * uu), -40 / gam, 40 / gam,
                             limit=400, points=[0.0])[0]
        k = av / gam
        assert np.isclose(num, (np.pi * k / m) / np.sinh(np.pi * k / m), rtol=1e-8)


@pytest.mark.parametrize("profile,frac", [("tanh", 0.189), ("erf", 0.112), ("sG", 0.156)])
def test_soft_parton_energy_fraction_draft(profile, frac):
    """[DERIVED, section 3.3] fraction of E/A carried by partons with p < 0.1 gamma m: int_0^0.1 W / int_0^inf W
    (E/A = gamma m Dphi^2/pi int dx W); the paper quotes 11--19%."""
    from scipy import integrate
    W = pt.PROFILES[profile]["W"]
    f = integrate.quad(W, 0, 0.1)[0] / integrate.quad(W, 0, 60, limit=200)[0]
    assert np.isclose(f, frac, atol=1.5e-3)
    assert 0.11 <= f <= 0.19


@pytest.mark.parametrize("profile,case,mu10,g1", [
    ("tanh", "scalar", 0.51, 0.0020), ("sG", "scalar", 0.65, 0.015), ("erf", "scalar", 0.83, 0.044),
    ("tanh", "conformal", 0.40, None), ("tanh", "dirac", 0.42, None)])
def test_g_mu_benchmarks_draft(profile, case, mu10, g1):
    """[DERIVED, eq. (3.14), section 3.3 and figure 1] g(mu) = int dzeta K c(mu^2/4zeta) / (c0 (int W)^2):
    g(0) = 1; mu at g = 0.1 and g(1) as quoted (2 significant figures)."""
    from gravdm import xsec_analytic as xa
    from scipy import integrate
    from scipy.optimize import brentq
    W = pt.PROFILES[profile]["W"]
    Wi = integrate.quad(W, 0, 60, limit=200)[0]
    zt, Kt = pt.K_table(profile, zmin=1e-12, zmax=60.0, n=600)
    cf = lambda r: xa.c_of_r(case, r)
    norm = xa.c0(case) * Wi ** 2
    g = lambda m: pt.yield_integral(cf, m, profile, zt, Kt) / norm
    assert np.isclose(g(0.0), 1.0, rtol=1e-7)
    assert round(brentq(lambda m: g(m) - 0.1, 0.05, 3), 2) == mu10
    if g1 is not None:
        assert float(f"{g(1.0):.2g}") == g1


@pytest.mark.parametrize("case,edge,spread", [("scalar", 0.51, 3.43), ("dirac", 0.42, 3.21), ("conformal", 0.40, 2.58)])
def test_robust_core_profile_spread_draft(case, edge, spread):
    """[DERIVED, section 3.3, profile-robust range] tanh reaches g = 0.1 first; max over mu in the core of
    max_profile g / min_profile g is <= 3.4 (paper: 'at most a factor 3.4')."""
    from gravdm import xsec_analytic as xa
    from scipy.optimize import brentq
    cf = lambda r: xa.c_of_r(case, r)
    G = {}
    for p in pt.PROFILES:
        zt, Kt = pt.K_table(p, zmin=1e-12, zmax=60.0, n=600)
        I0 = pt.yield_integral(cf, 0.0, p, zt, Kt)
        G[p] = lambda m, p=p, zt=zt, Kt=Kt, I0=I0: pt.yield_integral(cf, m, p, zt, Kt) / I0
    e = brentq(lambda m: G["tanh"](m) - 0.1, 0.05, 3)
    assert round(e, 2) == edge
    for p in ("erf", "sG"):
        assert G[p](e) > 0.1
    s = max(max(G[p](m) for p in G) / min(G[p](m) for p in G) for m in np.linspace(0.01, e, 40))
    assert np.isclose(s, spread, atol=0.01) and s <= 3.45


def test_fig_univ_caption_xi_labels():
    """Figure 1 caption/legend [DERIVED]: (i) the Proca g(mu) equals the xi_X=0 scalar g(mu) because
    c_Proca(r) = 3 c_scalar(r; xi_X=0) at xi_phi=0 for every r (checked against the full amplitude, gravxsec.c_hat);
    (ii) xi_X=1/6 is the conformal value: the massless scalar coefficient is 1/6 for any xi_phi (spin-0 channel off),
    and xi_X=0 gives 1 at xi_phi=0; (iii) the plotted Proca and scalar curves agree to quadrature accuracy."""
    from gravdm import gravxsec as gx
    from gravdm import xsec_analytic as xa
    for r in (1e-4, 0.03, 0.1, 0.2, 0.24):
        cs, cv = gx.c_hat(0, r, xi=0.0), gx.c_hat(1, r)
        assert np.isclose(cv / cs, 3.0, rtol=1e-8)
        assert np.isclose(xa.c_of_r("vector", r), 3 * xa.c_of_r("scalar", r), rtol=1e-12)
        assert np.isclose(cs, xa.c_of_r("scalar", r), rtol=1e-6)
    u = 960 * np.pi
    for xp in (0.0, 1 / 6, 1.0):
        assert np.isclose(gx.c_hat(0, 1e-10, xi=1 / 6, xi_phi=xp) * u, 1 / 6, rtol=1e-6)
    assert np.isclose(gx.c_hat(0, 1e-10, xi=0.0) * u, 1.0, rtol=1e-6)
    d = np.load(pathlib.Path(__file__).resolve().parents[1] / "output/01_universality.npz")
    m = d["mu"] <= 1.0
    for p in ("tanh", "sG", "erf"):
        assert np.max(np.abs(d[f"g_{p}_vector"][m] / d[f"g_{p}_scalar"][m] - 1)) < 1e-4
