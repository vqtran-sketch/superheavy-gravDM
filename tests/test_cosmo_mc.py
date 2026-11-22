"""Tests for anli.py and bubbles_mc.py (sections 4.2, 5.1-5.2 and appendices C.2, D.1)."""
import sys, pathlib
import numpy as np
from gravdm import anli, bubbles_mc as mc


def test_pure_kination_limit():
    """alpha -> 0: a(tau) = (1 + 2 tau)^{1/2} for n = 6 with H(0) = 1, a(0) = 1 [DERIVED: a^2 H = da/dtau, H ∝ a^-3]."""
    r = anli.solve(1e-9, n=6, N=3000)
    t = r["tau"]; m = t > -0.3          # avoid the grid error next to the a -> 0 singularity at tau = -1/2
    assert np.allclose(r["a"][m], np.sqrt(1 + 2 * t[m]), rtol=3e-3)   # trapezoid error in ln a near the singularity (0.2% at tau=-0.3, 0.03% at tau>0)


def test_false_vacuum_fraction_monotone_and_completes():
    r = anli.solve(0.05, n=6, N=3000)
    assert np.all(np.diff(r["f"]) <= 1e-12) and r["f"][0] > 0.999 and r["f"][-1] < 1e-6


def test_energy_bookkeeping():
    """L f_FV + released energy (×a^4) is conserved comovingly up to redshift: a^4 rho_rad = L ∫ a^4 (-df)."""
    r = anli.solve(0.05, n=6, N=6000)
    tot = r["a"][-1] ** 4 * r["rho_rad"][-1] / r["L"]
    direct = np.trapezoid(r["a"] ** 4 * (-np.gradient(r["f"], r["tau"])), r["tau"])
    assert np.isclose(tot, direct, rtol=1e-2)


def test_e_hat_flat_limit():
    """a = 1: e_hat = r/3 (E/A = L R/3, An-Li S22 flat limit)."""
    bg = mc.Background(dict(tau=np.linspace(0, 5, 2001), a=np.ones(2001)))
    assert np.isclose(bg.e_hat(np.array([0.2]), np.array([1.7]))[0], 0.5, rtol=1e-4)


def test_two_bubble_collision_surface_area():
    """Flat space, two bubbles nucleated simultaneously at distance D in a large box: the first-collision surface is the bisecting plane;
    A/V from the MC estimator = L^2/L^3 (one plane per periodic box crossing... two planes by periodicity) -> 2/Lbox; h = 1 on the axis."""
    bg = mc.Background(dict(tau=np.linspace(0, 20, 4001), a=np.ones(4001)))
    Lb = 4.0
    tn = np.array([0.0, 0.0]); xn = np.array([[1.0, 2.0, 2.0], [3.0, 2.0, 2.0]])
    q = mc.surface_Q(bg, tn, xn, Lb, np.random.default_rng(0), npts=400_000, eps=0.02, chunk=100_000)
    assert abs(q["AV"] / (2 / Lb) - 1) < 0.05
    # walls are spheres: on the bisector plane the normals are radial, h = |n1 - n2|/2 = (D/2)/r (=1 only on the axis)
    S = q["samples"]
    assert np.allclose(S[:, 6], 1.0 / S[:, 1], rtol=2e-2)


def test_exit_distance_two_bubbles():
    """Simultaneous nucleation at distance D: along the axis toward the other bubble the wall exits at s* = D/2; away from it, never."""
    tn = np.array([0.0, 0.0]); xn = np.array([[1.0, 2.0, 2.0], [3.0, 2.0, 2.0]])
    toward = np.array([[1.0, 0, 0]]); away = np.array([[-1.0, 0, 0]])
    assert np.isclose(mc.exit_distance(0.0, xn[0], toward, tn, xn, 40.0, 0)[0], 1.0)
    # away from it: in infinite space never; in the periodic box (side 40) the image of the other bubble at x = 3 - 40 is met at
    # s = (40 - 2)/2 = 19 (periodic images included)
    assert np.isclose(mc.exit_distance(0.0, xn[0], away, tn, xn, 40.0, 0)[0], 19.0)
    # earlier-nucleated other bubble (dt = -0.5): tau_k + s = tau_l + |D - s|  ->  s = (D - 0.5)/2... solve: s = (D^2 - dt^2)/(2 dt + 2 D)
    tn2 = np.array([0.0, -0.5])
    assert np.isclose(mc.exit_distance(0.0, xn[0], toward, tn2, xn, 40.0, 0)[0], (4 - 0.25) / (1.0 + 4.0))


def test_e_hat_capped_reduces_to_e_hat():
    r = anli.solve(0.05, n=6, N=3000); bg = mc.Background(r)
    ti = np.array([0.1, 0.3]); t = np.array([0.9, 1.2])
    assert np.allclose(mc.e_hat_capped(bg, ti, t, np.inf), bg.e_hat(ti, t), rtol=1e-10)
    flat = mc.Background(dict(tau=np.linspace(0, 5, 2001), a=np.ones(2001)))
    # flat: energy gained up to radius s*, diluted as r^-2: e = s*^3/(3 r^2)
    assert np.isclose(mc.e_hat_capped(flat, np.array([0.0]), np.array([2.0]), 1.0)[0], 1.0 / (3 * 4.0), rtol=1e-4)


def test_surface_estimator_poisson_voronoi_and_eps_extrapolation():
    """Simultaneous nucleation (tn = 0) gives a Poisson-Voronoi tessellation: exact S_V = 2.9104 lambda^{1/3} [LIT: Meijering 1953 via
    Moller 1992 eq. (4.2) with Lambda = delta]. surface_Q has an O(eps) deficit at triple lines (appendix C.2):
    S_V(eps) ≈ S_V (1 - 0.9 eps/0.01 %) for lambda = 1. The linear extrapolation 2 S(eps) - S(2 eps) removes it."""
    bg = mc.Background(dict(tau=np.linspace(0, 20, 4001), a=np.ones(4001)))
    L = 6.0
    rng = np.random.default_rng(3); N = rng.poisson(L ** 3); xn = rng.uniform(0, L, (N, 3)); tn = np.zeros(N)
    exact = 2.9104 * (N / L ** 3) ** (1 / 3)
    A = {e: mc.surface_Q(bg, tn, xn, L, np.random.default_rng(7), npts=600_000, eps=e, chunk=20000)["AV"] for e in (0.02, 0.01)}
    assert -0.035 < A[0.02] / A[0.01] - 1 < -0.002                 # O(eps) deficit present (expected ≈ -1%)
    assert abs((2 * A[0.01] - A[0.02]) / exact - 1) < 0.03          # extrapolated value agrees with the exact density


def test_flat_false_vacuum_fraction_closed_form():
    """Flat MC nucleation (Gamma = beta^4 e^{beta t}, beta = 1, script 11 background) reproduces
    f_FV(t) = exp(-int dt' Gamma(t') (4 pi/3)(t - t')^3) = exp(-8 pi e^t)   [DERIVED: int_0^inf x^3 e^{-x} dx = 6].
    Points sampled uniformly; a point is in the false vacuum at t if no bubble has arrived, min_k(tau_k + |x - x_k|) > t.
    Also checks that rejecting nucleations inside bubbles does not change the union (Poisson 'phantom' bubbles)."""
    t = np.linspace(-30.0, 30.0, 12001)
    bg = mc.Background(dict(tau=t, a=np.ones_like(t))); bg.cum_a4 = np.exp(t) - np.exp(t[0])
    # Var(-ln f) per box = (4 pi/3)^2 e^t int x^6 e^-x dx / V = (16 pi^2/9) 720 e^t / V: few early, large bubbles dominate (0.35 rms
    # at 8 pi e^t = 1 for V = 16^3), so average over many cheap boxes (nucleation only up to the last test time).
    Lb = 16.0; times = np.log(np.array([0.25, 1.0, 2.0]) / (8 * np.pi))     # 8 pi e^t = 0.25, 1, 2
    frac = np.zeros(len(times)); nseed = 300
    for sd in range(nseed):
        tn, xn = mc.nucleate(bg, Lb, np.random.default_rng(sd), tau_max=times[-1])
        x = np.random.default_rng(10_000 + sd).uniform(0, Lb, (3000, 3))
        d = np.abs(x[:, None, :] - xn[None, :, :]); d = np.minimum(d, Lb - d)
        tarr = np.min(tn[None, :] + np.sqrt((d ** 2).sum(-1)), axis=1)
        frac += np.array([np.mean(tarr > tt) for tt in times])
    frac /= nseed
    exact = np.exp(-8 * np.pi * np.exp(times))
    assert np.allclose(np.log(frac), np.log(exact), atol=0.06), (frac, exact)      # sigma(ln f) ~ 0.01-0.03 for 300 boxes


def test_exit_distance_periodic_image():
    """Simultaneous bubbles at (0,3,3) and (2,3,3) in a box of side 6; along -x from the first, the first
    wall met is the image of the second at (-4,3,3), at s = 2 (the minimum image relative to the centre would give s = inf)."""
    tn = np.array([0.0, 0.0]); xn = np.array([[0.0, 3, 3], [2.0, 3, 3]])
    s = mc.exit_distance(tn[0], xn[0], np.array([[-1.0, 0, 0], [1.0, 0, 0]]), tn, xn, 6.0, 0)
    assert np.allclose(s, [2.0, 1.0])


def test_surface_Q_npts_not_multiple_of_chunk():
    """Q and A/V must not depend on how npts is split into chunks (exactly npts points are drawn)."""
    t = np.linspace(0, 5, 2001); bg = mc.Background(dict(tau=t, a=np.ones_like(t)))
    tn = np.array([0.0, 0.0]); xn = np.array([[2.0, 3, 3], [4.0, 3, 3]]); L = 6.0
    q1 = mc.surface_Q(bg, tn, xn, L, np.random.default_rng(3), npts=40_000, eps=0.05, chunk=40_000)
    q2 = mc.surface_Q(bg, tn, xn, L, np.random.default_rng(3), npts=40_000, eps=0.05, chunk=30_000)
    assert np.isclose(q1["AV"], q2["AV"], rtol=0.1) and np.isclose(q1["Q"], q2["Q"], rtol=0.15)
