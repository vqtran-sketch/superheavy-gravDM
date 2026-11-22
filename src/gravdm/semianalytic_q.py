"""Semi-analytic evaluation of the first-collision surface integrals Q_beta, Q_head and A/V
(eq. (4.9) and appendix C.1 of the paper; cross-check of the Monte Carlo of bubbles_mc.surface_Q / script 11).

Setting [ASSUMED, same as script 11]: flat space, walls at the speed of light, nucleation rate per unit 4-volume Gamma(t)
in the false vacuum, bubbles nucleated only in the false vacuum. Units: beta = 1 for the exponential rate
Gamma = Gamma0 e^{beta t}; lengths and times in 1/beta, A/V in beta, Q in beta (Q_beta = Q/beta is dimensionless).

Master formula [DERIVED; cf. eq. (C.2)]: for a wall weight w(r) per bubble (w = 1 for area, w = r/3 = e_hat for the yield)
and an oblique-factor power h^k,
    Q[w, k] = (1/V) int dA w(r1) w(r2) h^k
            = (1/2) int dt  e^{-I(t)}  [Omega_pair  int dmu |n1-n2| h^k]  F_w(t)^2,                          (M)
    F_w(t) = int_0^inf dr r^{d-1} w(r) Gamma(t - r),       I(t) = int_0^inf dr Gamma(t - r) V_d(r),
with h = |n1 - n2|/2 = sqrt((1 - mu)/2), mu = n1.n2. In d = 3: Omega_pair = (4 pi)(2 pi), V_3 = 4 pi r^3/3, and
int_{-1}^{1} dmu |n1-n2| h^k = 8/(k+3). In d = 1 the angular part is the sum over n_i = +-1: sum |n1-n2| h^k = 4
(only opposite directions, h = 1), Omega_pair -> 1, V_1 = 2 r.

Closed form for Gamma = Gamma0 e^{t}, d = 3 [DERIVED] (C):
    Q[w, k] = m_w^2 / (2 (k+3)),   m_w = int_0^inf dr r^2 w(r) e^{-r}
 => A/V = 2/3,  Q_head = 2/3,  Q_beta = 1/3  (exact, independent of Gamma0).
Bubble density [DERIVED] (N): n_b = int dt Gamma e^{-I} = 1/(8 pi) for the exponential rate in d = 3.
"""
import numpy as np


# ---------------------------------------------------------------- quadrature helpers
def gl_composite(a, b, n_panels, order=16):
    """Composite Gauss-Legendre nodes/weights on [a, b] with n_panels equal panels of `order` points each."""
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(a, b, n_panels + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    nodes = 0.5 * (hi - lo) * x[None, :] + 0.5 * (hi + lo)
    weights = 0.5 * (hi - lo) * w[None, :]
    return nodes.ravel(), weights.ravel()


def exponential_rate(gamma0=1.0, beta=1.0):
    """Gamma(t) = gamma0 exp(beta t)  [per unit 4-volume; units beta^4 if gamma0 in beta^4]."""
    return lambda t: gamma0 * np.exp(beta * np.asarray(t, float))


# ---------------------------------------------------------------- geometric factors
def angular_factor(k, dim=3, n_panels=4):
    """Angular part of eq. (M).

    d = 3: Omega_pair * int_{-1}^{1} dmu |n1 - n2| h^k = 8 pi^2 * 8/(k+3), evaluated numerically with mu = 1 - 2h^2,
           dmu = 4 h dh, |n1 - n2| = 2h  ->  8 pi^2 int_0^1 8 h^{k+2} dh.
    d = 1: sum over n1, n2 in {+1, -1} of |n1 - n2| h^k = 4 (h = 1 for opposite directions, 0 contribution otherwise).
    Returns the full factor including Omega_pair (dimensionless)."""
    if dim == 1:
        return 4.0
    if dim != 3:
        raise ValueError("dim must be 1 or 3")
    h, wh = gl_composite(0.0, 1.0, n_panels)
    return 8 * np.pi ** 2 * np.sum(wh * 8 * h ** (k + 2))


def lightcone_volume(r, dim=3):
    """Spatial volume of a ball of radius r: 4 pi r^3/3 (d = 3), 2 r (d = 1)."""
    return 4 * np.pi * r ** 3 / 3 if dim == 3 else 2 * r


# ---------------------------------------------------------------- master formula, numerically
def _grids(t_range, r_max, n_panels):
    t, wt = gl_composite(*t_range, n_panels)
    r, wr = gl_composite(0.0, r_max, n_panels)
    return t, wt, r, wr


def false_vacuum_exponent(t, rate, dim=3, r_max=60.0, n_panels=8):
    """I(t) = int_0^{r_max} dr Gamma(t - r) V_d(r)  [eq. (I)]; P_FV(t) = exp(-I(t)).
    r_max truncates the past; for Gamma = e^t the neglected tail is ~ r_max^3 e^{-r_max}."""
    r, wr = gl_composite(0.0, r_max, n_panels)
    t = np.atleast_1d(np.asarray(t, float))
    return (rate(t[:, None] - r[None, :]) * lightcone_volume(r, dim)[None, :]) @ wr


def bubble_density(rate, t_range, dim=3, r_max=60.0, n_panels=8):
    """n_b = int dt Gamma(t) exp(-I(t))  [eq. (N)]: number of real (false-vacuum) nucleations per unit volume."""
    t, wt = gl_composite(*t_range, n_panels)
    return np.sum(wt * rate(t) * np.exp(-false_vacuum_exponent(t, rate, dim, r_max, n_panels)))


def surface_moment(rate, t_range, weight=lambda r: np.ones_like(r), k=0, dim=3, r_max=60.0, n_panels=8):
    """Q[w, k] = (1/V) int dA w(r1) w(r2) h^k over the first-collision surface, eq. (M), by composite Gauss-Legendre
    quadrature in t, r and the angle (no use of the factorisation of the exponential case).

    rate:    callable Gamma(t) per unit 4-volume (units beta^4 with beta = 1)
    t_range: (t_lo, t_hi) covering the support of Gamma e^{-I} F^2
    weight:  w(r) per bubble (r in 1/beta); w = 1 -> A/V (or point density in d = 1); w = r/3 -> e_hat
    Returns Q in units of beta (for w dimensionless)."""
    t, wt, r, wr = _grids(t_range, r_max, n_panels)
    I = false_vacuum_exponent(t, rate, dim, r_max, n_panels)
    jac = r ** (dim - 1) if dim == 3 else np.ones_like(r)
    F = (rate(t[:, None] - r[None, :]) * (jac * weight(r))[None, :]) @ wr
    return 0.5 * angular_factor(k, dim) * np.sum(wt * np.exp(-I) * F ** 2)


# ---------------------------------------------------------------- closed forms (exponential rate, d = 3)
def radial_moment(p):
    """m = int_0^inf dr r^{2+p} e^{-r} = (2+p)!  (weight w = r^p)."""
    from math import gamma
    return gamma(3 + p)


def closed_form_exponential(p=0, k=0, norm=1.0):
    """Eq. (C): Q[w, k] = m_w^2/(2(k+3)) for Gamma = Gamma0 e^t (beta = 1), d = 3, w = norm * r^p.
    A/V: p=0, k=0 -> 2/3.  Q_head: p=1, norm=1/3, k=0 -> 2/3.  Q_beta: p=1, norm=1/3, k=3 -> 1/3."""
    m = norm * radial_moment(p)
    return m ** 2 / (2 * (k + 3))


def surface_distributions():
    """Exact surface statistics for the exponential rate, d = 3 [DERIVED, appendix C.1]. On the first-collision surface the
    joint density per unit area factorises: u = 8 pi Gamma0 e^t ~ u e^{-u} (Gamma(2)), r1, r2 iid Gamma(3, 1), h ~ 3 h^2.
    Returns the means that the MC reports or that enter the mass reach."""
    return dict(
        mean_r_area=3.0,          # <r> per unit area (Gamma(3))
        mean_h_area=3 / 4,        # int h * 3h^2
        mean_h_slab=2 / 3,        # MC sample mean: slab points weigh dA/(2h) -> density 2h
        mean_r_slab=3.0,          # r independent of h
        mean_r_yield=4.0,         # weight r1 r2 h^3 dA: r ~ Gamma(4)
        mean_h_yield=6 / 7,       # weight h^3 dA: h ~ 6 h^5
    )


# ---------------------------------------------------------------- independent 1D Monte Carlo (test of the conditioning argument)
def mc_1d_collisions(rng, length, t_lo=-25.0, t_hi=3.5, gamma0=1.0):
    """Direct simulation in d = 1 on a ring of circumference `length` (units 1/beta) with Gamma = gamma0 e^t per unit
    length and time, nucleating only in the false vacuum (candidate kept iff not inside an earlier real bubble).
    Returns (n_bubbles, r_left, r_right): for each pair of neighbouring real bubbles the radii at their first collision.
    Used to test eq. (M) in d = 1, including the claim that the empty-past-light-cone condition alone selects
    first collisions of real bubbles. Requires length >> 1 so that wrap-around is irrelevant."""
    import bisect
    n_cand = rng.poisson(gamma0 * (np.exp(t_hi) - np.exp(t_lo)) * length)
    tc = np.sort(np.log(np.exp(t_lo) + rng.uniform(0, np.exp(t_hi) - np.exp(t_lo), n_cand)))
    xc = rng.uniform(0, length, n_cand)
    xs, ts = [], []                                       # real bubbles, sorted by position
    for t, x in zip(tc, xc):
        j = bisect.bisect(xs, x)
        inside = False
        for i in (j - 1, j % len(xs) if xs else 0):       # nearest real neighbours on each side decide coverage
            if xs:
                d = abs(x - xs[i]); d = min(d, length - d)
                inside |= d < t - ts[i]
        if not inside:
            xs.insert(j, x); ts.insert(j, t)
    xs = np.array(xs); ts = np.array(ts)
    D = np.diff(np.r_[xs, xs[0] + length])                # gap from bubble i to its right neighbour
    tn = np.r_[ts[1:], ts[0]]
    a = 0.5 * (D + tn - ts)                               # t_i + a = t_j + (D - a)
    return len(xs), a, D - a
