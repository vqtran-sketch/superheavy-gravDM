"""O(4) (zero-temperature, flat-space) bounce for the tilted quartic toy model of the An-Li false vacuum (appendix D.2).

Potential [ASSUMED toy model; barrier height b L and field excursion Delta phi = 2 v = nu L^{1/4}, as in appendix D.2]:
    V(phi) = lambda v^4 U(x),   x = phi / v,
    U(x) = (x^2 - 1)^2 / 4 + eps * t(x),   eps = L / (lambda v^4) = 1 / (4 b),
  tilt shapes t(x):
    "cubic":  t = (2 - 3x + x^3)/4   (t(-1) = 1, t(1) = 0, t'(+-1) = 0: minima stay at x = -1 (false), +1 (true), Delta U = eps exactly)
    "linear": t = (1 - x)/2          (minima shift; Delta U = eps + O(eps^3) is computed numerically, Delta phi likewise)
  b = (untilted barrier height lambda v^4/4) / L;  nu = Delta phi / L^{1/4} = 2 (4 b / lambda)^{1/4}.
  The paper uses the cubic tilt.
Rescaling rho = sqrt(lambda) v r: S_E = S_hat(b) / lambda with
    S_hat = 2 pi^2 int rho^3 [ x'^2 / 2 + U(x) - U(x_false) ] d rho,    x'' + (3/rho) x' = U'(x),  x'(0) = 0,  x(inf) = x_false
  [LIT: Coleman PRD 15 (1977) 2929]. Virial (Derrick, d = 4): S_hat = (pi^2 / 2) int rho^3 x'^2 d rho (second route, tested).
Thin-wall limit (eps -> 0) [LIT: Coleman Eq. (4.20)]: S_hat = 27 pi^2 sigma^4 / (2 eps^3), sigma = int_{-1}^{1} sqrt(2 U_0) dx = 2 sqrt2 / 3.
Units: dimensionless (S_hat); physical S = S_hat / lambda = S_hat nu^4 / (64 b).
Validity: flat space (H_c << m: corrections O(H_c^2/m^2) < 1e-9 here); tree-level potential.
"""
import numpy as np
from scipy import integrate, optimize, special


def U(x, eps, tilt="cubic"):
    t = (2 - 3 * x + x ** 3) / 4 if tilt == "cubic" else (1 - x) / 2
    return (x ** 2 - 1) ** 2 / 4 + eps * t


def dU(x, eps, tilt="cubic"):
    dt = (-3 + 3 * x ** 2) / 4 if tilt == "cubic" else -0.5 * np.ones_like(x)
    return x * (x ** 2 - 1) + eps * dt


def d2U(x, eps, tilt="cubic"):
    dt2 = 1.5 * x if tilt == "cubic" else 0.0 * x
    return 3 * x ** 2 - 1 + eps * dt2


def extrema(eps, tilt="cubic"):
    """Return (x_false, x_top, x_true) or None if the false minimum does not exist (spinodal reached).
    cubic tilt: false minimum at x = -1 exists iff U''(-1) = 2 - 3 eps/2 > 0, i.e. b > 3/16."""
    # U'(x) = x^3 - x + eps t'(x): cubic -> x^3 + (3 eps/4) x^2 - x - 3 eps/4 (roots -1, -3 eps/4, +1); linear -> x^3 - x - eps/2
    r = np.roots([1.0, 0.75 * eps, -1.0, -0.75 * eps] if tilt == "cubic" else [1.0, 0.0, -1.0, -0.5 * eps])
    r = np.sort(r[np.abs(r.imag) < 1e-12].real)
    if len(r) < 3:
        return None
    xf, xt, xT = r
    if d2U(np.array(xf), eps, tilt) <= 0:
        return None
    return xf, xt, xT


def _shoot(lndelta, eps, tilt, xf, xT, rho_max):
    """Integrate from near the centre with the linearised solution x - xT = -delta * 2 I1(m rho)/(m rho) (Bessel, valid while
    |x - xT| << 1). Returns (+1 overshoot / -1 undershoot, solution)."""
    m = np.sqrt(d2U(np.array(xT), eps, tilt))
    delta = np.exp(lndelta)
    # start where the linear deviation is 1e-6 (or at rho = 1e-6 if delta is already larger)
    f = lambda z: 2 * special.iv(1, z) / z if z > 1e-8 else 1.0 + z ** 2 / 8
    z0 = 1e-6
    if delta < 1e-6:
        z0 = optimize.brentq(lambda z: np.log(delta) + np.log(f(z)) - np.log(1e-6), 1e-6, 800.0)
    r0 = z0 / m
    x0 = xT - delta * f(z0)
    # derivative: d/drho [2 I1(z)/z] = m * 2 I2(z)/z
    v0 = -delta * m * (2 * special.iv(2, z0) / z0 if z0 > 1e-8 else z0 / 4)

    def rhs(r, y):
        return [y[1], dU(y[0], eps, tilt) - 3 * y[1] / r]

    over = lambda r, y: y[0] - xf          # field passes the false vacuum: overshoot
    over.terminal = True; over.direction = -1
    under = lambda r, y: y[1]
    under.terminal = True; under.direction = 1
    sol = integrate.solve_ivp(rhs, (r0, r0 + rho_max), [x0, v0], events=[over, under], rtol=1e-11, atol=1e-13, dense_output=True)
    if sol.t_events[0].size:
        return 1, sol
    return -1, sol


def bounce(b, tilt="cubic", rho_max=None, n_bisect=200):
    """O(4) bounce for given b. Returns dict(S_hat, S_hat_virial, R (wall radius, x crosses x_top), xf, xT, ok)."""
    eps = 1 / (4 * b)
    ex = extrema(eps, tilt)
    if ex is None:
        return dict(ok=False)
    xf, xt, xT = ex
    if rho_max is None:
        rho_max = 40.0 + 12.0 / eps          # integration range beyond the start of the linearised core
    lo, hi = -80.0, np.log(0.999 * (xT - xt))      # lo: tiny delta (overshoot expected), hi: start near the top (undershoot)
    s_lo, _ = _shoot(lo, eps, tilt, xf, xT, rho_max)
    s_hi, _ = _shoot(hi, eps, tilt, xf, xT, rho_max)
    if not (s_lo == 1 and s_hi == -1):
        return dict(ok=False, reason=f"bracket {s_lo},{s_hi}")
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        s, _ = _shoot(mid, eps, tilt, xf, xT, rho_max)
        if s == 1:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    _, sol = _shoot(lo, eps, tilt, xf, xT, rho_max)       # overshooting branch: follow until it nears x_false
    r = sol.t; x = sol.y[0]; v = sol.y[1]
    # keep the part before the field gets closer than 1e-3 of the vacuum separation to x_false (tail contributes ~1e-6)
    keep = x > xf + 1e-3 * (xT - xf)
    r, x, v = r[keep], x[keep], v[keep]
    rr = np.concatenate([np.linspace(0, r[0], 50)[:-1], np.geomspace(r[0], r[-1], 20000)])
    xx = np.concatenate([np.full(49, x[0]), sol.sol(rr[49:])[0]]); vv = np.concatenate([np.zeros(49), sol.sol(rr[49:])[1]])
    Uf = U(np.array(xf), eps, tilt)
    S = 2 * np.pi ** 2 * integrate.trapezoid(rr ** 3 * (0.5 * vv ** 2 + U(xx, eps, tilt) - Uf), rr)
    Sv = 0.5 * np.pi ** 2 * integrate.trapezoid(rr ** 3 * vv ** 2, rr)
    R = np.interp(xt, xx[::-1], rr[::-1])
    return dict(ok=True, S_hat=S, S_hat_virial=Sv, R=R, x0=xT - np.exp(lo), xf=xf, xt=xt, xT=xT, eps=eps, dU=Uf - U(np.array(xT), eps, tilt),
                rho=rr, x=xx, dx=vv)      # profile x(rho), x'(rho) up to |x - xf| = 1e-3 (xT - xf)


def S_hat_thin(b):
    """Thin-wall action (cubic tilt, Delta U = eps = 1/(4b)) [LIT: Coleman Eq. (4.20)] = 6739 b^3."""
    sigma = 2 * np.sqrt(2) / 3
    return 27 * np.pi ** 2 * sigma ** 4 / (2 * (1 / (4 * b)) ** 3)
