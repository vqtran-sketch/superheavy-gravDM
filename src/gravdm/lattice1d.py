"""1+1D lattice for planar wall collisions and the Fourier transform of the stress tensor (section 3.4 and appendix E of the paper).

Model [ASSUMED for the check]: V(phi) = (lam/4)(phi^2 - v^2)^2, degenerate vacua +-v, so the walls move at constant
gamma (no acceleration). Units: m = sqrt(2 lam) v = 1, v = 1 => lam = 1/2.
Kink: phi = v tanh(gamma m (x - x_c - u t)/2), wall tension sigma = (2/3) m v^2 (Dphi = 2v).
Configuration: outside +v, middle -v; left wall moving right, right wall moving left; collide at x = 0, t = t_c.

Stress tensor (4D, planar, k_perp = 0):
  T00 = phidot^2/2 + phi'^2/2 + V,  T01 = -phidot phi',  T11 = phidot^2/2 + phi'^2/2 - V,
  T22 = T33 = phidot^2/2 - phi'^2/2 - V.
Windowed transform (appendix E.1): T~(omega,k) = int dt dx w(t) e^{i omega t - i k x} T(t,x), w = exp(-(t-t_c)^2/(2 Tw^2)).
Invariants for production with k_perp = 0 (section 3.4):
  I2 = T~*_{mu nu} T~^{mu nu} - |T~^mu_mu|^2/3   (spin-2 part; conformal sectors)   I0 = |T~^mu_mu|^2.
Parton (free-passage cross term) prediction [eq. (3.10)]:
  T~^{mu nu} = (F(a) F(b) / (2 a b)) [p1^mu p2^nu + p2^mu p1^nu - eta^{mu nu} p1.p2],  p1 = a(1,1,0,0), p2 = b(1,-1,0,0),
  a = (omega+k)/2, b = (omega-k)/2, F(p) = Dphi (pi p/(gamma m))/sinh(pi p/(gamma m)).
"""
import numpy as np

ETA4 = np.diag([1.0, -1.0, -1.0, -1.0])


def _step(phi, v):
    """Quintic smooth step S(phi) = 1 - (6t^5 - 15t^4 + 10t^3), t = (1 + phi/v)/2: S(-v) = 1, S(+v) = 0, and S', S'' vanish at +-v.
    S is held at 1 for phi < -v and at 0 for phi > +v (C^2, since S', S'' vanish there), so V stays bounded below. Returns (S, dS/dphi)."""
    t = np.clip(0.5 * (1 + phi / v), 0.0, 1.0)
    return 1 - t ** 3 * (10 - 15 * t + 6 * t * t), -15 * t * t * (1 - t) ** 2 / v


def V(phi, lam=0.5, v=1.0, eps=0.0, tilt="linear"):
    """phi^4 potential plus a tilt that makes -v a false vacuum with V(-v) - V(+v) = eps (accelerating walls, appendix E.2).
    tilt="linear": (eps/2)(1 - phi/v); the false vacuum exists only for eps < 4 lam v^4/(3 sqrt 3) (Delta V/(sigma m) < 1/sqrt 3).
    tilt="step":   eps*S(phi) (quintic smooth step, _step); +-v stay minima with mass m for any eps (Delta V/(sigma m) = O(1) reachable;
                   the barrier moves towards -v and its height falls as eps grows; S is constant outside [-v, v]).
    tilt="cubic":  (eps/4)(2 - 3x + x^3), x = phi/v: the An--Li toy potential of appendix D.2 (bounce.py, U = lam v^4 U_hat
                   with eps = lam v^4 eps_bounce). +-v stay extrema; V''(-v) = m^2 - (3/2) eps/v^2 = m^2 [1 - Delta V/(sigma m)] (tanh sigma), so
                   the false vacuum exists for Delta V/(sigma m) < 1. The paper uses tilt="cubic"."""
    if tilt == "step":
        return 0.25 * lam * (phi * phi - v * v) ** 2 + eps * _step(phi, v)[0]
    if tilt == "cubic":
        x = phi / v
        return 0.25 * lam * (phi * phi - v * v) ** 2 + 0.25 * eps * (2 - 3 * x + x ** 3)
    return 0.25 * lam * (phi * phi - v * v) ** 2 + 0.5 * eps * (1 - phi / v)


def dV(phi, lam=0.5, v=1.0, eps=0.0, tilt="linear"):
    if tilt == "step":
        return lam * phi * (phi * phi - v * v) + eps * _step(phi, v)[1]
    if tilt == "cubic":
        x = phi / v
        return lam * phi * (phi * phi - v * v) + 0.75 * eps * (x * x - 1) / v
    return lam * phi * (phi * phi - v * v) - 0.5 * eps / v


def kink(x, t, xc, u, sgn, v=1.0, m=1.0):
    """Boosted kink sgn*v*tanh(gamma m (x - xc - u t)/2) and its time derivative."""
    g = 1 / np.sqrt(1 - u * u)
    z = g * m * (x - xc - u * t) / 2
    th = np.tanh(z)
    phi = sgn * v * th
    phidot = sgn * v * (1 - th * th) * g * m * (-u) / 2
    return phi, phidot


def two_walls(x, t, x0, u, v=1.0):
    """Free-passage superposition: left wall (+v -> -v) at -x0 moving +u, right wall (-v -> +v) at +x0 moving -u."""
    pL, dL = kink(x, t, -x0, u, -1, v)
    pR, dR = kink(x, t, +x0, -u, +1, v)
    return pL + pR + v, dL + dR


def lap4(f, dx):
    """4th-order periodic Laplacian."""
    return (-np.roll(f, 2) + 16 * np.roll(f, 1) - 30 * f + 16 * np.roll(f, -1) - np.roll(f, -2)) / (12 * dx * dx)


def grad4(f, dx):
    return (np.roll(f, 2) - 8 * np.roll(f, 1) + 8 * np.roll(f, -1) - np.roll(f, -2)) / (12 * dx)


def stress(phi, phidot, dx, eps=0.0, tilt="linear"):
    ph_x = grad4(phi, dx)
    K = 0.5 * phidot ** 2; G = 0.5 * ph_x ** 2; U = V(phi, eps=eps, tilt=tilt)
    return dict(T00=K + G + U, T01=-phidot * ph_x, T11=K + G - U, T22=K - G - U)


class FTAccumulator:
    """Accumulates T~(omega_j, k_j) for a list of target points (k_j on the FFT grid)."""

    def __init__(self, x, targets_kidx, targets_omega):
        self.x = x; self.dx = x[1] - x[0]; self.N = len(x)
        self.kidx = np.asarray(targets_kidx); self.om = np.asarray(targets_omega)
        self.acc = {c: np.zeros(len(self.om), complex) for c in ("T00", "T01", "T11", "T22")}

    def add(self, T, t, weight):
        # sum_x e^{-i k x} T(x) dx : numpy fft uses e^{-2 pi i j n/N}; phase for x[0] != 0
        for c, arr in T.items():
            F = np.fft.fft(arr) * self.dx
            kgrid = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
            Fk = F[self.kidx] * np.exp(-1j * kgrid[self.kidx] * self.x[0])
            self.acc[c] += weight * np.exp(1j * self.om * t) * Fk


def tensor_from_acc(acc):
    """Assemble 4x4 T~^{mu nu} (upper indices) for each target."""
    n = len(acc["T00"])
    T = np.zeros((n, 4, 4), complex)
    T[:, 0, 0] = acc["T00"]; T[:, 0, 1] = T[:, 1, 0] = acc["T01"]; T[:, 1, 1] = acc["T11"]
    T[:, 2, 2] = T[:, 3, 3] = acc["T22"]
    return T


def invariants(T):
    """I2 = T*_{mn}T^{mn} - |tr|^2/3,  I0 = |tr|^2,  tr = eta_{mn} T^{mn}."""
    Tl = np.einsum("am,nmk,kb->nab", ETA4, T, ETA4)
    TT = np.einsum("nab,nab->n", Tl.conj(), T).real
    tr = np.einsum("mm,nmm->n", ETA4, T)
    return TT - np.abs(tr) ** 2 / 3, np.abs(tr) ** 2


def parton_tensor_F(a, b, Fa, Fb):
    """Free-passage cross-term prediction [eq. (3.10)] for given wall spectra Fa = F_R(a), Fb = F_L(b) (e.g. measured on the lattice)."""
    a = np.atleast_1d(a); b = np.atleast_1d(b)
    T = np.zeros((len(a), 4, 4), complex)
    for i, (ai, bi) in enumerate(zip(a, b)):
        p1 = ai * np.array([1, 1, 0, 0.]); p2 = bi * np.array([1, -1, 0, 0.])
        S = np.outer(p1, p2) + np.outer(p2, p1) - np.linalg.inv(ETA4) * (p1 @ ETA4 @ p2)
        T[i] = Fa[i] * Fb[i] / (2 * ai * bi) * S
    return T


def parton_tensor(a, b, gamma, dphi=2.0, m=1.0):
    """Free-passage cross-term prediction T~^{mu nu}(a, b) [eq. (3.10)] (magnitude; phase irrelevant)."""
    def F(p):
        y = np.pi * p / (gamma * m)
        return dphi * np.where(np.abs(y) < 1e-12, 1.0, y / np.sinh(np.minimum(np.abs(y), 700)))
    a = np.atleast_1d(a); b = np.atleast_1d(b)
    T = np.zeros((len(a), 4, 4), complex)
    for i, (ai, bi) in enumerate(zip(a, b)):
        p1 = ai * np.array([1, 1, 0, 0.]); p2 = bi * np.array([1, -1, 0, 0.])
        S = np.outer(p1, p2) + np.outer(p2, p1) - np.linalg.inv(ETA4) * (p1 @ ETA4 @ p2)
        T[i] = F(ai) * F(bi) / (2 * ai * bi) * S
    return T


def run(gamma, Tw=4.0, tc_over_Tw=6.0, pts_per_wall=40, cfl=0.5, sample_dt=0.01, targets=None,
        free_passage=False, margin=12.0, eps=0.0, window_center=None, snap_ts=(), tilt="linear"):
    """Evolve (or evaluate free passage) and return T~ at targets [(a, b), ...].
    pts_per_wall: grid points per lab-frame wall thickness 2/(gamma m).
    snap_ts: times at which phi(x) is stored (returned as 'snaps', list of (t, phi)); used by script 25 to measure single-wall spectra."""
    u = np.sqrt(1 - 1 / gamma ** 2)
    tc = tc_over_Tw * Tw; x0 = u * tc
    Lh = x0 + margin
    dx = 2 / (gamma * pts_per_wall)
    N = int(2 * Lh / dx); N += N % 2
    x = -Lh + dx * np.arange(N)
    dt = cfl * dx
    nsteps = int(2 * tc / dt)
    stride = max(1, int(round(sample_dt / dt)))
    kgrid = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    a_t = np.array([t[0] for t in targets]); b_t = np.array([t[1] for t in targets])
    kidx = np.array([np.argmin(np.abs(kgrid - (a - b))) for a, b in zip(a_t, b_t)])
    ksn = kgrid[kidx]
    om = a_t + b_t
    # snap: keep omega, adopt grid k -> recompute a, b
    a_s = (om + ksn) / 2; b_s = (om - ksn) / 2
    accum = FTAccumulator(x, kidx, om)
    tw0 = tc if window_center is None else window_center      # accelerating walls (eps > 0) collide before tc
    w = lambda t: np.exp(-(t - tw0) ** 2 / (2 * Tw ** 2))
    phi_prev, _ = two_walls(x, -dt, x0, u)
    phi, _ = two_walls(x, 0.0, x0, u)
    E0 = None; Elog = []; snaps = []
    for n in range(nsteps + 1):
        t = n * dt
        if free_passage:
            if n % stride == 0:
                ph, pd = two_walls(x, t, x0, u)
                T = stress(ph, pd, dx)
                accum.add(T, t, w(t) * dt * stride)
            continue
        phi_next = 2 * phi - phi_prev + dt * dt * (lap4(phi, dx) - dV(phi, eps=eps, tilt=tilt))
        for ts in snap_ts:
            if abs(t - ts) < dt / 2:
                snaps.append((t, phi.copy()))
        if n % stride == 0:
            pd = (phi_next - phi_prev) / (2 * dt)
            T = stress(phi, pd, dx, eps, tilt)
            accum.add(T, t, w(t) * dt * stride)
            E = T["T00"].sum() * dx
            if E0 is None:
                E0 = E
            Elog.append((t, E))
        phi_prev, phi = phi, phi_next
    Tt = tensor_from_acc(accum.acc)
    return dict(a=a_s, b=b_s, omega=om, k=ksn, T=Tt, Elog=np.array(Elog), dx=dx, N=N, dt=dt, tc=tc, Tw=Tw,
                gamma=gamma, phi_final=phi if not free_passage else None, x=x, snaps=snaps)
