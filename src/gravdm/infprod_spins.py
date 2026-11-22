"""Gravitational production of a Dirac fermion and of a massive (Proca) vector of mass M at the inflation->kination
transition, on the SAME Chun-Scopel-Zaballa tanh(w) background as the minimal scalar (infprod.py); appendix D.4 of the paper.
Labels (S2)-(S14) mark the equations of the derivation summarised below; they are internal to this module.

Units: H0 = 1 (inflationary Hubble rate), a = e^y with a(y=0) = 1, conformal time eta (d eta = dy/(aH)), comoving k [H0], M [H0].
Background [LIT: CSZ 0904.0675 eqs. (3.5)-(3.9)], evaluated ANALYTICALLY here (infprod.Background tabulates the same formulas):
  w(y) = (wf-1)/2 + (wf+1)/2 tanh(2y/y0),  H = exp{-(3/4)(1+wf)[y + (y0/2)(ln cosh(2y/y0) + ln 2)]},  a''/a = (aH)^2 (1-3w)/2.

Mode equations [DERIVED]:
  Dirac (eq. S2):       i u_A' = m u_A + k u_B,  i u_B' = k u_A - m u_B,  m = a M,  |u_A|^2 + |u_B|^2 = 1.
  Proca transverse (S7): chi'' + (k^2 + m^2) chi = 0.
  Proca longitudinal (S9): phi'' + (k^2 + m^2 - f''/f) phi = 0,  f = m / sqrt(k^2 + m^2)   (phi = f A_L).
  Scalar-type modes reuse the minimal-scalar out-state formula; all |beta|^2 use FIRST-ORDER ADIABATIC out-states (S5, S11),
  which push the adiabatic floor from O(eps^2) to O(eps^4) (eps = omega'/omega^2); the zeroth-order value is returned too.
Number densities (comoving, a(0) = 1, units H0^3) (S6, S12):
  Dirac: n_X a^3 = (1/pi^2) int dk k^2 |beta_k|^2 (particles only, both helicities); n_{X+Xbar} = 2 n_X.
  Proca (real field): n a^3 = (1/(2 pi^2)) int dk k^2 (2 |beta_T|^2 + |beta_L|^2).
"""
import math
import numpy as np
from scipy import integrate


class AnalyticBackground:
    """Analytic CSZ tanh background (same formulas as infprod.Background). All quantities in H0 = 1 units."""

    def __init__(self, y0=1.0, wf=1.0):
        self.y0, self.wf = y0, wf

    @staticmethod
    def _logcosh(x):
        ax = abs(x)
        return ax + math.log1p(math.exp(-2 * ax)) - math.log(2.0)

    def H(self, y):
        y0, wf = self.y0, self.wf
        return math.exp(-0.75 * (1 + wf) * (y + 0.5 * y0 * (self._logcosh(2 * y / y0) + math.log(2.0))))

    def w(self, y):
        return (self.wf - 1) / 2 + (self.wf + 1) / 2 * math.tanh(2 * y / self.y0)

    def calH(self, y):
        """conformal Hubble aH = a'/a."""
        return math.exp(y) * self.H(y)

    def U(self, y):
        """a''/a = (aH)^2 (1 - 3w)/2  (CSZ eq. 3.9)."""
        h = self.calH(y)
        return h * h * (1 - 3 * self.w(y)) / 2

    def late_C(self, y=6.0):
        """H a^3 deep in kination (H = C a^-3); equals infprod.Background.late_C."""
        return self.H(y) * math.exp(3 * y)


# ------------------------------------------------------------------ Dirac
def _dirac_spinors(m, k):
    om = math.sqrt(k * k + m * m)
    return om, math.sqrt((om + m) / (2 * om)), math.sqrt((om - m) / (2 * om))


def beta2_dirac(bg, k, M, y_start=None, y_outs=(3.0,), rtol=1e-10, return_raw=False, in_order=1):
    """|beta_k|^2 of a Dirac fermion (one helicity; both helicities and X, Xbar are equal) [eqs. S2-S5].

    Integrates i u_A' = m u_A + k u_B, i u_B' = k u_A - m u_B (m = a M) in y, from the instantaneous positive-frequency
    spinor (u_A, u_B) = (sqrt((w+m)/2w), sqrt((w-m)/2w)) at y_start (deep inside the horizon, adiabatic) to each y_out.
    in_order = 1 (default): first-order adiabatic in-state u = A u_+ + B u_- with B = i phi' A/(2w) (inverse of eq. S5),
    which removes the O(eps_in^2) floor of the zeroth-order start (|beta|^2 ~ 1e-12 at k/aH = e^7, relevant for n a^3 < 1e-9).
    Out-state: decompose u = A u_+ + B u_- in instantaneous eigenspinors; zeroth order |beta|^2 = |B|^2 (eq. S4,
    = 1/2 - (m/2w)(|u_A|^2-|u_B|^2) - (k/w) Re(u_A u_B^*)); first-order adiabatic (eq. S5): |B - i phi' A/(2w)|^2,
    phi' = -m' k/(2 w^2), m' = aH m.  Returns array over y_outs (first-order); with return_raw also zeroth order.
    Units: k, M in H0 (a(0) = 1). Valid for any k, M with adiabatic in/out states (checked by varying y_out).
    """
    ys = max(min(math.log(k) - 8, -5.0), -13.0) if y_start is None else y_start
    m0 = math.exp(ys) * M
    om0, cp0, sp0 = _dirac_spinors(m0, k)
    Bin = 1j * (-(bg.calH(ys) * m0) * k / (2 * om0 * om0)) / (2 * om0) if in_order == 1 else 0.0
    Ain = math.sqrt(1 - abs(Bin) ** 2)
    uA0 = Ain * cp0 + Bin * sp0; uB0 = Ain * sp0 - Bin * cp0          # u = A u_+ + B u_-
    H, kk = bg.H, k

    def rhs(yy, s):
        a = math.exp(yy); f = 1.0 / (a * H(yy)); m = a * M
        uA = complex(s[0], s[1]); uB = complex(s[2], s[3])
        dA = -1j * f * (m * uA + kk * uB); dB = -1j * f * (kk * uA - m * uB)
        return [dA.real, dA.imag, dB.real, dB.imag]

    sol = integrate.solve_ivp(rhs, (ys, max(y_outs)), [uA0.real, uA0.imag, uB0.real, uB0.imag], method="DOP853", rtol=rtol, atol=1e-13,
                              t_eval=sorted(y_outs), dense_output=False)
    out, raw = [], []
    for j, ye in enumerate(sorted(y_outs)):
        uA = complex(sol.y[0, j], sol.y[1, j]); uB = complex(sol.y[2, j], sol.y[3, j])
        a = math.exp(ye); m = a * M
        om, cp, sp = _dirac_spinors(m, k)
        A = cp * uA + sp * uB            # overlap with u_+ = (cos phi, sin phi)
        B = sp * uA - cp * uB            # overlap with u_- = (sin phi, -cos phi)
        dphi = -(bg.calH(ye) * m) * k / (2 * om * om)
        raw.append(abs(B) ** 2)
        out.append(abs(B - 1j * dphi * A / (2 * om)) ** 2)
    out, raw = np.array(out), np.array(raw)
    return (out, raw) if return_raw else out


# ------------------------------------------------------------------ scalar-type modes (Proca T, L; minimal scalar)
def omega2_fn(bg, k, M, mode):
    """omega^2(y) for scalar-type modes: 'T': k^2+m^2 (S7); 'L': k^2+m^2-f''/f (S9-S10); 'S': k^2+m^2-a''/a."""
    k2 = k * k

    def om2(y):
        a = math.exp(y); m2 = (a * M) ** 2
        if mode == "T":
            return k2 + m2
        U = bg.U(y)
        if mode == "S":
            return k2 + m2 - U
        h2 = bg.calH(y) ** 2
        D = k2 + m2
        ff = (U - h2) * k2 / D - 2 * h2 * k2 * m2 / D ** 2 + h2 * k2 * k2 / D ** 2   # f''/f, eq. S10
        return k2 + m2 - ff
    return om2


def beta2_scalar_type(bg, k, M, mode, y_start=None, y_outs=(3.0,), rtol=1e-10, return_raw=False, dy_fd=1e-5, in_order=1):
    """|beta_k|^2 for chi'' + omega^2(eta) chi = 0 with omega^2 from omega2_fn (mode 'T', 'L' or 'S').

    In-state: adiabatic at y_start; in_order = 1 (default): first-order, chi = 1/sqrt(2w), chi' = (-i w - w'/(2w)) chi;
    in_order = 0: zeroth order (as infprod.beta2). Out-state (eq. S11): first-order adiabatic, |beta|^2 = (w/2)|chi - i D/w|^2,
    D = chi' + (w'/(2w)) chi, w' = aH dw/dy (central difference in y, step dy_fd). Zeroth order: same with w' = 0
    (= (|chi'|^2 + w^2|chi|^2)/(2w) - 1/2 used by infprod.beta2). Units as in beta2_dirac.
    """
    ys = max(min(math.log(k) - 8, -5.0), -13.0) if y_start is None else y_start
    om2 = omega2_fn(bg, k, M, mode)
    H = bg.H
    w0 = math.sqrt(om2(ys))
    c0 = 1 / math.sqrt(2 * w0)
    dw0 = bg.calH(ys) * (math.sqrt(om2(ys + dy_fd)) - math.sqrt(om2(ys - dy_fd))) / (2 * dy_fd) if in_order == 1 else 0.0

    def rhs(yy, s):
        f = 1.0 / (math.exp(yy) * H(yy)); o2 = om2(yy)
        return [f * s[2], f * s[3], -f * o2 * s[0], -f * o2 * s[1]]

    sol = integrate.solve_ivp(rhs, (ys, max(y_outs)), [c0, 0.0, -dw0 / (2 * w0) * c0, -w0 * c0], method="DOP853", rtol=rtol,
                              atol=1e-14 * c0, t_eval=sorted(y_outs))
    out, raw = [], []
    for j, ye in enumerate(sorted(y_outs)):
        c = complex(sol.y[0, j], sol.y[1, j]); dc = complex(sol.y[2, j], sol.y[3, j])
        w = math.sqrt(om2(ye))
        dw = bg.calH(ye) * (math.sqrt(om2(ye + dy_fd)) - math.sqrt(om2(ye - dy_fd))) / (2 * dy_fd)
        raw.append(w / 2 * abs(c - 1j * dc / w) ** 2)
        D = dc + dw / (2 * w) * c
        out.append(w / 2 * abs(c - 1j * D / w) ** 2)
    out, raw = np.array(out), np.array(raw)
    return (out, raw) if return_raw else out


# ------------------------------------------------------------------ number densities
def na3_dirac(bg, M, ks, y_outs=(3.0,), total=True, **kw):
    """Comoving number density [H0^3] (eq. S6). total=True: X + Xbar, 2 helicities: (2/pi^2) int dk k^2 |beta|^2;
    total=False: particles only, (1/pi^2) int. Trapezoid in k on the supplied grid. Returns (na3 per y_out, b2 table)."""
    b2 = np.array([beta2_dirac(bg, k, M, y_outs=y_outs, **kw) for k in ks])        # (nk, ny)
    pref = (2 if total else 1) / np.pi ** 2
    return pref * np.trapezoid(ks[:, None] ** 2 * b2, ks, axis=0), b2


def na3_proca(bg, M, ks, y_outs=(3.0,), **kw):
    """Comoving number density of a real Proca field [H0^3] (eq. S12): (1/(2 pi^2)) int dk k^2 (2|b_T|^2 + |b_L|^2).
    Returns (n_T, n_L) arrays over y_outs (n_T includes both transverse polarisations) and the b2 tables."""
    bT = np.array([beta2_scalar_type(bg, k, M, "T", y_outs=y_outs, **kw) for k in ks])
    bL = np.array([beta2_scalar_type(bg, k, M, "L", y_outs=y_outs, **kw) for k in ks])
    pref = 1 / (2 * np.pi ** 2)
    nT = pref * np.trapezoid(ks[:, None] ** 2 * 2 * bT, ks, axis=0)
    nL = pref * np.trapezoid(ks[:, None] ** 2 * bL, ks, axis=0)
    return nT, nL, bT, bL


# ------------------------------------------------------------------ Born approximation (checks)
def born_dirac(bg, k, M, y_lo=-14.0, y_hi=8.0, n=400001):
    """First-order (Born) Dirac result for a M << k throughout [eq. S13]:
    beta_k ~ (M/(2k)) int a' e^{-2ik eta} d eta = (M/(2k)) (1/(2ik)) int a'' e^{-2ik eta} d eta  (boundary term at eta_out
    dropped = first-order adiabatic out-state).  a'' d eta = a U dy/(aH) = U/H dy.  Returns |beta_k|^2."""
    y = np.linspace(y_lo, y_hi, n)
    H = np.array([bg.H(v) for v in y]); U = np.array([bg.U(v) for v in y]); a = np.exp(y)
    eta = integrate.cumulative_trapezoid(1 / (a * H), y, initial=0.0)
    I = integrate.simpson(U / H * np.exp(-2j * k * eta), x=y)
    return (M / (2 * k)) ** 2 * abs(I) ** 2 / (4 * k * k)


def dirac_smallM_log_coeff():
    """Leading small-M behaviour in kination (H = H0 a^-3) [eq. S14, DERIVED]:
    n_X a^3 (particles, 2 helicities) = M^2 H0 [A ln(H0/M) + B],  A = 1/(24 pi).  Returns A."""
    return 1 / (24 * np.pi)
