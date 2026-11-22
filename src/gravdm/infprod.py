"""Gravitational production of a minimally coupled scalar X of mass M at the inflation->kination transition (section 5.6 and appendix D.4 of the paper).

Background [LIT: Chun-Scopel-Zaballa 0904.0675, eqs. (3.5)-(3.9)]: w(y) = (wf-1)/2 + (wf+1)/2 tanh(2y/y0), y = ln a, wf = 1;
  H(y)/H0 = exp{-(3/4)(1+wf)[y + (y0/2)(ln cosh(2y/y0) + ln 2)]}  (eq. 3.7);  a''/a = (aH)^2 (1 - 3w)/2  (eq. 3.9).
Units: H0 = 1 (inflationary Hubble rate); a = e^y (a = 1 at the centre of the transition); conformal time eta, d eta = dy/(aH).
Mode equation: chi'' + omega^2 chi = 0, omega^2 = k^2 + a^2 M^2 - a''/a (xi = 0).
In-state: adiabatic (WKB) vacuum deep in inflation at y_in (valid for k/(aH) >> 1 or M >> H0).
Out-state: adiabatic at y_out; |beta_k|^2 = (|chi'|^2 + omega^2 |chi|^2)/(2 omega) - 1/2.  Floor ~ (omega'/omega^2)^2 at y_out: checked by
varying y_out (convergence test in script 08).
Outputs: n a^3 = (1/(2 pi^2)) int dk k^2 |beta_k|^2  [H0^3, comoving with a(y=0)=1]
         massless energy  a^4 rho = (1/(2 pi^2)) int dk k^3 |beta_k|^2  (k > kIR) -> compare Chun I = a^4 rho/H0^4 (their a = 1 at end of
         inflation x = -1; for a smooth transition we quote it at our normalisation and note the O(1) normalisation dependence).
"""
import numpy as np
from scipy import integrate


class Background:
    def __init__(self, y0=1.0, wf=1.0, y_in=-12.0, y_out=4.0, n=40001):
        self.y0, self.wf = y0, wf
        y = np.linspace(y_in, y_out, n)
        self.y = y
        self.H = np.exp(-0.75 * (1 + wf) * (y + 0.5 * y0 * (np.log(np.cosh(2 * y / y0)) + np.log(2))))
        self.w = (wf - 1) / 2 + (wf + 1) / 2 * np.tanh(2 * y / y0)
        self.a = np.exp(y)
        self.eta = np.concatenate([[0.0], integrate.cumulative_trapezoid(1 / (self.a * self.H), y)])
        self.U = (self.a * self.H) ** 2 * (1 - 3 * self.w) / 2          # a''/a

    def late_C(self):
        """H a^3 at late times (kination: H = C a^-3)."""
        return self.H[-1] * self.a[-1] ** 3

    def dt_transition(self):
        """Delta t = int_{-y0}^{y0} dy / H  (Chun definition below eq. 3.14), in 1/H0."""
        m = np.abs(self.y) <= self.y0
        return np.trapezoid(1 / self.H[m], self.y[m])


def beta2(bg, k, M, y_start=None, y_end=None, rtol=1e-10):
    """|beta_k|^2 for comoving k (units H0, a(0)=1), mass M (units H0). Integrates in y with chi, dchi/deta."""
    y = bg.y
    ys = y[0] if y_start is None else y_start
    ye = y[-1] if y_end is None else y_end
    a_f = lambda yy: np.exp(yy)
    H_f = lambda yy: np.interp(yy, y, bg.H)
    U_f = lambda yy: np.interp(yy, y, bg.U)
    om2 = lambda yy: k * k + a_f(yy) ** 2 * M * M - U_f(yy)
    w0 = np.sqrt(om2(ys))
    chi0 = 1 / np.sqrt(2 * w0); dchi0 = -1j * w0 * chi0
    def rhs(yy, s):
        c = s[0] + 1j * s[1]; dc = s[2] + 1j * s[3]
        f = 1 / (a_f(yy) * H_f(yy))                  # d eta / d y
        ddc = -om2(yy) * c
        return [f * dc.real, f * dc.imag, f * ddc.real, f * ddc.imag]
    sol = integrate.solve_ivp(rhs, (ys, ye), [chi0.real, chi0.imag, dchi0.real, dchi0.imag],
                              method="DOP853", rtol=rtol, atol=1e-14 * abs(chi0))
    c = sol.y[0, -1] + 1j * sol.y[1, -1]; dc = sol.y[2, -1] + 1j * sol.y[3, -1]
    w = np.sqrt(om2(ye))
    return max((abs(dc) ** 2 + w * w * abs(c) ** 2) / (2 * w) - 0.5, 0.0)
