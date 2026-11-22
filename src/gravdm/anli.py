"""Background evolution of the An-Li scenario [LIT: An-Li 2606.06587, Supplemental eqs. (S1)-(S7)]; appendix D.1 of the paper.

Conformal time eta, scale factor a(eta). Units: Gamma^{1/4} = 1 (Gamma = nucleation rate per physical 4-volume, constant),
M_P (reduced) given in these units through the ratio m_P = M_P / Gamma^{1/4}.
Components:
  rho_DS = rho_DS0 a^{-n}                                                 [S2]
  rho_V  = L f_FV(eta),  f_FV(eta) = exp[-(4 pi/3) int_{-inf}^{eta} d eta' Gamma a(eta')^4 (eta - eta')^3]   [S4, S5, conformal form]
  rho_w + rho_s = a^{-4} L int^{eta} d eta' a^4(eta') (-df_FV/d eta')     [S7: released vacuum energy redshifting as radiation]
  (da/d eta) = a^2 H,  H^2 = (rho_DS + rho_V + rho_w + rho_s)/(3 M_P^2)    [S3]
Normalisation: a = 1 at the time H^4 = Gamma ("eta_c", An-Li eq. 2), where alpha = L/rho_tot.
Solved by fixed-point iteration on a(eta) (f_FV depends on the whole history).
"""
import numpy as np
from scipy import integrate


def solve(alpha, n=6, mP=1e3, eta_span=None, N=6000, iters=60, tol=1e-10):
    """Return dict with eta grid, a, H, f_FV, rho components (units Gamma = 1).
    alpha = L/rho_tot at H = 1 (i.e. H^4 = Gamma). mP = M_P/Gamma^{1/4} (sets the conformal-time unit a H).
    Conformal time is measured in units of 1/(a_c H_c) by rescaling: we work with tau = eta * a_c H_c."""
    # Units: H_c = 1, a_c = 1 at tau = 0. rho_tot(c) = 3 mP^2. L = alpha * 3 mP^2.
    rho_c = 3 * mP ** 2
    L = alpha * rho_c
    rho_DS0 = (1 - alpha) * rho_c            # value of rho_DS at a = 1 if f_FV ~ 1 there (refined below)
    # pure DS power law: a = (1 + tau/p)^p, p = 2/(n-2); a -> 0 at tau = -p (big-bang singularity of the DS era).
    # Start just after it: nucleation there is negligible (Gamma a^4 -> 0).
    p = 2.0 / (n - 2)
    if eta_span is None:
        eta_span = (-0.9 * p, 8.0)
    tau = np.linspace(*eta_span, N); dtau = tau[1] - tau[0]
    a = np.clip(1 + tau / p, 1e-6, None) ** p
    for it in range(iters):
        # nucleation per comoving 4-volume = Gamma a^4 ; Gamma = 1 in units Gamma^{1/4} = 1, but tau is in 1/(a_c H_c) = 1 units:
        # H_c^4 = Gamma -> Gamma = 1 in these units too.
        w = a ** 4
        # f_FV(tau) = exp(-(4pi/3) sum_{tau'<tau} w(tau') (tau - tau')^3 dtau)  — O(N^2) but N ~ 6000 is fine
        # sum_{tau'<tau} w (tau - tau')^3 = tau^3 M0 - 3 tau^2 M1 + 3 tau M2 - M3, M_k cumulative moments (O(N), low memory)
        M = [np.cumsum(w * tau ** k) * dtau for k in range(4)]
        I = tau ** 3 * M[0] - 3 * tau ** 2 * M[1] + 3 * tau * M[2] - M[3]
        I = np.maximum(I, 0.0)
        f = np.exp(-(4 * np.pi / 3) * I)
        df = np.gradient(f, dtau)
        rel = np.cumsum(w * (-df)) * dtau / a ** 4
        rho_V = L * f; rho_rad = L * rel
        rho_DS = rho_DS0 * a ** (-n)
        H = np.sqrt((rho_DS + rho_V + rho_rad) / (3 * mP ** 2))
        a_new = np.exp(np.concatenate([[0.0], integrate.cumulative_trapezoid(a * H, tau)]))   # dln a/dtau = a H
        i0 = np.argmin(np.abs(tau)); a_new /= a_new[i0]
        # keep H(tau=0) = 1 exactly: rescale rho_DS0 so that rho_tot(0) = 3 mP^2
        rho_DS0 *= (3 * mP ** 2 - rho_V[i0] - rho_rad[i0]) / (rho_DS0)
        err = np.max(np.abs(a_new / a - 1))
        a = a_new
        if err < tol:
            break
    return dict(tau=tau, a=a, H=H, f=f, rho_V=rho_V, rho_rad=rho_rad, rho_DS=rho_DS, L=L, alpha=alpha, n=n, iters=it + 1, err=err)
