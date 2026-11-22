"""SM renormalisation-group running of (g1, g2, g3, y_t, lambda) in MS-bar, and the electroweak-vacuum barrier during kination
(Higgs curvature coupling condition, appendix D.5 of the paper).

[LIT] RGEs: Buttazzo et al., arXiv:1307.3536, App. B, Eqs. (96)-(99) and the y_t equation (gauge, quartic, top Yukawa up to 3 loops,
      pure-QCD 4-loop term in g3). Bottom and tau Yukawas neglected (their effect on lambda(mu) is below the 3-loop truncation error
      at the precision used here; tested against Buttazzo et al. Eq. (61) in tests/test_sm_rge.py).
      Conventions: V = lambda |H|^4 = lambda h^4/4 (same as Laverda-Rubio, arXiv:2402.06000, Eq. (2.1)), g1^2 = (5/3) gY^2.
      Buttazzo et al. write d x/d ln mu^2; here t = ln(mu/GeV), so d x/dt = 2 * (their right-hand side).
[LIT] Boundary values at mu = M_t (top pole mass): Buttazzo et al. Eqs. (55), (57)-(60) (NNLO matching, linear fits around
      M_h = 125.15, M_t = 173.34, alpha_s = 0.1184, M_W = 80.384 GeV).
[DERIVED] Barrier during kination (R = -6 H^2, L ⊃ -(1/2) xi R h^2 for H^†H = h^2/2): V = -3 xi H^2 h^2 + lambda(h) h^4/4 with mu = h.
      V'(h) = h [-6 xi H^2 + h^2 (lambda + beta_lambda/4)], beta_lambda = d lambda/d ln mu, so a barrier exists iff
          xi < xi_max(H) = max_h h^2 [lambda(h) + beta_lambda(h)/4] / (6 H^2)                                  (xi_max_barrier)
      With the one-loop log approximation lambda = -(3 y^4/16 pi^2) ln(h^2/mu_L^2) this gives exactly Laverda-Rubio Eq. (3.6),
          xi_max = y_t(mu_L)^4 mu_L^2 / (32 e^{3/2} pi^2 H^2)                                                    (xi_max_LR)
      (checked in tests). Validity: h >> H (mu = h, footnote 7 of Laverda-Rubio), xi H^2 h^2 << M_P^2 H^2.
Units: GeV.
"""
import numpy as np
from scipy.integrate import solve_ivp

K = 1 / (16 * np.pi ** 2)
REF = dict(Mt=173.34, Mh=125.15, alphas=0.1184, MW=80.384)


def couplings_at_Mt(Mt=172.57, Mh=125.20, alphas=0.1180, MW=80.369):
    """MS-bar (g1, g2, g3, yt, lam) at mu = M_t [LIT: Buttazzo et al. Eqs. (55), (57)-(60)]. Masses in GeV."""
    dt, dh, da, dw = Mt - REF["Mt"], Mh - REF["Mh"], (alphas - REF["alphas"]) / 0.0007, (MW - REF["MW"]) / 0.014
    lam = 0.12604 + 0.00206 * dh - 0.00004 * dt
    yt = 0.93690 + 0.00556 * dt - 0.00042 * da
    g2 = 0.64779 + 0.00004 * dt + 0.00011 * dw
    gY = 0.35830 + 0.00011 * dt - 0.00020 * dw
    g3 = 1.1666 + 0.00314 * da - 0.00046 * dt
    return np.array([np.sqrt(5 / 3) * gY, g2, g3, yt, lam])


def _rhs_mu2(c, loops):
    """d(g1^2, g2^2, g3^2, yt^2, lam)/d ln mu^2 [Buttazzo et al. App. B, y_b = y_tau = 0]."""
    g1, g2, g3, yt, l = c
    a, b, s, y = g1 ** 2, g2 ** 2, g3 ** 2, yt ** 2
    K2, K3 = K ** 2, K ** 3
    # one loop
    dg1 = K * a * a * 41 / 10
    dg2 = K * b * b * (-19 / 6)
    dg3 = K * s * s * (-7)
    dy = K * y * (9 * y / 2 - 8 * s - 9 * b / 4 - 17 * a / 20)
    dl = K * (l * (12 * l + 6 * y - 9 * b / 2 - 9 * a / 10) - 3 * y * y + 9 * b * b / 16 + 27 * a * a / 400 + 9 * b * a / 40)
    if loops >= 2:
        dg1 += K2 * a * a * (44 * s / 5 + 27 * b / 10 + 199 * a / 50 - 17 * y / 10)
        dg2 += K2 * b * b * (12 * s + 35 * b / 6 + 9 * a / 10 - 3 * y / 2)
        dg3 += K2 * s * s * (-26 * s + 9 * b / 2 + 11 * a / 10 - 2 * y)
        dy += K2 * y * (y * (-12 * y - 12 * l + 36 * s + 225 * b / 16 + 393 * a / 80)
                        + 6 * l ** 2 - 108 * s ** 2 - 23 * b ** 2 / 4 + 1187 * a ** 2 / 600 + 9 * s * b + 19 * s * a / 15 - 9 * b * a / 20)
        dl += K2 * (l ** 2 * (-156 * l - 72 * y + 54 * b + 54 * a / 5)
                    + l * y * (-3 * y / 2 + 40 * s + 45 * b / 4 + 17 * a / 4)
                    + l * (-73 * b ** 2 / 16 + 1887 * a ** 2 / 400 + 117 * b * a / 40)
                    + y ** 2 * (15 * y - 16 * s - 4 * a / 5)
                    + y * (-9 * b ** 2 / 8 - 171 * a ** 2 / 200 + 63 * b * a / 20)
                    + 305 * b ** 3 / 32 - 3411 * a ** 3 / 4000 - 289 * b ** 2 * a / 160 - 1677 * b * a ** 2 / 800)
    if loops >= 3:
        dg1 += K3 * a * a * (y * (189 * y / 16 - 29 * s / 5 - 471 * b / 32 - 2827 * a / 800) + l * (-9 * l / 5 + 9 * b / 10 + 27 * a / 50)
                             + 297 * s ** 2 / 5 + 789 * b ** 2 / 64 - 388613 * a ** 2 / 24000 - 3 * s * b / 5 - 137 * s * a / 75 + 123 * b * a / 160)
        dg2 += K3 * b * b * (y * (147 * y / 16 - 7 * s - 729 * b / 32 - 593 * a / 160) + l * (-3 * l + 3 * b / 2 + 3 * a / 10)
                             + 81 * s ** 2 + 324953 * b ** 2 / 1728 - 5597 * a ** 2 / 1600 + 39 * s * b - s * a / 5 + 873 * b * a / 160)
        dg3 += K3 * s * s * (y * (15 * y - 40 * s - 93 * b / 8 - 101 * a / 40)
                             + 65 * s ** 2 / 2 + 109 * b ** 2 / 8 - 523 * a ** 2 / 120 + 21 * s * b + 77 * s * a / 15 - 3 * b * a / 40) \
            + K ** 4 * s ** 5 * (-2472.28)
        dy += K3 * y * (y ** 2 * (58.6028 * y + 198 * l - 157 * s - 1593 * b / 16 - 2437 * a / 80)
                        + l * y * (15 * l / 4 + 16 * s - 135 * b / 2 - 127 * a / 10)
                        + y * (363.764 * s ** 2 + 16.990 * b ** 2 - 24.422 * a ** 2 + 48.370 * s * b + 18.074 * s * a + 34.829 * b * a)
                        + l ** 2 * (-36 * l + 45 * b + 9 * a) + l * (-171 * b ** 2 / 16 - 1089 * a ** 2 / 400 + 117 * b * a / 40)
                        - 619.35 * s ** 3 + 169.829 * b ** 3 + 16.099 * a ** 3 + 73.654 * s ** 2 * b - 15.096 * s ** 2 * a - 21.072 * s * b ** 2
                        - 22.319 * s * a ** 2 - 321 / 20 * s * b * a - 4.743 * b ** 2 * a - 4.442 * b * a ** 2)
        dl += K3 * (l ** 3 * (6011.35 * l + 873 * y - 387.452 * b - 77.490 * a)
                    + l ** 2 * y * (1768.26 * y + 160.77 * s - 359.539 * b - 63.869 * a)
                    + l ** 2 * (-790.28 * b ** 2 - 185.532 * a ** 2 - 316.64 * b * a)
                    + l * y ** 2 * (-223.382 * y - 662.866 * s - 5.470 * b - 21.015 * a)
                    + l * y * (356.968 * s ** 2 - 319.664 * b ** 2 - 74.8599 * a ** 2 + 15.1443 * s * b + 17.454 * s * a + 5.615 * b * a)
                    + l * b ** 2 * (-57.144 * s + 865.483 * b + 79.638 * a) + l * a ** 2 * (-8.381 * s + 61.753 * b + 28.168 * a)
                    + y ** 3 * (-243.149 * y + 250.494 * s + 74.138 * b + 33.930 * a)
                    + y ** 2 * (-50.201 * s ** 2 + 15.884 * b ** 2 + 15.948 * a ** 2 + 13.349 * s * b + 17.570 * s * a - 70.356 * b * a)
                    + y * s * (16.464 * b ** 2 + 1.016 * a ** 2 + 11.386 * b * a) + y * b ** 2 * (62.500 * b + 13.041 * a)
                    + y * a ** 2 * (10.627 * b + 11.117 * a) + s * (7.536 * b ** 3 + 0.663 * a ** 3 + 1.507 * b ** 2 * a + 1.105 * b * a ** 2)
                    - 114.091 * b ** 4 - 1.508 * a ** 4 - 37.889 * b ** 3 * a + 6.500 * b ** 2 * a ** 2 - 1.543 * b * a ** 3)
    return dg1, dg2, dg3, dy, dl


def beta(c, loops=3):
    """d(g1, g2, g3, yt, lam)/d ln mu (t = ln mu): factor 2 from ln mu^2 and d g = d(g^2)/(2 g)."""
    dg1, dg2, dg3, dy, dl = _rhs_mu2(c, loops)
    g1, g2, g3, yt, _ = c
    return np.array([dg1 / g1, dg2 / g2, dg3 / g3, dy / yt, 2 * dl])


def run(mu_max=1.3e19, loops=3, n=600, rtol=1e-10, **inputs):
    """Couplings on a log grid mu in [M_t, mu_max] GeV. Returns (mu, C) with C[:, i] = (g1, g2, g3, yt, lam)."""
    Mt = inputs.get("Mt", 172.57)
    t = np.linspace(np.log(Mt), np.log(mu_max), n)
    sol = solve_ivp(lambda tt, c: beta(c, loops), (t[0], t[-1]), couplings_at_Mt(**inputs), t_eval=t, rtol=rtol, atol=1e-14,
                    method="DOP853")
    return np.exp(t), sol.y.T


def mu_Lambda(mu, C):
    """Instability scale: first mu where lam(mu) = 0 (log-interpolated); NaN if lam stays positive. Also y_t there."""
    lam = C[:, 4]
    i = np.where(lam < 0)[0]
    if len(i) == 0:
        return np.nan, np.nan
    j = i[0]
    x = np.log(mu[j - 1]) - lam[j - 1] * (np.log(mu[j]) - np.log(mu[j - 1])) / (lam[j] - lam[j - 1])
    return np.exp(x), np.interp(x, np.log(mu), C[:, 3])


def barrier_function(mu, C, loops=3):
    """f(h) = h^2 [lambda(h) + beta_lambda(h)/4] (GeV^2) on the grid h = mu; V' = h [f - 6 xi H^2]."""
    bl = np.array([beta(c, loops)[4] for c in C])
    return mu ** 2 * (C[:, 4] + bl / 4)


def xi_max_from_lambda(H, h, lam, blam):
    """xi_max = max_h h^2 (lam + blam/4)/(6 H^2) for tabulated lam(h), blam = d lam/d ln h (h, H in GeV)."""
    return np.max(h ** 2 * (lam + blam / 4)) / (6 * np.asarray(H, float) ** 2)


def xi_max_barrier(H, mu, C, loops=3):
    """[DERIVED] Largest xi for which the kination potential -3 xi H^2 h^2 + lambda(h) h^4/4 keeps a barrier (H in GeV).
    xi_max = max_h f(h)/(6 H^2), f = barrier_function. For xi > xi_max, V' < 0 for all h > 0 (no barrier)."""
    return np.max(barrier_function(mu, C, loops)) / (6 * np.asarray(H, float) ** 2)


def xi_max_LR(H, muL, yL):
    """[LIT] Laverda-Rubio Eq. (3.6): y^4 mu_L^2/(32 e^{3/2} pi^2 H^2) (one-loop log approximation of lambda near mu_L)."""
    return yL ** 4 * muL ** 2 / (32 * np.exp(1.5) * np.pi ** 2 * np.asarray(H, float) ** 2)


def Lambda_V(mu, C):
    """Barrier height scale (max V)^{1/4} of the RG-improved tree potential lambda(h) h^4/4 between h = M_t and the first
    zero of lambda (GeV). Buttazzo et al. Eq. (67) define it with the full NNLO effective potential; tree level differs by O(1)."""
    lam = C[:, 4]
    k = np.where(lam < 0)[0]
    if len(k) == 0:
        return np.nan
    # the potential lambda h^4/4 is positive and growing below mu_L; with xi = 0 the barrier top is where (lambda h^4)' = 0
    V = lam * mu ** 4 / 4
    return np.max(V[: k[0] + 1]) ** 0.25
