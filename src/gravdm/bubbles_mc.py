"""Monte Carlo of bubble nucleation on the An-Li background and the collision-surface integral for the yield
(sections 4.2, 5.2 and appendix C.2 of the paper).

Units: H_c = 1, a_c = 1 (conformal time tau in 1/(a_c H_c)); nucleation per comoving 4-volume = Gamma a^4 = a^4
(Gamma = H_c^4, An-Li eq. 2). Walls move at the speed of light (gamma >> 1).
Wall energy per physical area of an uncollided wall of bubble i at conformal radius r [LIT: An-Li eq. S22]:
   E/A = (L/H_c) e_hat,  e_hat(tau_i, tau) = int_{tau_i}^{tau} a^4(tau') (tau'-tau_i)^2 dtau' / (a(tau)^3 (tau-tau_i)^2)
   (flat limit a = 1: e_hat = r/3, i.e. E/A = L R/3).
First-collision surface: points where the two earliest arrival times t_k(x) = tau_k + |x - x_k| coincide.
   Q = (1/V) int dA_com a(tau*)^2 e1 e2 h^3 = < delta_eps(t1 - t2) |n1 - n2| a^2 e1 e2 h^3 >_x     [eq. (5.3), appendix C.2]
   h = |n1 - n2|/2: oblique-collision factor [DERIVED, section 4.1]. Events per unit locus area for null sheets
   crossing at angle theta: (N1/A)(N2/A) sigma(s) (1 - cos theta)/|n1 - n2| = (N1/A)(N2/A) sigma h, with s = 4 p1 p2 h^2,
   so for sigma ∝ s: weight h^3 relative to head-on, and the mass cut-off scale gamma m -> gamma m h.
   (Q_head, without h^3, is returned for comparison with head-on collisions.)
Yield (2 X per event, sigma = c0 s/M_P^4 for M << gamma m) [eq. (5.3)]:
   n_X a^3 = 8 c0 L^2 Q / (M_P^4 H_c)    (comoving, a_c = 1)
"""
import numpy as np


class Background:
    def __init__(self, sol):
        self.tau = sol["tau"]; self.a = sol["a"]
        dt = self.tau[1] - self.tau[0]
        self.cum_a4 = np.concatenate([[0], np.cumsum(0.5 * (self.a[1:] ** 4 + self.a[:-1] ** 4)) * dt])

    def a_of(self, t):
        return np.interp(t, self.tau, self.a)

    def e_hat(self, ti, t, n=200):
        """Vectorised e_hat for arrays ti, t (t > ti)."""
        ti = np.asarray(ti, float); t = np.asarray(t, float)
        s = np.linspace(0, 1, n)[None, :]
        tp = ti[:, None] + (t - ti)[:, None] * s
        integrand = self.a_of(tp) ** 4 * (tp - ti[:, None]) ** 2
        I = np.trapezoid(integrand, dx=1.0 / (n - 1), axis=1) * (t - ti)
        r = t - ti
        return I / (self.a_of(t) ** 3 * r ** 2)


def nucleate(bg, Lbox, rng, tau_max=None):
    """Poisson nucleations with density a^4 per comoving 4-volume in the false vacuum of a periodic box."""
    tau = bg.tau
    tau_max = tau[-1] if tau_max is None else tau_max
    m = tau <= tau_max
    Ntot = rng.poisson(bg.cum_a4[m][-1] * Lbox ** 3)
    u = rng.uniform(0, bg.cum_a4[m][-1], Ntot)
    tn = np.sort(np.interp(u, bg.cum_a4[m], tau[m]))
    xn = rng.uniform(0, Lbox, (Ntot, 3))
    keep_t, keep_x = [], []
    for t, x in zip(tn, xn):
        if keep_t:
            d = np.abs(np.array(keep_x) - x); d = np.minimum(d, Lbox - d)
            if np.any(np.sqrt((d ** 2).sum(1)) < t - np.array(keep_t)):
                continue                                   # inside an existing bubble
        keep_t.append(t); keep_x.append(x)
    return np.array(keep_t), np.array(keep_x)


def surface_Q(bg, tn, xn, Lbox, rng, npts=2_000_000, eps=0.01, chunk=100_000):
    """Monte Carlo of Q [eq. (5.3)] and the collision-area density A/V (comoving), plus diagnostics."""
    acc_Q = 0.0; acc_Qh = 0.0; acc_A = 0.0; nsurf = 0
    samples = []
    for s in range(0, npts, chunk):
        x = rng.uniform(0, Lbox, (min(chunk, npts - s), 3))       # exactly npts points, also when chunk does not divide npts
        d = x[:, None, :] - xn[None, :, :]
        d -= Lbox * np.round(d / Lbox)
        r = np.sqrt((d ** 2).sum(2))
        t = tn[None, :] + r
        i2 = np.argpartition(t, 1, axis=1)[:, :2]
        t1 = np.take_along_axis(t, i2[:, :1], 1)[:, 0]; t2 = np.take_along_axis(t, i2[:, 1:2], 1)[:, 0]
        sel = np.abs(t1 - t2) < eps / 2
        if not np.any(sel):
            continue
        j1 = i2[sel, 0]; j2 = i2[sel, 1]; ts = 0.5 * (t1[sel] + t2[sel])
        n1 = d[sel, j1] / r[sel, j1][:, None]; n2 = d[np.where(sel)[0], j2] / r[sel, j2][:, None]
        gradn = np.linalg.norm(n1 - n2, axis=1)
        e1 = bg.e_hat(tn[j1], ts); e2 = bg.e_hat(tn[j2], ts)
        a2 = bg.a_of(ts) ** 2
        h = gradn / 2
        acc_Q += np.sum(gradn * a2 * e1 * e2 * h ** 3) / eps
        acc_Qh += np.sum(gradn * a2 * e1 * e2) / eps
        acc_A += np.sum(gradn) / eps
        nsurf += sel.sum()
        samples.append(np.c_[ts, r[sel, j1], r[sel, j2], e1, e2, gradn * a2 / eps, h])   # col 5: surface weight, col 6: h
    return dict(Q=acc_Q / npts, Q_head=acc_Qh / npts, AV=acc_A / npts, nsurf=nsurf, samples=np.concatenate(samples) if samples else None)


def exit_distance(tk, xk, n, tn, xn, Lbox, k):
    """Distance s* along the ray x_k + s n at which bubble k's wall first meets another bubble's wall (it then enters true vacuum and
    stops gaining energy) [DERIVED: tau_k + s = tau_l + |x_k + s n - x_l|  =>  s = (|d|^2 - dt^2)/(2 dt - 2 d.n), d = x_k - x_l,
    dt = tau_k - tau_l; physical roots have s > 0 and tau_k + s > tau_l]. All 27 periodic images of each other bubble (and the
    26 non-trivial images of bubble k itself) are tried: the minimum image relative to the centre need not be the first wall met
    along a ray (example: bubbles at (0,3,3), (2,3,3), L = 6, ray -x: exit at 2 via the image at (-4,3,3), not inf; tested).
    Vectorised over rays (n: (m,3))."""
    d0 = xk[None, :] - xn                                          # (N,3)
    d0 -= Lbox * np.round(d0 / Lbox)
    dt = tk - tn                                                   # (N,)
    best = np.full(n.shape[0], np.inf)
    for sh in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 3):
        d = d0 + Lbox * sh[None, :]
        dd = (d ** 2).sum(1)
        dn = n @ d.T                                               # (m,N)
        den = 2 * dt[None, :] - 2 * dn
        num = dd[None, :] - dt[None, :] ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            s = num / den
        ok = (den > 0) & (s > 0)
        if not np.any(sh):
            ok[:, k] = False                                       # bubble k itself (zero shift)
        best = np.minimum(best, np.where(ok, s, np.inf).min(1))
    return best


def e_hat_capped(bg, ti, t, scap, n=200):
    """Shell energy per physical area at conformal time t for a wall nucleated at ti, which gains energy only up to radius scap
    (its exit into true vacuum): int_{ti}^{ti+min(scap, t-ti)} a^4 (tau'-ti)^2 dtau' / (a(t)^3 (t-ti)^2)   [DERIVED; scap=inf -> e_hat]."""
    ti = np.asarray(ti, float); t = np.asarray(t, float)
    up = np.minimum(t - ti, scap)
    s = np.linspace(0, 1, n)[None, :]
    tp = ti[:, None] + up[:, None] * s
    I = np.trapezoid(bg.a_of(tp) ** 4 * (tp - ti[:, None]) ** 2, dx=1.0 / (n - 1), axis=1) * up
    return I / (bg.a_of(t) ** 3 * (t - ti) ** 2)


def crossing_Q(bg, tn, xn, Lbox, rng, npts=400_000, eps=0.02, chunk=10_000, tau_end=None, kths=(np.inf, 100.0, 10.0, 3.0), rmax=None, G=12):
    """All shell crossings (free passage): every pair of walls arriving simultaneously at x contributes a^2 e_k e_l h^3 (as in
    surface_Q), with e from e_hat_capped (energy gain stops at the exit point). Generation g = rank of the later shell's arrival at x
    (g = 1: first collision). Degradation of the hard component: weight f_k f_l with f = exp(-rank/k_th) [ASSUMED model].
    Returns dict: Q_by_gen (up to g = G) for k_th = inf, and Q_tot per k_th. rmax: max radius trusted (periodic images): Lbox/2."""
    rmax = Lbox / 2 if rmax is None else rmax
    Qg = np.zeros(G + 1); Qk = {kt: 0.0 for kt in kths}
    for s0 in range(0, npts, chunk):
        x = rng.uniform(0, Lbox, (min(chunk, npts - s0), 3))        # exactly npts points (see surface_Q)
        d = x[:, None, :] - xn[None, :, :]
        d -= Lbox * np.round(d / Lbox)
        r = np.sqrt((d ** 2).sum(2))
        t = tn[None, :] + r
        order = np.argsort(t, axis=1)
        ts = np.take_along_axis(t, order, 1)
        for g in range(1, G + 1):                                  # consecutive arrivals (rank g-1, g)
            gap = ts[:, g] - ts[:, g - 1]
            sel = np.abs(gap) < eps / 2
            if tau_end is not None:
                sel &= ts[:, g] < tau_end
            if not np.any(sel):
                continue
            rows = np.where(sel)[0]
            j1 = order[rows, g - 1]; j2 = order[rows, g]
            r1 = r[rows, j1]; r2 = r[rows, j2]
            keep = (r1 < rmax) & (r2 < rmax)
            if not np.any(keep):
                continue
            rows, j1, j2, r1, r2 = rows[keep], j1[keep], j2[keep], r1[keep], r2[keep]
            n1 = d[rows, j1] / r1[:, None]; n2 = d[rows, j2] / r2[:, None]
            gradn = np.linalg.norm(n1 - n2, axis=1); h = gradn / 2
            tc = 0.5 * (t[rows, j1] + t[rows, j2])
            s1 = np.array([exit_distance(tn[a], xn[a], n1[i:i + 1], tn, xn, Lbox, a)[0] for i, a in enumerate(j1)])
            s2 = np.array([exit_distance(tn[b], xn[b], n2[i:i + 1], tn, xn, Lbox, b)[0] for i, b in enumerate(j2)])
            e1 = e_hat_capped(bg, tn[j1], tc, s1); e2 = e_hat_capped(bg, tn[j2], tc, s2)
            w = gradn * bg.a_of(tc) ** 2 * e1 * e2 * h ** 3 / eps
            Qg[g] += w.sum()
            for kt in kths:
                f = np.exp(-(g - 1) / kt) * np.exp(-g / kt) if np.isfinite(kt) else 1.0
                Qk[kt] += (w * f).sum()
    return dict(Q_by_gen=Qg / npts, Q_tot={kt: v / npts for kt, v in Qk.items()})
