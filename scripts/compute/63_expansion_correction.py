"""63: correction to the flat-space first-collision yield (main-text Eq. (5), SM Eq. (S13)) from the expansion of the universe during
the transition, for exponential nucleation in a vacuum-dominated universe.

Main text: Bubble population (a yield lower by a factor close to e^{-4H/beta}, from 0.67 at H/beta = 0.1 to 0.95 at 0.01) and the
Dark-matter paragraph (curves move up by at most 14% in T_reh). SM: Sec. S4, "Expansion during the transition" (0.669 +- 0.006,
0.857 +- 0.012, 0.951 +- 0.011 from boxes of side 24/beta and 32/beta), Table SIII.

Setup [ASSUMED, as the benchmark]: runaway transition in vacuum, latent heat L, no other energy before the transition, walls at the speed
of light carrying all released vacuum energy, released energy redshifts as radiation (a^-4), entropy conserved after the transition.
Units beta = 1 (physical time t, comoving distance, conformal time tau with a = 1 at tau = 0). H0 = h beta is the de Sitter rate of
the false vacuum, H0^2 = L/(3 M_P^2).
Nucleation rate per physical volume and time: Gamma(t) = e^t/(8 pi) [flat: f_FV = exp(-e^t), transition at t ~ 0].
Background [DERIVED, same structure as gravdm.anli without the dark sector]:
  f(tau) = exp[-(4 pi/3) int Gamma(t(tau')) a(tau')^4 (tau - tau')^3 dtau'],
  rho_rad/L = a^-4 int a^4 (-df),  H = h sqrt(f + rho_rad/L),  dln a/dtau = a H,  dt/dtau = a.
Yield [same derivation as script 03 and gravdm.bubbles_mc]: n_X a^3 = 8 c0 L^2 Q/(M_P^4 beta) with Q = <a^2 e1 e2 h^3> over
first-collision surfaces (comoving), e from the released vacuum energy, and s a^3 from c_hat = lim a^4 rho_rad/L. Relative to the flat
formula (Q_flat -> Q_beta = 1/3, all L into radiation at T_reh):
  Y_exp/Y_flat = (Q_exp/Q_flat) * c_hat^(-3/4)      (fixed latent heat L and fixed beta of the rate).
Variance reduction [DERIVED]: paired Poisson processes. Points (u, x) of one unit-rate process in (cumulative weight, comoving space)
are mapped through each background's inverse cumulative weight, so flat and expanding configurations share their random numbers.
Checks: h -> 0 gives ratio 1 exactly per seed (same configuration); box size, surface thickness eps, grid resolution.
Settings can be passed as a JSON argument (the two stored runs are listed in README.md).
Output: output/63_expansion_correction{,_box32}.npz and .log"""
import sys, pathlib, time, json
import numpy as np
from scipy import integrate
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import bubbles_mc as mc

CFG = dict(hs=[0.1, 0.03, 0.01], box=24.0, seeds=list(range(1, 9)), eps=0.02, npts=400_000, chunk=5_000,
           tspan=(-14.0, 14.0), ngrid=6001, iters=200, tol=1e-11, fcut=1e-12, tag="")
if len(sys.argv) > 1:
    CFG.update(json.loads(sys.argv[1]))


def background(h):
    """Self-consistent a(tau), t(tau), f(tau), c_hat for de Sitter rate h (units beta = 1). h = 0: flat (a = 1)."""
    tau = np.linspace(*CFG["tspan"], CFG["ngrid"]); dtau = tau[1] - tau[0]
    if h == 0:
        a = np.ones_like(tau); t = tau.copy()
        f = np.exp(-np.exp(t)); return dict(tau=tau, a=a, t=t, f=f, c_hat=1.0, w=np.exp(t) / (8 * np.pi))
    a = np.exp(h * tau)
    for it in range(CFG["iters"]):
        t = integrate.cumulative_trapezoid(a, tau, initial=0.0); t -= np.interp(0.0, tau, t)
        w = np.exp(t) / (8 * np.pi) * a ** 4                       # nucleation per comoving 4-volume
        M = [np.cumsum(w * tau ** k) * dtau for k in range(4)]
        I = np.maximum(tau ** 3 * M[0] - 3 * tau ** 2 * M[1] + 3 * tau * M[2] - M[3], 0.0)
        f = np.exp(-(4 * np.pi / 3) * I)
        rel = np.cumsum(a ** 4 * (-np.gradient(f, dtau))) * dtau / a ** 4
        H = h * np.sqrt(f + rel)
        a_new = np.exp(integrate.cumulative_trapezoid(a * H, tau, initial=0.0)); a_new /= np.interp(0.0, tau, a_new)
        err = np.max(np.abs(a_new / a - 1)); a = a_new
        if err < CFG["tol"]:
            break
    t = integrate.cumulative_trapezoid(a, tau, initial=0.0); t -= np.interp(0.0, tau, t)
    w = np.exp(t) / (8 * np.pi) * a ** 4
    c_hat = a[-1] ** 4 * (rel[-1] + f[-1])
    return dict(tau=tau, a=a, t=t, f=f, c_hat=c_hat, w=w, iters=it + 1, err=err)


def make_bg(sol):
    bg = mc.Background(dict(tau=sol["tau"], a=sol["a"]))
    dtau = sol["tau"][1] - sol["tau"][0]
    w = np.where(sol["f"] > CFG["fcut"], sol["w"], 0.0)          # no false vacuum left: later nucleations are irrelevant
    bg.cum_a4 = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1])) * dtau])
    return bg


def nucleate_paired(bg, U, X, Lbox):
    """Map shared unit-rate points (U sorted, X) through the inverse cumulative weight of bg; keep those in the false vacuum."""
    tot = bg.cum_a4[-1] * Lbox ** 3
    m = U < tot
    tn = np.interp(U[m] / Lbox ** 3, bg.cum_a4, bg.tau); xn = X[m]
    keep_t, keep_x = [], []
    for tt, xx in zip(tn, xn):
        if keep_t:
            d = np.abs(np.array(keep_x) - xx); d = np.minimum(d, Lbox - d)
            if np.any(np.sqrt((d ** 2).sum(1)) < tt - np.array(keep_t)):
                continue
        keep_t.append(tt); keep_x.append(xx)
    return np.array(keep_t), np.array(keep_x)


if __name__ == "__main__":
    Lb = CFG["box"]; log = []
    sols = {0.0: background(0.0)}
    for h in CFG["hs"]:
        sols[h] = background(h)
        s = sols[h]
        log.append(f"h={h}: background iters={s['iters']} err={s['err']:.1e} c_hat={s['c_hat']:.4f} a(end)={s['a'][-1]:.3f} f(end)={s['f'][-1]:.1e}")
        print(log[-1], flush=True)
    bgs = {h: make_bg(s) for h, s in sols.items()}
    nexp = {h: float(np.trapezoid(sols[h]["w"] * sols[h]["f"], sols[h]["tau"])) for h in sols}   # exact <N_b>/V (comoving) [DERIVED]
    log.append("exact bubble density n_exp: " + ", ".join(f"h={h}: {v:.5f}" for h, v in nexp.items()) + f" (flat 1/(8 pi) = {1/(8*np.pi):.5f})")
    print(log[-1], flush=True)
    Umax = max(b.cum_a4[-1] for b in bgs.values()) * Lb ** 3
    rows = []
    for sd in CFG["seeds"]:
        rng = np.random.default_rng(sd)
        N = rng.poisson(Umax); U = np.sort(rng.uniform(0, Umax, N)); X = rng.uniform(0, Lb, (N, 3))
        Q = {}; NB = {}
        for h, bg in bgs.items():
            t0 = time.time()
            tn, xn = nucleate_paired(bg, U, X, Lb)
            q = mc.surface_Q(bg, tn, xn, Lb, np.random.default_rng(sd + 1000), npts=CFG["npts"], eps=CFG["eps"], chunk=CFG["chunk"])
            Q[h] = q["Q"]; NB[h] = len(tn) / Lb ** 3
            print(f"seed {sd} h={h}: bubbles={len(tn)} Q={q['Q']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        for h in CFG["hs"]:
            r = Q[h] / Q[0.0] * sols[h]["c_hat"] ** (-0.75)
            rows.append([h, sd, Q[0.0], Q[h], sols[h]["c_hat"], r, NB[0.0], NB[h]])
            log.append(f"seed {sd} h={h}: Q_flat={Q[0.0]:.4f} Q_exp={Q[h]:.4f} c_hat={sols[h]['c_hat']:.4f} ratio={r:.4f}")
            print(log[-1], flush=True)
    rows = np.array(rows)

    def cv(Qs, ns, n0):
        """Control variate Q - b (n_b - n_exp), b fitted on the seeds; jackknife error with refitted b [as script 11]."""
        def est(Q_, n_):
            b = np.cov(Q_, n_, ddof=1)[0, 1] / np.var(n_, ddof=1); return np.mean(Q_ - b * (n_ - n0)), b
        m, b = est(Qs, ns); N = len(Qs)
        jk = np.array([est(np.delete(Qs, i), np.delete(ns, i))[0] for i in range(N)])
        return m, np.sqrt((N - 1) / N * np.sum((jk - jk.mean()) ** 2)), b
    r0 = rows[rows[:, 0] == CFG["hs"][0]]
    qf, dqf, bf = cv(r0[:, 2], r0[:, 6], nexp[0.0])
    line = f"FLAT check: Q_beta(cv) = {qf:.4f} +- {dqf:.4f} (exact 1/3), raw {r0[:, 2].mean():.4f} +- {r0[:, 2].std(ddof=1)/np.sqrt(len(r0)):.4f}"
    log.append(line); print(line, flush=True)
    for h in CFG["hs"]:
        rr = rows[rows[:, 0] == h]
        qe, dqe, be = cv(rr[:, 3], rr[:, 7], nexp[h])
        ratio = qe / (1 / 3) * sols[h]["c_hat"] ** (-0.75); dratio = ratio * dqe / qe
        ratio2 = qe / qf * sols[h]["c_hat"] ** (-0.75); dratio2 = ratio2 * np.hypot(dqe / qe, dqf / qf)
        line = (f"RESULT h={h}: Q_exp(cv) = {qe:.4f} +- {dqe:.4f}; c_hat = {sols[h]['c_hat']:.4f}; Y_exp/Y_flat = {ratio:.3f} +- {dratio:.3f} "
                f"(vs exact 1/3), {ratio2:.3f} +- {dratio2:.3f} (vs flat MC); 4h = {4 * h:.3f}")
        log.append(line); print(line, flush=True)
    tag = CFG["tag"]
    np.savez(ROOT / f"output/63_expansion_correction{tag}.npz", rows=rows, header="h seed Q_flat Q_exp c_hat ratio nb_flat nb_exp", cfg=json.dumps(CFG))
    (ROOT / f"output/63_expansion_correction{tag}.log").write_text("\n".join(log) + "\n")
