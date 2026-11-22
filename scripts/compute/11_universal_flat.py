"""11: geometric factor Q_beta of a runaway vacuum transition with exponential nucleation in flat space (Monte Carlo).

Main text: "a Monte Carlo of the first-collision surfaces gives 0.336 +- 0.002". SM: Sec. S4 (Q_beta = 0.336 +- 0.002 from eight
realizations in a box of side 32/beta, with a control variate). Input to scripts 49, 50 and p62.

Flat space (beta >> H), nucleation rate Gamma ∝ e^{beta t}, walls at the speed of light, all latent heat in the walls. Units beta = 1.
Wall energy per area at radius r: E/A = L r/3 (e_hat = r/3).
n_X = 8 c(0) L^2 Q_beta R / (M_P^4 beta), Q_beta including eperp^3                                [SM Eq. (S11)]
Y_X = n_X/s with L = (pi^2 g*/30) T_reh^4, s = (2 pi^2/45) g* T_reh^3, H^2 = L/(3 M_P^2):
   Y_X = C_Y c(0) Q_beta (H/beta) (T_reh/M_P)^3,  C_Y = 8 sqrt(3) a^{3/2}/b,  a = pi^2 g*/30, b = 2 pi^2 g*/45   [SM Eq. (S13)]
The exact value is Q_beta = 1/3 (SM Eq. (S12)); this Monte Carlo checks it.
Output: output/11_universal_flat.npz (raw mean Qb, dQ; control-variate Qcv, b, dQcv_naive, dQcv_jk).
Post-processing only (no Monte Carlo): python scripts/compute/11_universal_flat.py --post"""
import sys, pathlib, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import bubbles_mc as mc
from gravdm.units import G_STAR_SM

CFG = dict(boxes=[24.0, 32.0], seeds=[1, 2, 3, 4, 5, 6, 7, 8], eps=[0.02], npts=1_000_000, tspan=(-30.0, 30.0), ngrid=12001)


def flat_background():
    t = np.linspace(*CFG["tspan"], CFG["ngrid"])
    bg = mc.Background(dict(tau=t, a=np.ones_like(t)))
    # nucleation density per volume per time: Gamma = e^{t} (beta = 1, origin of t arbitrary)
    bg.cum_a4 = np.exp(t) - np.exp(t[0])
    return bg


def control_variate(rows, Lb):
    """Control-variate estimate of Q_beta over the seeds of box Lb (SM Sec. S4: 0.336 +- 0.002) [DERIVED].
    Q_cv = Q - b (n_b - 1/(8 pi)), n_b = N_bubbles/L^3, whose exact mean for exponential nucleation at v = 1 is beta^3/(8 pi);
    b = cov(Q, n_b)/var(n_b) fitted on the same seeds. Errors: naive std(Q_cv)/sqrt(N) and a leave-one-out jackknife that refits b."""
    r = rows[(rows[:, 0] == Lb)]
    Q = r[:, 4]; nb = r[:, 3] / Lb ** 3; nexp = 1 / (8 * np.pi); N = len(Q)
    def est(Qs, ns):
        b = np.cov(Qs, ns, ddof=1)[0, 1] / np.var(ns, ddof=1)
        return np.mean(Qs - b * (ns - nexp)), b
    Qcv, b = est(Q, nb)
    naive = np.std(Q - b * (nb - nexp), ddof=1) / np.sqrt(N)
    jk = np.array([est(np.delete(Q, i), np.delete(nb, i))[0] for i in range(N)])
    djk = np.sqrt((N - 1) / N * np.sum((jk - jk.mean()) ** 2))
    return dict(Qcv=Qcv, b=b, dQcv_naive=naive, dQcv_jk=djk, nb_mean=nb.mean(), nb_exp=nexp, N=N)


def post(path):
    """Add the control-variate result to an existing output without rerunning the Monte Carlo."""
    d = dict(np.load(path)); rows = d["rows"]
    cv = control_variate(rows[rows[:, 2] == min(CFG["eps"])], max(CFG["boxes"]))
    print(f"control variate (box {max(CFG['boxes'])}, {cv['N']} seeds): Q_beta = {cv['Qcv']:.4f} +- {cv['dQcv_naive']:.4f} (naive) "
          f"+- {cv['dQcv_jk']:.4f} (jackknife, b refitted); b = {cv['b']:.3f}; <n_b> = {cv['nb_mean']:.5f} vs 1/(8 pi) = {cv['nb_exp']:.5f}; "
          f"raw mean {d['Qb']:.4f} +- {d['dQ']:.4f}")
    d.update({k: np.asarray(v) for k, v in cv.items()})
    np.savez(path, **d)


if __name__ == "__main__" and "--post" in sys.argv:
    post(ROOT / "output/11_universal_flat.npz")
elif __name__ == "__main__":
    bg = flat_background()
    rows = []
    for Lb in CFG["boxes"]:
        for sd in CFG["seeds"]:
            t0 = time.time()
            # cap nucleation at the time the box is surely filled (tau_max): percolation happens at Gamma ~ beta^4 /(8 pi) -> t ~ ln(1/V)
            tn, xn = mc.nucleate(bg, Lb, np.random.default_rng(sd), tau_max=np.log(50.0 / Lb ** 3) + 8)
            for eps in CFG["eps"]:
                q = mc.surface_Q(bg, tn, xn, Lb, np.random.default_rng(sd + 100), npts=CFG["npts"], eps=eps, chunk=20_000)   # chunk: memory ∝ chunk × N_bubbles
                S = q["samples"]
                rows.append([Lb, sd, eps, len(tn), q["Q"], q["Q_head"], q["AV"], np.mean(S[:, 1]), np.mean(S[:, 6])])
                print(f"L={Lb} seed={sd} eps={eps}: bubbles={len(tn)} Q_beta={q['Q']:.4f} (head-on {q['Q_head']:.4f}) A/V={q['AV']:.3f} "
                      f"<r>={np.mean(S[:,1]):.3f} <h>={np.mean(S[:,6]):.3f} ({time.time()-t0:.0f}s)", flush=True)
    rows = np.array(rows)
    a = np.pi ** 2 * G_STAR_SM / 30; b = 2 * np.pi ** 2 * G_STAR_SM / 45
    K = 8 * np.sqrt(3) * a ** 1.5 / b
    big = rows[(rows[:, 0] == max(CFG["boxes"])) & (rows[:, 2] == min(CFG["eps"]))]   # quoted result: 8 seeds in the box L = 32/beta
    Qb = big[:, 4].mean(); dQ = big[:, 4].std(ddof=1) / np.sqrt(len(big))
    print(f"K = {K:.2f};  Q_beta = {Qb:.4f} ± {dQ:.4f} (stat);  Y = {K*Qb:.3f} c0 (H/beta)(T_reh/M_P)^3 "
          f"= {K*Qb/(960*np.pi):.3e} c0hat (H/beta)(T_reh/M_P)^3")
    np.savez(ROOT / "output/11_universal_flat.npz", rows=rows, K=K, Qb=Qb, dQ=dQ,
             header="Lbox seed eps Nbub Q_beta Q_head A/V <r> <h>")
    post(ROOT / "output/11_universal_flat.npz")
