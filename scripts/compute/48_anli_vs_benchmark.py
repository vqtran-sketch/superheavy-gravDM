"""48: the flat-space benchmark and the An-Li background on common axes; the effective inverse duration beta_eff.

Main text: "A realization" (Eq. (5) still gives the yield to about 10% with beta replaced by beta_eff ≈ 3.5 H_c). SM: Sec. S7, "Yield"
(false-vacuum fraction from 0.9 to 0.1 in the same time as for an exponential rate; beta_eff ≈ 3.5 H_c for all three alpha).
The 1+1D cartoon data stored in the output are not used.

Benchmark [ASSUMED, as in the main text]: flat space, H constant during the transition (vacuum domination), Gamma = Gamma_* e^{beta t}.
  Flat-space false-vacuum fraction f = exp(-I), I(t) = (4 pi/3) int Gamma(t') (t - t')^3 dt' = 8 pi Gamma(t)/beta^4 [DERIVED],
  time origin t = 0 at I = 1  ->  f(t) = exp(-e^{beta t}),  Gamma/H^4 = (beta/H)^4 e^{beta t}/(8 pi); f falls from 0.9 to 0.1 in
  beta Delta t = ln(ln 10/ln(10/9)) ~ 3.08.
  Comoving released energy (a = e^{H t}, a = 1 at t = 0): E = int a^4 (-df) = Gamma_fn(1 + 4 H/beta) [DERIVED, analytic; checked numerically below].
An-Li [LIT: An:2026ghw, Eqs. (1)-(2), (S1)-(S7); solved by gravdm.anli]: Gamma constant, units H_c = 1 = Gamma^{1/4}, a_c = 1 at H = H_c.
  cosmic time t H_c = int a dtau (tau = conformal time in 1/(a_c H_c)); E = lim a^4 (rho_w + rho_s)/L.
  Wall energy per area hat e(tau_i, tau) = int_{tau_i}^tau a^4 (tau' - tau_i)^2 dtau' / [a^3(tau) (tau - tau_i)^2],
  physical radius H_c r = a(tau) (tau - tau_i); flat limit hat e = H_c r/3.
1+1D cartoons (illustration only): Poisson nucleation in (x, time) with rejection inside existing bubbles, walls at the speed of light.
Output: output/48_anli_vs_benchmark.npz"""
import sys, pathlib
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import gamma as Gfn
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import anli

CFG = dict(alphas=(0.05, 5e-3, 5e-4), n=6, Nbg=6000, Hbeta=(0.1, 0.01), tau_i=(-0.4, 0.0, 0.4, 0.8),
           cart_seed=(11, 12), cart_X=4.0, cart_Hbeta=0.1, cart_tmin=-0.5)

out = {}
# ---- An-Li backgrounds
for al in CFG["alphas"]:
    s = anli.solve(al, n=CFG["n"], N=CFG["Nbg"])
    tau, a, H, f = s["tau"], s["a"], s["H"], s["f"]
    t = cumulative_trapezoid(a, tau, initial=0.0); t -= np.interp(0.0, tau, t)
    rho = s["rho_DS"] + s["rho_V"] + s["rho_rad"]
    k = f"a{al:g}"
    out |= {f"{k}_tau": tau, f"{k}_t": t, f"{k}_a": a, f"{k}_H": H, f"{k}_f": f, f"{k}_xDS": s["rho_DS"] / rho,
            f"{k}_xV": s["rho_V"] / rho, f"{k}_xR": s["rho_rad"] / rho, f"{k}_Ecom": a ** 4 * s["rho_rad"] / s["L"],
            f"{k}_Ephys": s["rho_rad"] / s["L"], f"{k}_dfdtau": -np.gradient(f, tau)}
    fx = lambda y, v: np.interp(-v, -f, y)
    print(f"alpha={al:g}: E = {out[f'{k}_Ecom'][-1]:.3f}; f=0.5 at tau={fx(tau, .5):.3f}, tH_c={fx(t, .5):.3f}, a={fx(a, .5):.3f}"
          f" (a^4 = {fx(a, .5) ** 4:.2f}); Dt(f 0.9->0.1) H_c = {fx(t, .1) - fx(t, .9):.3f}")
    if al == CFG["alphas"][0]:
        # wall energy per area for a few nucleation times (alpha = 0.05)
        for ti in CFG["tau_i"]:
            m = tau > ti + 1e-3
            num = cumulative_trapezoid(a[tau >= ti] ** 4 * (tau[tau >= ti] - ti) ** 2, tau[tau >= ti], initial=0.0)
            tt, aa = tau[tau >= ti], a[tau >= ti]
            ehat = num[1:] / (aa[1:] ** 3 * (tt[1:] - ti) ** 2)
            out[f"ehat_{ti:g}_r"] = aa[1:] * (tt[1:] - ti); out[f"ehat_{ti:g}"] = ehat
        out["anli_tau_c"], out["anli_aH_inv"] = tau, 1.0 / (a * H)     # comoving Hubble radius in 1/(a_c H_c)

# ---- flat-space benchmark (H const during the transition)
for hb in CFG["Hbeta"]:
    t = np.linspace(-3.0, 1.0, 4001)            # in 1/H
    bt = t / hb
    f = np.exp(-np.exp(bt))
    a = np.exp(t)
    df = -np.gradient(f, t)
    Ecom = cumulative_trapezoid(a ** 4 * df, t, initial=0.0)
    out |= {f"b{hb:g}_t": t, f"b{hb:g}_f": f, f"b{hb:g}_G": np.exp(bt) / hb ** 4 / (8 * np.pi), f"b{hb:g}_Ecom": Ecom}
    print(f"benchmark H/beta={hb:g}: E numeric {Ecom[-1]:.4f} vs Gamma(1+4H/beta) = {Gfn(1 + 4 * hb):.4f};"
          f" Dt(f 0.9->0.1) H = {hb * (np.log(np.log(10)) - np.log(np.log(1 / 0.9))):.3f}")


# ---- 1+1D cartoons
def cartoon(rate, tgrid, X, rng):
    """Poisson nucleation on [0, X] x tgrid with rate(t) per unit length per unit time; reject points inside earlier bubbles.
    Light cones at 45 degrees. Returns array of (x, t) nucleation points."""
    pts = []
    for t0, t1 in zip(tgrid[:-1], tgrid[1:]):
        nexp = rate(0.5 * (t0 + t1)) * X * (t1 - t0)
        for _ in range(rng.poisson(nexp)):
            x, t = rng.uniform(0, X), rng.uniform(t0, t1)
            if all(abs(x - xj) > t - tj for xj, tj in pts):
                pts.append((x, t))
    return np.array(pts)


rng = np.random.default_rng(CFG["cart_seed"][0])
hb = CFG["cart_Hbeta"]; beta = 1 / hb
tg = np.linspace(CFG["cart_tmin"], 0.6, 1101)
out["cart_b"] = cartoon(lambda t: 0.5 * beta ** 2 * np.exp(beta * t), tg, CFG["cart_X"], rng)
# An-Li: calibrate lam so that 1D f(tau_half) = 1/2, f_1D = exp(-lam int a^4 (tau - tau') 2 dtau')
k = f"a{CFG['alphas'][0]:g}"
tau, a, f3 = out[f"{k}_tau"], out[f"{k}_a"], out[f"{k}_f"]
th = np.interp(-0.5, -f3, tau)
m = tau <= th
lam = np.log(2) / np.trapezoid(a[m] ** 4 * 2 * (th - tau[m]), tau[m])
rng = np.random.default_rng(CFG["cart_seed"][1])
tg = np.linspace(-0.6, 1.4, 2001)
out["cart_a"] = cartoon(lambda s: lam * np.interp(s, tau, a) ** 4, tg, CFG["cart_X"], rng)
out["cart_lam"] = lam
print(f"cartoons: benchmark {len(out['cart_b'])} bubbles, An-Li {len(out['cart_a'])} bubbles (lam = {lam:.3f})")
np.savez(ROOT / "output/48_anli_vs_benchmark.npz", **out, CFG=str(CFG))
