"""55: 1+1D collision of walls in a Higgs-like (Z2-even) potential: true vacuum at phi = 0, false vacua at phi = +-v.

SM: Sec. S3, "Other potentials" (in an even sextic potential, as for the radial Higgs field, the walls are trapped and turn back after
the first crossing). Also a helper for script 56, which imports its potential, wall and stress tensor.
Question: after the walls cross, does the field between them return to the false-vacuum manifold (phi ~ -v), so that the walls are
trapped and re-collide, and is the first-collision source still that of free passage?

Potential [ASSUMED toy]: V(phi) = lam phi^2 (phi^2 - v^2)^2 / v^2 + dV * S(phi^2/v^2), S(y) = y^2 (3 - 2y) for y <= 1, S = 1 for y > 1.
  Degenerate part: vacua at 0 and +-v, mass m^2 = V''(0) = 2 lam v^2 (mass 2m at +-v); kink phi(z) = v / sqrt(1 + exp(-2 m z)),
  tension sigma = m v^2 / 4. The tilt keeps the extrema and raises +-v by dV (S'(0) = S'(1) = 0).
Set-up: slice true | false | true, i.e. 0 | v | 0; walls start at +-x0 with Lorentz factor gamma0; superposition
  phi = phi_1 + phi_2 - v (exact far from the overlap). Free passage = the same superposition continued through the collision.
Outputs (units m = v = 1): phi(t, x=0); wall positions (where phi crosses v/2 nearest to the center); energy drift; Fourier invariants
  I2, I0 of the windowed T~(omega, k) around the first collision for the lattice and for the free-passage field (same pipeline as
  script 02) at targets on the diagonal p_1 = p_2. Output: output/55_lattice_higgs_like.{npz,log}.
Checks: energy conservation; dV = 0 run (degenerate: the region between crossed walls is the vacuum -v, walls pass through);
  free-passage pipeline vs lattice in the hard region."""
import sys, pathlib, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import lattice1d as L

CFG = dict(gamma0=10.0, dV_over_sigma_m=[0.0, 0.16], v=1.0, m=1.0, pts_per_wall=20, cfl=0.5, x0=24.0, t_extra=140.0, margin=30.0,
           Tw=4.0, targets=[4.0, 8.0, 16.0], sample_dt=0.01, snap_every=1.0)


def pot(phi, lam, v, dV):
    y = np.clip(phi ** 2 / v ** 2, 0, 1)
    return lam * phi ** 2 * (phi ** 2 - v ** 2) ** 2 / v ** 2 + dV * y ** 2 * (3 - 2 * y)


def dpot(phi, lam, v, dV):
    y = phi ** 2 / v ** 2
    dS = np.where(y <= 1, 6 * y * (1 - y), 0.0) * 2 * phi / v ** 2          # d/dphi of S(phi^2/v^2)
    return lam * (2 * phi * (phi ** 2 - v ** 2) ** 2 + 4 * phi ** 3 * (phi ** 2 - v ** 2)) / v ** 2 + dV * dS


def wall(x, t, xc, u, gamma, sgn, v, m):
    """Boosted kink between 0 and v: sgn=+1 rises 0->v towards +x, sgn=-1 falls v->0; returns phi, phidot."""
    z = sgn * gamma * (x - xc - u * t)
    e = np.exp(-2 * m * np.clip(z, -200, 200))
    phi = v / np.sqrt(1 + e)
    dphidz = v * m * e / (1 + e) ** 1.5
    return phi, dphidz * sgn * gamma * (-u)


def stress(phi, phidot, dx, lam, v, dV):
    ph_x = L.grad4(phi, dx)
    K = 0.5 * phidot ** 2; G = 0.5 * ph_x ** 2; U = pot(phi, lam, v, dV)
    return dict(T00=K + G + U, T01=-phidot * ph_x, T11=K + G - U, T22=K - G - U)


def run(dVsm):
    v, m, g0 = CFG["v"], CFG["m"], CFG["gamma0"]
    lam = m ** 2 / (2 * v ** 2); sigma = m * v ** 2 / 4; dV = dVsm * sigma * m
    u = np.sqrt(1 - 1 / g0 ** 2); x0 = CFG["x0"]; tc = x0 / u
    Lh = x0 + CFG["margin"] + CFG["t_extra"]
    dx = 2 / (g0 * m * CFG["pts_per_wall"]); N = int(2 * Lh / dx); N += N % 2
    x = -Lh + dx * np.arange(N); dt = CFG["cfl"] * dx
    tend = tc + CFG["t_extra"]; nsteps = int(tend / dt); stride = max(1, int(round(CFG["sample_dt"] / dt)))
    kgrid = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    tg = np.array(CFG["targets"]); kidx = np.array([np.argmin(np.abs(kgrid)) for _ in tg]); om = 2 * tg
    accL = L.FTAccumulator(x, kidx, om); accF = L.FTAccumulator(x, kidx, om)
    w = lambda t: np.exp(-(t - tc) ** 2 / (2 * CFG["Tw"] ** 2))
    def init(t):
        a, ad = wall(x, t, -x0, u, g0, +1, v, m); b, bd = wall(x, t, +x0, -u, g0, -1, v, m)
        return a + b - v, ad + bd
    # free-passage reference at the Lorentz factor at the collision: thin-wall d(sigma gamma v)/dt = dV gives
    # gamma_c ~= gamma0 + (dV/sigma) t_c (v ~ 1); walls placed to meet at the same t_c (the false vacuum accelerates them)
    gc = g0 + dV / sigma * tc; uc = np.sqrt(1 - 1 / gc ** 2)
    def free(t):
        a, ad = wall(x, t, -uc * tc, uc, gc, +1, v, m); b, bd = wall(x, t, +uc * tc, -uc, gc, -1, v, m)
        return a + b - v, ad + bd
    phi_prev, _ = init(-dt); phi, _ = init(0.0)
    i0 = np.argmin(np.abs(x)); ts, phi0, xw, E = [], [], [], []
    for n in range(nsteps + 1):
        t = n * dt
        phi_next = 2 * phi - phi_prev + dt * dt * (L.lap4(phi, dx) - dpot(phi, lam, v, dV))
        if n % stride == 0:
            pd = (phi_next - phi_prev) / (2 * dt)
            T = stress(phi, pd, dx, lam, v, dV)
            if abs(t - tc) < 6 * CFG["Tw"]:
                accL.add(T, t, w(t) * dt * stride)
                pf, pfd = free(t); accF.add(stress(pf, pfd, dx, lam, v, dV), t, w(t) * dt * stride)
            if n % int(round(CFG["snap_every"] / dt)) == 0:
                ts.append(t); phi0.append(phi[i0]); E.append(T["T00"].sum() * dx)
                # right wall position: first crossing of |phi - phi_far| at half height, searching outward from the centre
                half = np.abs(phi[i0:]) - 0.5 * v; s = np.where(np.diff(np.sign(half)) != 0)[0]
                xw.append(x[i0 + s[0]] if len(s) else np.nan)
        phi_prev, phi = phi, phi_next
    I2l, I0l = L.invariants(L.tensor_from_acc(accL.acc)); I2f, I0f = L.invariants(L.tensor_from_acc(accF.acc))
    E = np.array(E)
    return dict(t=np.array(ts), phi0=np.array(phi0), xw=np.array(xw), Edrift=np.max(np.abs(E / E[0] - 1)), tc=tc,
                targets=tg, gamma_c=gc, I2_ratio=I2l / I2f, I0_ratio=I0l / I0f, dV=dV, sigma=sigma, N=N, dx=dx)


if __name__ == "__main__":
    out = {}; lines = []
    for dVsm in CFG["dV_over_sigma_m"]:
        t0 = time.time(); r = run(dVsm); key = f"dV{dVsm:g}"
        for k, v in r.items(): out[f"{key}_{k}"] = v
        after = r["t"] > r["tc"] + 5
        msg = (f"dV/(sigma m) = {dVsm:g}: N = {r['N']}, energy drift {r['Edrift']:.1e}, run {time.time()-t0:.0f} s\n"
               f"  first collision at t_c = {r['tc']:.1f}/m; phi(x=0) after crossing: min {np.nanmin(r['phi0'][after]):.3f}, "
               f"max {np.nanmax(r['phi0'][after]):.3f} (v = 1)\n"
               f"  right-wall position after crossing: max {np.nanmax(r['xw'][after]):.2f}/m at t = {r['t'][after][np.nanargmax(r['xw'][after])]:.1f}/m, "
               f"final {r['xw'][-1]:.2f}/m at t = {r['t'][-1]:.1f}/m\n"
               f"  free-passage reference at gamma_c = {r['gamma_c']:.2f}; lattice/free-passage at p_R = p_L = {list(r['targets'])} m: I2 {np.round(r['I2_ratio'], 3)}, I0 {np.round(r['I0_ratio'], 3)}")
        print(msg, flush=True); lines.append(msg)
    np.savez(ROOT / "output/55_lattice_higgs_like.npz", **out)
    (ROOT / "output/55_lattice_higgs_like.log").write_text("\n".join(lines) + "\n")
