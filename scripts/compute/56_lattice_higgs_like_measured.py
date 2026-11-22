"""56: free passage at the FIRST collision of accelerating Higgs-like (Z2-even) walls, tested with single-wall spectra measured on the
lattice just before the collision (method of scripts 25 and 39; potential of script 55).

SM: Sec. S3, "Other potentials" (in an even sextic potential the first collision is not affected by trapping, and the parton cross
term reproduces the lattice there within 4%).

Potential [ASSUMED toy, identical to script 55]: V(phi) = lam phi^2 (phi^2 - v^2)^2 / v^2 + dV S(phi^2/v^2),
  S(y) = y^2 (3 - 2y) (y <= 1), S = 1 (y > 1).  Units m = v = 1 (lengths and times in 1/m, energies in m, phi in v).
  m^2 = V''(0) = 2 lam v^2 -> lam = 1/2.  Degenerate kink (dV = 0) [DERIVED, BPS]: phi' = sqrt(2V) = m phi (1 - phi^2/v^2),
  phi(z) = v / sqrt(1 + exp(-2 m z)), tension sigma = int_0^v sqrt(2V) dphi = m v^2 / 4.  Strength R = dV/(sigma m) (dV = R m^2 v^2/4).
  The tilt keeps the extrema (S'(0) = S'(1) = 0) and raises +-v by dV.  Inside |phi| < v: V''(v) = 4 m^2 - 24 dV/v^2 (mass of the
  false vacuum, > 0 for R < 2/3: metastable); for |phi| > v: V'' = 4 m^2.  V is C^1 at |phi| = v (V'' jumps by 24 dV/v^2), as in script 55.
Slice 0 | v | 0 (true | false | true): the false vacuum lies BETWEEN the walls, the walls accelerate towards each other.
  Thin-wall kinematics [approximation]: d(sigma gamma u)/dt = dV (script 16 `kinematics`), gives t_col and gamma_c (used to place targets).
Single-wall spectrum [DERIVED here; checked numerically in `kink_check`]: for phi(gamma z), F(p) = int dx phi' e^{i p x}
  = (v/2) B(1 - i q/2, 1/2 + i q/2), q = p/(gamma m) (substitution w = e^{-2 gamma m z}), hence
  |F(p)|^2 = v^2 (pi q)/sinh(pi q)   (cf. tanh kink of lattice1d: |F| = Dphi (pi q)/sinh(pi q); the sextic kink is NOT tanh).
Parton (free-passage cross-term) prediction [SM Eq. (S8) at xi_phi = 0, lattice1d.parton_tensor_F]:
  T~^{mu nu} = F_1(a) F_2(b)/(2ab) [p1^mu p2^nu + p2^mu p1^nu - eta^{mu nu} p1.p2], a = (omega + k)/2, b = (omega - k)/2,
  with F_1, F_2 the MEASURED |F| of the right-moving (x < 0) and left-moving (x > 0) walls (script 25 `measured_F`: snapshots at
  t_col - snap_dt, |F| extrapolated linearly in t to t_col).  Lattice invariants I2, I0 of the windowed transform (Gaussian window Tw
  centered at t_col) as in lattice1d.run; targets (p_1, p_2) = targets_frac x gamma_c m.
Validity: 1+1D (planar walls, k_perp = 0), classical field, first collision only; hard targets a, b >~ 4 m (potential cross term neglected
  in the parton formula); the measured-F method needs the walls well separated at the last snapshot (half-gap monitored).
Checks: (i) analytic sextic F vs numerical transform (kink_check); (ii) R = 0 runs at constant gamma (lattice vs parton with ANALYTIC
  and with measured F); (iii) method check per accelerating point: dV = 0 run at gamma_c, lattice vs measured-F parton (and analytic F);
  (iv) energy conservation; (v) false vacuum between the walls (max|phi - v|) at t_col/2, t_col - 2, t_col - 1; (vi) snapshot sets
  base/close/far from the same run; (vii) resolution x2 (mode conv).  Also reported: gamma from the measured wall momentum
  P = int_{x<0} T^{01} dx = sigma gamma u (rigid wall), extrapolated to t_col, and gamma_F from inverting |F_1(a)|^2 at each target.
Usage: python 56_lattice_higgs_like_measured.py [one | main | conv | all]
Output: output/56_lattice_higgs_like_measured{,_one,_conv}.{npz,log}"""
import sys, time, pathlib, importlib.util
import numpy as np
from scipy import optimize
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import lattice1d as L

CFG = dict(v=1.0, m=1.0, R=[0.0, 0.16, 0.4], gamma0s=[10.0, 20.0],
           Tw=3.0, tc_over_Tw=6.0, margin=12.0, PTS=20, cfl=0.5, sample_dt=0.01,
           targets_frac=[(0.4, 0.4), (0.6, 0.6), (0.8, 0.8), (0.4, 0.8)],
           snap_sets={"base": (1.5, 1.0, 0.5), "close": (0.8, 0.6, 0.4), "far": (3.0, 2.0, 1.0)},
           monitor_before=(None, 2.0, 1.0), monitor_inner_margin=0.5,
           one_point=(10.0, 0.4), conv_points=[(10.0, 0.4), (20.0, 0.4)], conv_PTS=40,
           kink_check=dict(q=[0.4, 0.8, 1.6, 3.2], half_box=40.0, dx=2e-3), barrier_grid=200001, gF_qmax=60.0)


def _load(name, fname):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "compute" / fname)
    mod = importlib.util.module_from_spec(s); s.loader.exec_module(mod); return mod


s25 = _load("s25", "25_lattice_strong_accel.py")      # measured_F; s25.s16.kinematics
s55 = _load("s55", "55_lattice_higgs_like.py")        # pot, dpot, wall, stress of the Higgs-like potential
LOG = []


def out(msg):
    print(msg, flush=True); LOG.append(msg)


def params():
    v, m = CFG["v"], CFG["m"]
    return v, m, m ** 2 / (2 * v ** 2), m * v ** 2 / 4          # v, m, lam, sigma


def F_sextic(p, gamma):
    """|F(p)| of the boosted sextic kink [DERIVED, docstring]: v sqrt(pi q / sinh(pi q)), q = p/(gamma m). Units of v."""
    v, m, _, _ = params()
    y = np.pi * np.abs(np.asarray(p, float)) / (gamma * m)
    return v * np.sqrt(np.where(y < 1e-12, 1.0, y / np.sinh(np.minimum(y, 700))))


def gamma_from_F(p, F):
    """Invert |F(p)|^2/v^2 = pi q/sinh(pi q) for q, return gamma_F = p/(m q) (nan if |F| >= v or not invertible)."""
    v, m, _, _ = params(); f = (F / v) ** 2; res = []
    for pi_, fi in zip(np.atleast_1d(p), np.atleast_1d(f)):
        g = lambda q: np.pi * q / np.sinh(np.pi * q) - fi
        res.append(pi_ / (m * optimize.brentq(g, 1e-8, CFG["gF_qmax"])) if 0 < fi < 1 and g(CFG["gF_qmax"]) < 0 else np.nan)
    return np.array(res)


def kink_check():
    """Numerical F of the static kink (analytic phi') vs the closed form; returns max relative deviation."""
    v, m, _, _ = params(); kc = CFG["kink_check"]
    x = np.arange(-kc["half_box"], kc["half_box"], kc["dx"])
    e = np.exp(-2 * m * x); dphi = v * m * e / (1 + e) ** 1.5
    q = np.array(kc["q"]); Fn = np.abs(np.exp(1j * np.outer(q * m, x)) @ dphi * kc["dx"])
    return np.max(np.abs(Fn / F_sextic(q * m, 1.0) - 1)), Fn


def barrier(R):
    """Barrier between 0 and v of the tilted potential: (phi_top, V_top, V(v), V_top - V(v), m_f^2 inside)."""
    v, m, lam, sigma = params(); dV = R * sigma * m
    ph = np.linspace(0, v, CFG["barrier_grid"]); V = s55.pot(ph, lam, v, dV)
    i = np.argmax(V[1:-1]) + 1
    return ph[i], V[i], V[-1], V[i] - V[-1], 4 * m ** 2 - 24 * dV / v ** 2


def evolve(g0, dV, pts, targets, tw0, snap_ts):
    """Leapfrog evolution (as lattice1d.run, 4th-order stencils) of the Higgs-like slice 0 | v | 0 with boosted sextic kinks at
    -+x0 moving with +-u0; returns the windowed T~ at the targets, snapshots {t: phi}, half-box momenta {t: (P_R, P_L)}, energy log."""
    v, m, lam, sigma = params(); Tw = CFG["Tw"]
    u = np.sqrt(1 - 1 / g0 ** 2); tc0 = CFG["tc_over_Tw"] * Tw; x0 = u * tc0
    Lh = x0 + CFG["margin"]; dx = 2 / (g0 * m * pts); N = int(2 * Lh / dx); N += N % 2
    x = -Lh + dx * np.arange(N); dt = CFG["cfl"] * dx; nsteps = int(2 * tc0 / dt)
    stride = max(1, int(round(CFG["sample_dt"] / dt)))
    kgrid = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    a_t = np.array([t[0] for t in targets]); b_t = np.array([t[1] for t in targets])
    kidx = np.array([np.argmin(np.abs(kgrid - (a - b))) for a, b in zip(a_t, b_t)]); ksn = kgrid[kidx]; om = a_t + b_t
    acc = L.FTAccumulator(x, kidx, om)
    w = lambda t: np.exp(-(t - tw0) ** 2 / (2 * Tw ** 2))

    def init(t):
        a, ad = s55.wall(x, t, -x0, u, g0, +1, v, m); b, bd = s55.wall(x, t, +x0, -u, g0, -1, v, m)
        return a + b - v, ad + bd
    phi_prev, _ = init(-dt); phi, _ = init(0.0)
    snaps, P, E = {}, {}, []
    left = x < 0
    for n in range(nsteps + 1):
        t = n * dt
        phi_next = 2 * phi - phi_prev + dt * dt * (L.lap4(phi, dx) - s55.dpot(phi, lam, v, dV))
        hit = [ts for ts in snap_ts if abs(t - ts) < dt / 2]
        if hit or n % stride == 0:
            pd = (phi_next - phi_prev) / (2 * dt)
        if hit:
            ph_x = L.grad4(phi, dx); T01 = -pd * ph_x
            for ts in hit:
                snaps[ts] = (t, phi.copy()); P[ts] = (T01[left].sum() * dx, T01[~left].sum() * dx)
        if n % stride == 0:
            T = s55.stress(phi, pd, dx, lam, v, dV); acc.add(T, t, w(t) * dt * stride); E.append(T["T00"].sum() * dx)
        phi_prev, phi = phi, phi_next
    E = np.array(E)
    return dict(a=(om + ksn) / 2, b=(om - ksn) / 2, T=L.tensor_from_acc(acc.acc), x=x, dx=dx, N=N, snaps=snaps, P=P,
                Edrift=np.max(np.abs(E / E[0] - 1)), tc0=tc0)


def wall_positions(x, ph):
    """Positions of the kink centres (phi = v/sqrt 2): left wall = first crossing at x < 0, right wall = last crossing at x > 0."""
    v = CFG["v"]; h = ph - v / np.sqrt(2); s = np.where(np.sign(h[:-1]) != np.sign(h[1:]))[0]
    xs = x[s] - h[s] * (x[s + 1] - x[s]) / (h[s + 1] - h[s])
    xl = xs[xs < 0]; xr = xs[xs > 0]
    return (xl[0] if len(xl) else np.nan), (xr[-1] if len(xr) else np.nan)


def spectra(run, tcol, dts):
    sub = dict(x=run["x"], dx=run["dx"], snaps=[run["snaps"][tcol - d] for d in dts])
    return s25.measured_F(sub, tcol, run["a"], run["b"])


def run_point(g0, R, PTS):
    v, m, lam, sigma = params(); dV = R * sigma * m
    u0 = np.sqrt(1 - 1 / g0 ** 2); tc0 = CFG["tc_over_Tw"] * CFG["Tw"]; x0 = u0 * tc0
    tcol, gc = s25.s16.kinematics(g0, dV, x0, sigma)
    tg = [(fa * gc, fb * gc) for fa, fb in CFG["targets_frac"]]
    pts = int(np.ceil(PTS * gc / g0))
    mon = [tcol / 2 if b is None else tcol - b for b in CFG["monitor_before"]]
    dts_all = sorted({d for s in CFG["snap_sets"].values() for d in s})
    t0 = time.time()
    ra = evolve(g0, dV, pts, tg, tcol, mon + [tcol - d for d in dts_all])
    if R > 0:
        r0 = evolve(gc, 0.0, PTS, tg, tc0, [tc0 - d for d in dts_all]); tm = tc0
    else:
        r0, tm = ra, tcol
    wall = time.time() - t0
    x = ra["x"]; dev = []
    for tt in mon:
        _, ph = ra["snaps"][tt]; xl, xr = wall_positions(x, ph)
        inner = (x > xl + CFG["monitor_inner_margin"]) & (x < xr - CFG["monitor_inner_margin"])
        dev.append(np.max(np.abs(ph[inner] - v)) if np.isfinite(xl) and np.isfinite(xr) and inner.any() else np.inf)
    xl, xr = wall_positions(x, ra["snaps"][tcol - min(dts_all)][1]); halfgap = (xr - xl) / 2
    # gamma from the measured momentum of the right-moving wall (x < 0), extrapolated linearly to t_col (base snapshot set)
    dts = CFG["snap_sets"]["base"]; ts = np.array([tcol - d for d in dts])
    PR = np.array([ra["P"][t][0] for t in ts]); PL = np.array([-ra["P"][t][1] for t in ts])
    Pc = 0.5 * (np.polyval(np.polyfit(ts, PR, 1), tcol) + np.polyval(np.polyfit(ts, PL, 1), tcol))
    gP = np.sqrt(1 + (Pc / sigma) ** 2)
    A2, A0 = L.invariants(ra["T"]); Z2, Z0 = L.invariants(r0["T"])
    P2, P0 = L.invariants(L.parton_tensor_F(ra["a"], ra["b"], F_sextic(ra["a"], gc), F_sextic(ra["b"], gc)))
    Q2, Q0 = L.invariants(L.parton_tensor_F(r0["a"], r0["b"], F_sextic(r0["a"], gc), F_sextic(r0["b"], gc)))
    res = dict(g0=g0, R=R, dV=dV, gamma_c=gc, t_col=tcol, gamma_P=gP, pts=pts, N=ra["N"], dev=np.array(dev), halfgap=halfgap,
               Edrift=ra["Edrift"], Edrift_method=r0["Edrift"], a=ra["a"], b=ra["b"],
               I2_analytic=A2 / P2, I0_analytic=A0 / P0, I2_method_analytic=Z2 / Q2, I0_method_analytic=Z0 / Q0, wall_s=wall)
    for name, dts in CFG["snap_sets"].items():
        FR, FL, FR1, FL1 = spectra(ra, tcol, dts)
        M2, M0 = L.invariants(L.parton_tensor_F(ra["a"], ra["b"], FR, FL))
        N2, N0 = L.invariants(L.parton_tensor_F(ra["a"], ra["b"], FR1, FL1))
        GR, GL, _, _ = spectra(r0, tm, dts)
        V2, V0 = L.invariants(L.parton_tensor_F(r0["a"], r0["b"], GR, GL))
        res.update({f"I2_{name}": A2 / M2, f"I0_{name}": A0 / M0, f"I2_{name}_last": A2 / N2, f"I0_{name}_last": A0 / N0,
                    f"I2_method_{name}": Z2 / V2, f"I0_method_{name}": Z0 / V0, f"FR_{name}": FR, f"FL_{name}": FL,
                    f"gammaF_{name}": gamma_from_F(ra["a"], FR)})
    f3 = lambda arr: " ".join(f"{z:.3f}" for z in arr)
    out(f"g0={g0:g} R={R:g} dV={dV:.3f}: t_col={tcol:.2f} gamma_c(thin)={gc:.2f} gamma_P={gP:.2f} gamma_F(base)=[{f3(res['gammaF_base'])}] "
        f"pts={pts} N={ra['N']} wall {wall:.0f} s")
    out(f"   targets (a,b)/gamma_c = {CFG['targets_frac']}")
    for name in CFG["snap_sets"]:
        out(f"   [{name:5s}] lattice/measured-F parton: I2 {f3(res[f'I2_{name}'])} | I0 {f3(res[f'I0_{name}'])} | last snap I2 "
            f"{f3(res[f'I2_{name}_last'])} | method (dV=0 at gamma_c) I2 {f3(res[f'I2_method_{name}'])} I0 {f3(res[f'I0_method_{name}'])}")
    out(f"   lattice/analytic-sextic parton at gamma_c: I2 {f3(res['I2_analytic'])} I0 {f3(res['I0_analytic'])} | method run "
        f"(dV=0, gamma_c) vs analytic: I2 {f3(res['I2_method_analytic'])} I0 {f3(res['I0_method_analytic'])}")
    out(f"   false vacuum max|phi - v| between walls at t_col/2, t_col-2, t_col-1: {f3(dev)}; half-gap at last snapshot "
        f"{halfgap:.3f}/m (expected ~{min(dts_all):g}); energy drift {ra['Edrift']:.1e} (method run {r0['Edrift']:.1e})")
    return res


def save(results, tag):
    keys = results[0].keys()
    np.savez(ROOT / f"output/56_lattice_higgs_like_measured{tag}.npz",
             **{k: np.array([r[k] for r in results]) for k in keys}, targets_frac=np.array(CFG["targets_frac"]),
             snap_sets=np.array([f"{k}:{v}" for k, v in CFG["snap_sets"].items()]))
    (ROOT / f"output/56_lattice_higgs_like_measured{tag}.log").write_text("\n".join(LOG) + "\n")


def header():
    v, m, lam, sigma = params()
    err, _ = kink_check()
    out(f"Units m = v = 1: lam = {lam:g}, sigma = {sigma:g}. Kink-spectrum check |F_num/F_analytic - 1| max = {err:.1e} at q = {CFG['kink_check']['q']}")
    for R in CFG["R"]:
        pt, Vt, Vf, dB, mf2 = barrier(R)
        out(f"R = {R:g}: dV = {R * sigma * m:.4f}; barrier top phi = {pt:.4f} v, V_top = {Vt:.4f}, V(v) = {Vf:.4f}, V_top - V(v) = {dB:.4f} "
            f"({'inf' if Vf == 0 else f'{dB / Vf:.3f}'} x V(v)); m_f^2 (inside) = {mf2:.3f} m^2 -> {'metastable' if dB > 0 and mf2 > 0 else 'UNSTABLE'}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    if mode in ("one",):
        header(); save([run_point(*CFG["one_point"], CFG["PTS"])], "_one")
    if mode in ("main", "all"):
        LOG.clear(); header(); T0 = time.time()
        res = [run_point(g0, R, CFG["PTS"]) for g0 in CFG["gamma0s"] for R in CFG["R"]]
        out(f"total wall time {time.time() - T0:.0f} s"); save(res, "")
    if mode in ("conv", "all"):
        LOG.clear(); T0 = time.time(); res = []
        for g0, R in CFG["conv_points"]:
            for P in (CFG["PTS"], CFG["conv_PTS"]):
                out(f"--- resolution PTS = {P}"); res.append(run_point(g0, R, P))
        out(f"total wall time {time.time() - T0:.0f} s"); save(res, "_conv")
