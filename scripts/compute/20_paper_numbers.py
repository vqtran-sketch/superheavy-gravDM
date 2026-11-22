"""20: print the window-dependent numbers of the An-Li realization, directly from the stored outputs.

Main text: "A realization" (M_X between 5e12 and 2e17 GeV for a Dirac fermion and nu >= 2; H_inf below 2e10 to 2e11 GeV); "Gravitational
waves" (3e4 to 5e5 Hz). SM: Sec. S7 (window over all spins, conditions on the inflationary era, GW peak frequencies), Table SII;
Sec. S5 (abrupt-end range 70 to 180).

Reads output/04, 06, 07, 10 (+_power), 14, 17, 21. Window = src/gravdm/window.py (freeze-in <= 10% of collisions, R >= 0.1,
L^{1/4} < alpha^{1/4} V_inf^{1/4}, M < M_P/4); the rows with the yield above Lambda_cut = 0.46 M_P discarded come from script 21.
Sections printed: Dirac window (n_DS = 6, Table SII), all-spin envelope (nu >= 2), GW f_peak ranges, H_inf conditions (end of
inflation, friction, parametrized false-vacuum retention), freeze-in reference values, windows with a hard cut on the parton sqrt(s).
Output: output/20_paper_numbers.txt (also printed).
"Edgeless" cases: the core reaches the lowest scanned mass, i.e. freeze-in stays below 10% of the collisions for all M_X << T_*
(no lower edge; the grid start is not a physical edge). For these, ranges are also quoted for M_X above the Dirac edge at the same alpha."""
import re
import sys, pathlib, io
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import window, freezein as fz
from gravdm.units import M_P

CFG = dict(table_cases={0.05: (1.5, 2.0, 3.0), 0.005: (1.0, 2.0, 3.0), 0.0005: (1.0, 2.0, 3.0)}, spins=("dirac", "scalar", "conformal", "vector"),
           Vinf4=1.4e16, Mmax_frac=0.25,
           Yflat=6.8e-3)   # main-text Eq. (5) coefficient with Q_beta = 1/3 [DERIVED]
d6 = np.load(ROOT / "output/06_dm_plane.npz"); Md = np.load(ROOT / "output/04_mass_dependence.npz")
M = d6["M_over_MP"] * M_P
buf = io.StringIO()
def out(*a):
    print(*a); print(*a, file=buf)


def core(alpha, nu, sp):
    k = f"a{alpha}_nu{nu}_{sp}"
    L4 = d6[k + "_L4"]; R = Md[f"R_a{alpha}_nu{nu}_{sp}"]
    return (np.isfinite(L4) & (R >= window.R_CORE) & (d6[k + "_fcoll"] >= window.FCOLL_MIN) & (L4 < alpha ** 0.25 * CFG["Vinf4"])
            & (M < CFG["Mmax_frac"] * M_P)), k


def edges(alpha, nu, sp):
    """Interpolated window edges (window.interp_edges) and M_min/T_* at the lower edge."""
    c, k = core(alpha, nu, sp)
    lo, hi = window.interp_edges(M, c, d6[k + "_fcoll"], Md[f"R_a{alpha}_nu{nu}_{sp}"], L4=d6[k + "_L4"],
                                 L4max=alpha ** 0.25 * CFG["Vinf4"], Mmax=CFG["Mmax_frac"] * M_P)
    r = lo / np.exp(np.interp(np.log(lo), np.log(M), np.log(d6[k + "_Tstar"]))) if lo else np.nan
    return lo, hi, r


def edgeless(alpha, sp):
    return bool(core(alpha, 3.0, sp)[0][0])


def m_ok(alpha, sp):
    """mask of masses to quote: all, or M_X >= Dirac lower edge (nu = 2) for edgeless cases."""
    return M >= M[core(alpha, 2.0, "dirac")[0]].min() if edgeless(alpha, sp) else np.ones_like(M, bool)


def parse(k):
    a, nu, sp = re.search(r"a([0-9.e-]+)_nu([0-9.]+)_([a-z]+)_", k).groups()
    return float(a), float(nu), sp


out(f"Window: FI <= {window.FI_MAX_OVER_COLL:.0%} of collisions (f_coll >= {window.FCOLL_MIN:.4f}), R >= {window.R_CORE}")
out("\n[SM Table SII] Dirac, n = 6:  M_X range | L^{1/4} range | f_peak range | M_min/T*")
for a, nus in CFG["table_cases"].items():
    for nu in nus:
        c, k = core(a, nu, "dirac")
        if not c.any():
            out(f"  alpha={a} nu={nu}: empty"); continue
        i0 = np.where(c)[0][0]
        out(f"  alpha={a:<7g} nu={nu}: M {M[c].min():.2e}-{M[c].max():.2e} | L4 {np.nanmin(d6[k+'_L4'][c]):.2e}-{np.nanmax(d6[k+'_L4'][c]):.2e}"
            f" | f {np.nanmin(d6[k+'_f0'][c]):.2e}-{np.nanmax(d6[k+'_f0'][c]):.2e} | M_min/T* {M[i0]/d6[k+'_Tstar'][i0]:.2f}")
        lo, hi, r = edges(a, nu, "dirac")
        out(f"      interpolated edges: M {lo:.2e}-{hi:.2e} GeV, M_min/T* {r:.2f}")
out("\n[Edgeless cases] (core contains the lowest scanned M_X; massless-limit collision share f_coll and FI/coll):")
for a in CFG["table_cases"]:
    for sp in CFG["spins"]:
        k = f"a{a}_nu3.0_{sp}"
        if edgeless(a, sp):
            out(f"  alpha={a:g} {sp}: f_coll(M_min) = {d6[k + '_fcoll'][0]:.4f}, FI/coll = {1 / d6[k + '_fcoll'][0] - 1:.3f} (criterion 0.1)")
for a in CFG["table_cases"]:
    for sp in CFG["spins"]:
        c, k = core(a, 2.0, sp)
        out(f"  lower edge alpha={a:g} {sp}: " + ("none in grid" if edgeless(a, sp) else
            f"{M[c].min():.2e} GeV (grid), {edges(a, 2.0, sp)[0]:.2e} GeV (interpolated), M_min/T* {edges(a, 2.0, sp)[2]:.2f}"))
lowM = M <= d6["a0.05_nu2.0_dirac_Tstar"]
out(f"  collision share, Dirac, alpha=0.05, M_X <= T*: {d6['a0.05_nu2.0_dirac_fcoll'][lowM].min():.3f}-{d6['a0.05_nu2.0_dirac_fcoll'][lowM].max():.3f}")
for excl in (False, True):
    out(f"\n[Envelope] all spins, nu >= 2" + (", edgeless cases cut at the Dirac edge:" if excl else ", as computed (grid-limited if edgeless):"))
    Ms, Ls = [], []
    for a in CFG["table_cases"]:
        fs = []
        for nu in (2.0, 3.0):
            for sp in CFG["spins"]:
                c, k = core(a, nu, sp)
                if excl:
                    c = c & m_ok(a, sp)
                Ms += list(M[c]); Ls += list(d6[k + "_L4"][c]); fs += list(d6[k + "_f0"][c])
        out(f"  alpha={a:g}: f_peak {min(fs):.2e}-{max(fs):.2e} Hz (nu >= 2, all spins)")
    out(f"  all alpha: M_X {min(Ms):.2e}-{max(Ms):.2e} GeV, L^1/4 {min(Ls):.2e}-{max(Ls):.2e} GeV")
out("\n[Lower edges, Dirac] M_X at FI = 10% (interpolated): " + ", ".join(f"alpha={a:g}: {edges(a, 2.0, 'dirac')[0]:.2e}" for a in CFG["table_cases"]))
his = [edges(a, nu, sp)[1] for a in CFG["table_cases"] for nu in (2.0, 3.0) for sp in CFG["spins"]]
los = [edges(a, nu, sp)[0] for a in CFG["table_cases"] for nu in (2.0, 3.0) for sp in CFG["spins"] if not edgeless(a, sp)]
out(f"  envelope nu >= 2, all spins (interpolated): lowest lower edge {min(los):.2e}, highest upper edge {max(h for h in his if h):.2e} GeV")
for name, f in (("end of inflation, scalar, exp", "10_inflation_condition"), ("end of inflation, scalar, power", "10_inflation_condition_power")):
    d = np.load(ROOT / f"output/{f}.npz")
    out(f"\n[{name}] (script 10)")
    for k in d.files:
        if k.endswith("_Hmax"):
            h = d[k]; m = np.isfinite(h)
            if m.any():
                out(f"  {k}: H_inf,max {np.nanmin(h):.2e}-{np.nanmax(h):.2e} GeV; M/H {np.nanmin(M[m]/h[m]):.1f}-{np.nanmax(M[m]/h[m]):.1f}")
                a, nu, sp = parse(k)
                if edgeless(a, sp):
                    m2 = m & m_ok(a, sp)
                    out(f"    (M_X >= Dirac edge) H_inf,max {np.nanmin(h[m2]):.2e}-{np.nanmax(h[m2]):.2e} GeV; M/H {np.nanmin(M[m2]/h[m2]):.1f}-{np.nanmax(M[m2]/h[m2]):.1f}")
d14 = np.load(ROOT / "output/14_inflation_condition_spins.npz")
out("\n[end of inflation, Dirac/Proca] (script 14)")
for k in d14.files:
    if k.endswith("_Hmax"):
        h = d14[k]; m = np.isfinite(h)
        if m.any():
            out(f"  {k}: H_inf,max {np.nanmin(h):.2e}-{np.nanmax(h):.2e} GeV; M/H {np.nanmin(M[m]/h[m]):.1f}-{np.nanmax(M[m]/h[m]):.1f}")
d7 = np.load(ROOT / "output/07_validity.npz")
out("\n[friction] (script 07, minimal scalar lines)")
for k in d7.files:
    if k.endswith("_Hfric"):
        c = d7[k.replace("_Hfric", "_core")]
        out(f"  {k}: H_inf < {d7[k][c].min():.2e}-{d7[k][c].max():.2e} GeV")
        a, nu, sp = parse(k)
        if edgeless(a, sp):
            c2 = c & m_ok(a, sp)
            out(f"    (M_X >= Dirac edge) H_inf < {d7[k][c2].min():.2e}-{d7[k][c2].max():.2e} GeV")
d17 = np.load(ROOT / "output/17_higgs_trapping.npz")
out("\n[Higgs trapping] (script 17)")
for k in d17.files:
    out(f"  {k}: {np.nanmin(d17[k]):.2e}-{np.nanmax(d17[k]):.2e} GeV")
out("\n[freeze-in reference] Y_FI/(T_reh/M_P)^3 at M/T = 0, 1, 3, 10 (T_max = T_reh, xi_H = 0):")
T = 1e12
for sp in CFG["spins"]:
    ys = [fz.yield_freezein_X(T, x * T, sp) / (T / M_P) ** 3 for x in (0.0, 1.0, 3.0, 10.0)]
    out(f"  {sp:9s}: " + ", ".join(f"{y:.2e}" for y in ys) + f"   [flat-space collisions/FI(0) = {CFG['Yflat'] / ys[0]:.0f} chat (H/beta)]")
d21 = np.load(ROOT / "output/21_cutoff_window.npz")
out("\n[hard cut on parton sqrt(s)] (script 21): q50/q90 of sqrt(s)/M_P | yield fraction above cuts | "
    "Dirac core under each cut")
for a in CFG["table_cases"]:
    for nu in (1.0, 1.5, 2.0, 3.0):
        kk = f"a{a}_nu{nu}"
        out(f"  alpha={a:<7g} nu={nu}: q50 {float(d21[kk + '_sqrts_q50']):.3f} q90 {float(d21[kk + '_sqrts_q90']):.3f} | above " +
            "/".join(f"{float(d21[f'{kk}_cut{lam}_frac_above']):.2f}" for lam in d21["cuts"]))
        for tag in ["nocut"] + [f"cut{lam}" for lam in d21["cuts"]]:
            c = d21[f"{kk}_dirac_{tag}_core"]; L = d21[f"{kk}_dirac_{tag}_L4"]
            lo, hi = window.interp_edges(M, c, d21[f"{kk}_dirac_{tag}_fcoll"], d21[f"{kk}_dirac_{tag}_R"], L4=L,
                                         L4max=a ** 0.25 * CFG["Vinf4"], Mmax=CFG["Mmax_frac"] * M_P) if c.any() else (None, None)
            out(f"      {tag:8s} M {M[c].min():.2e}-{M[c].max():.2e} (interpolated {lo or np.nan:.2e}-{hi or np.nan:.2e}) "
                f"L4 {np.nanmin(L[c]):.2e}-{np.nanmax(L[c]):.2e}" if c.any() else f"      {tag}: empty")
for tag in ["nocut"] + [f"cut{lam}" for lam in d21["cuts"]]:
    Ms, Ls = [], []
    for a in CFG["table_cases"]:
        for nu in (2.0, 3.0):
            for sp in CFG["spins"]:
                c = d21[f"a{a}_nu{nu}_{sp}_{tag}_core"]; Ms += list(M[c]); Ls += list(d21[f"a{a}_nu{nu}_{sp}_{tag}_L4"][c])
    out(f"  envelope nu >= 2, all spins, {tag}: M_X {min(Ms):.2e}-{max(Ms):.2e} GeV, L^1/4 {min(Ls):.2e}-{max(Ls):.2e} GeV")
(ROOT / "output/20_paper_numbers.txt").write_text(buf.getvalue())
