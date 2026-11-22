"""64: end-of-inflation condition along the cores of the flat-space benchmark DM curves.

Main text: Dark-matter paragraph (production at the end of inflation adds less than 10% above M_X ≈ 2e15 GeV for any H_inf allowed by
the CMB; below that mass it requires H_inf below a few percent of M_X). SM: Sec. S5, "Production at the end of inflation"
(M_X/H_inf >~ 13 to 44; H_inf <~ 4e11 GeV at M_X = 1e13 GeV, T_reh = 1e12 GeV; threshold 2e15 GeV).

Estimate [LIT, Ema:2019yrd, Majorana fermion, M_X <~ H_inf, reheating after H falls to M_X]:
    rho_X/s ~ A M_X^2 T_R/(4 M_P^2),  A ~ 1e-3;  Dirac: x2;  above H_inf A -> A (M_X/H_inf)^2 exp(-c M_X/H_inf), c = 1 [ASSUMED].
Condition [DERIVED]: rho_X/s <= f_max rho_DM/s with rho_DM/s = units.MY_DM (Omega h^2 = 0.12), T_R = T_reh along the core of each DM
curve (script 50, yield above Lambda_cut discarded). Solved for x = M_X/H_inf (the larger root of x^2 exp(-c x) = rhs).
CMB [DERIVED, SM Eq. (S18)]: H_inf < pi M_P sqrt(A_s r/2) = 4.7e13 GeV. The condition holds for every allowed H_inf where
M_X/H_inf_max >= x_req.
Units: GeV. Output: output/64_end_of_inflation_condition.{npz,log}."""
import sys, pathlib
import numpy as np
from scipy import optimize
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm.units import M_P, MY_DM

CFG = dict(A=1e-3, dirac_factor=2.0, c=1.0, f_max=0.1, A_s=2.1e-9, r_max=0.036, check_point=(1e13, 1e12))

H_inf_max = np.pi * M_P * np.sqrt(CFG["A_s"] * CFG["r_max"] / 2)


def x_required(M, T):
    """Smallest M_X/H_inf (> 2/c, on the falling side of x^2 e^{-cx}) that keeps rho_X/s below f_max of the DM. M, T in GeV."""
    pref = CFG["dirac_factor"] * CFG["A"] * M ** 2 * T / (4 * M_P ** 2)
    rhs = CFG["f_max"] * MY_DM / pref
    f = lambda x: 2 * np.log(x) - CFG["c"] * x - np.log(rhs)
    if f(2 / CFG["c"]) <= 0:            # condition already met at the peak of x^2 e^{-cx}
        return 2 / CFG["c"]
    return optimize.brentq(f, 2 / CFG["c"], 1e4)


D = np.load(ROOT / "output/50_benchmark_dm_plane.npz")
M = D["M"]
lines, out = [], {}
xlo, xhi, Mcross = [], [], []
for hb in D["Hbeta"]:
    for nu in D["nus"]:
        key = f"hb{hb:g}_nu{nu:g}_cut"
        core = D[key + "_core"]; T = D[key + "_T"]
        x = np.array([x_required(m, t) for m, t in zip(M[core], T[core])])
        ok = M[core] / H_inf_max >= x                      # CMB bound alone suffices
        mc = M[core][ok].min() if ok.any() else np.nan
        xlo.append(x.min()); xhi.append(x.max()); Mcross.append(mc)
        out[key + "_x"] = x; out[key + "_M"] = M[core]
        lines.append(f"H/beta={hb:g} nu={nu:g}: core M = {M[core].min():.2e}-{M[core].max():.2e} GeV, "
                     f"M/H_inf required {x.min():.1f}-{x.max():.1f}; CMB bound alone suffices above M = {mc:.2e} GeV")
xc = x_required(*CFG["check_point"])
lines += [f"all cores: M/H_inf required {min(xlo):.0f}-{max(xhi):.0f}; CMB alone suffices above {np.nanmin(Mcross):.1e}-{np.nanmax(Mcross):.1e} GeV",
          f"check point M = {CFG['check_point'][0]:.0e}, T_reh = {CFG['check_point'][1]:.0e} GeV: M/H_inf >= {xc:.1f}",
          f"H_inf_max = {H_inf_max:.2e} GeV"]
(ROOT / "output/64_end_of_inflation_condition.log").write_text("\n".join(lines) + "\n")
np.savez(ROOT / "output/64_end_of_inflation_condition.npz", H_inf_max=H_inf_max, x_check=xc, Mcross=np.array(Mcross), **out)
print("\n".join(lines))
