"""58: data for main-text Fig. 3: the (M_X, T_reh) plane of a gravity-only Dirac X, thermal freeze-in against first collisions of
runaway bubbles, with the CMB bound on the reheating temperature.

Main text: Fig. 3; Dark-matter paragraph (freeze-in needs T_reh above about 4e12 GeV; its reach near 1e17 GeV; H_inf^max ≃ 5e13 GeV;
up to six orders of magnitude between M_X and T_reh). SM: Sec. S5 (H_inf < 4.7e13 GeV, T_reh < 5.8e15 GeV, freeze-in DM needs
T_reh >= 3.7e12 GeV with M_X/T_reh <= 21 and heaviest mass 1.2e17 GeV; lowest core T_reh 5.6e10 GeV; M_X/T_reh 3e4 to 4e6 on the cores).
Input to p58 and p62.

Inputs:
  output/50_benchmark_dm_plane.npz (script 50): freeze-in-only relic curve T_FIonly(M_X) (instantaneous reheating, T_max = T_reh) and the
  collision relic curves T_reh(M_X) for H/beta = 0.1, 0.03, 0.01, nu = 2, 3, with the yield above Lambda_cut = 0.46 M_P discarded ('cut')
  and the core mask (conditions of SM Sec. S5).
CMB bound [LIT]: r_0.05 < 0.036 (95% CL, BICEP:2021xfz) and A_s = 2.1e-9 (Planck:2018vyg) give
  H_inf < pi M_P sqrt(A_s r / 2) = 4.7e13 GeV  [DERIVED from V = (3 pi^2/2) A_s r M_P^4 and H^2 = V/(3 M_P^2); SM Eq. (S18)].
  With instantaneous reheating, 3 H^2 M_P^2 = (pi^2 g_*/30) T^4, so T_reh < (90/(pi^2 g_*))^{1/4} sqrt(H_inf M_P) [DERIVED].
  The same bound applies to the collision curves: the vacuum energy of the transition, L = 3 H^2 M_P^2, cannot exceed the inflationary one.
Annotation numbers [DERIVED here]: the largest M_X/T_reh on the freeze-in curve below the CMB bound and on the cores of the collision
  curves; the lowest T_reh of each; the heaviest freeze-in DM allowed by the CMB bound.
Units: GeV. Output: output/58_opening_plane.npz; output/58_opening_plane.log is captured standard output."""
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm.units import M_P, G_STAR_SM, CMB_R_MAX, CMB_A_S

CFG = dict(r_max=CMB_R_MAX, A_s=CMB_A_S, g_star=G_STAR_SM)

D = np.load(ROOT / "output/50_benchmark_dm_plane.npz")
M = D["M"]
H_inf_max = np.pi * M_P * np.sqrt(CFG["A_s"] * CFG["r_max"] / 2)
T_cmb_max = (90 / (np.pi ** 2 * CFG["g_star"])) ** 0.25 * np.sqrt(H_inf_max * M_P)
# check the closed form against energy density: 3 H^2 M_P^2 = (pi^2 g/30) T^4
assert abs(3 * H_inf_max ** 2 * M_P ** 2 / (np.pi ** 2 * CFG["g_star"] / 30 * T_cmb_max ** 4) - 1) < 1e-12

out = dict(M=M, T_FI=D["T_FIonly"], H_inf_max=H_inf_max, T_cmb_max=T_cmb_max, Hbeta=D["Hbeta"], nus=D["nus"])
for hb in D["Hbeta"]:
    for nu in D["nus"]:
        k = f"hb{hb:g}_nu{nu:g}_cut"
        out[f"{k}_T"] = D[f"{k}_T"]; out[f"{k}_core"] = D[f"{k}_core"]


def reach_at(T_line, mask, T0):
    """Largest M_X [GeV] where the relic line T_line(M) crosses T0 within mask (log interpolation); nan if none."""
    lT = np.log(T_line); lM = np.log(M); best = np.nan
    for i in range(len(M) - 1):
        if mask[i] and mask[i + 1] and np.isfinite(lT[i]) and np.isfinite(lT[i + 1]) and (lT[i] - np.log(T0)) * (lT[i + 1] - np.log(T0)) <= 0:
            f = (np.log(T0) - lT[i]) / (lT[i + 1] - lT[i]) if lT[i + 1] != lT[i] else 0.0
            best = np.exp(lM[i] + f * (lM[i + 1] - lM[i]))
    return best


fi_ok = np.isfinite(out["T_FI"]) & (out["T_FI"] <= T_cmb_max)
M_FI_cmb = reach_at(out["T_FI"], np.isfinite(out["T_FI"]), T_cmb_max)
out |= dict(M_FI_cmb=M_FI_cmb, FI_ratio_max=np.nanmax((M / out["T_FI"])[fi_ok]), FI_Tmin=np.nanmin(out["T_FI"]))
ratio_core = []; Tmin_core = []
for hb in D["Hbeta"]:
    for nu in D["nus"]:
        k = f"hb{hb:g}_nu{nu:g}_cut"; c = out[f"{k}_core"]
        if c.any():
            ratio_core.append(np.nanmax((M / out[f"{k}_T"])[c])); Tmin_core.append(np.nanmin(out[f"{k}_T"][c]))
out |= dict(coll_ratio_max=max(ratio_core), coll_ratio_min_of_max=min(ratio_core), coll_Tmin=min(Tmin_core))
print(f"H_inf max = {H_inf_max:.2e} GeV (r < {CFG['r_max']}, A_s = {CFG['A_s']}); T_reh max (instantaneous) = {T_cmb_max:.2e} GeV")
print(f"Freeze-in line: lowest T_reh = {out['FI_Tmin']:.2e} GeV; heaviest DM below the CMB bound M_X = {M_FI_cmb:.2e} GeV; "
      f"max M_X/T_reh below the bound = {out['FI_ratio_max']:.1f}")
print(f"Collision cores: lowest T_reh = {out['coll_Tmin']:.2e} GeV; max M_X/T_reh per line from {out['coll_ratio_min_of_max']:.1e} "
      f"to {out['coll_ratio_max']:.1e}")
np.savez(ROOT / "output/58_opening_plane.npz", **out)
