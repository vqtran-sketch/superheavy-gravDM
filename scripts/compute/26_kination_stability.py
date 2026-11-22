"""26: Higgs stability during kination along the An-Li Dirac DM curves.

SM: Sec. S7, "Conditions on the inflationary era" (the tachyonic mass -6 xi_H H^2 during kination must destabilize neither the false
vacuum nor the electroweak vacuum afterward, which allows xi_H <~ 1 at the top-quark mass).

Two vacua are involved in the An-Li scenario [An:2026ghw]:
 (A) before H_c the SM sits in the An-Li false vacuum v1 (energy L above ours, distance Delta phi = nu L^{1/4}); its potential is not
     specified by An-Li. Parametrized by a barrier b L and width Delta phi (quartic double well: m_1^2 = 32 b L^{1/2}/nu^2).
     Kination condition [ESTIMATE]: the tachyonic curvature mass must not exceed the v1 curvature at the onset of kination,
        6 |xi_H| H_inf^2 < m_1^2   ->   H_inf < H_kin,v1 = sqrt(32 b L^{1/2}/(6 nu^2 |xi_H|)),
     compared with the Hawking-Moss bound. H decreases during kination, so H = H_inf is the worst case. This condition is weaker than
     the Hawking-Moss bound for xi_H <= 1.
 (B) after the transition (H <= H_c) the bubble interiors are in the electroweak vacuum while the kinating dark sector still
     dominates (alpha < 1); there the SM-only barrier condition of Laverda and Rubio applies [LIT: Laverda:2024qjt, Eq. (3.6)], evaluated
     with the 3-loop RG-improved tree potential (sm_rge.xi_max_barrier [DERIVED]) at H = H_c (worst case; thermal masses from the
     shells only help and are ignored). R = -6 H^2 is used (pure kination; the true |R| = 6 H^2 (1 - 3 alpha) is smaller, so this is
     conservative). For nu_LR = sqrt(3 xi/2) < 5 barrier existence is a necessary condition; the non-crossing criterion of that
     reference is calibrated only for nu_LR >= 5.
 H_c = sqrt(L/(3 alpha))/M_P (L = 3 alpha M_P^2 H_c^2). Units: GeV.
Inputs: output/06_dm_plane.npz, output/04_mass_dependence.npz.
Output: output/26_kination_stability.npz, printed summary (stored as output/26_kination_stability.log)."""
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import window, sm_rge as S
from gravdm.units import M_P

# SM inputs [LIT: PDG 2026, ParticleDataGroup:2026mpi]: top pole mass from cross sections 172.1 +- 0.6 GeV (the matching of
# Buttazzo:2013uya uses the pole mass); direct-measurement average 172.60 +- 0.27 GeV (pole-mass proxy, ~0.5 GeV scheme ambiguity);
# M_h = 125.13 +- 0.11 GeV; M_W = 80.3625 GeV; alpha_s(M_Z) = 0.1180 +- 0.0009.
CFG = dict(alphas=[0.05, 0.005, 0.0005], nus=[2.0, 3.0], spin="dirac", Mts=[171.5, 172.0, 172.1, 172.6, 172.7, 173.0], Mt_pole=172.1,
           Mt_direct=172.6, alphas_s_var=[0.1171, 0.1189],
           loops=[2, 3], xis=[0.1, 1.0, 5.0], bs=[0.1, 1.0], Ne=60, Mh=125.13, alphas_s=0.1180, MW=80.3625)


def H_HM(L4, b, Ne):
    """Hawking-Moss bound of script 17: 8 pi^2 b L/(3 H^4) > 3 Ne + ln Ne."""
    return (8 * np.pi ** 2 * b * L4 ** 4 / (3 * (3 * Ne + np.log(Ne)))) ** 0.25


def H_kin_v1(L4, nu, b, xi):
    """[ESTIMATE] 6 |xi| H^2 < m_1^2 = 32 b L^{1/2}/nu^2 (quartic double well of height b L and width nu L^{1/4})."""
    return np.sqrt(32 * b * L4 ** 2 / (6 * nu ** 2 * abs(xi)))


# (B) electroweak vacuum: f(h) tables per (Mt, loops); xi_max(H) = fmax/(6 H^2)
FMAX, MUL = {}, {}
for Mt in CFG["Mts"]:
    for lp in CFG["loops"]:
        mu, C = S.run(loops=lp, Mt=Mt, Mh=CFG["Mh"], alphas=CFG["alphas_s"], MW=CFG["MW"])
        FMAX[(Mt, lp)] = np.max(S.barrier_function(mu, C, lp)); MUL[(Mt, lp)] = S.mu_Lambda(mu, C)[0]
        print(f"Mt={Mt} {lp}-loop: mu_Lambda={MUL[(Mt, lp)]:.2e} GeV, xi_max(H) = {FMAX[(Mt, lp)] / 6 / 1e16:.3g} (1e8 GeV/H)^2", flush=True)

for Mt in (CFG["Mt_pole"], CFG["Mt_direct"]):                       # alpha_s(M_Z) +- 1 sigma at fixed M_t
    for a_s in CFG["alphas_s_var"]:
        mu, C = S.run(loops=3, Mt=Mt, Mh=CFG["Mh"], alphas=a_s, MW=CFG["MW"])
        FMAX[(Mt, 3, a_s)] = np.max(S.barrier_function(mu, C, 3))
        print(f"Mt={Mt} alpha_s={a_s} 3-loop: xi_max(H) = {FMAX[(Mt, 3, a_s)] / 6 / 1e16:.3g} (1e8 GeV/H)^2", flush=True)

d = np.load(ROOT / "output/06_dm_plane.npz"); Md = np.load(ROOT / "output/04_mass_dependence.npz")
M = d["M_over_MP"] * M_P
out = dict(Mts=np.array(CFG["Mts"]), fmax=np.array([[FMAX[(m, l)] for l in CFG["loops"]] for m in CFG["Mts"]]),
           muL=np.array([[MUL[(m, l)] for l in CFG["loops"]] for m in CFG["Mts"]]))
for alpha in CFG["alphas"]:
    for nu in CFG["nus"]:
        k = f"a{alpha}_nu{nu}_{CFG['spin']}"
        L4 = d[k + "_L4"]; R = Md[f"R_a{alpha}_nu{nu}_{CFG['spin']}"]
        core = np.isfinite(L4) & (R >= window.R_CORE) & (d[k + "_fcoll"] >= window.FCOLL_MIN) & (M < 0.25 * M_P)
        i = np.where(core)[0]; L = L4[i]
        Hc = L ** 2 / (np.sqrt(3 * alpha) * M_P)
        out[k + "_M"] = M[i]; out[k + "_L4"] = L; out[k + "_Hc"] = Hc
        print(f"\nalpha={alpha} nu={nu}: M_X {M[i[0]]:.1e}-{M[i[-1]]:.1e} GeV, L^1/4 {L.min():.1e}-{L.max():.1e}, H_c {Hc.min():.1e}-{Hc.max():.1e} GeV")
        for Mt in CFG["Mts"]:
            for lp in CFG["loops"]:
                xm = FMAX[(Mt, lp)] / (6 * Hc ** 2)
                out[f"{k}_ximaxEW_Mt{Mt}_L{lp}"] = xm
            xm3 = out[f"{k}_ximaxEW_Mt{Mt}_L3"]
            print(f"  (B) EW vacuum after H_c, Mt={Mt} (3-loop): xi_H < {xm3.min():.2g} (at max H_c) ... {xm3.max():.2g} (at min H_c); "
                  f"2-loop min {out[f'{k}_ximaxEW_Mt{Mt}_L2'].min():.2g}")
        for Mt in (CFG["Mt_pole"], CFG["Mt_direct"]):
            for a_s in CFG["alphas_s_var"]:
                xm = FMAX[(Mt, 3, a_s)] / (6 * Hc ** 2); out[f"{k}_ximaxEW_Mt{Mt}_L3_as{a_s}"] = xm
                print(f"  (B) Mt={Mt}, alpha_s={a_s} (3-loop): xi_H < {xm.min():.2g} (at max H_c)")
        for b in CFG["bs"]:
            hm = H_HM(L, b, CFG["Ne"]); out[f"{k}_b{b}_HHM"] = hm
            for xi in CFG["xis"]:
                hk = H_kin_v1(L, nu, b, xi); out[f"{k}_b{b}_xi{xi}_Hkinv1"] = hk
                top = np.minimum(hm, hk)
                print(f"  (A) b={b} xi_H={xi}: H_kin,v1 {hk.min():.1e}-{hk.max():.1e}, H_HM {hm.min():.1e}-{hm.max():.1e}; "
                      f"open H_inf range H_c < H_inf < min: min ratio min/H_c = {np.min(top / Hc):.1e}", flush=True)
np.savez(ROOT / "output/26_kination_stability.npz", **out)
