"""49: relic-abundance crossings and consistency checks for the flat-space benchmark at fixed T_reh.

Main text: Fig. 2 (circles at the crossings with the relic line for H/beta = 0.1; crossing near 1e5 T_reh). Input to p62.

Total yield: Y = [coeff (H/beta) R(M/Mbar) + Y_FI(M/T)/(T/M_P)^3] (T/M_P)^3, coeff = K Q_beta/(960 pi) (script 11, main-text Eq. (5)),
R from script 47, Y_FI from script 46 (the same T_reh). Relic condition M_X Y = units.MY_DM [Omega h^2 = 0.12, SM Eq. (S17)].
Mbar_coll = 2 sqrt(3) S0 (H/beta) M_P/nu^2 [DERIVED, main-text Eq. (6), SM Eq. (S14)], independent of T_reh for Dphi = nu L^{1/4}.
Checks per benchmark (H/beta, nu, T_reh):
  * T_reh^min = [MY_DM M_P^3/(coeff (H/beta) Mbar max_mu(mu R))]^{1/3} (no relic solution below it) [DERIVED];
  * parton sqrt(s) of the massless yield: q50, q90 and fractions above 0.1, 0.2, 0.46 M_P (Lambda_cut of SM Sec. S5),
    from the single-collision cdf of sqrt(s)/M_coll (parton.K, tanh) folded over the yield-weighted M_coll distribution (rspread);
  * M_X < M_P/4, 2 M_X vs the species scale 0.46 M_P;
  * L^{1/4} = A_g^{1/4} T_reh, Dphi = nu L^{1/4} < M_P, H = sqrt(L/3)/M_P, beta/T_reh (Eq. (5) needs M_X >> beta);
  * X becomes non-relativistic at T_nr ~ T_reh M_X/p0 with p0 ~ q50/2 (free streaming, rough) [estimate].
Usage: python 49_universal_relic.py [T_reh in GeV]  (default 1e12; main-text Fig. 2 uses 1e11, which needs the matching script 46 output).
Output: output/49_universal_relic.npz (or _T1e11.npz) and printed table."""
import sys, pathlib
import numpy as np
from scipy import optimize
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import rspread as rs, parton as pt
from gravdm.units import M_P, MY_DM, G_STAR_SM

CFG = dict(Treh=1e12, Hbeta=(0.1, 0.01), nu=3.0, Q_beta=1.0 / 3.0, profile="tanh", spin="dirac", cuts=(0.1, 0.2, 0.46),
           Mmax_frac=0.25, zgrid=(-10.0, np.log10(30.0), 4000))
if len(sys.argv) > 1:                      # T_reh override (main-text Fig. 2 uses 1e11 GeV); default output unchanged
    CFG["Treh"] = float(sys.argv[1])
TAG = "" if CFG["Treh"] == 1e12 else f"_T{CFG['Treh']:.0e}".replace("+", "")
T = CFG["Treh"]; norm = (T / M_P) ** 3
coeff = CFG["Q_beta"] * float(np.load(ROOT / "output/11_universal_flat.npz")["K"]) / (960 * np.pi)
S = np.load(ROOT / "output/47_R_spread.npz"); assert str(S["spin"]) == CFG["spin"] and str(S["profile"]) == CFG["profile"]
Rf = lambda m: np.exp(np.interp(np.log(m), np.log(S["mu"]), np.log(np.maximum(S["R"], 1e-300)), left=0.0, right=-700.0))
F = np.load(ROOT / f"output/46_freezein_curve{TAG}.npz"); assert float(F["Treh"]) == T and str(F["spin"]) == CFG["spin"]
Fi = lambda x: np.exp(np.interp(np.log(x), np.log(F["x"]), np.log(np.maximum(F["Y"], 1e-300)), left=np.log(F["Y"][0]), right=-700.0))

# single-collision cdf of sqrt(s)/M_coll for the massless yield (as script 21), folded over ln(M_coll/Mbar)
z = np.logspace(*CFG["zgrid"]); lz = np.log(z)
Kz = np.array([pt.K(zi, CFG["profile"]) for zi in z])
f = z * Kz; c = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(lz))])
y_s, cdf_s = 2 * np.sqrt(z), c / c[-1]
u, p = rs.lns_density()
above = lambda lam: np.trapezoid(p * (1 - np.interp(lam * np.exp(-u), y_s, cdf_s, left=0.0, right=1.0)), u)   # lam = Lambda/Mbar
quant = lambda q: np.exp(optimize.brentq(lambda l: (1 - above(np.exp(l))) - q, -12.0, 6.0))

muR_max = np.max(S["mu"] * S["R"])
Ag = np.pi ** 2 * G_STAR_SM / 30
L4 = Ag ** 0.25 * T; H = np.sqrt(L4 ** 4 / 3) / M_P
out = dict(Treh=T, nu=CFG["nu"], Hbeta=np.array(CFG["Hbeta"]), L4=L4, H=H, muR_max=muR_max)
print(f"T_reh = {T:.1e} GeV: L^1/4 = {L4:.3e} GeV, Dphi = {CFG['nu'] * L4:.2e} GeV = {CFG['nu'] * L4 / M_P:.1e} M_P, H = {H:.2e} GeV;"
      f" max mu R = {muR_max:.4f}")
for hb in CFG["Hbeta"]:
    Mb = 2 * np.sqrt(3) * rs.S0 * hb * M_P / CFG["nu"] ** 2
    Tmin = (MY_DM * M_P ** 3 / (coeff * hb * Mb * muR_max)) ** (1 / 3)
    lnres = lambda lnM: np.log(np.exp(lnM) * norm * (coeff * hb * Rf(np.exp(lnM) / Mb) + Fi(np.exp(lnM) / T)) / MY_DM)
    Ms = np.logspace(np.log10(0.1 * T), np.log10(M_P), 3000); v = np.array([lnres(np.log(m)) for m in Ms])
    roots = np.array([np.exp(optimize.brentq(lnres, np.log(Ms[i]), np.log(Ms[i + 1]), xtol=1e-12))
                      for i in np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]])
    q50, q90 = quant(0.5) * Mb, quant(0.9) * Mb
    fr = [above(cut * M_P / Mb) for cut in CFG["cuts"]]
    k = f"hb{hb:g}"
    out |= {f"{k}_Mbar": Mb, f"{k}_Tmin": Tmin, f"{k}_roots": roots, f"{k}_q50": q50, f"{k}_q90": q90, f"{k}_frac_above": np.array(fr),
            f"{k}_beta": H / hb}
    print(f"H/beta = {hb}: Mbar = {Mb:.3e} GeV = {Mb / M_P:.4f} M_P = {Mb / T:.2e} T_reh; T_reh^min = {Tmin:.2e} GeV; beta/T_reh = {H / hb / T:.1e}")
    print(f"   parton sqrt(s) (massless yield): q50 = {q50 / M_P:.3f} M_P, q90 = {q90 / M_P:.3f} M_P; fraction above "
          + ", ".join(f"{cut} M_P: {x:.2g}" for cut, x in zip(CFG["cuts"], fr)))
    for r in roots:
        share = Fi(r / T) / (coeff * hb * Rf(r / Mb) + Fi(r / T))
        ok = r < CFG["Mmax_frac"] * M_P
        Tnr = T * r / (0.5 * q50)
        print(f"   relic root M_X = {r:.3e} GeV = {r / T:.3g} T_reh = {r / M_P:.2g} M_P; R = {Rf(r / Mb):.3g}; freeze-in share {share:.1e};"
              f" M_X < M_P/4: {ok}; 2M_X/(0.46 M_P) = {2 * r / (0.46 * M_P):.2g}; T_nr ~ {Tnr:.1e} GeV")
np.savez(ROOT / f"output/49_universal_relic{TAG}.npz", **out)
