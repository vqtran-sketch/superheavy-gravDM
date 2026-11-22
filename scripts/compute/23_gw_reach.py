"""23: reach of GW detectors for the signal of the An-Li scenario along the DM curves.

SM: Sec. S7, "Gravitational waves" (planned detectors reach only the low-frequency tail; the peak lies above the quantum limit of
interferometric detection for f_peak <~ 1e5 Hz with photon counting and <~ 6e5 Hz at the channel limit; Delta N_eff ≈ 5e-3).

Signal: h^2 Omega_peak = 5e-5 C_p alpha (Eq. (13) in An:2026ghw), f_peak from output/06 (C_k = 4.6), spectral template of
gravdm.gw_reach.
Ground-based: SNR from Eq. (1.1) in Schmitz:2020syl with the noise spectra of data/schmitz2020 (Zenodo 10.5281/zenodo.3689582; ET
cross-correlation, CE auto-correlation as in that data set, which is optimistic for CE).
High-frequency: quantum limits of Fig. 1 in Guo:2025cza (arXiv:2501.18146; ten years, single detector), as straight-line fits in log h_c.
Delta N_eff from the integrated spectrum (Eq. (125) in Caprini:2018mtu). The foreground is not included in the SNR; the residual
binary-neutron-star background (Sachdev:2020bkk, arXiv:2002.05365) is compared separately at 100 Hz.
Output: output/23_gw_reach.npz, output/23_gw_reach.log"""
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import gw_reach as gr
from gravdm.units import M_P

YR = 3.156e7
CFG = dict(data=ROOT / "data/schmitz2020", alphas=("0.05", "0.0005"), nus=("2.0", "3.0"), spins=("scalar", "dirac", "vector"),
           detectors=(("ET", 2), ("CE", 1), ("HLVK", 2)), t_obs_yr=(1, 10), S_max={"0.05": 4.3, "0.0005": 5.1}, freq_ambiguity=(2.3, 4.8),
           # log10 h_c = a log10 f + b; pixel fits to Fig. 1 in Guo:2025cza (photon counting: seeded fit, the labels overlap the colour)
           qlimit={"channel": (1.000, -33.246), "photon": (1.000, -31.750), "linear": (1.250, -31.732)},
           # Unresolved BNS residual for a CE+ET network, ~1e-11 at 10 Hz (Sachdev:2020bkk), scaled as f^(2/3) [LIT, assumed slope]
           bns=dict(omega10=1e-11, slope=2 / 3))
d = np.load(ROOT / "output/06_dm_plane.npz")
log = []


def say(s):
    print(s); log.append(s)


def load(n):
    a = np.loadtxt(CFG["data"] / f"strain_{n}.dat", comments="#"); p = np.loadtxt(CFG["data"] / f"plis_{n}.dat", comments="#")
    return 10**a[:, 0], 10**a[:, 1], 10**p[:, 0], 10**p[:, 1]


DET = {n: load(n) for n, _ in CFG["detectors"]}
hc_limit = lambda name, f: 10**(CFG["qlimit"][name][0] * np.log10(f) + CFG["qlimit"][name][1])


def all_snr(fp, Op, ir, T):
    return {n: gr.snr(DET[n][0], DET[n][1], gr.template(DET[n][0], fp, Op, ir), nd, T * YR) for n, nd in CFG["detectors"]}


rows = []
for a in CFG["alphas"]:
    Op, ir = float(d[f"a{a}_Omega_peak"]), a == "0.05"
    say(f"\nalpha={a}: Omega_peak={Op:.2e}; int dlnf Omega={gr.integrated_omega(1e5, Op, ir):.2e}; DeltaNeff={gr.integrated_omega(1e5, Op, ir) / 5.6e-6:.1e}")
    for nu in CFG["nus"]:
        for sp in CFG["spins"]:
            f0 = d[f"a{a}_nu{nu}_{sp}_f0"]; L4 = d[f"a{a}_nu{nu}_{sp}_L4"]
            i = int(np.nanargmin(np.where(f0 > 0, f0, np.nan))); fp = float(f0[i]); M = float(d["M_over_MP"][i] * M_P)
            s1, s10 = all_snr(fp, Op, ir, 1), all_snr(fp, Op, ir, 10)
            fs = fp * CFG["S_max"][a]**(-1 / 3); ss = all_snr(fs, Op, ir, 10)
            margin = {k: Op / gr.omega_of_hc(fp, hc_limit(k, fp)) for k in CFG["qlimit"]}
            say(f" nu={nu} {sp:6s} lowest fp={fp:.2e} Hz (M={M:.1e}, L4={L4[i]:.1e}): SNR 1yr ET={s1['ET']:.3f} CE={s1['CE']:.2f}; 10yr CE={s10['CE']:.2f}; "
                f"10yr CE with S_max shift (fp={fs:.1e})={ss['CE']:.2f}; Omega_peak/QL channel={margin['channel']:.1e} photon={margin['photon']:.1e} linear={margin['linear']:.1e}")
            rows.append((float(a), float(nu), fp, M, s1["ET"], s1["CE"], s10["CE"], ss["CE"], margin["channel"], margin["photon"], margin["linear"]))
say("\nFrequency scan, alpha=0.05 (10 yr):")
for fp in (1.2e4, 2e4, 4e4, 1e5, 1.4e5, 3e5, 8e5):
    s = all_snr(fp, 2.5e-8, True, 10)
    m = {k: 2.5e-8 / gr.omega_of_hc(fp, hc_limit(k, fp)) for k in CFG["qlimit"]}
    f1 = 100.0
    say(f" fp={fp:.1e}: SNR ET={s['ET']:.2e} CE={s['CE']:.2e} HLVK={s['HLVK']:.1e}; Omega(100 Hz)={gr.template(f1, fp, 2.5e-8):.1e} vs BNS residual {CFG['bns']['omega10'] * (f1 / 10)**CFG['bns']['slope']:.1e}; "
        f"margin QL channel {m['channel']:.1e} photon {m['photon']:.1e} linear {m['linear']:.1e}")
say("\nCross-check: PLISC at 100 Hz (SNR=1, 1 yr): " + ", ".join(f"{n}={np.interp(2, np.log10(DET[n][2]), np.log10(DET[n][3])):.2f} (log10)" for n in DET))
for a, Op in (("0.05", 2.5e-8), ("0.0005", 2.5e-10)):
    fc = np.sqrt(np.sqrt(Op / gr.A_OMEGA) / 10**CFG["qlimit"]["channel"][1])
    say(f"Peak crosses the channel quantum limit at fp = {fc:.2e} Hz for alpha = {a}")
np.savez(ROOT / "output/23_gw_reach.npz", rows=np.array(rows), cfg_qlimit=np.array(list(CFG["qlimit"].values())))
(ROOT / "output/23_gw_reach.log").write_text("\n".join(log) + "\n")
