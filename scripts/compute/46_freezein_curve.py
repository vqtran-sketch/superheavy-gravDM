"""46: thermal gravitational freeze-in yield Y_FI/(T_reh/M_P)^3 versus M_X/T_reh.

Main text: Dark-matter paragraph (Y_FI = 4.8e-5 (T_reh/M_P)^3 for a Dirac fermion), Figs. 2 and 3 (dashed and dash-dotted freeze-in
curves). SM: Sec. S5, Eq. (S15). Input to scripts 49, 50 and p62.

gravdm.freezein.yield_freezein_X: spin-2 channel of the Standard Model plasma plus the Higgs trace channel (xi_H), Maxwell-Boltzmann
statistics, radiation domination with constant g_*, T_max = T_reh (instantaneous reheating).
Check [DERIVED]: an independent implementation of Eq. (S15) with the closed-form coefficients of SM Eq. (S4) gives the same values at
M/T_reh = 0, 1, 3, 10 to 4 digits (Dirac 4.810e-5, 3.312e-5, 4.140e-6, 6.695e-11).
Usage: python 46_freezein_curve.py [T_reh in GeV]  (default 1e12; main-text Fig. 2 uses 1e11).
Output: output/46_freezein_curve.npz (T_reh = 1e12 GeV) or output/46_freezein_curve_T1e11.npz etc."""
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import freezein as fz
from gravdm.units import M_P

CFG = dict(Treh=1e12, spin="dirac", xi_H=0.0, x_grid=(-2.0, np.log10(40.0), 140))   # M/T_reh, log10 range; Y_FI ~ 1e-15 at 40
if len(sys.argv) > 1:                      # T_reh override (main-text Fig. 2 uses 1e11 GeV); default output unchanged
    CFG["Treh"] = float(sys.argv[1])
TAG = "" if CFG["Treh"] == 1e12 else f"_T{CFG['Treh']:.0e}".replace("+", "")
T = CFG["Treh"]
x = np.logspace(*CFG["x_grid"])
y = np.array([fz.yield_freezein_X(T, xi * T, CFG["spin"], xi_H=CFG["xi_H"]) for xi in x]) / (T / M_P) ** 3
np.savez(ROOT / f"output/46_freezein_curve{TAG}.npz", x=x, Y=y, Treh=T, spin=CFG["spin"], xi_H=CFG["xi_H"])
print(f"Y_FI/(T/M_P)^3: x={x[0]:.2g} -> {y[0]:.4e}; x={x[-1]:.3g} -> {y[-1]:.3e}")
