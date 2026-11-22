"""08: gravitational production of a heavy minimal scalar X at the transition from inflation to kination.

SM: Sec. S7, "Conditions on the inflationary era" (Bogoliubov calculation on the inflation-kination background; minimal scalar,
via scripts 10 and 20). Long run.

n a^3 (M/H0) from gravdm.infprod on the inflation-kination background of Chun:2009yu (tanh transition of width y0), y0 in {0.1, 1}.
Convergence: two late-time matching points y_out (stored as *_na3 and *_na3_yout_lo), and the k-grid doubled at one mass (conv_k).
Conversion to the yield of the An-Li scenario [DERIVED]: kination H = C a^-3 (C from the background), a_c = (C H0/H_c)^{1/3}, SM
comoving entropy (2 pi^2/45) g* (a_c T_*)^3 with T_* from script 06  =>  Y_inf = (n a^3 / C) H0^2 H_c / ((2 pi^2/45) g* T_*^3).
Resumable: results are appended to output/08_inflationary_X_y0<y0>.npz per (y0, M)."""
import sys, pathlib, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
from gravdm import infprod as ip

CFG = dict(y0s=[0.1, 1.0], Ms=[1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0], nk=28, y_outs=[2.5, 3.0])
# run time grows as e^{3 y_out}. Usage: python scripts/compute/08_inflationary_X.py <y0>  (one process per y0)
if len(sys.argv) > 1:
    CFG["y0s"] = [float(sys.argv[1])]
OUT = ROOT / f"output/08_inflationary_X_y0{CFG['y0s'][0]}.npz"
res = dict(np.load(OUT)) if OUT.exists() else {}


def na3(bg, M, nk, y_out):
    kmax = 3.0 + 2.5 * M
    ks = np.geomspace(0.02, kmax, nk)
    b2 = np.array([ip.beta2(bg, k, M, y_start=max(min(np.log(k) - 7, -5.0), -12.0), y_end=y_out) for k in ks])
    return np.trapezoid(ks ** 2 * b2, ks) / (2 * np.pi ** 2), ks, b2


for y0 in CFG["y0s"]:
    bg = ip.Background(y0=y0, y_out=max(CFG["y_outs"]))
    res[f"y0{y0}_C"] = bg.late_C()
    for M in CFG["Ms"]:
        key = f"y0{y0}_M{M}"
        if key + "_na3" in res:
            continue
        t0 = time.time()
        vals = [na3(bg, M, CFG["nk"], yo)[0] for yo in CFG["y_outs"]]
        res[key + "_na3"] = vals[-1]; res[key + "_na3_yout_lo"] = vals[0]
        print(f"y0={y0} M/H0={M}: n a^3 = {vals[-1]:.3e} H0^3 (y_out={CFG["y_outs"][0]}: {vals[0]:.3e}) ({time.time()-t0:.0f}s)", flush=True)
        np.savez(OUT, **res)
# k-grid convergence at one point
if "conv_k" not in res and CFG["y0s"] == [1.0]:
    bg = ip.Background(y0=1.0, y_out=3.0)
    res["conv_k"] = np.array([na3(bg, 3.0, CFG["nk"], 3.0)[0], na3(bg, 3.0, 2 * CFG["nk"], 3.0)[0]])
    print("k-grid convergence (M=3, y0=1):", res["conv_k"], flush=True)
    np.savez(OUT, **res)
