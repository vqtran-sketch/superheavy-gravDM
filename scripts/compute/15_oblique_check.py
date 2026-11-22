"""15: direct numerical check of the oblique-collision flux factor.

SM: Sec. S4, "Oblique factor" (a direct integration over two crossing sheets of Gaussian thickness reproduces the flux factor within
1.2% for theta between 30 and 180 degrees).

Two planar null sheets of quanta moving along unit vectors n1, n2 (angle theta), column densities N1/A, N2/A, Gaussian thickness w:
   n_i(x, t) = (N_i/A) g_w(n_i . x - t).
Event density per 4-volume = n1 n2 sigma v_Moller, v_Moller = 1 - cos(theta).
The events are integrated over a space-time box and divided by the area of the collision surface swept by the intersection line of
the sheets (the plane (n1 - n2) . x = 0 inside the spatial box). Prediction [DERIVED, SM Eq. (S10)]:
events/area = (N1/A)(N2/A) sigma h, h = eperp = |n1 - n2|/2 (head-on: h = 1). With sigma ∝ s = 4 p1 p2 h^2 the total weight is h^3.
Units: N_i/A = 1, sigma = 1 (constant here; the h^2 from sigma ∝ s is kinematic and exact).
Output: printed table + output/15_oblique_check.npz."""
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root
CFG = dict(thetas_deg=[180, 150, 120, 90, 60, 30], w=0.15, half=1.5, ngrid=81, nt=241)


def events_per_area(theta, w, half, ngrid, nt):
    n1 = np.array([1.0, 0, 0]); n2 = np.array([np.cos(theta), np.sin(theta), 0])
    g = lambda u: np.exp(-u ** 2 / (2 * w * w)) / np.sqrt(2 * np.pi * w * w)
    xs = np.linspace(-half, half, ngrid); dx = xs[1] - xs[0]
    X, Y = np.meshgrid(xs, xs, indexing="ij")             # the problem is z-independent: work per unit z-length
    ts = np.linspace(-3 * half, 3 * half, nt); dt = ts[1] - ts[0]
    tot = 0.0
    for t in ts:
        u1 = n1[0] * X + n1[1] * Y - t; u2 = n2[0] * X + n2[1] * Y - t
        tot += np.sum(g(u1) * g(u2)) * dx * dx * dt
    tot *= (1 - np.cos(theta))                             # Moller flux, sigma = 1
    # locus plane (n1 - n2).x = 0 within the square (per unit z): its length inside [-half, half]^2
    d = n1 - n2; d = d / np.linalg.norm(d)
    tvec = np.array([-d[1], d[0]])                         # direction of the line in the xy-plane
    smax = min(half / abs(tvec[0]) if abs(tvec[0]) > 1e-12 else np.inf, half / abs(tvec[1]) if abs(tvec[1]) > 1e-12 else np.inf)
    length = 2 * smax
    return tot / length


rows = []
for th in CFG["thetas_deg"]:
    th_r = np.radians(th)
    h = np.sqrt((1 - np.cos(th_r)) / 2)                    # |n1 - n2|/2
    val = events_per_area(th_r, CFG["w"], CFG["half"], CFG["ngrid"], CFG["nt"])
    val2 = events_per_area(th_r, CFG["w"], CFG["half"], 2 * CFG["ngrid"] - 1, 2 * CFG["nt"] - 1)   # convergence (x2 grid)
    rows.append([th, h, val, val2])
    print(f"theta={th:3d} deg: h={h:.4f}  events/area = {val:.4f} (x2 grid {val2:.4f})  ratio to h: {val/h:.4f}")
np.savez(ROOT / "output/15_oblique_check.npz", rows=np.array(rows), header="theta_deg h events_per_area events_per_area_x2grid")
