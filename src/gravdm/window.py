"""Definition of the robust core of the DM lines, shared by scripts 06, 07, 10, 14, 17 and p06 (sections 4.4 and 5.4 of the paper).

Lower edge: thermal freeze-in below 10% of the collision yield, Y_FI <= 0.1 Y_coll, i.e. f_coll = Y_coll/(Y_coll + Y_FI) >= 1/1.1
(condition (ii) of section 4.4). Upper edge: R(M) >= 0.1 (condition (i)). The callers also apply an upper cap on L^{1/4}
(alpha^{1/4} x 1.4e16 GeV, the CMB bound on the inflationary energy density) and on M_X (passed as Mmax).
"""
FI_MAX_OVER_COLL = 0.1
FCOLL_MIN = 1.0 / (1.0 + FI_MAX_OVER_COLL)
R_CORE = 0.1


def interp_edges(M, core, fcoll, R, L4=None, L4max=None, Mmax=None):
    """Window edges of the first contiguous core segment, interpolated linearly in ln M between grid points
    (the nearest grid point inside the window, spacing x1.14, would shift either edge by up to 14%).
    Lower edge: crossing of f_coll through FCOLL_MIN (None if the core starts at the first grid point: no edge in the grid).
    Upper edge: crossing of R through R_CORE, or the cap Mmax, or the crossing of L4 through L4max, whichever fails first.
    M in GeV (increasing grid); returns (M_lo or None, M_hi or None)."""
    import numpy as np
    idx = np.where(core)[0]
    if len(idx) == 0:
        return None, None
    i0 = idx[0]; i1 = i0
    while i1 + 1 < len(M) and core[i1 + 1]:
        i1 += 1
    lM = np.log(M)

    def x(y, j, thr):          # crossing of y through thr between j and j+1
        return float(np.exp(lM[j] + (thr - y[j]) * (lM[j + 1] - lM[j]) / (y[j + 1] - y[j])))

    lo = None
    if i0 > 0:
        lo = x(fcoll, i0 - 1, FCOLL_MIN) if np.isfinite(fcoll[i0 - 1]) and fcoll[i0 - 1] < FCOLL_MIN else float(M[i0])
    hi = None
    if i1 + 1 < len(M):
        j = i1
        cands = []
        if R[j + 1] < R_CORE:
            cands.append(x(R, j, R_CORE))
        if Mmax is not None and M[j + 1] >= Mmax:
            cands.append(float(Mmax))
        if L4 is not None and L4max is not None and np.isfinite(L4[j + 1]) and L4[j + 1] >= L4max:
            cands.append(x(L4, j, L4max))
        hi = min(cands) if cands else float(M[i1])
    return lo, hi
