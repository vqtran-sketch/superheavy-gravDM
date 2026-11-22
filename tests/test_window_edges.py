"""window.interp_edges: window edges interpolated between grid points."""
import sys, pathlib
import numpy as np
from gravdm import window


def test_linear_profiles_recover_exact_edges():
    """f_coll and R linear in ln M: the interpolated edges equal the exact crossings for any grid offset."""
    M = np.logspace(10, 18, 81)
    lM = np.log(M)
    fcoll = window.FCOLL_MIN + 0.02 * (lM - np.log(3.3e12))     # crosses at 3.3e12
    R = window.R_CORE - 0.05 * (lM - np.log(7.7e16))            # crosses at 7.7e16
    core = (fcoll >= window.FCOLL_MIN) & (R >= window.R_CORE)
    lo, hi = window.interp_edges(M, core, fcoll, R)
    assert np.isclose(lo, 3.3e12, rtol=1e-9) and np.isclose(hi, 7.7e16, rtol=1e-9)


def test_cap_and_edgeless():
    M = np.logspace(10, 18, 81)
    fcoll = np.ones_like(M); R = np.ones_like(M)
    core = M < 6e17
    lo, hi = window.interp_edges(M, core, fcoll, R, Mmax=6e17)
    assert lo is None and hi == 6e17
