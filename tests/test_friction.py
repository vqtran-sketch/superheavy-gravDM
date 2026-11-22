"""Tests for friction.py."""
import sys, pathlib
import numpy as np
from gravdm import friction as fr


def test_T_equality_reproduces_chun_eq57():
    """[LIT: Chun-Scopel-Zaballa 0904.0675 eq. (5.7)] our derivation of T_* from rho_ratio = 1 + entropy conservation."""
    for H0, I, b1 in [(1e13, 0.1, 2.0), (3e10, 0.03, 5.0)]:
        assert np.isclose(fr.T_equality(H0, I, b1), fr.T_equality_chun(H0, I, b1), rtol=1e-12)


def test_scalings():
    assert np.isclose(fr.T_at(1e5, 2e12) / fr.T_at(1e5, 1e12), 2 ** (2 / 3), rtol=1e-12)     # T ∝ H0^{2/3}
    assert np.isclose(fr.rho_ratio_at(1e4, 1e12) / fr.rho_ratio_at(1e5, 1e12), 10 ** (2 / 3), rtol=1e-12)  # ∝ a^2 ∝ Hc^{-2/3}
    assert np.isclose(fr.rho_ratio_at(1e5, 1e12, thermalisation="entropy") / fr.rho_ratio_at(1e5, 1e12),
                      (4 / fr.G_STAR_SM) ** (1 / 3), rtol=1e-12)
    H = fr.H0_max_runaway(1e12, 0.05, 1e17)
    assert np.isclose(fr.T_at(np.sqrt(1e48 / 0.15) / fr.M_P, H, 0.19),
                      (1e48 / (0.65 ** 2 * 0.65e12 * 1e5)) ** (1 / 3), rtol=1e-10)
