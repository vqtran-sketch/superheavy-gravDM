"""Output-level checks of the cut yields (cutoff Lambda_cut, section 6): a pure removal cut must satisfy 0 <= R_cut <= R,
be monotone in Lambda_cut, and vanish for 2 M_X >= Lambda_cut. Reads stored outputs of scripts 21 and 50 (skipped if absent)."""
import pathlib
import numpy as np
import pytest

OUT = pathlib.Path(__file__).resolve().parents[1] / "output"
M_P = 2.435e18


def _load(name):
    p = OUT / name
    if not p.exists():
        pytest.skip(f"{name} not produced")
    return np.load(p)


def test_cutoff_window_21_removal_cut():
    d = _load("21_cutoff_window.npz")
    cuts = sorted(float(c) for c in d["cuts"])
    MoP = d["M_over_MP"]
    bases = sorted({k[: -len("_nocut_R")] for k in d.files if k.endswith("_nocut_R")})
    assert bases
    for b in bases:
        R = d[b + "_nocut_R"]
        prev = np.zeros_like(R)
        for lam in cuts:
            Rc = d[f"{b}_cut{lam:g}_R"]
            assert np.all(Rc >= 0)
            assert np.all(Rc <= R * (1 + 1e-12) + 1e-300), b
            assert np.all(Rc >= prev * (1 - 1e-12)), (b, lam)       # larger cut keeps at least as much
            assert np.all(Rc[2 * MoP >= lam] == 0), (b, lam)
            prev = Rc


def test_benchmark_50_removal_cut():
    d = _load("50_benchmark_dm_plane.npz")
    M = d["M"]; lam = float(d["cut"]) * M_P
    keys = [k[: -len("_nocut_R")] for k in d.files if k.endswith("_nocut_R")]
    assert keys
    for k in keys:
        R, Rc = d[k + "_nocut_R"], d[k + "_cut_R"]
        assert np.all(Rc >= 0) and np.all(Rc <= R * (1 + 1e-12) + 1e-300), k
        assert np.all(Rc[2 * M >= lam] == 0), k
