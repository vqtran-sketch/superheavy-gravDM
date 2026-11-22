"""Numerical check of the oblique-collision flux factor h = |n1 - n2|/2 (section 4.1; script 15)."""
import pathlib
import numpy as np
SRC = (pathlib.Path(__file__).resolve().parents[1] / "scripts/compute/15_oblique_check.py").read_text().split("rows = []")[0]
SRC = SRC.replace("ROOT = pathlib.Path(__file__).resolve().parents[2]   # repository root", "")
ns = {}
exec(SRC, ns)


def test_flux_factor_h():
    for th in (180.0, 120.0, 60.0):
        v = ns["events_per_area"](np.radians(th), 0.15, 1.5, 81, 241)
        assert abs(v / np.sqrt((1 - np.cos(np.radians(th))) / 2) - 1) < 0.02     # 1.2% discretisation at this grid
