# gravdm v2

Code, stored numerical results and figure scripts for the main text and its Supplemental Material

> C.-W. Chiang and V. Q. Tran, *Superheavy gravitational dark matter from runaway bubble collisions* (2026), arXiv number to be added.

This version contains only the code and data that the main text and its Supplemental Material use. `docs/PROVENANCE.md` maps every
figure, table and quantitative statement of the main text and of the Supplemental Material (Secs. S1–S8) to the script and stored output
that produce it.

## What the paper computes

A stable particle X that couples only through gravity is pair-produced by s-channel graviton exchange between the field quanta of
colliding runaway bubble walls. The main text and its Supplemental Material

1. give the graviton-mediated cross sections for scalar, Dirac and Proca X, with the curvature couplings (Sec. S1);
2. derive the number of pairs per unit wall area of two colliding planar walls and test free passage with nonlinear lattice simulations
   in 1+1 dimensions at gamma = 10, 40, 80 and 160 (Secs. S2, S3; main-text Fig. 1);
3. sum the yield over the first-collision surfaces of a runaway vacuum transition with exponential nucleation, including the expansion
   during the transition, and compare it with thermal freeze-in (Secs. S4, S5; main-text Figs. 2, 3);
4. apply the result to the decay of a Standard Model false vacuum after inflation (the An–Li scenario) with a cosmological Monte Carlo
   (Sec. S7), and estimate the effects left out of the first-collision yield (Sec. S8).

## Repository layout

| path | content |
|---|---|
| `src/gravdm/` | physics package: cross sections, parton luminosity, 1+1-dimensional lattice, bubble Monte Carlo, freeze-in, An–Li background, bounce, end-of-inflation production, SM running, GW reach, plot style |
| `scripts/compute/NN_*.py` | compute scripts; each writes `output/NN_*.npz` (and `.log`/`.txt` where noted) |
| `scripts/plots/pNN_*.py` | figure scripts; they read `output/` only and write `plots/pNN_*.pdf` and `.png` |
| `output/` | stored results of all compute scripts; `env.txt` is the `pip freeze` of the environment used |
| `plots/` | the three figures of the main text and the five figures of the Supplemental Material |
| `tests/` | pytest suite for `src/gravdm` (limits, closed forms, literature normalizations, analytic-versus-numerical checks) |
| `data/` | third-party data, with sources and licences in `data/README.md` |
| `docs/PROVENANCE.md` | main-text and Supplemental Material element -> script -> output, run order |

Script numbers (`NN`) are not consecutive. They are kept fixed so that output files and scripts stay in one-to-one correspondence
with the companion long paper. The docstring of each script names the figures, equations, sections and statements of the main text and
of the Supplemental Material that it supports. Docstrings inside `src/gravdm/` and `tests/` cite the equations of the companion long
paper, which derives the same formulas in more detail.

## Installation

Python >= 3.10 and numpy >= 2.0. The results were produced with Python 3.13 and the versions pinned in `requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # or: export PYTHONPATH=$PWD/src
```

All commands below are run from the repository root.

## Tests

```bash
python -m pytest            # about 2.5 minutes; 280 passed, 1 skipped
```

## Figures (fast, from the stored outputs)

```bash
python scripts/plots/p02h_lattice_diag_yield_plot.py   # main-text Fig. 1 and SM Fig. S3 -> plots/p02h_lattice_diag_yield_perlog{,_mu0p2}
python scripts/plots/p62_yield_plot.py                 # main-text Fig. 2 -> plots/p62_yield
python scripts/plots/p58_opening_plane_plot.py         # main-text Fig. 3 -> plots/p58_opening_plane
python scripts/plots/p01b_universality_plot.py         # SM Fig. S1 -> plots/p01b_universality
python scripts/plots/p02e_lattice_invariants_plot.py   # SM Fig. S2 -> plots/p02e_lattice_invariants
python scripts/plots/p50_benchmark_dm_plane_plot.py    # SM Fig. S4 -> plots/p50_benchmark_dm_plane
python scripts/plots/p53_dm_plane_spins_plot.py        # SM Fig. S5 -> plots/p53_dm_plane_spins
```

Each takes a few seconds. With matplotlib 3.11.2 and the default macOS backend the figures are pixel-identical to those of the main text
and the Supplemental Material, except Fig. S2: the paper version was made with matplotlib 3.10.0, and with 3.11.2 the figure is one
point taller and the "0.99" contour label sits at a different point of the same contour. The data drawn are identical. With the `Agg`
backend the bounding boxes can differ by a few pixels.

## Compute scripts

Most numbers can be checked from the stored outputs without rerunning anything. The stored `.log` files of 02f, 03, 06, 07, 13, 14,
21, 24, 26, 39 and 58 are captured standard output: run these scripts as `python ... | tee output/NN_name.log`. To rerun, the order and the inputs of each script are
listed in `docs/PROVENANCE.md` (section 4). Runtime classes on a single laptop core:

- *fast* (< 1 min): 01, 02g, 02h, 05, 07, 10, 14, 15, 17, 23, 24b, 26, 28, 45, 47–50, 57, 58, 64 and all plots;
- *medium* (1 min to about 1 h): 02, 02i, 04, 06, 20, 21, 24, 27, 39, 46, 55, 56;
- *heavy* (several hours): lattice runs 02f at gamma = 80 and 160; Monte Carlo 03, 11, 18, 19, 63; end-of-inflation Bogoliubov
  calculations 08, 13. Script 18 needs about 0.5 GB of memory.

The lattice runs behind Fig. 1 and Figs. S2, S3 are

```bash
S=scripts/compute
for g in 10 40 80 160; do python $S/02f_lattice_dense.py $g; python $S/02f_lattice_dense.py $g 20; done      # 40 and 20 points per wall
for g in 80 160; do
  python $S/02f_lattice_dense.py $g 20 '{"Tw": 5.0, "tc_over_Tw": 4.8}' win                                # wider Fourier window
  python $S/02f_lattice_dense.py $g 20 '{"sample_dt": 0.00125}' dt2                                         # half sampling step
done
python $S/02g_lattice_yield_ratio.py; python $S/02h_lattice_diag_yield.py; python $S/02i_potential_term_check.py
```

The expansion correction (Sec. S4) is (the script writes its own log)

```bash
S=scripts/compute
python $S/63_expansion_correction.py '{"box": 24.0, "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}'
python $S/63_expansion_correction.py '{"box": 32.0, "seeds": [21, 22, 23, 24, 25, 26, 27, 28], "tag": "_box32"}'
```

Fig. 2 uses the reheating temperature 1e11 GeV: `python scripts/compute/46_freezein_curve.py 1e11` and
`python scripts/compute/49_universal_relic.py 1e11` write the `_T1e11` outputs (without an argument, 1e12 GeV).

## Data credits

- Detector noise and power-law-integrated sensitivity curves for ET, CE and LIGO–Virgo–KAGRA, used by script 23: K. Schmitz, Zenodo,
  DOI 10.5281/zenodo.3689582 (CC BY 4.0); K. Schmitz, JHEP 01 (2021) 097, arXiv:2002.04615.
- Quantum limit of interferometric detection (script 23): X. Guo et al., arXiv:2501.18146.

## Citing

If you use this code or its results, please cite the paper (see also `CITATION.cff`):

```bibtex
@article{Chiang:2026gravdm,
    author        = "Chiang, Cheng-Wei and Tran, Van Que",
    title         = "{Superheavy gravitational dark matter from runaway bubble collisions}",
    year          = "2026",
    note          = "arXiv number to be added"
}
```

## Licence

The code (`src/`, `scripts/`, `tests/`) is released under the MIT licence. The stored results (`output/`) and figures (`plots/`) are
released under CC BY 4.0. See `LICENSE` and `data/README.md`.
