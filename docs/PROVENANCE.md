# Provenance: main text and Supplemental Material -> script -> output

"Main text" is the body of the paper, "SM" its Supplemental Material (Secs. S1–S8, Figs. S1–S5, Tables SI–SIII). Script paths are relative to
`scripts/compute/` (numbers `NN`) or `scripts/plots/` (`pNN`). Outputs are in `output/`, figures in `plots/`. "Closed form" means a
formula of the text evaluated by hand or by a `gravdm` function. "[LIT]" marks a value taken from the cited reference.

## 1. Figures

| element | plot script | compute script(s) (direct <- upstream) | output file(s) read |
|---|---|---|---|
| main-text Fig. 1, lattice vs parton along p_1 = p_2, Dirac X, M_X = 1e-3 gamma m | `p02h_lattice_diag_yield_plot.py` | `02h` <- `02f` (gamma = 80, 160; 40 and 20 points per wall; `win`, `dt2` variants) <- `02` | `02h_lattice_diag_yield.npz` |
| main-text Fig. 2, yield vs M_X/T_reh at T_reh = 1e11 GeV | `p62_yield_plot.py` | `11`, `46 1e11`, `47` (<- 01), `49 1e11` (<- 11, 46, 47), `58` (<- 50) | `11_universal_flat.npz`, `46_freezein_curve_T1e11.npz`, `47_R_spread.npz`, `49_universal_relic_T1e11.npz`, `58_opening_plane.npz` |
| main-text Fig. 3, reheating temperature for the observed DM density | `p58_opening_plane_plot.py` | `58` (<- 50 <- 11, 46, 47) | `58_opening_plane.npz` |
| SM Fig. S1, mass suppression g(mu) | `p01b_universality_plot.py` | `01` | `01_universality.npz` |
| SM Fig. S2, lattice-to-parton I_2 and I_0 | `p02e_lattice_invariants_plot.py` | `02f` (gamma = 40, 80, 160; 40 and 20 points per wall) | `02f_lattice_dense_g{40,80,160}{,_n20}.npz` |
| SM Fig. S3, as Fig. 1 for M_X = 0.2 gamma m | `p02h_lattice_diag_yield_plot.py` | `02h` <- `02f` | `02h_lattice_diag_yield.npz` |
| SM Fig. S4, benchmark DM curves and GW peak frequency | `p50_benchmark_dm_plane_plot.py` | `50` (<- 11, 46, 47) | `50_benchmark_dm_plane.npz` |
| SM Fig. S5, An–Li curves for three spins and GW peak frequency | `p53_dm_plane_spins_plot.py` | `06` (<- 04 <- 01; 05 <- 03, 19), `04` | `06_dm_plane.npz`, `04_mass_dependence.npz` |

## 2. Tables

| table | content | script(s) | output(s) |
|---|---|---|---|
| SI | massless coefficients c_hat(xi_phi), threshold behavior | closed forms in `gravdm.xsec_analytic`; independent numerical amplitudes in `gravdm.gravxsec` | tests `test_gravxsec.py`, `test_amp_structure.py`, `test_literature_xsec.py`, `test_dirac_xi.py` |
| SII | An–Li window per (alpha, nu): M_X, L^{1/4}, f_peak, fraction above Lambda_cut | `20` (reads 04, 06, 07, 10, 14, 17, 21); the cut and the fraction above 0.46 M_P from `21` | `20_paper_numbers.txt`, `21_cutoff_window.npz` |
| SIII | effects beyond the first-collision yield | S_cr: `18`; pre-thermal scatterings: `57`; T ~ 100 MeV: closed form with [LIT] Saikawa–Shirai g_*; expansion: `63`; loops, contact terms, strong gravity, black holes: closed forms | see section 3 |

## 3. Quantitative statements

### Main text
| statement | source |
|---|---|
| Y_X ≃ 6.8e-3 (H/beta)(T_reh/M_P)^3; Q_beta = 1/3; Monte Carlo 0.336 ± 0.002 | closed form, `gravdm.semianalytic_q`; `11` |
| factor 1.5 vs a single bubble size; 1.4 from the spread of radii | `gravdm.benchdm` (`tests/test_benchdm.py`) |
| expansion factor ≈ e^{-4H/beta}: 0.67, 0.86, 0.95; curves move up in T_reh by ≤ 14% | `63` (section 5) |
| M̄_coll ≃ 11 (H/beta) M_P/nu^2; yield halved at M̄_coll/6, a tenth at 0.45 M̄_coll | `gravdm.benchdm`; `47`, `49` |
| Lambda_cut = 0.46 M_P | closed form (species scale) |
| Y_FI = 4.8e-5 (T_reh/M_P)^3; factor 142 H/beta | `gravdm.freezein`, `46` |
| crossing near 1e5 T_reh at H/beta = 0.1 (Fig. 2) | `49 1e11` |
| freeze-in needs T_reh ≳ 4e12 GeV; reach near 1e17 GeV; H_inf^max ≃ 5e13 GeV | `58` |
| cores 1e10 to 3e17 GeV; up to six orders of magnitude below M_X | `50`, `58` |
| end of inflation: < 10% above ≈ 2e15 GeV; H_inf below a few percent of M_X; abrupt end 3e15 to 8e15 GeV | `64`; abrupt end: `14` via `20` times H_inf^max |
| lattice: within 0.3% up to 0.7 gamma m, uncertainty 1 to 2% at gamma m; 2(m/p)^2; ≤ 0.5% and 0.1% vs superposition; 1.3% integrated | `02h`, `02f`, `02i`, `02g` |
| gamma ≃ M̄_coll/m ≳ 1e3 | `50` (M̄_coll), closed form |
| realization: beta_eff ≈ 3.5 H_c, Eq. (5) to about 10%; window 5e12 to 2e17 GeV (nu ≥ 2); H_inf below 2e10 to 2e11 GeV | `48`; `20`, `21`; `20`, `28` |
| xi_phi lowers the yield by up to 20%; curves up to 8% higher in T_reh | `24b` |
| loops 16% times a logarithm; contact terms and strong gravity about 1% or less | closed forms (SM S8) |
| shell crossings up to a factor of five; scatterings up to 20% and 80% | `18`; `57` |
| GW: 1e4 to 1e7 Hz (envelope, benchmark); 3e4 to 5e5 Hz (bulk flow, alpha = 0.05) | `50`; `20` |

### SM S1 (cross sections)
| statement | source |
|---|---|
| eqs. (S3)–(S4), Table SI | `gravdm.xsec_analytic`, `gravdm.gravxsec`; tests above |
| xi_phi = 1/6 lowers R by ≤ 20% at the upper edge of the core, < 10% for M_X ≲ 0.1 M̄_coll | `24b` (`24b_dirac_xi_benchmark.log`) |

### SM S2 (one collision)
| statement | source |
|---|---|
| g(0) = 1 for any profile | `01` (`theorem_check`) |
| partons with p < 0.1 gamma m carry 11% (erf) to 19% (tanh) of E/A | integrals of the `gravdm.parton` spectra (section 5) |
| Dirac 10% points mu = 0.42, 0.53, 0.68; within 10% at fixed wall energy; yields within 1.3 | `01_universality.npz` (section 5) |
| classical-source derivation | analytic (companion paper); no script |

### SM S3 (lattice)
| statement | source |
|---|---|
| grid 28 × 28 from m/2 to 3 gamma m; 20 and 40 points per wall; error ratio about four | `02f` |
| extrapolation uncertainty ≤ 0.3% for 0.1–0.7 gamma m, 1 to 2% at gamma m; window and time step < 1e-4 | `02f` (`_n20`, `_win`, `_dt2`), `02h` (`sigR`, `sig_win`, `sig_dt2`) |
| superposition without potential reproduces the parton prediction to 5e-5 | `02i` |
| I_2 within 9.4/1.9/0.6%, I_0 within 29/4.6/1.3% (gamma = 40/80/160); lattice/superposition 1.3/0.3/0.1% (I_2), 0.5/0.1% (I_0) | `02f` |
| soft deficit ≈ 2(m/p)^2; superposition with potential within 20% of it | `02i`, `02f` (`I2f`) |
| Fig. S2(b): within 0.7% for p_1 p_2 > (0.1 gamma m)^2 | `02f` (gamma = 160) |
| gamma = 10: I_2 excess 6 to 8% at 6–8 m; integrated yield up to 13% near mu ≈ 0.3 | `02f` (gamma = 10); `02g` |
| f_2 from 1 to 2/3; yield ratio f_2 r_2 + (1 − f_2) r_0 | closed form; `02g` |
| diagonal: 80% of the yield above 0.2 gamma m; within 0.3% up to 0.7 gamma m; M_X = 0.2 gamma m within 0.12% | `02h` |
| integrated: 0.8% (gamma = 80, mu ≤ 1); 0.2% (mu ≤ 0.5) and 1.3% (mu = 1) at gamma = 160 | `02g` |
| quantum scattering ≈ 0.03 lambda m/p | closed-form estimate |
| tilted quartic: 6% (I_2), 13% (I_0) at gamma_0 = 20; near rest inconclusive | `39` (<- 25 <- 16) |
| even sextic potential: first collision within 4% | `56` (<- 55, 25) |

### SM S4 (oblique collisions, first-collision sum)
| statement | source |
|---|---|
| flux factor within 1.2% for 30° to 180° | `15` (`tests/test_oblique.py`) |
| geometric factor of An and Jiang equals the flux factor | `45` |
| A_locus/V = 2 beta/3, Gamma(3) radii, <eta^3> = 1/2, Q_beta = 1/3 | `gravdm.semianalytic_q`, `gravdm.rspread`; `tests/test_semianalytic_q.py`, `tests/test_rspread.py` |
| Monte Carlo 0.336 ± 0.002 (eight realizations, box 32/beta) | `11` (`python scripts/compute/11_universal_flat.py --post`) |
| line-crossing estimator 0.3326 ± 0.0023 | section 5 |
| factor (8 pi)^{1/3}/(6 Q_beta) ≈ 1.5; 4/(8 pi)^{1/3} ≈ 1.37 | closed forms, `gravdm.benchdm` |
| Gamma(4) radii, 6 eta^5, M̄_coll = 525 pi L/(256 beta Dphi^2); R = 0.9/0.5/0.1 at 0.04/0.17/0.45 | `gravdm.rspread`, `gravdm.benchdm`; `47` |
| M̄_coll = 0.12 M_P and 0.012 M_P (nu = 3) | `50` |
| expansion: 0.669 ± 0.006, 0.857 ± 0.012, 0.951 ± 0.011; e^{-4.2 H/beta} within 3%; ≤ 1.14 in T_reh | `63` (two boxes; section 5) |
| comparison with Redi et al.: about thirty times larger | closed form with [LIT] Eq. (51) of Redi et al. |

### SM S5 (freeze-in, DM curves)
| statement | source |
|---|---|
| Y_FI = 4.8e-5 c_FI (T_reh/M_P)^3; 283 = 4 + 3 × 45 + 12 × 12; Y_X/Y_FI = 142; H/beta ≳ 7e-3 | `gravdm.freezein`, `46`; `tests/test_freezein.py` |
| cores 1e10 to 3e17 GeV; T_reh a few e10 to 1e13 GeV; lowest 5.6e10 GeV | `50`, `58` |
| species scale 0.46 M_P (0.09 M_P with M_P); unitarity at 2.3 M_P | closed forms |
| H_inf < 4.7e13 GeV; T_reh < 5.8e15 GeV; freeze-in DM needs T_reh ≥ 3.7e12 GeV, M_X/T_reh ≤ 21, heaviest 1.2e17 GeV; M_X/T_reh 3e4 to 4e6 on the cores | `58` (`58_opening_plane.log`) |
| end of inflation: M_X/H_inf ≳ 13 to 44 (Ema et al. estimate); 2e15 GeV; abrupt end 70 to 180 and (3–8)e15 GeV | `64`; `14` (<- 13) via `20` |
| black-hole collapse probability < 1e-40 | closed-form extrapolation of Eq. (15) of Gouttenoire and Volansky |
| GW envelope h^2 Omega_peak ~ 1e-6 (H/beta)^2, 1e4 to 1e7 Hz | `gravdm.benchdm`, `50` |
| bulk flow 0.56 beta, slopes 1.05 and −2.10 | [LIT] Table I of Lewicki and Vaskonen |
| t_stop/R_c ≈ 13/(k nu^2) | closed form with [LIT] Eq. (28) of Strumia |

### SM S6 (runaway walls)
All values are [LIT] (Bödeker–Moore, Gouttenoire et al., Höche et al., Espinosa et al., Ellis et al.). No script.

### SM S7 (An–Li realization)
| statement | source |
|---|---|
| gamma m/M_P = 6 sqrt(3 alpha) e_c/nu^2 | closed form; `04` |
| beta_eff ≈ 3.5 H_c; Eq. (S13) to about 10% | `48` |
| Monte-Carlo coefficient to about 1% | `05` (<- 03, 19) |
| factor 1.6 for thermalization at 100 MeV | closed form, Y_X ∝ g_*rho^{3/4}/g_*s with [LIT] g_*rho ≈ 17.7, g_*s ≈ 17.3 at 100 MeV (tabulated data of Saikawa and Shirai, not redistributed) |
| window 4e12 to 2e17 GeV, L^{1/4} 1e11 to 9e12 GeV (all spins, nu ≥ 2); 28% above Lambda_cut for nu = 1; Table SII | `20`, `21`, `06` |
| xi_H = 1 raises the upper edge by ≈ 1.7 | `24` |
| end of inflation 70 to 180 (abrupt), 9 to 21 (smooth) | `14` (<- 13), `10` (<- 08), via `20` |
| retention: H_inf ≲ (0.14–0.19) L^{1/4} (nu = 2), (0.05–0.07) L^{1/4} (nu = 3) | `28` (<- 27) |
| H_inf bound 2e10 to 2e11 GeV | `20` (with `07`, `14`, `28`) |
| xi_H ≲ 1 at the top-quark mass | `26` (`gravdm.sm_rge`) |
| black holes: < 1e-10 of the DM | closed-form estimate |
| h^2 Omega_peak 2.5e-8, 2.5e-10 | [LIT] An and Li |
| f_peak 3e4 to 5e5 Hz (alpha = 0.05), 1e6 to 8e6 Hz (5e-4) | `20` (Table SII) |
| quantum limit: f_peak ≲ 1e5 Hz (photon counting), ≲ 6e5 Hz (channel limit); Delta N_eff ≈ 5e-3 | `23` |

### SM S8 (beyond the first collision)
| statement | source |
|---|---|
| S_cr ≈ 4.3 (alpha = 0.05), 5.1 (5e-4) | `18` |
| pre-thermal hard–soft scatterings up to 20% and 80% | `57` |
| loops ≤ N Lambda_cut^2/(16 pi^2 M_P^2) ≈ 16% | closed form |
| contact terms ≲ 1e-10 (realization), ≲ 1e-2 (light end of the benchmark); (E/A)/(M_P^2 p) ≲ 1e-2 | closed-form estimates with M̄_coll from `50` and q90 from `21` |
| focusing distance, shock-wave deflection ∝ (H/beta)^2 or alpha | closed forms |

## 4. Run order and script catalogue

Run in this order to rebuild everything from scratch (later scripts read earlier outputs). The stored `.log` files of scripts that do
not write their own log (03, 06, 07, 13, 14, 21, 24, 26, 39, 63) are captured standard output: run them as
`python ... | tee output/NN_name.log`.

| script | reads | writes | gravdm modules | runtime |
|---|---|---|---|---|
| `01_universality_compute.py` | — | `01_universality.npz` | parton, xsec_analytic | fast |
| `02_lattice_compute.py` | — | `02_lattice.npz` | lattice1d | medium |
| `02f_lattice_dense.py GAMMA [PTS] [JSON TAG]` | imports `one()` from `02` | `02f_lattice_dense_g{GAMMA}[_n{PTS}][_{TAG}].{npz,log}` | lattice1d | heavy at gamma = 80, 160 |
| `02g_lattice_yield_ratio.py` | `02f` (gamma = 10, 80, 160) | `02g_lattice_yield_ratio.{npz,log}` | gravxsec, parton | fast |
| `02h_lattice_diag_yield.py` | `02f` (gamma = 80, 160, incl. `_win`, `_dt2`); imports `02g` | `02h_lattice_diag_yield.npz` | gravxsec, parton (via 02g) | fast |
| `02i_potential_term_check.py` | — | `02i_potential_term_check.{npz,log}` | lattice1d | medium |
| `03_yield_mc.py` | — | `03_yield_mc{TAG}.npz` | anli, bubbles_mc, units | heavy |
| `19_kappa_recheck.py` | — | `19_kappa_recheck_L{6,8}.jsonl` | anli, bubbles_mc, units | heavy |
| `05_kappaY_summary.py` | `03`, `19` | `05_kappaY.npz` | anli | fast |
| `04_mass_dependence.py` | `01` | `04_mass_dependence.npz` | anli, bubbles_mc | medium |
| `06_dm_plane.py` | `04`, `05` | `06_dm_plane.npz` | freezein, units, xsec_analytic | medium |
| `07_validity.py` | `04`, `06` | `07_validity.npz` | friction, units, window | fast |
| `08_inflationary_X.py` | — | `08_inflationary_X_y0{y0}.npz` | infprod | heavy |
| `10_inflation_condition.py` | `04`, `06`, `08` | `10_inflation_condition{,_power}.npz` | units, window | fast |
| `13_inflationary_spins.py` | — | `13_inflationary_spins.npz` | infprod_spins | heavy |
| `14_inflation_condition_spins.py` | `04`, `06`, `10`, `13` | `14_inflation_condition_spins.npz` | infprod_spins, units, window | fast |
| `17_higgs_trapping.py` | `04`, `06` | `17_higgs_trapping.npz` | units, window | fast |
| `21_cutoff_window.py` | `01`, `04`, `05`, `06` | `21_cutoff_window.npz` | freezein, parton, units, window, xsec_analytic | medium |
| `20_paper_numbers.py` | `04`, `06`, `07`, `10`, `14`, `17`, `21` | `20_paper_numbers.txt` | freezein, units, window | medium |
| `11_universal_flat.py` | — | `11_universal_flat.npz` | bubbles_mc, units | heavy |
| `15_oblique_check.py` | — | `15_oblique_check.npz` | — | fast |
| `18_shell_crossings_mc.py` | — | `18_shell_crossings_mc.npz` | anli, bubbles_mc | heavy (~3 h, ~0.5 GB) |
| `23_gw_reach.py` | `06`, `data/schmitz2020` | `23_gw_reach.{log,npz}` | gw_reach, units | fast |
| `24_dirac_xi.py` | `04`, `05` | `24_dirac_xi.npz` | freezein, parton, units, window, xsec_analytic | medium |
| `24b_dirac_xi_benchmark.py` | — | `24b_dirac_xi_benchmark.{npz,log}` | parton, xsec_analytic | fast |
| `27_bounce_window.py` | `04`, `06` | `27_bounce_window.{log,npz}` | bounce, units, window | medium |
| `28_trapping_bounce.py` | `04`, `06`, `27` | `28_trapping_bounce.{log,npz}` | bounce, units, window | fast |
| `26_kination_stability.py` | `04`, `06` | `26_kination_stability.npz` | sm_rge, units, window | fast |
| `16_lattice_accelerating.py` | — | — | lattice1d | helper (imported by 25) |
| `25_lattice_strong_accel.py` | — | — | lattice1d | helper (imported by 39, 56) |
| `39_lattice_anli_cubic.py` | — | `39_lattice_anli_cubic*.npz` | lattice1d | medium |
| `45_oblique_aj_checks.py` | — | — (prints) | — | fast |
| `46_freezein_curve.py [T_reh]` | — | `46_freezein_curve{,_T1e11}.npz` | freezein, units | medium |
| `47_R_spread.py` | `01` | `47_R_spread.npz` | rspread | fast |
| `48_anli_vs_benchmark.py` | — | `48_anli_vs_benchmark.npz` | anli | fast |
| `49_universal_relic.py [T_reh]` | `11`, `46`, `47` | `49_universal_relic{,_T1e11}.npz` | parton, rspread, units | fast |
| `50_benchmark_dm_plane.py` | `11`, `46`, `47` | `50_benchmark_dm_plane.npz` | benchdm, parton, rspread, units, window, xsec_analytic | fast |
| `55_lattice_higgs_like.py` | — | `55_lattice_higgs_like.{log,npz}` | lattice1d | medium |
| `56_lattice_higgs_like_measured.py` | — | `56_lattice_higgs_like_measured{tag}.{log,npz}` | lattice1d | medium |
| `57_prethermal_estimate.py` | `50` | `57_prethermal_estimate.{log,npz}` | units | fast |
| `58_opening_plane.py` | `50` | `58_opening_plane.{log,npz}` | units | fast |
| `63_expansion_correction.py [JSON]` | — | `63_expansion_correction{,_box32}.{log,npz}` | bubbles_mc | heavy |
| `64_end_of_inflation_condition.py` | `50` | `64_end_of_inflation_condition.{log,npz}` | units | fast |
| `p01b_universality_plot.py` | `01` | `plots/p01b_universality.{pdf,png}` | paper_plotstyle | fast |
| `p02e_lattice_invariants_plot.py` | `02f` | `plots/p02e_lattice_invariants.{pdf,png}` | paper_plotstyle | fast |
| `p02h_lattice_diag_yield_plot.py` | `02h` | `plots/p02h_lattice_diag_yield_perlog{,_mu0p2}.{pdf,png}` | paper_plotstyle | fast |
| `p50_benchmark_dm_plane_plot.py` | `50` | `plots/p50_benchmark_dm_plane.{pdf,png}` | paper_plotstyle, units | fast |
| `p53_dm_plane_spins_plot.py` | `04`, `06` | `plots/p53_dm_plane_spins.{pdf,png}` | paper_plotstyle, units, window | fast |
| `p58_opening_plane_plot.py` | `58` | `plots/p58_opening_plane.{pdf,png}` | paper_plotstyle, units | fast |
| `p62_yield_plot.py` | `11`, `46`, `47`, `49`, `58` | `plots/p62_yield.{pdf,png}` | paper_plotstyle, units | fast |

The Monte-Carlo invocations of scripts 03 and 19 (Table SII normalization) are those of the companion long paper:

```bash
S=scripts/compute
python $S/03_yield_mc.py '{"points": [[0.05, 6]],   "boxes": [6.0], "seeds": [201, ..., 224], "eps": [0.01, 0.005], "npts": 2000000, "tag": "c1"}'
python $S/03_yield_mc.py '{"points": [[0.0005, 6]], "boxes": [6.0], "seeds": [201, ..., 224], "eps": [0.01, 0.005], "npts": 2000000, "tag": "c2"}'
python $S/03_yield_mc.py '{"points": [[0.05, 5]],   "boxes": [6.0], "seeds": [201, ..., 224], "eps": [0.01, 0.005], "npts": 2000000, "tag": "c3"}'
python $S/03_yield_mc.py '{"points": [[0.05, 6]],   "boxes": [8.0], "seeds": [301, ..., 310], "eps": [0.01, 0.005], "npts": 2000000, "tag": "c4"}'
python $S/19_kappa_recheck.py 6 201,...,224 0.01,0.005 2000000 20000 > output/19_kappa_recheck_L6.jsonl
python $S/19_kappa_recheck.py 8 101,...,116 0.01,0.005 2000000 10000 > output/19_kappa_recheck_L8.jsonl
python $S/05_kappaY_summary.py
```
(`...` stands for the full comma-separated list of consecutive seeds. The number of sample points `npts` is not stored; 2×10^6 is
within the (1–2)×10^6 used.)

## 5. Numbers without a dedicated script or stored output

These come from closed forms or short checks. Each can be reproduced with a few lines using `gravdm` and the stored outputs:

- the soft-parton energy fraction 11–19% (integrals of the `gravdm.parton` spectra);
- the Dirac 10% points and the profile comparison at fixed wall energy (10% points within 10%, yields within a factor 1.3), from
  `01_universality.npz`;
- the line-crossing (Crofton) estimate Q_beta = 0.3326 ± 0.0023;
- the combined expansion factors of Sec. S4: inverse-variance means of the two box sizes in `63_expansion_correction.log` (box 24/beta)
  and `63_expansion_correction_box32.log` (box 32/beta), with half their difference as a systematic error;
- the factor 1.6 for thermalization at 100 MeV, from the tabulated g_*rho and g_*s of Saikawa and Shirai (arXiv:1803.01038,
  ancillary file `standardmodel2018.dat`);
- the primordial-black-hole collapse probability and the Hawking-radiation estimate;
- the estimates of Sec. S8 (loops, contact terms, focusing, shock-wave deflection) and the quantum-scattering estimate of Sec. S3.
