# Third-party data

## `schmitz2020/` — detector noise and sensitivity curves (redistributed)

| file | content | used by |
|---|---|---|
| `strain_{CE,ET,HLVK}.dat` | strain noise power spectra | `scripts/compute/23_gw_reach.py` (signal-to-noise ratios and Delta N_eff, SM Sec. S7) |
| `plis_{CE,ET,HLVK}.dat` | power-law-integrated sensitivity curves (SNR = 1, 1 yr) | `scripts/compute/23_gw_reach.py` |

- **Source:** K. Schmitz, *New Sensitivity Curves for Gravitational-Wave Experiments*, Zenodo dataset, version v1 (27 Feb 2020),
  DOI [10.5281/zenodo.3689582](https://doi.org/10.5281/zenodo.3689582). The files here come from the archives
  `strain_noise_power_spectra.tar.gz` and `power-law-integrated_sensitivities.tar.gz` of that record. They are unmodified,
  including their comment headers.
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the Zenodo record.
- **Please cite:** K. Schmitz, "New Sensitivity Curves for Gravitational-Wave Signals from Cosmological Phase Transitions",
  JHEP 01 (2021) 097, arXiv:2002.04615, doi:10.1007/JHEP01(2021)097, together with the Zenodo DOI above.
