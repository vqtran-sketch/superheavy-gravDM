"""Physical constants and unit conversions (natural units, GeV). Single source of truth for the package."""
import math

GEVINV_CM = 1.97327e-14          # 1 GeV^-1 in cm
GEVM2_CM2 = 3.8938e-28           # 1 GeV^-2 in cm^2
GEV_TO_HZ = 1.0 / (6.582119569e-25) / (2 * math.pi)   # f = k/(2π): k[GeV] -> f[Hz]  (ħ = 6.582e-25 GeV s)

M_PL = 1.220890e19               # Planck mass G^-1/2 [GeV]
M_P = M_PL / math.sqrt(8 * math.pi)  # reduced Planck mass [GeV] (= 2.435e18)

# Present-day cosmology [LIT: standard values, PDG astrophysical constants; Planck 2018]
T0 = 2.7255 * 8.617333e-14       # CMB temperature today [GeV]
GS0 = 3.909                      # entropy d.o.f. today
S0_CM3 = 2891.2                  # entropy density today [cm^-3]
RHOC_H2 = 1.05368e-5             # critical density / h^2 [GeV cm^-3]
OMEGA_DM_H2 = 0.120              # Planck 2018, arXiv:1807.06209 [LIT]
MY_DM = OMEGA_DM_H2 * RHOC_H2 / S0_CM3   # M_X * Y_X required [GeV]
G_STAR_SM = 106.75
