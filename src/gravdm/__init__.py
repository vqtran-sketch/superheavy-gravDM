"""gravdm: physics modules for 'Gravitational dark matter from runaway bubble collisions'.

Submodules (imported explicitly, e.g. ``from gravdm import parton``):

  units            constants and unit conversions (natural units, GeV)
  xsec_analytic    closed-form phi phi -> h* -> X X cross sections (eqs. (2.3), (2.4), table 1, appendix A)
  gravxsec         independent numerical evaluation of the same amplitudes (cross-check)
  parton           wall spectra, parton luminosity K(zeta), mass-suppression function g(mu)
  lattice1d        1+1D nonlinear wall-collision lattice and Fourier invariants I2, I0
  bubbles_mc       first-collision-surface Monte Carlo (flat and cosmological)
  semianalytic_q   closed-form / quadrature first-collision integrals (Q_beta = 1/3)
  anli             self-consistent background of the An-Li scenario
  rspread          yield-weighted spread of the collision scale (R(M_X) of the benchmark)
  benchdm          closed forms of the flat-space benchmark (Mbar_coll, minimum T_reh, envelope GWs)
  freezein         thermal gravitational freeze-in yield
  window           robust-core criteria and interpolated window edges
  friction, infprod, infprod_spins, bounce, sm_rge, gw_reach
                   scenario conditions, end-of-inflation production, O(4)/O(3,1) bounces, SM running, GW reach
  paper_plotstyle  matplotlib style of the paper figures
"""
__version__ = "1.0.0"
