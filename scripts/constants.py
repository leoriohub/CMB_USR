"""
Centralized physical constants and model defaults for CMB anomaly analysis.

All constants reference Planck 2018 results (Aghanim et al. 2020).
The xi/lam ratio is fixed by As normalization to the CMB amplitude.
"""

# ── Physical constants ──────────────────────────────────────────────────────

S = 5e-5                    # conformal time unit (dimensionless)
As = 2.1e-9                 # Planck scalar amplitude at pivot
k_pivot_phys = 0.05         # Mpc^-1, Planck pivot scale
r_ls = 14000.0              # Mpc, comoving distance to last scattering
T_cmb = 2.7255              # K, CMB temperature

# ── Higgs model CMB-normalized defaults ─────────────────────────────────────
# xi/lam ratio is fixed by As normalization to Planck 2018.
# Changing one requires re-normalizing the other.

lam_default = 0.13
xi_default = 15000.0

# ── Initial condition defaults ──────────────────────────────────────────────

phi0_default = 5.70         # field value at start of integration
yi_usr_default = -0.10      # USR trigger (deep negative velocity)
yi_sr_default = -0.001      # standard slow-roll velocity

# ── Simulation defaults ─────────────────────────────────────────────────────

N_star_default = 60         # e-folds before end where pivot exits
ell_max_default = 29        # low-l TT range (Commander likelihood)
num_k_default = 80          # default k-mode count
k_min_default = 1e-5        # Mpc^-1
k_max_default = 1.0         # Mpc^-1
bg_steps_default = 10000    # background ODE integration steps
ms_steps_default = 5000     # Mukhanov-Sasaki ODE integration steps
T_max_default = 5000.0      # max conformal time
