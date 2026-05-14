"""
Centralized physical constants and model defaults for CMB anomaly analysis.

All constants reference Planck 2018 results (Aghanim et al. 2020).
The xi/lam ratio is fixed by As normalization to the CMB amplitude.
"""

import os

# ── Project root — computed, never hardcoded ────────────────────────────────

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Physical constants ──────────────────────────────────────────────────────

S = 5e-5                    # conformal time unit (dimensionless)

k_pivot_phys = 0.002        # Mpc^-1, large-scale pivot (ell≈28) for low-ell anomaly analysis
                            # Planck default is 0.05 (minimizes A_s–n_s correlation at high-ell);
                            # 0.002 anchors near quadrupole/octupole where suppression is observed.
                            # Shifts N_* by +ln(0.05/0.002) ≈ +3.22 e-folds vs 0.05 pivot.

As_planck = 2.1e-9          # Planck 2018 A_s at k_* = 0.05 Mpc^-1 (TT,TE,EE+lowE)
As = As_planck * (k_pivot_phys / 0.05) ** (0.965 - 1.0)  # extrapolated to k_pivot_phys
r_ls = 14000.0              # Mpc, comoving distance to last scattering
T_cmb = 2.7255              # K, CMB temperature

# ── Higgs model CMB-normalized defaults ─────────────────────────────────────
# xi/lam ratio is fixed by As normalization to Planck 2018.
# Changing one requires re-normalizing the other.

lam_default = 0.13
xi_default = 15000.0

# ── Initial condition defaults ──────────────────────────────────────────────

phi0_default = 5.70         # field value at start of integration
y0_usr_default = -0.10      # USR trigger (deep negative velocity, dx/dT at T=0)
y0_sr_default = -0.001      # standard slow-roll velocity

# ── Simulation defaults ─────────────────────────────────────────────────────

N_star_default = 60         # e-folds before end where pivot exits
ell_max_default = 29        # low-l TT range (Commander likelihood)
num_k_default = 80          # default k-mode count
k_min_default = 1e-5        # Mpc^-1
k_max_default = 1.0         # Mpc^-1
bg_steps_default = 10000    # background ODE integration steps
ms_steps_default = 5000     # Mukhanov-Sasaki ODE integration steps
T_max_default = 5000.0      # max conformal time
