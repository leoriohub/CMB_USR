"""
Centralized physical constants and model defaults for CMB anomaly analysis.

All constants reference Planck 2018 results (Aghanim et al. 2020).
The xi/lam ratio is fixed by As normalization to the CMB amplitude.
"""

import os

# ── Project root — computed, never hardcoded ────────────────────────────────

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Physical constants ──────────────────────────────────────────────────────

S = 5e-5  # conformal time unit (dimensionless)

k_pivot_phys = 0.002  # Mpc^-1, large-scale pivot (ell≈28) for low-ell anomaly analysis
# Planck default is 0.05 (minimizes A_s–n_s correlation at high-ell);
# 0.002 anchors near quadrupole/octupole where suppression is observed.
# Shifts N_* by +ln(0.05/0.002) ≈ +3.22 e-folds vs 0.05 pivot.

As_planck = 2.1e-9  # Planck 2018 A_s at k_* = 0.05 Mpc^-1 (TT,TE,EE+lowE)
As = As_planck * (k_pivot_phys / 0.05) ** (0.965 - 1.0)  # extrapolated to k_pivot_phys
ns_sr_default = 0.965  # Higgs inflation slow-roll prediction at N_* ≈ 55-60.
# Coincides with Planck 2018 TT+lowE best fit (0.965 ± 0.004).
# Used as ΛCDM baseline spectral index.
ns_method_default = "lsq"  # n_s extraction: "lsq" (least-squares fit) or "sr" (slow-roll formula)
r_ls = 14000.0  # Mpc, comoving distance to last scattering
T_cmb = 2.7255  # K, CMB temperature

# ── Physical constants (cgs) ──────────────────────────────────────────────
G = 6.67430e-8
M_SUN = 1.98892e33
MPC_CM = 3.085677581e24
K_B = 1.380649e-16  # erg/K
M_P = 1.6735575e-24
KM_PER_S = 1.0e5
C_LIGHT = 2.99792458e10
SIGMA_SB = 5.670374419e-5
A_R = 4.0 * SIGMA_SB / C_LIGHT  # erg/cm³/K⁴
GAMMA_B = 5.0 / 3.0
MU = 1.22
N_EFF = 3.046

# ── Planck 2018 ΛCDM cosmology (TT+lowE best fit) ───────────────────────────
# Used by CAMB for both ΛCDM baseline and custom P_S(k) C_ell computations.
# Values from Aghanim et al. 2020, Table 2 (TT+lowE column).
CAMB_COSMOLOGY = {
    "H0": 67.66,
    "ombh2": 0.02242,
    "omch2": 0.11933,
    "tau": 0.054,
    "mnu": 0.06,
}

# ── Higgs model CMB-normalized defaults ─────────────────────────────────────
# xi/lam ratio is fixed by As normalization to Planck 2018.
# Changing one requires re-normalizing the other.

lam_default = 0.13
xi_default = 15000.0

# ── Initial condition defaults ──────────────────────────────────────────────

phi0_default = 5.70  # field value at start of integration
y0_usr_default = -0.10  # USR trigger (deep negative velocity, dx/dT at T=0)
y0_sr_default = -0.001  # standard slow-roll velocity

# ── Simulation defaults ─────────────────────────────────────────────────────

N_star_default = 60  # e-folds before end where pivot exits
ell_max_default = 29  # low-l TT range (Commander likelihood)
num_k_default = 80  # default k-mode count
k_min_default = 1e-5  # Mpc^-1
k_max_default = 1.0  # Mpc^-1
bg_steps_default = 10000  # background ODE integration steps
ms_steps_default = 5000  # Mukhanov-Sasaki ODE integration steps
T_max_default = 5000.0  # max conformal time

# ── PBH parameters (Ezquiaga et al. 1705.04861, Sec. III) ───────────────────

zeta_c_default = 0.052  # critical collapse threshold for PBH formation
gamma_default = 0.4  # PBH mass efficiency factor
k_eq_default = 0.0104  # Mpc^-1, comoving wavenumber at matter-radiation equality
# Computed from CAMB_COSMOLOGY: k_eq = a_eq * H_eq
M_eq_default = 3.0e17  # M_sun, horizon mass at matter-radiation equality
# M_eq = c^3/(2G) * 1/H_eq (approximate from ΛCDM)
ACCRETION = 3e7  # mass growth from equality to present day (Chisholm 2006)
