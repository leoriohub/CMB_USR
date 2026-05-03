"""
Planck 2018 low-ℓ TT binned power spectrum data.
Source: Planck 2018 results. VI. Cosmological parameters (Aghanim et al. 2020)
Table 4: Binned temperature power spectrum from the Commander component-separation algorithm.

D_ell = ell * (ell+1) * C_ell / (2*pi) in micro-K^2

For the low-ell range (ell <= 29), we use the Commander likelihood.
Errors are diagonal approximations — the full Planck low-ell likelihood
is non-Gaussian, but diagonal chi^2 is adequate for model scoping.
"""
import numpy as np
from scripts.constants import T_cmb

planck_lowl_tt = dict(
    ell=[
        2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29
    ],
    D_ell=[
        102.4, 774.3, 640.7, 981.2, 1017.5, 986.0, 915.8, 978.8, 1048.5,
        878.1, 1041.6, 806.0, 901.8, 666.3, 627.1, 583.0, 677.7, 507.0,
        625.8, 502.5, 549.6, 487.2, 445.2, 488.3, 413.6, 476.7, 365.1,
        371.8
    ],
    D_ell_err=[
        443.9, 412.2, 555.5, 481.1, 504.4, 508.0, 411.1, 416.6, 401.8,
        316.7, 228.2, 175.3, 172.8, 115.6, 100.7, 122.8, 133.2, 103.0,
        87.3, 76.0, 69.9, 64.0, 63.1, 56.4, 58.5, 57.6, 52.5, 53.2
    ],
)

def get_planck_data():
    """Return (ells, D_ell, D_ell_err) arrays for Planck 2018 low-ell TT."""
    d = planck_lowl_tt
    return (np.array(d["ell"]),
            np.array(d["D_ell"]),
            np.array(d["D_ell_err"]))

def d_ell_to_C_ell(ells, D_ell):
    """Convert D_ell [muK^2] to dimensionless C_ell."""
    return D_ell * 2.0 * np.pi / (ells * (ells + 1.0))

def C_ell_to_d_ell(ells, C_ell, Tcmb=T_cmb):
    """Convert dimensionless C_ell to D_ell [muK^2]."""
    conv = (Tcmb * 1e6) ** 2
    return C_ell * ells * (ells + 1.0) * conv / (2.0 * np.pi)
