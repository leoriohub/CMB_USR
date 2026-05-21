"""
Planck 2018 low-ℓ TT binned power spectrum data.
Source: Planck 2018 results. V. CMB power spectra and likelihoods
(Aghanim et al. 2020, A&A 641, A5).
Data file: COM_PowerSpect_CMB-TT-full_R3.01.txt from IRSA Planck Release 3
(https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/)

For ℓ = 2-29, the spectrum is derived from the Commander component-separation
algorithm applied to Planck 2018 temperature maps between 30 and 857 GHz.

D_ell = ℓ(ℓ+1) C_ℓ T_cmb^2 / (2π)  [μK^2]

The errors are asymmetric 68% confidence limits including foreground
subtraction uncertainties. For the low-ℓ range (ℓ ≤ 29), we use the
Commander likelihood. A diagonal chi^2 approximation is used for model
scoping (the full Planck low-ℓ likelihood is non-Gaussian).
"""
import os

import numpy as np
from scripts.constants import T_cmb

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "Planck")
DATA_FILE = os.path.join(DATA_DIR, "planck_2018_low_ell_tt.csv")

# PLIK binned/unbinned likelihood data from Planck 2018 Release 3.01
_PLIK_UNBINNED = os.path.join(DATA_DIR, "COM_PowerSpect_CMB-TT-full_R3.01.txt")
_PLIK_BINNED = os.path.join(DATA_DIR, "COM_PowerSpect_CMB-TT-binned_R3.01.txt")

_planck_data = None


def _load_data():
    """Load low-ℓ Commander data from CSV, caching the result."""
    global _planck_data
    if _planck_data is not None:
        return _planck_data
    data = np.loadtxt(DATA_FILE, skiprows=1, delimiter=",")
    _planck_data = {
        "ell": data[:, 0].astype(int),
        "D_ell": data[:, 1],
        "D_ell_err_lower": data[:, 2],
        "D_ell_err_upper": data[:, 3],
    }
    return _planck_data


def get_planck_data():
    """Return (ells, D_ell, D_ell_err_sym) for Planck 2018 low-ℓ TT.

    D_ell_err_sym is the average of the lower and upper 68% confidence
    limits. Use `get_planck_data_asymmetric()` for the full asymmetric
    error bars.
    """
    d = _load_data()
    err_sym = 0.5 * (d["D_ell_err_lower"] + d["D_ell_err_upper"])
    return (d["ell"], d["D_ell"], err_sym)


def get_planck_data_asymmetric():
    """Return (ells, D_ell, D_ell_err_lower, D_ell_err_upper).

    The errors are the 68% confidence limits from the Commander
    component-separation algorithm.
    """
    d = _load_data()
    return (d["ell"], d["D_ell"], d["D_ell_err_lower"], d["D_ell_err_upper"])


def d_ell_to_C_ell(ells, D_ell, Tcmb=T_cmb):
    """Convert D_ell [μK^2] to dimensionless C_ell."""
    return D_ell * 2.0 * np.pi / (ells * (ells + 1.0)) / (Tcmb * 1e6) ** 2


def C_ell_to_d_ell(ells, C_ell, Tcmb=T_cmb):
    """Convert dimensionless C_ell to D_ell [μK^2]."""
    conv = (Tcmb * 1e6) ** 2
    return C_ell * ells * (ells + 1.0) * conv / (2.0 * np.pi)


def load_planck_unbinned():
    """Load Planck 2018 unbinned TT spectrum (ℓ=2-2508), returns (ells, D_ell)."""
    data = np.loadtxt(_PLIK_UNBINNED, skiprows=1)
    return data[:, 0].astype(int), data[:, 1]


def load_planck_binned():
    """Load Planck 2018 binned TT spectrum (ℓ≈47-2500), returns (ells, D_ell, D_err_sym)."""
    data = np.loadtxt(_PLIK_BINNED, skiprows=1)
    return data[:, 0].astype(int), data[:, 1], data[:, 2]
