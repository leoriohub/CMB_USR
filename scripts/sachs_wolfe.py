"""
Sachs-Wolfe CMB angular power spectrum from primordial P_S(k).

Computes C_ell^TT for ell <= 30 using the Sachs-Wolfe approximation:

    C_ell = (4π/25) ∫ dk/k  P_R(k)  j_ell²(k · r_ls)

The factor 4π/25 = (2/9) · (2π²/25) after converting the integral measure.
This approximation is valid only at large angular scales (ell ≲ 30),
where acoustic physics at last scattering is subdominant.

Conversion to D_ell (muK²) for comparison with Planck binned data:

    D_ell = ell(ell+1) C_ell T_cmb² / (2π)

References
----------
- Sachs & Wolfe (1967), ApJ, 147, 73
- Planck 2018 results. VI. Cosmological parameters (Aghanim et al. 2020)
"""

import json
import os
import sys

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.special import spherical_jn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from scripts.pspectrum_pipeline import load_pspectrum
from scripts.constants import As, k_pivot_phys, r_ls, T_cmb

# Planck 2018 best-fit LCDM (Aghanim et al. 2020, TT+lowE)
PLANCK_LCDM = dict(
    As=As,
    ns=0.965,
    k_pivot=k_pivot_phys,
    r_ls=r_ls,
    Tcmb=T_cmb,
)


def interpolate_ps(k_phys, P_S, k_min=None, k_max=None, n_fine=30000):
    """
    Interpolate P_S(k) onto a dense log-spaced grid for SW integration.

    Uses cubic spline in log(k) space and clamps to non-negative values.
    Default range spans the input k-grid with safety margins.
    The fine grid (default 30000 points) ensures the spherical Bessel
    integral converges, especially at higher ell where j_ell oscillates.
    """
    if k_min is None:
        k_min = max(k_phys[0], 1e-6)
    if k_max is None:
        k_max = min(k_phys[-1], 5.0)
    k_dense = np.logspace(np.log10(k_min), np.log10(k_max), n_fine)
    interp = interp1d(np.log(k_phys), P_S, kind='cubic', bounds_error=False,
                      fill_value='extrapolate')
    P_S_dense = interp(np.log(k_dense))
    P_S_dense = np.clip(P_S_dense, 0, None)
    return k_dense, P_S_dense


def compute_cl_sw(data, ell_max=30, r_ls=r_ls, n_fine=30000):
    """
    CMB angular power spectrum via Sachs-Wolfe integral.

    Parameters
    ----------
    data : dict with keys "k_phys" and "P_S" from run_pspectrum_pipeline
    ell_max : int, max multipole (recommended <= 30 for SW validity)
    r_ls : float, comoving distance to last scattering (Mpc)
    n_fine : int, k-grid density for Bessel integral

    Returns
    -------
    ells : ndarray, shape (ell_max - 1,), multipoles 2..ell_max
    C_ell_TT : ndarray, dimensionless C_ell

    Notes
    -----
    Uses C_ell = (4π/25) ∫ d(log k) P_R(k) j_ell²(k · r_ls).
    Normalisation 4π/25 = 8π/(9·25) accounts for the transfer function
    pre-factor and the conversion from curvature to temperature.
    """
    k_phys = np.asarray(data["k_phys"])
    P_S = np.asarray(data["P_S"])
    k_dense, P_S_dense = interpolate_ps(k_phys, P_S, n_fine=n_fine)

    ells = np.arange(2, ell_max + 1)
    C_ell_TT = np.zeros(len(ells))

    for i, ell in enumerate(ells):
        x = k_dense * r_ls
        j_ell = spherical_jn(ell, x)
        integrand = P_S_dense * j_ell ** 2
        C_ell_TT[i] = (4.0 * np.pi / 25.0) * simpson(
            integrand, x=np.log(k_dense)
        )

    return ells, C_ell_TT


def compute_cl_sw_powerlaw(k_min=1e-5, k_max=5.0, As=As, ns=0.965,
                           k_pivot=k_pivot_phys, ell_max=30, r_ls=r_ls, n_fine=30000):
    """
    Sachs-Wolfe C_ell for a power-law primordial spectrum.

    Used as the LCDM baseline for chi² comparisons.
    P_R(k) = As * (k / k_pivot)^(ns - 1).

    Returns (ells, C_ell_TT, P_R_grid) where P_R_grid is the power-law
    spectrum evaluated on the dense k-grid.
    """
    k_dense = np.logspace(np.log10(k_min), np.log10(k_max), n_fine)
    Ps_pl = As * (k_dense / k_pivot) ** (ns - 1.0)

    ells = np.arange(2, ell_max + 1)
    C_ell_TT = np.zeros(len(ells))

    for i, ell in enumerate(ells):
        x = k_dense * r_ls
        j_ell = spherical_jn(ell, x)
        integrand = Ps_pl * j_ell ** 2
        C_ell_TT[i] = (4.0 * np.pi / 25.0) * simpson(
            integrand, x=np.log(k_dense)
        )

    return ells, C_ell_TT, Ps_pl


def compute_cl_sw_from_file(path, ell_max=30, r_ls=r_ls):
    """Load a P_S(k) JSON file and compute its Sachs-Wolfe C_ell."""
    data = load_pspectrum(path)
    return compute_cl_sw(data, ell_max=ell_max, r_ls=r_ls)


def save_cl_results(ells, C_ell_TT, k_grid, Ps_grid, metadata, output_path):
    """Save C_ell results to JSON for later analysis or plotting."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    record = {
        "metadata": metadata,
        "ells": ells.tolist(),
        "C_ell_TT": C_ell_TT.tolist(),
        "k_grid": k_grid.tolist(),
        "Ps_grid": Ps_grid.tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)
    return output_path


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("pspectrum_path",
                        help="Path to P_S(k) JSON file or directory")
    parser.add_argument("--ell-max", type=int, default=30)
    parser.add_argument("--r-ls", type=float, default=r_ls)
    parser.add_argument("--output-dir", default="outputs/simulations/c_ell")
    args = parser.parse_args()

    if os.path.isdir(args.pspectrum_path):
        files = sorted(glob.glob(os.path.join(args.pspectrum_path, "*.json")))
    else:
        files = [args.pspectrum_path]

    for fpath in files:
        data = load_pspectrum(fpath)
        ells, C_ell = compute_cl_sw(data, ell_max=args.ell_max, r_ls=args.r_ls)
        bn = os.path.splitext(os.path.basename(fpath))[0]
        out_path = os.path.join(args.output_dir, f"C_ell_{bn}.json")
        save_cl_results(ells, C_ell, data["k_phys"], data["P_S"],
                        data["metadata"], out_path)
        print(f"Saved: {out_path}")
