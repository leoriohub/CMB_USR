"""
CAMB wrapper: full C_ell computation from custom primordial P_S(k).

Uses CAMB's set_initial_power_table() to inject the P_S(k) computed
by the inflation pipeline, then computes the full CMB temperature
angular power spectrum including:
  - Sachs-Wolfe effect (primordial)
  - Integrated Sachs-Wolfe effect (dark energy)
  - Acoustic oscillations
  - Silk damping
  - CMB lensing

References
----------
- Howlett et al. (2012), JCAP 04, 027 (CAMB paper)
- https://camb.readthedocs.io/
"""

import json
import os

import numpy as np

from scripts.constants import As, k_pivot_phys, ns_sr_default, CAMB_COSMOLOGY
from pspectrum_pipeline import load_pspectrum
from scripts.planck_data import C_ell_to_d_ell
from scripts.plotting import OUTPUT_DIRS


_LCDM_CACHE = {}
_CAMB_PARAMS_CACHE = {}


def _make_camb_params(ell_max=2500):
    """Create a CAMBparams object with Planck 2018 LCDM cosmology. Cached."""
    if ell_max not in _CAMB_PARAMS_CACHE:
        import camb
        params = camb.CAMBparams()
        params.set_cosmology(
            H0=CAMB_COSMOLOGY["H0"], ombh2=CAMB_COSMOLOGY["ombh2"],
            omch2=CAMB_COSMOLOGY["omch2"], tau=CAMB_COSMOLOGY["tau"],
            mnu=CAMB_COSMOLOGY["mnu"],
        )
        params.set_for_lmax(ell_max)
        params.Want_CMB = True
        params.WantScalars = True
        params.WantTensors = False
        _CAMB_PARAMS_CACHE[ell_max] = params
    return _CAMB_PARAMS_CACHE[ell_max].copy()


def _extend_pspectrum(k_phys, P_S, k_min=1e-6, k_max=10.0, n_extend=200):
    """
    Extend P_S(k) beyond the computed range by power-law extrapolation.

    CAMB requires the primordial spectrum over a wide k-range. The
    inflation pipeline typically covers ~1e-5 to 1 Mpc^-1. This function
    extrapolates at both ends using the local spectral index.
    """
    k_phys = np.asarray(k_phys)
    P_S = np.asarray(P_S)

    finite = np.isfinite(P_S) & (P_S > 0)
    k_ok = k_phys[finite]
    P_S_ok = P_S[finite]

    if len(k_ok) < 3:
        raise ValueError(f"Too few finite P_S values ({len(k_ok)}) for extrapolation")

    n_edge = min(5, max(3, len(k_ok) // 3))

    # Low-k end spectral index
    log_k_lo = np.log(k_ok[:n_edge])
    log_P_lo = np.log(P_S_ok[:n_edge])
    ns_lo = float(np.polyfit(log_k_lo, log_P_lo, 1)[0])

    # High-k end spectral index
    log_k_hi = np.log(k_ok[-n_edge:])
    log_P_hi = np.log(P_S_ok[-n_edge:])
    ns_hi = float(np.polyfit(log_k_hi, log_P_hi, 1)[0])

    ns_lo = np.clip(ns_lo, -0.5, 1.5)
    ns_hi = np.clip(ns_hi, -0.5, 1.5)

    P_S_at_kmin = P_S_ok[0]
    P_S_at_kmax = P_S_ok[-1]
    k_at_min = k_ok[0]
    k_at_max = k_ok[-1]

    # Only extend at ends the data doesn't already cover.
    # If k_at_min < k_min the low-k extension would create a
    # descending array → unsorted k_full → CAMB spline failure.
    if k_at_min > k_min:
        k_lo = np.logspace(np.log10(k_min), np.log10(k_at_min), n_extend, endpoint=False)
        P_S_lo = P_S_at_kmin * (k_lo / k_at_min) ** ns_lo
    else:
        k_lo, P_S_lo = np.array([]), np.array([])

    if k_at_max < k_max:
        k_hi = np.logspace(np.log10(k_at_max), np.log10(k_max), n_extend, endpoint=False)[1:]
        P_S_hi = P_S_at_kmax * (k_hi / k_at_max) ** ns_hi
    else:
        k_hi, P_S_hi = np.array([]), np.array([])

    k_full = np.concatenate([k_lo, k_ok, k_hi])
    P_S_full = np.concatenate([P_S_lo, P_S_ok, P_S_hi])

    if not np.all(np.diff(k_full) > 0):
        raise ValueError(
            f"Extended k-grid is not monotonic (k_min={k_min:.1e}, "
            f"k_at_min={k_at_min:.1e}, k_max={k_max:.1e}, k_at_max={k_at_max:.1e})"
        )

    return k_full, P_S_full


def compute_cl_full_camb(pspectrum_data, ell_max=2500, k_min=1e-6, k_max=10.0):
    """
    Compute full C_ell^TT via CAMB using a custom primordial P_S(k).

    Parameters
    ----------
    pspectrum_data : dict with keys "k_phys" and "P_S"
        Output from run_pspectrum_pipeline or load_pspectrum.
    ell_max : int, maximum multipole (default 2500)
    k_min : float, minimum k for CAMB spline (default 1e-6)
    k_max : float, maximum k for CAMB spline (default 10.0)

    Returns
    -------
    ells : ndarray, multipoles 2..ell_max
    C_ell_TT : ndarray, dimensionless C_ell^TT
    C_ell_TE : ndarray, C_ell^TE
    C_ell_EE : ndarray, C_ell^EE
    """
    import camb

    k_phys = np.asarray(pspectrum_data["k_phys"])
    P_S = np.asarray(pspectrum_data["P_S"])

    finite = np.isfinite(P_S) & (P_S > 0)
    k_ok = k_phys[finite]
    P_S_ok = P_S[finite]

    if len(k_ok) < 10:
        raise ValueError(f"Too few valid P_S values ({len(k_ok)})")

    k_full, P_S_full = _extend_pspectrum(k_ok, P_S_ok, k_min=k_min, k_max=k_max)

    params = _make_camb_params(ell_max=ell_max)
    params.set_initial_power_table(k_full, P_S_full)

    results = camb.get_results(params)
    cls = results.get_cmb_power_spectra(lmax=ell_max)

    ells = np.arange(2, ell_max + 1)
    total = cls["total"]
    # CAMB default returns C_ell * ell(ell+1)/(2pi) in dimensionless units.
    # Convert to conventional dimensionless C_ell for codebase consistency:
    ell_factor = ells * (ells + 1) / (2 * np.pi)
    # CAMB columns: [TT, EE, BB, TE]
    C_ell_TT = total[ells, 0] / ell_factor
    C_ell_TE = total[ells, 3] / ell_factor
    C_ell_EE = total[ells, 1] / ell_factor

    return ells, C_ell_TT, C_ell_TE, C_ell_EE


def compute_cl_camb_powerlaw(ell_max=2500, As=As, ns=ns_sr_default, k_pivot=k_pivot_phys):
    """
    Compute full C_ell^TT for a power-law primordial spectrum via CAMB.

    This is the LCDM baseline for chi^2 comparisons, computed with the
    same CAMB setup as the custom P_S(k) runs.
    """
    import camb

    params = _make_camb_params(ell_max=ell_max)
    params.InitPower.set_params(As=As, ns=ns, r=0, pivot_scalar=k_pivot)

    results = camb.get_results(params)
    cls = results.get_cmb_power_spectra(lmax=ell_max)

    ells = np.arange(2, ell_max + 1)
    total = cls["total"]
    # Convert from CAMB convention to conventional C_ell
    ell_factor = ells * (ells + 1) / (2 * np.pi)
    C_ell_TT = total[ells, 0] / ell_factor
    C_ell_TE = total[ells, 3] / ell_factor
    C_ell_EE = total[ells, 1] / ell_factor

    return ells, C_ell_TT, C_ell_TE, C_ell_EE



def compute_chi2_camb(pspectrum_data, ell_max=29):
    """
    Compute chi^2 vs Planck 2018 low-ell TT using full CAMB C_ell.

    Uses asymmetric Commander errors, same convention as the SW-only chi^2.
    """
    from scripts.chi2_analysis import chi2_model_lcdm

    ells, C_ell, _, _ = compute_cl_full_camb(pspectrum_data, ell_max=ell_max)
    D_ell = C_ell_to_d_ell(ells, C_ell)

    # LCDM baseline — cached (constant As, ns)
    if ell_max not in _LCDM_CACHE:
        _LCDM_CACHE[ell_max] = compute_cl_camb_powerlaw(ell_max=ell_max)
    ells_pl, C_ell_pl, _, _ = _LCDM_CACHE[ell_max]
    D_ell_pl = C_ell_to_d_ell(ells_pl, C_ell_pl)

    chi2_m, chi2_l = chi2_model_lcdm(
        D_ell, ells,
        D_lcdm=D_ell_pl, ells_lcdm=ells_pl,
        ell_max=ell_max,
    )

    return chi2_m, chi2_l, chi2_m - chi2_l


def save_camb_results(ells, C_ell_TT, C_ell_TE, C_ell_EE, metadata, output_path):
    """Save CAMB C_ell results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    record = {
        "_type": "result",
        "format_version": 2,
        "metadata": metadata,
        "c_ell": {
            "ells": ells.tolist(),
            "C_ell_TT": C_ell_TT.tolist(),
            "C_ell_TE": C_ell_TE.tolist(),
            "C_ell_EE": C_ell_EE.tolist(),
        },
    }
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)
    return output_path


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(
        description="Compute full C_ell via CAMB from P_S(k) JSON files"
    )
    parser.add_argument("pspectrum_path",
                        help="Path to P_S(k) JSON file or directory")
    parser.add_argument("--ell-max", type=int, default=2500)
    parser.add_argument("--output-dir", default=OUTPUT_DIRS["c_ell"])
    args = parser.parse_args()

    if os.path.isdir(args.pspectrum_path):
        files = sorted(glob.glob(os.path.join(args.pspectrum_path, "*.json")))
    else:
        files = [args.pspectrum_path]

    for fpath in files:
        data = load_pspectrum(fpath)
        print(f"Processing: {fpath}")

        ells, C_TT, C_TE, C_EE = compute_cl_full_camb(data, ell_max=args.ell_max)
        md = data["metadata"]
        phi0 = md.get("x0", 0)
        y0 = md.get("y0", 0)
        nstar = md.get("N_star", 0)
        out_path = os.path.join(args.output_dir, make_filename("camb", phi0, y0, nstar, ".json"))
        save_camb_results(ells, C_TT, C_TE, C_EE, data["metadata"], out_path)
        print(f"  Saved: {out_path}")

        chi2_m, chi2_l, dchi2 = compute_chi2_camb(data)
        print(f"  chi2 (model) = {chi2_m:.2f},  chi2 (LCDM) = {chi2_l:.2f},  Delta = {dchi2:+.2f}")
