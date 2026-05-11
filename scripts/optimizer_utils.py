"""Shared utilities for Higgs USR optimizers.

Consolidates find_k_dip, compute_chi2, and _write_log that were
duplicated across higgs_usr_optimizer.py, overnight_scan.py, and
usr_chi2_optimizer.py.
"""

import json

import numpy as np

from scripts.sachs_wolfe import compute_cl_sw
from scripts.planck_data import get_planck_data, C_ell_to_d_ell


def find_k_dip(k_phys, P_S):
    """Find the k value at the USR dip (local minimum for k < 0.01).

    Returns -1.0 if no clear dip is found.
    """
    mask = k_phys < 0.01
    if not np.any(mask):
        return -1.0
    k_sub = k_phys[mask]
    ps_sub = P_S[mask]
    i_dip = int(np.argmin(ps_sub))
    if i_dip == 0 or i_dip == len(ps_sub) - 1:
        return -1.0
    return float(k_sub[i_dip])


def compute_chi2(pipeline_result, ell_max=29):
    """Diagonal chi^2 vs Planck 2018 low-ell TT (Sachs-Wolfe approximation)."""
    data = {
        "k_phys": pipeline_result["k_phys"],
        "P_S": pipeline_result["P_S"],
    }
    ells_model, C_ell = compute_cl_sw(data, ell_max=ell_max)
    D_model = C_ell_to_d_ell(ells_model, C_ell)

    ells_pl, D_pl, D_err = get_planck_data()
    mask = ells_pl <= ell_max
    D_pl_m = D_pl[mask]
    D_err_m = D_err[mask]
    D_interp = np.interp(ells_pl[mask], ells_model, D_model)
    return float(np.sum(((D_interp - D_pl_m) / D_err_m) ** 2))


def _write_log(log_file, entry):
    """Write a single evaluation result to the JSONL log."""
    log_file.write(json.dumps(entry) + "\n")
    log_file.flush()
