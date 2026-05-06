"""Higgs USR Grid-Scan Optimizer: find (phi0, y0) that puts the P_S(k) dip
at k ≈ 1-5e-4 Mpc^-1 and minimizes Planck low-ell chi^2.

Unlike usr_chi2_optimizer.py (differential_evolution), this script does a
deterministic grid scan because the Higgs USR dip is subtle (epsH ~ 2e-4)
and the valid parameter range is narrow and well-characterized.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.pspectrum_pipeline import run_pspectrum_pipeline, build_weighted_kgrid
from scripts.sachs_wolfe import compute_cl_sw, compute_cl_sw_powerlaw
from scripts.planck_data import get_planck_data, C_ell_to_d_ell
from scripts.constants import As, k_pivot_phys
from models import HiggsModel

# ── Helpers ──────────────────────────────────────────────────────────────────

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
    data = {"k_phys": pipeline_result["k_phys"], "P_S": pipeline_result["P_S"]}
    ells_model, C_ell = compute_cl_sw(data, ell_max=ell_max)
    D_model = C_ell_to_d_ell(ells_model, C_ell)
    ells_pl, D_pl, D_err = get_planck_data()
    mask = ells_pl <= ell_max
    D_interp = np.interp(ells_pl[mask], ells_model, D_model)
    return float(np.sum(((D_interp - D_pl[mask]) / D_err[mask]) ** 2))


def compute_dip_suppression(k_phys, P_S):
    """Percentage suppression at the dip relative to the pivot amplitude."""
    k_dip = find_k_dip(k_phys, P_S)
    if k_dip <= 0:
        return 0.0, k_dip
    ps_dip = np.interp(k_dip, k_phys, P_S)
    ps_pivot = np.interp(k_pivot_phys, k_phys, P_S)
    suppression = float((1.0 - ps_dip / ps_pivot) * 100)
    return suppression, k_dip
