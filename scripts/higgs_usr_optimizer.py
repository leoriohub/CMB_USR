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
