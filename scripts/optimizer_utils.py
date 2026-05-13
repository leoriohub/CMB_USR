"""Shared utilities for optimizers: find_k_dip and _write_log."""

import json

import numpy as np


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


def _write_log(log_file, entry):
    """Write a single evaluation result to the JSONL log."""
    log_file.write(json.dumps(entry) + "\n")
    log_file.flush()
