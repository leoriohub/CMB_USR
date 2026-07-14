#!/usr/bin/env python3
r"""Benchmark comparison of ``beta_f_method="sigma0"`` vs ``"ps"``.

Loads the asteroid PS from cache (``ps_phi8.00_y0-0.000_nstar*.json``) or
generates a toy log-normal peak if the real one is not cached, then runs
:func:`beta_f_compaction` with both methods and prints a comparison table.
"""

from __future__ import annotations

import json
import os
import glob

import numpy as np

from scripts.compaction import beta_f_compaction

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "simulations", "pspectra")
EVIDENCE_DIR = os.path.join(PROJECT_ROOT, ".omo", "evidence")
EVIDENCE_FILE = os.path.join(EVIDENCE_DIR, "task-3-fix-erfc-compaction.txt")


def find_asteroid_ps() -> dict | None:
    """Search for an asteroid PS JSON file.

    Returns the loaded dict, or ``None`` if none is found.
    """
    pattern = os.path.join(OUTPUT_DIR, "ps_phi8.00_y0-0.000_nstar*.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    # Load the last (largest N_star) match
    with open(matches[-1]) as f:
        return json.load(f)


def toy_asteroid_ps(nk: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Generate a toy log-normal asteroid-like P_S(k).

    Parameters
    ----------
    nk : int
        Number of k-modes.

    Returns
    -------
    k : ndarray, shape (nk,)
        Log-spaced wavenumbers in [1e8, 1e14] Mpc⁻¹.
    ps : ndarray, shape (nk,)
        Log-normal peak with amplitude 1e-4 at k=1e10 and width 1 dex.
    """
    k = np.logspace(8, 14, nk)
    # Log-normal peak
    ps = 1e-4 * np.exp(-0.5 * ((np.log(k / 1e10)) / 1.0) ** 2)
    return k, ps


def main() -> None:
    # --- load or generate PS ---
    data = find_asteroid_ps()
    if data is not None:
        spec = data["spectrum"]
        k = np.array(spec["k_phys"])
        ps = np.array(spec["P_S"])
        fname = os.path.basename(
            sorted(glob.glob(os.path.join(OUTPUT_DIR, "ps_phi8.00_y0-0.000_nstar*.json")))[-1]
        )
        source = f"asteroid PS file ({fname})"
    else:
        k, ps = toy_asteroid_ps()
        source = "toy log-normal (asteroid-like)"
        print(f"[info] Real asteroid PS not found at {OUTPUT_DIR} — using {source}")

    print(f"[info] Using {source}")
    print(f"[info] k range: {k[0]:.3e} – {k[-1]:.3e}  ({len(k)} modes)")
    print()

    # --- run both methods ---
    print("Computing beta_f with method='sigma0' ...")
    beta_sigma0, M_sigma0, meta_sigma0 = beta_f_compaction(
        k, ps, beta_f_method="sigma0"
    )

    print("Computing beta_f with method='ps' ...")
    beta_ps, M_ps, meta_ps = beta_f_compaction(
        k, ps, beta_f_method="ps"
    )

    # --- statistics ---
    n_coll_sigma0 = int(np.sum(M_sigma0 > 0))
    n_coll_ps = int(np.sum(M_ps > 0))

    max_beta_sigma0 = float(np.max(beta_sigma0))
    max_beta_ps = float(np.max(beta_ps))

    max_M_sigma0 = float(np.max(M_sigma0)) if n_coll_sigma0 > 0 else 0.0
    max_M_ps = float(np.max(M_ps)) if n_coll_ps > 0 else 0.0

    # --- comparison table ---
    sep = "-" * 68
    header = f"{'Method':<16} | {'beta_f_max':<16} | {'n_collapsing':<13} | {'M_pbh_max [M_sun]':<16}"
    row_sigma0 = f"{'sigma0':<16} | {max_beta_sigma0:<16.6e} | {n_coll_sigma0:<13} | {max_M_sigma0:<16.6e}"
    row_ps = f"{'ps':<16} | {max_beta_ps:<16.6e} | {n_coll_ps:<13} | {max_M_ps:<16.6e}"

    print(sep)
    print(header)
    print(sep)
    print(row_sigma0)
    print(row_ps)
    print(sep)
    print()

    # --- sanity check ---
    if max_beta_sigma0 > max_beta_ps:
        verdict = (
            "VERIFIED: sigma0 method produces larger beta_f "
            f"(ratio = {max_beta_sigma0 / max(max_beta_ps, 1e-300):.3e})"
        )
    elif max_beta_sigma0 == 0 and max_beta_ps == 0:
        verdict = "NOTE: both methods produce zero beta_f"
    else:
        verdict = "NOTE: ps method produced larger or equal beta_f (opposite of expected)"

    print(verdict)

    # --- save evidence ---
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    lines = [
        "=" * 70,
        "Task-3: beta_f_method comparison (sigma0 vs ps)",
        "=" * 70,
        "",
        f"Source: {source}",
        f"k modes: {len(k)}  ({k[0]:.3e} – {k[-1]:.3e})",
        "",
        sep,
        header,
        sep,
        row_sigma0,
        row_ps,
        sep,
        "",
        verdict,
        "",
    ]
    with open(EVIDENCE_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[saved] {EVIDENCE_FILE}")


if __name__ == "__main__":
    main()
