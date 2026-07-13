#!/usr/bin/env python
"""
Benchmark harness for compaction PBH abundance computation.

Compares three approaches:
  1. Scalar ``compute_zeta_r_profile`` (original for-loop)
  2. Vectorized ``_compute_zeta_profile_vectorized`` (matrix multiply)
  3. Press-Schechter ``compute_pbh_abundance`` (baseline)

Usage::

    python scripts/bench_compaction.py

Output: timing summary table with per-mode (ms) and total (s) for each method.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_cached_ps() -> tuple[np.ndarray, np.ndarray]:
    """Load the cached asteroid-config primordial power spectrum."""
    ps_path = os.path.join(
        ROOT,
        "outputs/simulations/pspectra",
        "ps_phi8.00_y0-0.000_nstar79.7.json",
    )
    if not os.path.exists(ps_path):
        raise FileNotFoundError(
            f"Cached PS file not found: {ps_path}\n"
            "Run the MS solver first to generate it."
        )
    with open(ps_path) as fh:
        rec = json.load(fh)
    k_phys = np.array(rec["spectrum"]["k_phys"], dtype=float)
    P_S = np.array(rec["spectrum"]["P_S"], dtype=float)
    print(f"Loaded {os.path.basename(ps_path)}: {len(k_phys)} k-modes")
    return k_phys, P_S


def _bench_scalar_zeta_profile(
    k_arr: np.ndarray, P_S_k: np.ndarray,
    k_modes: np.ndarray, r_profile: np.ndarray, zeta_c: float,
) -> float:
    """Run compute_zeta_r_profile for each mode and return mean total time.

    Uses *full* ``k_arr`` / ``P_S_k`` arrays (same as ``beta_f_compaction``)
    but iterates only over ``k_modes`` (e.g. PBH-relevant subset).
    """
    from scripts.compaction import compute_zeta_r_profile

    n_modes = len(k_modes)
    # Cap total runs to ~5s worth of computation
    n_runs = max(1, min(5, 500 // (n_modes * 30)))  # ~30ms/mode for scalar
    n_runs = max(1, n_runs)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        for ki in k_modes:
            R = 1.0 / float(ki)
            compute_zeta_r_profile(
                k_arr, P_S_k, r_profile, R_smooth=R, zeta_c=zeta_c,
            )
    dt = time.perf_counter() - t0
    return dt / n_runs


def _bench_vectorized_zeta_profile(
    ln_k: np.ndarray,
    w: np.ndarray,
    k_arr: np.ndarray,
    P_S_k: np.ndarray,
    k_modes: np.ndarray,
    r_profile: np.ndarray,
    zeta_c: float,
) -> float:
    """Run vectorized profile for each mode and return mean total time."""
    from scripts.compaction import _compute_zeta_profile_vectorized

    n_modes = len(k_modes)
    # Cap total runs to ~3s worth of computation  
    n_runs = max(1, min(10, 300 // (n_modes * 15)))  # ~15ms/mode for vectorized
    n_runs = max(1, n_runs)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        for ki in k_modes:
            R = 1.0 / float(ki)
            _compute_zeta_profile_vectorized(
                ln_k, w, k_arr, P_S_k, r_profile, R, zeta_c,
            )
    dt = time.perf_counter() - t0
    return dt / n_runs


def _bench_beta_f(
    k_arr: np.ndarray, P_S_k: np.ndarray, zeta_c: float,
    n_workers: int = 1,
) -> float:
    """Run beta_f_compaction and return total time."""
    from scripts.compaction import beta_f_compaction

    t0 = time.perf_counter()
    beta_f_compaction(k_arr, P_S_k, mu=zeta_c, n_workers=n_workers)
    return time.perf_counter() - t0


def _bench_press_schechter(
    k_arr: np.ndarray, P_S_k: np.ndarray, zeta_c: float,
) -> float:
    """Run Press-Schechter abundance once and return total time."""
    from scripts.full_pbh_pipeline import compute_pbh_abundance

    gamma_default = 0.4
    k_eq_default = 0.0104
    M_eq_default = 3.0e17

    t0 = time.perf_counter()
    compute_pbh_abundance(
        k_arr, P_S_k, gamma_default, zeta_c, k_eq_default, M_eq_default,
    )
    return time.perf_counter() - t0


def main() -> None:
    k_phys, P_S = _load_cached_ps()

    # Use a subset of PBH-relevant modes for per-profile benchmarks
    pbh_mask = k_phys > 1e6
    k_pbh = k_phys[pbh_mask]
    n_modes = len(k_pbh)
    print(f"PBH modes (k > 10⁶): {n_modes}")

    zeta_c = 1.0
    r_profile = np.logspace(-3.0, 1.5, 500)

    # Precompute ln_k and Simpson weights (used by vectorized version)
    from scripts.compaction import _simpson_weights
    ln_k = np.log(k_phys)
    w = _simpson_weights(ln_k)
    print(f"Grid: {len(ln_k)} points, Simpson weights computed")

    print("\n--- Per-profile timing ---")
    t_scalar = _bench_scalar_zeta_profile(
        k_phys, P_S, k_pbh, r_profile, zeta_c,
    )
    per_mode_scalar = t_scalar / n_modes * 1000
    print(f"  Scalar compute_zeta_r_profile:  {t_scalar:.4f}s total, "
          f"{per_mode_scalar:.4f}ms per mode")

    t_vec = _bench_vectorized_zeta_profile(
        ln_k, w, k_phys, P_S, k_pbh, r_profile, zeta_c,
    )
    per_mode_vec = t_vec / n_modes * 1000
    print(f"  Vectorized profile:             {t_vec:.4f}s total, "
          f"{per_mode_vec:.4f}ms per mode")

    if t_scalar > 0:
        speedup_per_mode = t_scalar / t_vec if t_vec > 0 else float("inf")
        print(f"  Per-mode speedup (vectorized vs scalar): {speedup_per_mode:.1f}×")

    print("\n--- Full pipeline timing ---")
    t_beta_1 = _bench_beta_f(k_phys, P_S, zeta_c, n_workers=1)
    print(f"  beta_f_compaction (n_workers=1):  {t_beta_1:.3f}s")

    t_beta_2 = _bench_beta_f(k_phys, P_S, zeta_c, n_workers=2)
    print(f"  beta_f_compaction (n_workers=2):  {t_beta_2:.3f}s")

    t_beta_10 = _bench_beta_f(k_phys, P_S, zeta_c, n_workers=10)
    print(f"  beta_f_compaction (n_workers=10): {t_beta_10:.3f}s")

    t_ps = _bench_press_schechter(k_phys, P_S, zeta_c)
    print(f"  Press-Schechter:                  {t_ps:.3f}s")

    # Summary table
    base_for_speedup = t_beta_1 if t_beta_1 > 0 else 1.0
    print("\n" + "=" * 90)
    print(f"{'Method':<35} {'n_workers':>10} {'total(s)':>10} "
          f"{'speedup vs sc':>14} {'speedup vs PS':>14}")
    print("-" * 90)
    print(f"{'Scalar compute_zeta_r_profile':<35} {'—':>10} "
          f"{t_scalar:>10.4f} {'—':>14} {'—':>14}")
    print(f"{'Vectorized profile':<35} {'—':>10} "
          f"{t_vec:>10.4f} "
          f"{t_scalar / t_vec if t_vec > 0 else float('inf'):>13.1f}× "
          f"{'—':>14}")
    print(f"{'beta_f_compaction':<35} {'1':>10} "
          f"{t_beta_1:>10.3f} "
          f"{base_for_speedup / t_beta_1:>13.1f}× "
          f"{t_ps / t_beta_1 if t_beta_1 > 0 else float('inf'):>13.1f}×")
    print(f"{'beta_f_compaction':<35} {'2':>10} "
          f"{t_beta_2:>10.3f} "
          f"{base_for_speedup / t_beta_2:>13.1f}× "
          f"{t_ps / t_beta_2 if t_beta_2 > 0 else float('inf'):>13.1f}×")
    print(f"{'beta_f_compaction':<35} {'10':>10} "
          f"{t_beta_10:>10.3f} "
          f"{base_for_speedup / t_beta_10:>13.1f}× "
          f"{t_ps / t_beta_10 if t_beta_10 > 0 else float('inf'):>13.1f}×")
    print(f"{'Press-Schechter':<35} {'—':>10} "
          f"{t_ps:>10.3f} {'—':>14} {'1.0×':>14}")
    print("=" * 90)


if __name__ == "__main__":
    main()
