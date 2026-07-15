"""
BoTorch Bayesian optimization for PBH parameter tuning in Ezquiaga CHI model.

Optimizes (x_c, c, β, χ₀, N_star, ζ_c) to find configurations where the P_S(k)
peak falls in a target mass gap (sub-solar [10⁻⁶, 10⁻²] M_⊙ or asteroid
[10⁻¹⁷, 10⁻¹⁵] M_⊙) with maximal PBH abundance f_total while maintaining
n_s compatibility with Planck.

Usage
-----
Local run:
    python scripts/optimize_pbh_botorch.py \\
        --x-c-lo 0.75 --x-c-hi 0.85 \\
        --c-lo 0.5 --c-hi 10.0 \\
        --beta-lo 1e-6 --beta-hi 9e-4 \\
        --chi0-lo 4.0 --chi0-hi 8.0 \\
        --N-star-lo 50 --N-star-hi 70 \\
        --zeta-c-lo 0.04 --zeta-c-hi 0.10 \\
        --target subsolar \\
        --n-trials 200 --n-init 32 --q-batch 4 \\
        --workers 8 --seed 42

Lab machine (nohup + ssh):
    ssh uni "cd ~/Documentos/CMB_USR && git pull && \\
        source ~/miniconda3/etc/profile.d/conda.sh && conda activate cmb-anomaly && \\
        nohup python scripts/optimize_pbh_botorch.py \\
            --x-c-lo 0.75 --x-c-hi 0.85 \\
            --target asteroid \\
            --n-trials 500 > ~/pbh_opt_asteroid.log 2>&1 & echo PID=\\$!"

BoTorch and GPyTorch are imported lazily inside functions that need them.
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

from scripts.constants import (
    As_planck,
    k_pivot_phys,
    gamma_default,
    k_eq_default,
    M_eq_default,
)


def _build_bounds(args):
    """Build (d, 2) bounds tensor from CLI args, with beta/chi0 in log-space.

    Dimension order: [x_c, c, beta, chi0, N_star, zeta_c].
    beta and chi0 are stored as logarithms internally for BoTorch.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments with ``_lo`` and ``_hi`` for each dim.

    Returns
    -------
    bounds : torch.Tensor, shape (6, 2)
        Column 0 = lower bounds, column 1 = upper bounds.
        beta and chi0 rows are in log-space.

    Raises
    ------
    ValueError
        If any lo >= hi, or any log-dim lo <= 0.
    """
    dims = [
        ("x_c", args.x_c_lo, args.x_c_hi, False),
        ("c", args.c_lo, args.c_hi, False),
        ("beta", args.beta_lo, args.beta_hi, True),
        ("chi0", args.chi0_lo, args.chi0_hi, True),
        ("N_star", args.N_star_lo, args.N_star_hi, False),
        ("zeta_c", args.zeta_c_lo, args.zeta_c_hi, False),
    ]

    bounds = []
    for name, lo, hi, use_log in dims:
        if lo >= hi:
            raise ValueError(
                f"Bounds violation for '{name}': lo={lo} >= hi={hi}. "
                f"Ensure lo < hi for every dimension."
            )
        if use_log:
            if lo <= 0:
                raise ValueError(
                    f"Log-space bounds violation for '{name}': lo={lo} <= 0. "
                    f"Log-space dimensions require lo > 0."
                )
            bounds.append([np.log(lo), np.log(hi)])
        else:
            bounds.append([lo, hi])

    return torch.tensor(bounds, dtype=torch.float64)


def _resolve_mass_bin(args):
    """Resolve target mass bin bounds [M_sun] from CLI args.

    ``--mass-bin-lo`` / ``--mass-bin-hi`` override ``--target`` when set.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments with ``target``, ``mass_bin_lo``, ``mass_bin_hi``.

    Returns
    -------
    lo : float
        Lower bound of target mass range [M_sun].
    hi : float
        Upper bound of target mass range [M_sun].
    """
    if args.mass_bin_lo is not None and args.mass_bin_hi is not None:
        return (args.mass_bin_lo, args.mass_bin_hi)

    targets = {
        "subsolar": (1e-6, 1e-2),
        "asteroid": (1e-17, 1e-15),
        "all": (1e-17, 1e2),
    }
    return targets.get(args.target, (1e-17, 1e2))


def main():
    p = argparse.ArgumentParser(
        description="BoTorch Bayesian optimization for PBH parameter tuning "
                    "in the Ezquiaga CHI model"
    )

    # ── Bounds ────────────────────────────────────────────────────────────────
    p.add_argument("--x-c-lo", type=float, default=0.75,
                    help="Lower bound for x_c (inflection point coordinate)")
    p.add_argument("--x-c-hi", type=float, default=0.85,
                    help="Upper bound for x_c (inflection point coordinate)")
    p.add_argument("--c-lo", type=float, default=0.5,
                    help="Lower bound for c = xi0 * kappa^2 mu^2")
    p.add_argument("--c-hi", type=float, default=10.0,
                    help="Upper bound for c = xi0 * kappa^2 mu^2")
    p.add_argument("--beta-lo", type=float, default=1e-6,
                    help="Lower bound for beta (deviation from inflection, physical)")
    p.add_argument("--beta-hi", type=float, default=9e-4,
                    help="Upper bound for beta (deviation from inflection, physical)")
    p.add_argument("--chi0-lo", type=float, default=4.0,
                    help="Lower bound for chi0 (initial field, physical)")
    p.add_argument("--chi0-hi", type=float, default=8.0,
                    help="Upper bound for chi0 (initial field, physical)")
    p.add_argument("--N-star-lo", type=float, default=50.0,
                    help="Lower bound for N_star (e-folds before end where pivot exits)")
    p.add_argument("--N-star-hi", type=float, default=70.0,
                    help="Upper bound for N_star")
    p.add_argument("--zeta-c-lo", type=float, default=0.04,
                    help="Lower bound for zeta_c (collapse threshold)")
    p.add_argument("--zeta-c-hi", type=float, default=0.10,
                    help="Upper bound for zeta_c (collapse threshold)")

    # ── Fixed params ──────────────────────────────────────────────────────────
    p.add_argument("--y0", type=float, default=-1e-4,
                    help="Initial field velocity (default: -1e-4)")

    # ── Target ────────────────────────────────────────────────────────────────
    p.add_argument("--target", choices=["subsolar", "asteroid", "all"],
                    default="subsolar",
                    help="Target mass range for PBH peak placement. "
                         "subsolar=[1e-6, 1e-2] M_sun, asteroid=[1e-17, 1e-15] M_sun, "
                         "all=[1e-17, 1e2] M_sun")

    # ── Constraints ───────────────────────────────────────────────────────────
    p.add_argument("--N-total-min", type=float, default=65.0,
                    help="Minimum N_total to accept a config")
    p.add_argument("--f-total-min", type=float, default=0.0,
                    help="Minimum f_total (PBH abundance) to accept")
    p.add_argument("--f-total-max", type=float, default=1.0,
                    help="Maximum f_total (PBH abundance) to accept")
    p.add_argument("--n-s-min", type=float, default=0.94,
                    help="Minimum n_s for Planck compatibility")
    p.add_argument("--n-s-max", type=float, default=0.99,
                    help="Maximum n_s for Planck compatibility")

    # ── Mass bin override ─────────────────────────────────────────────────────
    p.add_argument("--mass-bin-lo", type=float, default=None,
                    help="Override lower bound of target mass bin [M_sun]. "
                         "When set, overrides --target.")
    p.add_argument("--mass-bin-hi", type=float, default=None,
                    help="Override upper bound of target mass bin [M_sun]. "
                         "When set, overrides --target.")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    p.add_argument("--k-pivot", type=float, default=0.05,
                    help="Pivot k_pivot_phys (Mpc^-1). Drives BOTH A_s "
                         "normalization and n_s extraction (single-pivot invariant). "
                         "Planck default 0.05; Higgs low-ell 0.002.")
    p.add_argument("--pivot-k", type=float, default=None,
                    help="Deprecated alias for --k-pivot (back-compat).")
    p.add_argument("--ns-method", choices=["lsq", "derivative"], default="lsq",
                    help="n_s extraction method: lsq=window fit (default), "
                         "derivative=log-derivative at k_pivot.")
    p.add_argument("--ns-window", type=float, default=3.0,
                    help="Fit half-width for lsq method: [k_pivot/w, k_pivot*w]. "
                         "Ignored for derivative method.")
    p.add_argument("--workers", type=int, default=8,
                    help="MS solver parallel workers per config")

    # ── BoTorch ───────────────────────────────────────────────────────────────
    p.add_argument("--n-trials", type=int, default=200,
                    help="Number of BoTorch Bayesian optimization trials")
    p.add_argument("--n-init", type=int, default=32,
                    help="Number of random initialization (Sobol) points")
    p.add_argument("--q-batch", type=int, default=4,
                    help="q-batch size for parallel acquisition function evaluation")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")

    # ── Output / config ───────────────────────────────────────────────────────
    p.add_argument("--output-dir", default="outputs/plots/pbh",
                    help="Output directory for diagnostic plots")
    p.add_argument("--log", default="outputs/simulations/logs/pbh_optimizer.jsonl",
                    help="Path for incremental JSONL optimization log")
    p.add_argument("--no-plot", action="store_true",
                    help="Skip all plotting")
    p.add_argument("--config", type=str, default=None,
                    help="JSON config file. CLI args override file values.")
    p.add_argument("--resume", action="store_true",
                    help="Skip parameter sets already recorded in the log file")
    p.add_argument("--f-max", type=float, default=1.0,
                    help="Exclude configs with f_total > this threshold (default: 1.0)")

    # Pre-parse --config before full argument parse
    pre_args, _ = p.parse_known_args()
    if pre_args.config:
        with open(pre_args.config) as f:
            config = json.load(f)
        p.set_defaults(**config)

    args = p.parse_args()

    # Back-compat: --pivot-k aliases --k-pivot
    if args.pivot_k is not None:
        args.k_pivot = args.pivot_k

    # Build bounds tensor
    bounds = _build_bounds(args)
    mass_lo, mass_hi = _resolve_mass_bin(args)

    # ── Print config summary ──────────────────────────────────────────────────
    print("=" * 60)
    print("PBH BoTorch Optimizer — Config")
    print("=" * 60)
    print(f"  Target:          {args.target}")
    print(f"  Mass bin:        [{mass_lo:.3e}, {mass_hi:.3e}] M_sun")
    print(f"  Bounds (d={bounds.shape[0]}):")
    dim_names = ["x_c", "c", "beta", "chi0", "N_star", "zeta_c"]
    log_dims = {"beta", "chi0"}
    for i, name in enumerate(dim_names):
        if name in log_dims:
            lo_phys = float(np.exp(bounds[i, 0].item()))
            hi_phys = float(np.exp(bounds[i, 1].item()))
        else:
            lo_phys = float(bounds[i, 0].item())
            hi_phys = float(bounds[i, 1].item())
        print(f"    {name:>8s}: [{lo_phys:.6g}, {hi_phys:.6g}]")
    print(f"  y0:              {args.y0}")
    print(f"  N_total min:     {args.N_total_min}")
    print(f"  f_total range:   [{args.f_total_min}, {args.f_total_max}]")
    print(f"  n_s range:       [{args.n_s_min}, {args.n_s_max}]")
    print(f"  k_pivot:         {args.k_pivot}")
    print(f"  ns_method:       {args.ns_method}")
    print(f"  ns_window:       {args.ns_window}")
    print(f"  BoTorch:")
    print(f"    n_trials:      {args.n_trials}")
    print(f"    n_init:        {args.n_init}")
    print(f"    q_batch:       {args.q_batch}")
    print(f"    seed:          {args.seed}")
    print(f"  Workers:         {args.workers}")
    print(f"  Log:             {args.log}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Resume:          {args.resume}")
    print("=" * 60)

    # TODO: Implement forward model, pre-filter, GP, and acquisition.
    # BoTorch / GPyTorch imported lazily here when those functions are added.


if __name__ == "__main__":
    main()
