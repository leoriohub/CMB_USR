#!/usr/bin/env python3
"""
n_s extraction comparison: LSQ vs log-derivative across ALL cached P_S(k) files.

Reads every existing P_S(k) JSON from outputs/simulations/pspectra/,
computes n_s both ways (LSQ and derivative) at k_pivot, and produces:
  - Table summary (stdout)
  - Scatter plot: outputs/plots/paper/ns_method_comparison.png

Usage:
    python scripts/compare_ns_vs_x0.py
    python scripts/compare_ns_vs_x0.py --k-pivot 0.05 --ns-window 3.0
    python scripts/compare_ns_vs_x0.py --dir /custom/path --output my_plot.png
"""

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline

from scripts.plotting import TOL, PAPER_RCPARAMS, OUTPUT_DIRS, save_fig, get_path
from scripts.constants import k_pivot_phys, As
from scripts.observables import extract_ns


def load_ps_json(path):
    """Load a P_S(k) JSON file, return (k_phys, P_S, metadata) or (None, None, None)."""
    try:
        with open(path) as f:
            rec = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: can't read {path}: {e}")
        return None, None, None

    spec = rec.get("spectrum", {})
    k_phys = np.array(spec.get("k_phys", []), dtype=float)
    P_S = np.array(spec.get("P_S", []), dtype=float)
    meta = rec.get("metadata", {})

    # Handle partial checkpoints
    if spec.get("partial", False):
        n_comp = spec.get("n_completed", 0)
        if n_comp < 10:
            return None, None, None

    if len(k_phys) < 5 or len(P_S) < 5:
        return None, None, None

    # Filter to finite values only
    ok = np.isfinite(P_S) & (P_S > 1e-30) & np.isfinite(k_phys)
    if ok.sum() < 5:
        return None, None, None
    return k_phys[ok], P_S[ok], meta


def main():
    parser = argparse.ArgumentParser(
        description="n_s extraction comparison: LSQ vs log-derivative across all cached P_S(k)"
    )
    parser.add_argument("--k-pivot", type=float, default=k_pivot_phys,
                        help="Pivot wavenumber (default: %(default)s)")
    parser.add_argument("--ns-window", type=float, default=4.0,
                        help="Fit half-width for lsq method [k_p/w, k_p*w] (ignored for derivative)")
    parser.add_argument("--ns-method", choices=["lsq", "derivative"], default="lsq",
                        help="n_s extraction method: lsq=window fit (default), derivative=log-derivative at k_pivot")
    parser.add_argument("--dir", type=str, default=None,
                        help="P_S(k) JSON directory (default: outputs/simulations/pspectra/)")
    parser.add_argument("--output", type=str, default="ns_method_comparison",
                        help="Output filename (no extension, default: %(default)s)")
    args = parser.parse_args()

    # Resolve P_S directory
    if args.dir is not None:
        ps_dir = args.dir
    else:
        ps_dir = get_path("pspectra", "")
    if not os.path.isdir(ps_dir):
        print(f"ERROR: directory not found: {ps_dir}")
        sys.exit(1)

    # Find all JSON files (both new and legacy naming conventions)
    json_files = sorted(glob.glob(os.path.join(ps_dir, "ps_*.json")))
    json_files += sorted(glob.glob(os.path.join(ps_dir, "PS_Higgs_*.json")))
    json_files += sorted(glob.glob(os.path.join(ps_dir, "Higgs_Inflation_*.json")))
    json_files = sorted(set(json_files))  # deduplicate

    if not json_files:
        print(f"No P_S(k) JSON files found in {ps_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} P_S(k) file(s) in {ps_dir}")
    print(f"k_pivot = {args.k_pivot}, ns_window = {args.ns_window}")
    print()

    # Process each file
    rows = []
    for fpath in json_files:
        k_phys, P_S, meta = load_ps_json(fpath)
        if k_phys is None:
            continue

        fname = os.path.basename(fpath)

        # Extract model parameters from metadata
        x0 = meta.get("x0", meta.get("phi0", None))
        y0 = meta.get("y0", None)
        N_star = meta.get("N_star", None)
        N_total = meta.get("N_total", None)

        # Derive model label from filename or metadata
        label = meta.get("model", "")
        if x0 is not None and y0 is not None:
            label += f" x₀={x0:.3g} y₀={y0:.3g}"
            if N_star is not None:
                label += f" N*={N_star:.1f}"

        # n_s via user-selected method
        ns_val, ns_meta = extract_ns(k_phys, P_S, args.k_pivot,
                                     ns_window=args.ns_window if args.ns_method == "lsq" else None,
                                     method=args.ns_method)

        if ns_val is None:
            continue

        k_range_str = ns_meta.get("k_range", ["?", "?"])
        k_range_str = f"[{k_range_str[0]:.2e}, {k_range_str[1]:.2e}]"

        rows.append({
            "file": fname,
            "label": label,
            "x0": x0,
            "y0": y0,
            "N_star": N_star,
            "N_total": N_total,
            "ns": ns_val,
            "n_modes": ns_meta.get("n_modes", 0) if ns_meta else 0,
            "k_range": k_range_str,
        })

    if not rows:
        print("No valid P_S(k) files could be processed (k_pivot may be outside grid).")
        sys.exit(1)

    # ── Print summary table ──
    print(f"{'File':<50} {'n_s':>10} {'n_modes':>8} {'N*':>7} {'k_range':>25}")
    print("-" * 102)
    for r in rows:
        ns_str = f"{r['ns']:.4f}" if r['ns'] is not None else "FAIL"
        n_modes_str = f"{r['n_modes']}"
        nstar_str = f"{r['N_star']:.1f}" if r['N_star'] is not None else "?"
        print(f"{r['file']:<50} {ns_str:>10} {n_modes_str:>8} {nstar_str:>7} {r['k_range']:>25}")

    # ── Summary stats ──
    valid_ns = [r['ns'] for r in rows if r['ns'] is not None]
    if valid_ns:
        vals = np.array(valid_ns)
        print(f"\nSummary: {len(valid_ns)} valid extractions ({args.ns_method} method)")
        print(f"  n_s:  mean={np.mean(vals):.6f}  std={np.std(vals):.6f}  "
              f"range=[{np.min(vals):.6f}, {np.max(vals):.6f}]")

    # ── Plot: histogram of n_s values ──
    valid = [r for r in rows if r['ns'] is not None]
    if valid:
        ns_vals = np.array([r['ns'] for r in valid])

        with plt.rc_context(PAPER_RCPARAMS):
            fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))

            ax.hist(ns_vals, bins=max(10, len(ns_vals) // 3),
                    color=TOL["red"], alpha=0.7, edgecolor="none")
            ax.axvline(np.mean(ns_vals), color=TOL["blue"], ls="--", lw=0.8,
                       label=f"mean={np.mean(ns_vals):.4f}")
            ax.set_xlabel(rf"$n_s$ ({args.ns_method})", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.grid(True, alpha=0.2, which="both")
            ax.legend(fontsize=7)

            # Annotation
            ax.text(
                0.95, 0.95,
                f"N={len(ns_vals)}\n"
                f"μ={np.mean(ns_vals):.4f}\n"
                f"σ={np.std(ns_vals):.4f}",
                transform=ax.transAxes, fontsize=6, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="none", alpha=0.7),
            )

            save_fig(fig, args.output, "paper")

    print(f"\nPlot: {get_path('paper', args.output + '.png')}")


if __name__ == "__main__":
    main()
