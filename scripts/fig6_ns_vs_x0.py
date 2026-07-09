#!/usr/bin/env python3
"""
n_s vs x₀: SR and MS comparison (user-selectable method).

Scan x₀ at fixed y₀ and N*, computing n_s three ways at each point:
  1. SR algebraic:  n_s = 1 + 2η_H − 4ε_H at N_pivot
  2. MS LSQ:        polyfit ln P vs ln k over [k_pivot/w, k_pivot*w]
  3. MS derivative: CubicSpline derivative at k_pivot

Output: diagnostic plot at
    outputs/plots/diagnostics/ns_vs_x0_y0{Y0}_nstar{N*}_{x0min}-{x0max}_n{N}_modes{M}_wf{W}.png

Usage:
    python scripts/legacy/fig6_ns_vs_x0.py \
        --y0 -0.10 --N-star 60 \
        --x0-min 5.71 --x0-max 5.80 \
        --n-points 30 --n-cores 8 --n-modes 1000
"""

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plotting import TOL, PAPER_RCPARAMS, save_fig, get_path
from scripts.constants import k_pivot_phys, As
from scripts.observables import extract_ns

from models import HiggsModel

# ── Imports that must come after model for Numba compile ──────────────────

import inf_dyn_background as bg_solver
from pspectrum_pipeline import (
    run_pspectrum_pipeline,
    find_end_of_inflation,
    get_k_pivot_code,
    build_weighted_kgrid,
)


def compute_sr_ns_at_pivot(derived_bg, N_star, N_total):
    """SR algebraic n_s at the pivot exit.

    Returns (n_s, N_pivot) or (None, None) if the pivot is outside the
    background grid or N_total < N_star.
    """
    N_arr = derived_bg["N"]
    if N_total < N_star or N_arr[-1] < N_total - N_star:
        return None, None
    N_pivot = N_total - N_star
    eps_pivot = float(np.interp(N_pivot, N_arr, derived_bg["epsH"]))
    eta_pivot = float(np.interp(N_pivot, N_arr, derived_bg["etaH"]))
    ns = 1.0 + 2.0 * eta_pivot - 4.0 * eps_pivot
    return ns, N_pivot


def compute_ns_at_x0(x0, y0, N_star, ns_window, ms_steps, pivot_k,
                     k_grid, ns_method="lsq", xi=15000.0, lam=0.13):
    """Compute n_s for a single (x₀, y₀, N*) configuration.

    Uses a FIXED k_grid (same for all x₀) to avoid random k-subset noise
    in the derivative n_s.

    Returns dict with keys:
        x0, y0, N_star, ns_method,
        ns_SR, ns_MS_val,
        N_total, pivot_k, status
    """
    model = HiggsModel(xi=xi, lam=lam)

    try:
        result = run_pspectrum_pipeline(
            model=model,
            phi0=x0,
            y0=y0,
            k_phys_grid=k_grid,
            k_pivot_phys=pivot_k,
            N_star=N_star,
            ms_steps=ms_steps,
            normalize_to_As=True,
            As=As,
            save_outputs=False,
            n_workers=1,  # already parallelised by x₀
            use_numba=True,
            ms_method='dp5',
        )
    except Exception as e:
        return {
            "x0": x0, "y0": y0, "N_star": N_star,
            "ns_SR": None, "ns_MS_val": None,
            "ns_MS_method": ns_method,
            "N_total": None, "pivot_k": pivot_k,
            "status": f"pipeline failed: {e}",
        }

    if result["status"] != "success":
        return {
            "x0": x0, "y0": y0, "N_star": N_star,
            "ns_SR": None, "ns_MS_val": None,
            "ns_MS_method": ns_method,
            "N_total": result.get("metadata", {}).get("N_total"),
            "pivot_k": pivot_k,
            "status": result.get("message", result["status"]),
        }

    k_phys = result["k_phys"]
    P_S = result["P_S"]
    meta = result["metadata"]
    N_total = meta["N_total"]

    # 1. SR n_s from background
    ns_sr, _ = compute_sr_ns_at_pivot(result["derived_bg"], N_star, N_total)

    # 2. MS n_s via selected method
    ns_kw = dict(ns_window=ns_window) if ns_method == "lsq" else dict()
    ns_ms, ns_meta = extract_ns(k_phys, P_S, pivot_k,
                                method=ns_method, **ns_kw)

    msg_parts = [f"x0={x0:.4f}"]
    msg_parts.append(f"ns_SR={ns_sr:.4f}" if ns_sr is not None else "ns_SR=FAIL")
    msg_parts.append(f"ns_MS={ns_ms:.4f}" if ns_ms is not None else "ns_MS=FAIL")
    print(f"  [{','.join(msg_parts)}]")

    return {
        "x0": x0, "y0": y0, "N_star": N_star,
        "ns_SR": ns_sr, "ns_MS_val": ns_ms, "ns_MS_method": ns_method,
        "N_total": N_total, "pivot_k": pivot_k,
        "status": "ok",
    }


def _worker(args):
    """Top-level worker function for ProcessPoolExecutor (must be picklable).

    args: (x0, y0, N_star, ns_window, ms_steps, pivot_k, k_grid, ns_method, xi, lam)
    """
    return compute_ns_at_x0(*args)


def plot_ns_comparison(results, y0, N_star, x0_min, x0_max,
                       n_points, ms_steps, ns_window, pivot_k, ns_method,
                       filename):
    """Produce the 2-panel comparison plot (main + residuals)."""
    x0s = np.array([r["x0"] for r in results])
    order = np.argsort(x0s)
    x0s = x0s[order]

    ns_sr = np.array([r["ns_SR"] if r["ns_SR"] is not None else np.nan
                      for r in results], dtype=float)[order]
    ns_ms = np.array([r["ns_MS_val"] if r["ns_MS_val"] is not None else np.nan
                      for r in results], dtype=float)[order]

    valid_sr = np.isfinite(ns_sr)
    valid_ms = np.isfinite(ns_ms)

    # Mask out-of-range data so connecting lines don't create
    # visual artifacts at the y-axis boundaries.
    ns_sr_plot = np.where((ns_sr >= 0.9) & (ns_sr <= 1.0), ns_sr, np.nan)
    ns_ms_plot = np.where((ns_ms >= 0.9) & (ns_ms <= 1.0), ns_ms, np.nan)

    # Trim x-range: only show region where EITHER curve has valid data
    # within [0.9, 1.0]. Everything outside is blank space.
    both_ok = np.isfinite(ns_sr_plot) & np.isfinite(ns_ms_plot)
    if both_ok.any():
        x_lo = float(x0s[both_ok][0])
        x_hi = float(x0s[both_ok][-1])
    else:
        x_lo, x_hi = x0s.min(), x0s.max()

    # MS method label for plot
    if ns_method == "lsq":
        ms_label = rf"MS (LSQ, $k_p/\gamma$..$\gamma k_p$, $\gamma$={ns_window})"
        ms_res_label = rf"LSQ $\gamma$={ns_window}"
    else:
        ms_label = r"MS ($d\ln P/d\ln k$ at $k_p$)"
        ms_res_label = "Derivative"

    with plt.rc_context(PAPER_RCPARAMS):
        fig, (ax_main, ax_res) = plt.subplots(
            2, 1, figsize=(4.5, 4.2), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        )

        # Use markers only for sparse scans, lines-only for dense
        use_markers = len(x0s) <= 60
        marker_kw = dict(ms=3.5) if use_markers else dict()
        if valid_sr.any():
            ax_main.plot(x0s[valid_sr], ns_sr_plot[valid_sr],
                         "s-" if use_markers else "-",
                         color=TOL["blue"], lw=1.0,
                         label=r"SR ($n_s = 1 + 2\eta_H - 4\varepsilon_H$)",
                         zorder=3, **marker_kw)
        if valid_ms.any():
            ax_main.plot(x0s[valid_ms], ns_ms_plot[valid_ms],
                         "D-" if use_markers else "-",
                         color=TOL["teal"], lw=1.0,
                         label=ms_label,
                         zorder=2, **marker_kw)

        ax_main.axhline(0.965, color=TOL["grey"], ls="--", lw=0.6, alpha=0.5)
        ax_main.text(x_lo, 0.966, r"$n_s=0.965$ (Planck)",
                     fontsize=6, color=TOL["grey"], alpha=0.6)
        ax_main.set_ylabel(r"$n_s$", fontsize=9)

        # Hard y-limits: [0.9, 1.0]. Anything outside this range is
        # unphysical for n_s and gets clipped from view.
        y_lo, y_hi = 0.9, 1.0
        ax_main.set_ylim(y_lo, y_hi)
        ax_main.set_xlim(x_lo, x_hi)
        ax_main.legend(fontsize=6, loc="lower left", framealpha=0.8)
        ax_main.grid(True, alpha=0.2, which="both")

        # ── Bottom panel: residuals vs SR ──
        use_markers = len(x0s) <= 60
        marker_kw_res = dict(ms=3) if use_markers else dict()
        nan_mask = np.isfinite(ns_ms) & np.isfinite(ns_sr) & (np.abs(ns_sr) > 1e-10)
        if nan_mask.any():
            res_ms = ns_ms[nan_mask] - ns_sr[nan_mask]
            ax_res.plot(x0s[nan_mask], res_ms,
                        "D-" if use_markers else "-",
                        color=TOL["teal"], lw=0.8,
                        label=ms_res_label, zorder=2, **marker_kw_res)

        ax_res.axhline(0, color=TOL["grey"], ls="--", lw=0.6, alpha=0.4)
        ax_res.set_xlabel(r"$x_0$ (initial field value)", fontsize=9)
        ax_res.set_ylabel(r"$\Delta n_s$ (MS $-$ SR)", fontsize=8)
        ax_res.legend(fontsize=6, loc="lower right", framealpha=0.8)
        ax_res.grid(True, alpha=0.2, which="both")

        # Annotate run config
        config_str = (
            rf"$y_0={y0:.3f}$, $N^*={N_star:.0f}$, "
            rf"$k_p={pivot_k:.3f}$, modes={ms_steps}, window={ns_window}, "
            rf"ns_method={ns_method}"
        )
        ax_main.text(0.98, 0.98, config_str, transform=ax_main.transAxes,
                     fontsize=5, ha="right", va="top", alpha=0.6,
                     bbox=dict(boxstyle="round,pad=0.2", fc="w",
                               ec="none", alpha=0.6))

        save_fig(fig, filename, "diagnostics")


def main():
    parser = argparse.ArgumentParser(
        description="n_s vs x₀: SR and MS comparison (user-selectable method)",
    )
    parser.add_argument("--y0", type=float, default=-0.10, help="Initial velocity")
    parser.add_argument("--N-star", type=float, default=60,
                        help="e-folds before end for pivot exit")
    parser.add_argument("--x0-min", type=float, default=5.71, help="Min x₀ to scan")
    parser.add_argument("--x0-max", type=float, default=5.80, help="Max x₀ to scan")
    parser.add_argument("--n-points", type=int, default=30, help="Number of x₀ points")
    parser.add_argument("--n-cores", type=int, default=8, help="Parallel workers")
    parser.add_argument("--n-modes", type=int, default=1000,
                        help="MS integration steps per k-mode")
    parser.add_argument("--ns-window", type=float, default=4.0,
                        help="Fit half-width for lsq method [k_pivot/w, k_pivot*w] (ignored for derivative)")
    parser.add_argument("--xi", type=float, default=15000.0,
                        help="Non-minimal coupling ξ")
    parser.add_argument("--lam", type=float, default=0.13, help="Higgs quartic λ")
    parser.add_argument("--k-pivot", type=float, default=k_pivot_phys,
                        help="Pivot wavenumber (same for As and n_s)")
    parser.add_argument("--ns-method", choices=["lsq", "derivative"], default="lsq",
                        help="n_s extraction method: lsq=window fit (default), derivative=log-derivative at k_pivot")
    parser.add_argument("--k-dense", type=int, default=500,
                        help="Number of dense k-modes near the pivot (default 500)")
    parser.add_argument("--k-outer", type=int, default=200,
                        help="Number of outer k-modes on each side (default 200)")
    args = parser.parse_args()

    pivot_k = args.k_pivot
    x0s = np.linspace(args.x0_min, args.x0_max, args.n_points)

    print(f"n_s scan: x₀ ∈ [{args.x0_min:.4f}, {args.x0_max:.4f}] "
          f"({args.n_points} pts), y₀={args.y0:.3f}, N*={args.N_star:.0f}")
    print(f"  Workers: {args.n_cores}, MS modes: {args.n_modes}, "
          f"ns_window: {args.ns_window}, k-grid: {args.k_dense}+2x{args.k_outer}")
    t_start = time.time()

    # Build a FIXED k-grid (same for ALL x₀) to eliminate random-subsample noise
    k_grid = build_weighted_kgrid(
        k_min=pivot_k / 20.0,
        k_max=pivot_k * 20.0,
        k_pivot_phys=pivot_k,
        dense_min=pivot_k / 10.0,
        dense_max=pivot_k * 10.0,
        n_dense=args.k_dense,
        n_outer=args.k_outer,
    )
    print(f"  Fixed k-grid: {len(k_grid)} modes spanning "
          f"[{k_grid[0]:.2e}, {k_grid[-1]:.2e}]")

    # Build worker args: (x0, y0, N_star, ns_window, ms_steps, pivot_k, k_grid, ns_method, xi, lam)
    worker_args = [
        (x0, args.y0, args.N_star, args.ns_window, args.n_modes,
         pivot_k, k_grid, args.ns_method, args.xi, args.lam)
        for x0 in x0s
    ]

    results = []
    n_workers = min(args.n_cores, len(worker_args))
    if n_workers > 1:
        print(f"\nRunning {len(worker_args)} x₀ points on {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, wa): wa[0] for wa in worker_args}
            for future in as_completed(futures):
                r = future.result()
                results.append(r)
                if r["status"] != "ok":
                    print(f"  ✗ x0={r['x0']:.4f}: {r['status']}")
    else:
        print(f"\nRunning {len(worker_args)} x₀ points serially...")
        for wa in worker_args:
            r = _worker(wa)
            results.append(r)
            if r["status"] != "ok":
                print(f"  ✗ x0={r['x0']:.4f}: {r['status']}")

    elapsed = time.time() - t_start
    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nDone: {n_ok}/{len(results)} OK in {elapsed:.1f}s "
          f"({elapsed / max(n_ok, 1):.1f}s/pt avg)")

    # Generate plot
    filename = (
        f"ns_vs_x0_y0{args.y0:+.3f}_nstar{args.N_star:.0f}_"
        f"{args.x0_min:.2f}-{args.x0_max:.2f}_"
        f"n{args.n_points}_modes{args.n_modes}_wf{args.ns_window:.1f}"
    )
    plot_ns_comparison(results, args.y0, args.N_star,
                       args.x0_min, args.x0_max,
                       args.n_points, args.n_modes, args.ns_window,
                       pivot_k, args.ns_method, filename)

    # Save data to scan JSON
    data_path = get_path("scans", filename + ".json")
    scan_data = {
        "parameter": "x0",
        "y0": args.y0,
        "N_star": args.N_star,
        "x0_range": [args.x0_min, args.x0_max],
        "n_points": args.n_points,
        "ns_window": args.ns_window,
        "ms_modes": args.n_modes,
        "k_pivot": pivot_k,
        "ns_method": args.ns_method,
        "results": [
            {
                "x0": r["x0"],
                "ns_SR": r["ns_SR"],
                "ns_MS_val": r["ns_MS_val"],
                "ns_MS_method": r["ns_MS_method"],
                "N_total": r["N_total"],
                "status": r["status"],
            }
            for r in sorted(results, key=lambda x: x["x0"])
        ],
    }
    with open(data_path, "w") as f:
        json.dump(scan_data, f, indent=2)
    print(f"  Data: {data_path}")


if __name__ == "__main__":
    main()
