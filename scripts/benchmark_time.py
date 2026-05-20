"""
Time benchmark: Python vs Numba MS solver, serial and parallel.

Usage:
    # Local
    python -m scripts.benchmark_time

    # Lab machine (inside tmux)
    python -m scripts.benchmark_time --large > ~/bench_time.log 2>&1
    Ctrl+b d
    ssh uni "cat ~/bench_time.log"
"""
import argparse
import time
import sys

import numpy as np

from models import HiggsModel
from pspectrum_pipeline import run_pspectrum_pipeline, build_weighted_kgrid
from scripts.constants import As, k_pivot_phys
from numba_ms_solver import HAVE_NUMBA


GRID_SMALL = dict(n_dense=40, n_outer=20)   # ~59 modes
GRID_LARGE = dict(n_dense=200, n_outer=100)  # ~299 modes


def _grid(name):
    return build_weighted_kgrid(1e-5, 1.0, k_pivot_phys, **GRID_SMALL if name == "small" else GRID_LARGE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true", help="Use large grid (299 modes)")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16])
    args = parser.parse_args()

    if not HAVE_NUMBA:
        print("ERROR: Numba not available"); sys.exit(1)

    grid_name = "large" if args.large else "small"
    k = _grid(grid_name)
    n_modes = len(k)
    print(f"Grid: {grid_name} ({n_modes} modes)")
    print(f"Config: Higgs φ₀=5.70 y₀=-0.170 N*=52")
    print(f"ms_steps=5000, 1 run per row")
    print()
    print(f"{'Solver':<10} {'Workers':>8} {'Time':>8} {'Modes/s':>10} {'Speedup':>8}")
    print("-" * 50)

    m = HiggsModel(lam=0.13, xi=15000.0)
    baseline = None

    for solver, use_nb in [("Python", False), ("Numba", True)]:
        for nw in args.workers:
            sys.stdout.flush()
            t0 = time.time()
            r = run_pspectrum_pipeline(
                m, phi0=5.70, y0=-0.170, N_star=52.0,
                k_phys_grid=k,
                normalize_to_As=True, As=As,
                n_workers=nw, use_numba=use_nb,
                ms_steps=5000, save_outputs=False,
            )
            t = time.time() - t0
            ok = r["metadata"]["n_completed"]
            rate = ok / max(t, 0.01)

            if solver == "Python" and nw == 1:
                baseline = t
                speedup = 1.0
            elif baseline:
                speedup = baseline / t
            else:
                speedup = 0

            notes = ""
            if ok < n_modes:
                notes = f" ({n_modes-ok} failed)"
            print(f"{solver:<10} {nw:>8d} {t:>8.1f}s {rate:>10.1f} {speedup:>7.1f}x{notes}")
            sys.stdout.flush()

    print("-" * 50)
    print(f"Baseline: Python 1w = {baseline:.1f}s")
    print()


if __name__ == "__main__":
    main()
