"""
Benchmark Numba MS solver vs Python across grid sizes and worker counts.

Usage:
    python scripts/benchmark_numba.py [--n-dense N] [--n-outer N]
"""
import argparse
import time
import sys

import numpy as np

from models import HiggsModel
from pspectrum_pipeline import run_pspectrum_pipeline, build_weighted_kgrid
from scripts.constants import As, k_pivot_phys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-dense", type=int, default=120, help="modes in dense zone")
    parser.add_argument("--n-outer", type=int, default=60, help="modes outside dense zone")
    parser.add_argument("--workers", type=int, default=None, nargs="+",
                        help="worker counts to test (default: [1, 4, 8, 16])")
    args = parser.parse_args()

    if args.workers is None:
        args.workers = [1, 4, 8, 16]

    k_grid = build_weighted_kgrid(
        1e-5, 1.0, k_pivot_phys,
        dense_zone=(1e-4, 1e-2),
        n_dense=args.n_dense,
        n_outer=args.n_outer,
    )
    n_modes = len(k_grid)
    print(f"Grid: {n_modes} modes ({args.n_dense} dense + {args.n_outer} outer × 2)")
    print(f"Config: phi0=5.70 y0=-0.170 N*=52.0")
    print(f"{'Solver':<12} {'Workers':>8} {'Time':>8} {'Modes/s':>10} {'Notes'}")
    print("-" * 60)

    m = HiggsModel(lam=0.13, xi=15000.0)

    for label, use_nb in [("Python", False), ("Numba", True)]:
        for nw in args.workers:
            t0 = time.time()
            r = run_pspectrum_pipeline(
                m, phi0=5.70, y0=-0.170, N_star=52.0,
                k_phys_grid=k_grid,
                normalize_to_As=True, As=As,
                n_workers=nw, use_numba=use_nb,
                ms_steps=5000, save_outputs=False,
            )
            t = time.time() - t0
            ok = r["metadata"]["n_completed"]
            rate = ok / max(t, 0.01)
            notes = ""
            if ok < n_modes:
                notes = f" ({n_modes-ok} failed)"
            print(f"{label:<12} {nw:>8d} {t:>8.1f}s {rate:>10.1f}{notes}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
