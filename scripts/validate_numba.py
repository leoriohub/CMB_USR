"""
Compare Python vs Numba MS solvers on a set of known-good configs.
Reports maximum relative deviation in P_S(k) for each config.

Usage:
    python scripts/validate_numba.py [--workers N]
"""
import sys
import time
import argparse
import multiprocessing
import numpy as np

from models import HiggsModel, PunctuatedInflationModel
from pspectrum_pipeline import (
    run_pspectrum_pipeline, build_weighted_kgrid,
)
from scripts.constants import As, k_pivot_phys
from numba_ms_solver import HAVE_NUMBA

TEST_CONFIGS = [
    ("Higgs", dict(phi0=6.40, y0=-0.475, N_star=59.0)),
    ("Higgs", dict(phi0=5.70, y0=-0.170, N_star=52.0)),
    ("Higgs", dict(phi0=5.75, y0=-0.170, N_star=55.0)),
    ("Higgs", dict(phi0=6.55, y0=-0.780, N_star=50.0)),
    ("Higgs", dict(phi0=7.00, y0=-0.500, N_star=57.0)),
]


def _build_model(model_name):
    if model_name == "Higgs":
        return HiggsModel(lam=0.13, xi=15000.0)
    if model_name == "Punctuated":
        return PunctuatedInflationModel(m=1.1323e-7, lam=3.3299e-15)
    raise ValueError(f"Unknown model: {model_name}")


def run_config(model, phi0, y0, N_star, use_numba, n_workers):
    k_grid = build_weighted_kgrid(
        1e-5, 1.0, k_pivot_phys,
        dense_zone=(1e-4, 1e-2), n_dense=40, n_outer=20,
    )
    result = run_pspectrum_pipeline(
        model=model, phi0=phi0, y0=y0, N_star=N_star,
        k_pivot_phys=k_pivot_phys, k_phys_grid=k_grid,
        normalize_to_As=True, As=As,
        n_workers=n_workers, use_numba=use_numba,
        ms_steps=5000, save_outputs=False,
    )
    if result["status"] != "success":
        return None, result.get("message", "unknown")
    return result["P_S"], None


def _warmup_numba():
    """Run a minimal config to trigger Numba JIT compilation."""
    m_warm = HiggsModel(lam=0.13, xi=15000.0)
    _ = run_pspectrum_pipeline(
        m_warm, phi0=6.0, y0=-0.1, N_star=50.0,
        k_pivot_phys=k_pivot_phys,
        k_phys_grid=np.logspace(-5, 0, 20),
        normalize_to_As=True, As=As,
        n_workers=1, use_numba=True,
        ms_steps=500, save_outputs=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Validate Numba MS solver")
    parser.add_argument("--workers", type=int,
                        default=max(1, multiprocessing.cpu_count() - 1),
                        help="Parallel workers (default: n_cores - 1)")
    args = parser.parse_args()

    if not HAVE_NUMBA:
        print("ERROR: Numba not available. Install with: conda install numba")
        sys.exit(1)

    print(f"Numba MS Solver Validation  (workers={args.workers})")
    print("=" * 65)
    print(f"{'Config':<50} {'Max |ratio-1|':>18}  Speedup")
    print("=" * 65)

    print(" Warming up Numba JIT...", end=" ", flush=True)
    _warmup_numba()
    print("done.")

    max_dev_all = 0.0
    worst_config = None

    for idx, (model_name, params) in enumerate(TEST_CONFIGS):
        model = _build_model(model_name)
        phi0 = params["phi0"]
        y0 = params["y0"]
        ns = params["N_star"]
        label = f"[{idx+1}/{len(TEST_CONFIGS)}] {model_name} phi0={phi0:.2f} y0={y0:+.3f} N*={ns:.1f}"

        print(f"\n{label}")
        print(f"  Python: ", end="", flush=True)
        t0 = time.time()
        ps_py, err = run_config(model, phi0, y0, ns, use_numba=False,
                                n_workers=args.workers)
        t_py = time.time() - t0
        if ps_py is None:
            print(f"FAILED: {err}")
            continue
        print(f"{t_py:.1f}s")

        print(f"  Numba:  ", end="", flush=True)
        t0 = time.time()
        ps_nb, err = run_config(model, phi0, y0, ns, use_numba=True,
                                n_workers=args.workers)
        t_nb = time.time() - t0
        if ps_nb is None:
            print(f"FAILED: {err}")
            continue
        print(f"{t_nb:.1f}s")

        finite = np.isfinite(ps_py) & (ps_py > 0) & np.isfinite(ps_nb)
        if np.sum(finite) < 10:
            print(f"  {'INSUFFICIENT FINITE POINTS':>50}")
            continue

        ratio = ps_nb[finite] / ps_py[finite]
        max_dev = float(np.max(np.abs(ratio - 1.0)))
        speedup = t_py / max(t_nb, 0.01)

        print(f"  max |ratio-1| = {max_dev:.2e}  ({speedup:.1f}x speedup)")

        if max_dev > max_dev_all:
            max_dev_all = max_dev
            worst_config = label

    print("\n" + "=" * 65)
    if max_dev_all > 1e-12:
        print(f"WARNING: Max deviation {max_dev_all:.2e} exceeds 1e-12!")
        print(f"Worst config: {worst_config}")
        sys.exit(1)

    print(f"All configs pass. Max deviation: {max_dev_all:.2e}")
    print("Validation OK.")
    sys.exit(0)


if __name__ == "__main__":
    main()
