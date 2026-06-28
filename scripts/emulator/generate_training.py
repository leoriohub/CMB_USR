#!/usr/bin/env python3
"""Generate training data for NN emulator via Latin-hypercube sampling of Ezquiaga CHI parameters.

Samples (x_c, c, beta, chi0, n_star) → runs Fortran/Numba MS solver → extracts P_S(k) on FIXED_K_GRID.
Saves params + P_S as compressed .npz with JSONL progress log.
"""

import argparse
import json
import os
import signal
import sys
import time
import warnings

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.emulator.grid import FIXED_K_GRID
from scripts.constants import As_planck


class TimeoutError_(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError_


def with_timeout(seconds):
    """Decorator/context to raise TimeoutError_ after `seconds`."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        return wrapper
    return decorator

try:
    from scipy.stats.qmc import LatinHypercube
    HAS_LHS = True
except ImportError:
    HAS_LHS = False
    warnings.warn("scipy.stats.qmc not available, falling back to uniform random")

import pspectrum_pipeline as pp
from models import EzquiagaCHIModel, inflection_parameters
from fortran_ms_solver import HAVE_FORTRAN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate NN emulator training data from Ezquiaga CHI parameters"
    )
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of Latin-hypercube samples (default: 500)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=8,
                        help="Pipeline workers per sample (default: 8)")
    parser.add_argument("--output", type=str,
                        default="outputs/emulator/training_data.npz",
                        help="Output .npz path (default: outputs/emulator/training_data.npz)")
    parser.add_argument("--k-min", type=float, default=1e-3,
                        help="Min k for computation grid (default: 1e-3)")
    parser.add_argument("--k-max", type=float, default=1e19,
                        help="Max k for computation grid (default: 1e19)")
    parser.add_argument("--num-k", type=int, default=300,
                        help="Points in computation k-grid (default: 300)")
    parser.add_argument("--k-pivot", type=float, default=0.05,
                        help="Pivot scale in Mpc^-1 (default: 0.05)")
    parser.add_argument("--no-fortran", action="store_true",
                        help="Force Numba backend even if Fortran is available")
    parser.add_argument("--usr-fraction", type=float, default=0.5,
                        help="Fraction of samples from USR-active neighborhood (default: 0.5)")
    return parser.parse_args()


def build_model(x_c, c, beta, chi0):
    model = EzquiagaCHIModel(c=c)
    a_new, b_new = inflection_parameters(x_c, c, beta)
    model.a = a_new
    model.b = b_new
    model.v0 = model._V0 * model.a / (model.b * model.c)**2
    model.x0 = chi0
    model.y0 = -1e-4
    model.patch_background_solver()
    return model


# Known-good reference configurations for Ezquiaga CHI (verified working)
# Each entry: (x_c, c, beta, chi0, n_star)
KNOWN_GOOD_CONFIGS = [
    (0.784, 0.77, 1.8e-4, 8.0, 72.0),     # asteroid PBH
    (0.79, 1.86, 3e-4, 8.0, 55.0),        # high-c resolved peak
    (0.784, 0.77, 1e-5, 8.0, 60.0),       # near-inflection
]


def sample_parameters(n_samples, seed=None, usr_fraction=0.5):
    """Stratified sample: usr_fraction from USR-active neighborhood,
    rest from Latin-hypercube covering the full parameter space."""
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    params_list = []

    # ── Known good configs (always included) ──
    n_known = min(len(KNOWN_GOOD_CONFIGS), n_samples)
    known = np.array([list(cfg) for cfg in KNOWN_GOOD_CONFIGS[:n_known]])
    known[:, 2] = np.log10(known[:, 2])
    params_list.append(known)

    remaining = n_samples - n_known
    if remaining <= 0:
        return np.vstack(params_list)[:n_samples]

    # ── USR-focused samples (perturbed around known good configs) ──
    n_usr = max(0, int(remaining * usr_fraction))
    if n_usr > 0:
        usr_centers = np.array([
            [0.784, 0.77, np.log10(2e-5), 8.0, 65.0],    # subsolar
            [0.784, 0.77, np.log10(1.8e-4), 8.0, 72.0],   # asteroid
            [0.79, 1.86, np.log10(3e-4), 8.0, 55.0],      # high-c
            [0.784, 0.77, np.log10(1e-5), 8.0, 60.0],      # near-inflection
            [0.78, 0.77, np.log10(5e-5), 8.0, 68.0],       # mid-range
        ])
        # Spread: x_c ±0.015, c ±0.08, log10(β) ±0.4, χ₀ ±0.5, N* ±3
        scales = np.array([0.015, 0.08, 0.4, 0.5, 3.0])
        usr_params = np.zeros((n_usr, 5))
        for i in range(n_usr):
            center = usr_centers[rng.integers(len(usr_centers))]
            pt = center + rng.normal(0, scales)
            pt[0] = np.clip(pt[0], 0.75, 0.85)   # x_c bounds
            pt[1] = np.clip(pt[1], 0.7, 5.0)     # c bounds
            pt[2] = np.clip(pt[2], -6.0, -3.0)   # log10(beta) bounds
            pt[3] = np.clip(pt[3], 5.0, 9.0)     # chi0 bounds
            pt[4] = np.clip(pt[4], 50.0, 75.0)   # N_star bounds
            usr_params[i] = pt
        params_list.append(usr_params)

    # ── LHC samples (full parameter space coverage) ──
    n_lhc = remaining - n_usr
    if n_lhc > 0:
        if HAS_LHS:
            lh_seed = (seed or 0) + 2 if seed is not None else None
            sampler = LatinHypercube(d=5, seed=lh_seed)
            samples = sampler.random(n=n_lhc)
        else:
            samples = rng.uniform(size=(n_lhc, 5))
        lhc_params = np.zeros((n_lhc, 5))
        lhc_params[:, 0] = 0.75 + samples[:, 0] * 0.10
        lhc_params[:, 1] = 0.7 + samples[:, 1] * 4.3
        lhc_params[:, 2] = -6.0 + samples[:, 2] * 2.9542
        lhc_params[:, 3] = 5.0 + samples[:, 3] * 4.0
        lhc_params[:, 4] = 50.0 + samples[:, 4] * 25.0
        params_list.append(lhc_params)

    all_params = np.vstack(params_list)[:n_samples]
    rng.shuffle(all_params)
    return all_params


def evaluate_sample(params_row, k_compute, k_pivot, n_workers, backend):
    x_c, c, log_beta, chi0, n_star = params_row
    beta = 10.0 ** log_beta
    model = build_model(x_c, c, beta, chi0)
    result = pp.run_pspectrum_pipeline(
        model=model,
        k_phys_grid=k_compute.copy(),
        k_pivot_phys=k_pivot,
        N_star=n_star,
        normalize_to_As=True,
        As=As_planck,
        n_workers=n_workers,
        backend=backend,
        save_outputs=False,
    )
    if result["status"] != "success":
        raise RuntimeError(result["message"])
    return result["k_phys"], result["P_S"]


def interpolate_to_fixed_grid(k_out, P_S_out):
    from scipy.interpolate import interp1d
    valid = np.isfinite(P_S_out) & (P_S_out > 0)
    n_valid = int(np.sum(valid))
    if n_valid < 3:
        raise RuntimeError(f"Too few valid P_S points for interpolation: {n_valid}")
    interp = interp1d(
        np.log(k_out[valid]),
        np.log(P_S_out[valid]),
        kind='cubic',
        bounds_error=False,
        fill_value=np.nan,
    )
    log_P_S = interp(np.log(FIXED_K_GRID))
    in_range = (FIXED_K_GRID >= k_out[valid].min()) & (FIXED_K_GRID <= k_out[valid].max())
    result = np.exp(np.where(in_range, log_P_S, np.nan))
    return result


def main():
    args = parse_args()
    backend = 'numba' if args.no_fortran else ('fortran' if HAVE_FORTRAN else 'numba')
    print(f"Backend: {backend} (Fortran available: {HAVE_FORTRAN})")
    print(f"FIXED_K_GRID: {len(FIXED_K_GRID)} points from {FIXED_K_GRID[0]:.1e} to {FIXED_K_GRID[-1]:.1e}")
    k_compute = np.logspace(np.log10(args.k_min), np.log10(args.k_max), args.num_k)
    print(f"Computation k-grid: {len(k_compute)} points from {args.k_min:.1e} to {args.k_max:.1e}")
    n_usr = max(0, int(args.n_samples * args.usr_fraction))
    n_lhc = args.n_samples - min(len(KNOWN_GOOD_CONFIGS), args.n_samples) - n_usr
    print(f"Generating {args.n_samples} samples (USR-neighborhood: ~{n_usr}, LHC: ~{n_lhc}) seed={args.seed}...")
    params = sample_parameters(args.n_samples, args.seed, usr_fraction=args.usr_fraction)
    P_S_all = np.full((args.n_samples, len(FIXED_K_GRID)), np.nan)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    log_path = args.output.replace('.npz', '.jsonl')
    with open(log_path, 'w') as log_file:
        header = {
            "_type": "header",
            "n_samples": args.n_samples,
            "seed": args.seed,
            "backend": backend,
            "k_pivot": args.k_pivot,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "num_k": args.num_k,
            "fixed_k_grid_min": float(FIXED_K_GRID[0]),
            "fixed_k_grid_max": float(FIXED_K_GRID[-1]),
            "fixed_k_grid_size": len(FIXED_K_GRID),
        }
        log_file.write(json.dumps(header) + "\n")
        for i in range(args.n_samples):
            x_c = params[i, 0]
            c_val = params[i, 1]
            log_beta = params[i, 2]
            chi0 = params[i, 3]
            n_star = params[i, 4]
            beta = 10.0 ** log_beta
            t0 = time.time()
            print(f"[{i+1}/{args.n_samples}] x_c={x_c:.4f} c={c_val:.4f} "
                  f"beta={beta:.6e} chi0={chi0:.4f} N*={n_star:.1f} ...", end=" ", flush=True)
            status = "error"
            n_valid = 0
            elapsed_s = 0.0
            try:
                k_out, P_S_out = with_timeout(15)(evaluate_sample)(
                    params[i], k_compute, args.k_pivot,
                    min(args.workers, 8), backend
                )
                P_S_fixed = interpolate_to_fixed_grid(k_out, P_S_out)
                P_S_all[i] = P_S_fixed
                n_valid = int(np.sum(np.isfinite(P_S_fixed)))
                status = "ok"
                elapsed_s = time.time() - t0
                print(f"OK {n_valid}/{len(FIXED_K_GRID)} [{elapsed_s:.1f}s]")
            except Exception as e:
                elapsed_s = time.time() - t0
                print(f"FAIL [{elapsed_s:.1f}s] {e}")
            record = {
                "_type": "data",
                "status": status,
                "x_c": float(x_c),
                "c": float(c_val),
                "log10_beta": float(log_beta),
                "chi0": float(chi0),
                "n_star": float(n_star),
                "n_valid": n_valid,
                "elapsed_s": round(elapsed_s, 1),
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
    np.savez_compressed(
        args.output,
        params=params,
        P_S=P_S_all,
        k_grid=FIXED_K_GRID,
        n_samples=args.n_samples,
        seed=args.seed or -1,
    )
    n_ok = int(np.sum(np.isfinite(P_S_all[:, 0])))
    print(f"\nSaved: {args.output}")
    print(f"  Params shape: {params.shape}")
    print(f"  P_S shape:    {P_S_all.shape}")
    print(f"  OK samples:   {n_ok}/{args.n_samples}")
    print(f"  Log:          {log_path}")


if __name__ == "__main__":
    main()
