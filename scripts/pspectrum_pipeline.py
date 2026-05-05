"""
Primordial power spectrum pipeline: background + Mukhanov-Sasaki per k-mode.

Orchestrates:
1. Background integration (inf_dyn_background)
2. End-of-inflation detection and pivot-scale mapping
3. Mukhanov-Sasaki integration for each k-mode (inf_dyn_MS_full)
4. Normalisation to Planck A_s at pivot
5. Caching to JSON

K-space conventions
-------------------
- k_phys : physical wavenumber (Mpc^-1), same convention as Planck
- k_code : code-unit wavenumber used internally by the MS solver
- pivot : Planck pivot scale k_* = 0.05 Mpc^-1 at which P_R(k_*) = A_s

The weighted k-grid concentrates modes in the USR dip region
(10^-4 to 10^-2 Mpc^-1) to resolve the spectral feature.
"""

import argparse
import json
import os
import sys
import time
import uuid
import warnings

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from scripts.constants import As, k_pivot_phys, N_star_default

import inf_dyn_background as bg_solver
import inf_dyn_MS_full as ms_solver
from models import HiggsModel, FullHiggsModel, PunctuatedInflationModel

# Fast CubicSpline-based interpolator (avoids interp1d overhead for per-step calls)
from scipy.interpolate import CubicSpline


def build_bg_interpolators_fast(bg_sol, T_span):
    """
    Build interpolation functions using CubicSpline instead of interp1d.
    
    CubicSpline avoids ~60% of the Python overhead that interp1d adds for
    scalar per-step lookups in the ODE RHS function. Physics is identical
    (max diff < 1e-15).
    """
    x_interp = CubicSpline(T_span, bg_sol[0], bc_type='not-a-knot', extrapolate=True)
    y_interp = CubicSpline(T_span, bg_sol[1], bc_type='not-a-knot', extrapolate=True)
    z_interp = CubicSpline(T_span, bg_sol[2], bc_type='not-a-knot', extrapolate=True)
    n_interp = CubicSpline(T_span, bg_sol[3], bc_type='not-a-knot', extrapolate=True)
    return x_interp, y_interp, z_interp, n_interp


def find_end_of_inflation(epsH):
    """
    Find index where epsilon_H last crosses 1 (true end of inflation).

    Returns -1 if inflation never begins or never ends within the window.
    Scans backward so the USR peak (eps > 1 in the middle) isn't mistaken
    for the end of inflation.
    """
    in_inflation = False
    start_idx = -1
    for idx, eps in enumerate(epsH):
        if not in_inflation:
            if eps < 1.0:
                in_inflation = True
                start_idx = idx
        else:
            pass
    if start_idx == -1:
        return -1
    # Scan backward from end to find the LAST time eps >= 1
    for idx in range(len(epsH) - 1, start_idx - 1, -1):
        if epsH[idx] >= 1.0:
            return idx
    return -1


def ensure_k_pivot(k_grid, k_pivot_phys, rtol=1e-6):
    """Ensure the pivot scale is exactly present in k_grid. Adds it if missing."""
    if np.any(np.isclose(k_grid, k_pivot_phys, rtol=rtol, atol=0.0)):
        pivot_idx = int(np.where(np.isclose(k_grid, k_pivot_phys, rtol=rtol, atol=0.0))[0][0])
        return k_grid, pivot_idx
    k_grid = np.sort(np.append(k_grid, k_pivot_phys))
    pivot_idx = int(np.where(k_grid == k_pivot_phys)[0][0])
    return k_grid, pivot_idx


def build_weighted_kgrid(k_min, k_max, k_pivot_phys, dense_zone=(1e-4, 1e-2), n_dense=120, n_outer=60):
    """
    Build a k-grid with dense logarithmic sampling in the USR dip zone.

    The USR feature typically appears at k ~ 10^-4 to 10^-2 Mpc^-1.
    Standard uniform log-spacing would undersample this region.
    This grid concentrates ~2/3 of modes there for adequate resolution.
    """
    k_low = np.logspace(np.log10(k_min), np.log10(dense_zone[0]), n_outer // 2)
    k_dense = np.logspace(np.log10(dense_zone[0]), np.log10(dense_zone[1]), n_dense)
    k_high = np.logspace(np.log10(dense_zone[1]), np.log10(k_max), n_outer // 2)
    k_grid = np.unique(np.concatenate([k_low, k_dense, k_high]))
    if not np.any(np.isclose(k_grid, k_pivot_phys)):
        k_grid = np.sort(np.append(k_grid, k_pivot_phys))
    return k_grid


def get_k_pivot_code(bg_sol, derived_bg, end_idx, N_star):
    """
    Find the code-unit wavenumber k_code at pivot exit.

    The pivot exits N_star e-folds before the end of inflation.
    k_code = a_pivot * z_pivot (dimensionless code units),
    used to scale all physical k-modes into the MS solver.

    Returns (k_pivot_code, pivot_bg_idx, N_total) or (None, None, None)
    if total e-folds are insufficient for N_star.
    """
    N_total = derived_bg["N"][end_idx]
    if N_total < N_star:
        return None, None, None
    N_pivot = N_total - N_star
    pivot_idx = int(np.argmin(np.abs(derived_bg["N"][:end_idx] - N_pivot)))
    z_pivot = bg_sol[2][pivot_idx]
    a_pivot = np.exp(bg_sol[3][pivot_idx])
    k_pivot_code = a_pivot * z_pivot
    return k_pivot_code, pivot_idx, N_total


def extract_mode_initial_conditions(bg_sol, T_span_bg, end_idx, k_code, k_start_factor):
    """
    Extract Bunch-Davies initial conditions for a given k-mode.

    Finds the time when k/(aH) = k_start_factor (typically 100), i.e.
    sufficiently deep inside the horizon for vacuum initial conditions.
    Uses the background trajectory to locate this crossing.
    """
    n_bg = bg_sol[3]
    z_bg = bg_sol[2]
    log_az = n_bg + np.log(z_bg)
    target_start = np.log(k_code) - np.log(k_start_factor)
    start_idx = int(np.argmin(np.abs(log_az[:end_idx] - target_start)))
    start_idx = max(start_idx, 0)
    xi = bg_sol[0][start_idx]
    y0 = bg_sol[1][start_idx]
    zi = bg_sol[2][start_idx]
    ni = bg_sol[3][start_idx]
    t_start = T_span_bg[start_idx]
    t_end = T_span_bg[end_idx]
    return xi, y0, zi, ni, t_start, t_end, start_idx


def _compute_single_mode(args):
    """Worker function for parallel k-mode execution. Returns (idx, P_S, P_T, start_idx, error)."""
    idx, k_code, bg_sol, T_span_bg, end_idx, k_start_factor, ms_steps, model = args
    try:
        interp = build_bg_interpolators_fast(bg_sol, T_span_bg)
        xi, y0_val, zi, ni, t_start, t_end, start_idx = extract_mode_initial_conditions(
            bg_sol, T_span_bg, end_idx, k_code, k_start_factor
        )
        T_ms = np.linspace(t_start, t_end, ms_steps)
        ms_sol = ms_solver.run_ms_simulation(interp, ni, T_ms, k_code, model)
        derived_ms = ms_solver.get_ms_derived_quantities_with_bg(ms_sol, interp, T_ms, model, k_code, ni)
        ps = float(derived_ms["P_S"][-1])
        pt = float(derived_ms["P_T"][-1])
        if not (np.isfinite(ps) and ps > 0):
            return idx, None, None, start_idx, f"non-finite P_S={ps}"
        return idx, ps, pt, start_idx, None
    except Exception as e:
        return idx, None, None, -1, str(e)


def run_pspectrum_pipeline(
    model,
    phi0=None,
    y0=None,
    k_min=1e-5,
    k_max=1.0,
    num_k=80,
    k_pivot_phys=k_pivot_phys,
    N_star=N_star_default,
    k_start_factor=100.0,
    T_span_bg=None,
    bg_steps=10000,
    T_max=5000.0,
    ms_steps=5000,
    normalize_to_As=True,
    As=As,
    output_dir="outputs/simulations/pspectra",
    save_outputs=True,
    k_phys_grid=None,
    n_workers=1,
):
    """
    Compute P_S(k) for a grid of k-modes for a given inflation model.

    Pipeline:
        1. Integrate background ODE
        2. Find when inflation ends (eps_H = 1)
        3. Locate pivot exit N_star e-folds before end
        4. For each k-mode: extract initial conditions, integrate MS equations,
           read off P_S and P_T at end of inflation
        5. Normalise to Planck A_s at the pivot scale

    Parameters
    ----------
    model : InflationModel instance
    phi0, y0 : float, optional — overrides model defaults
    k_min, k_max : float, physical k-range (Mpc^-1)
    num_k : int, modes per decade if k_phys_grid not provided
    k_pivot_phys : float, Planck pivot scale (default 0.05 Mpc^-1)
    N_star : float, e-folds before end where pivot exits
    k_start_factor : float, k/(aH) at which to start MS integration (default 100)
    T_span_bg : array or None, background time grid
    bg_steps, T_max : background integration parameters
    ms_steps : int, steps per k-mode MS integration
    normalize_to_As : bool, rescale P_S to match As at pivot
    As : float, target amplitude at pivot (Planck 2018: 2.1e-9)
    output_dir : str, where to save JSON cache
    save_outputs : bool, write JSON file
    k_phys_grid : array or None, custom physical k-grid (overrides num_k)
    n_workers : int, parallel workers (1 = serial)

    Returns
    -------
    dict with keys:
        "status" : "success" or "error"
        "message" : error description if status is "error"
        "k_phys", "P_S", "P_T" : arrays
        "metadata" : dict of run configuration
        "output_file" : path to saved JSON (if save_outputs=True)
    """
    if phi0 is not None:
        model.phi0 = float(phi0)
    if y0 is not None:
        model.y0 = float(y0)

    if T_span_bg is None:
        T_span_bg = np.linspace(0.0, T_max, bg_steps)

    bg_sol = bg_solver.run_background_simulation(model, T_span_bg)
    derived_bg = bg_solver.get_derived_quantities(bg_sol, model)

    end_idx = find_end_of_inflation(derived_bg["epsH"])
    if end_idx == -1:
        return {"status": "error", "message": "Inflation did not end in background window."}

    k_pivot_code, pivot_bg_idx, N_total = get_k_pivot_code(bg_sol, derived_bg, end_idx, N_star)
    if k_pivot_code is None:
        return {
            "status": "error",
            "message": f"Total inflation ({derived_bg['N'][end_idx]:.2f}) is less than N_star ({N_star}).",
        }

    if not (k_min <= k_pivot_phys <= k_max):
        return {
            "status": "error",
            "message": "k_pivot_phys must be within [k_min, k_max] to normalize spectrum.",
        }

    if k_phys_grid is not None and len(k_phys_grid) > 0:
        k_phys_grid = np.asarray(k_phys_grid, dtype=float)
    else:
        k_phys_grid = np.logspace(np.log10(k_min), np.log10(k_max), num_k)
    k_phys_grid, pivot_idx = ensure_k_pivot(k_phys_grid, k_pivot_phys)
    k_code_grid = k_pivot_code * (k_phys_grid / k_pivot_phys)

    P_S_raw = np.full_like(k_phys_grid, np.nan)
    P_T_raw = np.full_like(k_phys_grid, np.nan)
    start_indices = np.full_like(k_phys_grid, -1, dtype=int)

    # ── Per-mode hot path optimizations ─────────────────────────────────────
    # Pre-bind hot functions to local variables (avoids global/dict lookups)
    _linspace = np.linspace
    _run_ms = ms_solver.run_ms_simulation
    _get_derived = ms_solver.get_ms_derived_quantities_with_bg
    _extract = extract_mode_initial_conditions

    n_modes = len(k_code_grid)
    checkpoint_interval = max(n_modes // 10, 1)
    t_start_all = time.time()
    failed_modes = []
    errors = []

    # Build background interpolation for serial mode
    bg_interp = build_bg_interpolators_fast(bg_sol, T_span_bg)

    # Build metadata early (needs pivot info from above, which we have)
    run_id = str(uuid.uuid4())[:8]
    metadata = {
        "model": model.name,
        "phi0": float(model.phi0),
        "y0": float(model.y0),
        "xi": getattr(model, "xi_val", None),
        "m": getattr(model, "m", None),
        "lam": getattr(model, "lam", None),
        "v_vev": getattr(model, "v_vev", None),
        "k_min": float(k_min),
        "k_max": float(k_max),
        "num_k": int(len(k_phys_grid)),
        "k_pivot_phys": float(k_pivot_phys),
        "k_pivot_code": float(k_pivot_code),
        "N_star": float(N_star),
        "N_pivot": float(N_total - N_star),
        "N_total": float(N_total),
        "pivot_bg_idx": int(pivot_bg_idx),
        "pivot_k_idx": int(pivot_idx),
        "k_start_factor": float(k_start_factor),
        "normalize_to_As": bool(normalize_to_As),
        "As_target": float(As),
        "bg_steps": int(bg_steps),
        "ms_steps": int(ms_steps),
        "T_max": float(T_max),
        "run_id": run_id,
    }

    def save_checkpoint(iteration):
        if not save_outputs:
            return
        if iteration % checkpoint_interval != 0:
            return
        n_comp = int(np.sum(np.isfinite(P_S_raw)))
        partial = {
            "metadata": metadata,
            "spectrum": {
                "k_phys": k_phys_grid.tolist(),
                "k_code": k_code_grid.tolist(),
                "P_S": [None if np.isnan(x) else x for x in P_S_raw.tolist()],
                "P_T": [None if np.isnan(x) else x for x in P_T_raw.tolist()],
                "start_idx": start_indices.tolist(),
                "failed_modes": failed_modes,
                "errors": errors,
                "n_completed": n_comp,
            },
            "partial": True,
        }
        partial_path = os.path.join(output_dir, f"_checkpoint_{run_id}.json")
        try:
            with open(partial_path, "w") as f:
                json.dump(partial, f, indent=2)
        except (OSError, IOError) as e:
            warnings.warn(f"Checkpoint save failed: {e}")

    def _solve_one_mode_fast(idx, k_code_val):
        """Tight helper: extract + solve in one call, returns (ps, pt, si, err)."""
        try:
            xi, y0v, zi, ni, t_start, t_end, si = _extract(
                bg_sol, T_span_bg, end_idx, k_code_val, k_start_factor
            )
            T_ms = _linspace(t_start, t_end, ms_steps)
            ms_sol = _run_ms(bg_interp, ni, T_ms, k_code_val, model)
            d = _get_derived(ms_sol, bg_interp, T_ms, model, k_code_val, ni)
            ps = float(d["P_S"][-1])
            pt = float(d["P_T"][-1])
            if np.isfinite(ps) and ps > 0:
                return ps, pt, si, None
            return None, None, si, f"non-finite P_S={ps}"
        except Exception as e:
            return None, None, -1, str(e)

    def record_result(idx, ps, pt, si, err_msg):
        """Store result from a mode computation."""
        if err_msg is not None:
            failed_modes.append(idx)
            errors.append(f"mode {idx} (k={k_code_grid[idx]:.4e}): {err_msg}")
            return
        if np.isfinite(ps) and ps > 0:
            P_S_raw[idx] = ps
            P_T_raw[idx] = pt
            start_indices[idx] = si
        else:
            failed_modes.append(idx)
            errors.append(f"mode {idx} (k={k_code_grid[idx]:.4e}): non-finite P_S={ps}")

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        n_actual = min(n_workers, n_modes, multiprocessing.cpu_count())
        print(f"  Computing {n_modes} k-modes on {n_actual} workers...")
        tasks = [
            (i, kc, bg_sol, T_span_bg, end_idx, k_start_factor, ms_steps, model)
            for i, kc in enumerate(k_code_grid)
        ]
        done_count = 0
        with ProcessPoolExecutor(max_workers=n_actual) as executor:
            futures = {executor.submit(_compute_single_mode, t): t[0] for t in tasks}
            for future in as_completed(futures):
                done_count += 1
                idx, ps, pt, si, err = future.result()
                record_result(idx, ps, pt, si, err)
                if done_count % checkpoint_interval == 0:
                    print(f"    {done_count}/{n_modes} modes done")
                    save_checkpoint(done_count)
    else:
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(k_code_grid), total=n_modes, desc="  k-modes", unit="mode")
        except ImportError:
            iterator = enumerate(k_code_grid)
            print(f"  Computing {n_modes} k-modes (serial)...")

        for idx, k_code in iterator:
            ps, pt, si, err = _solve_one_mode_fast(idx, k_code)
            record_result(idx, ps, pt, si, err)
            try:
                save_checkpoint(idx + 1)
            except Exception:
                pass

    elapsed = time.time() - t_start_all
    n_ok = n_modes - len(failed_modes)
    print(f"\n  Completed {n_ok}/{n_modes} modes in {elapsed:.1f}s ({elapsed/max(n_ok,1):.1f}s/mode avg)")
    if failed_modes:
        print(f"  {len(failed_modes)} failed mode(s):")
        for e in errors[:5]:
            print(f"    {e}")
        if len(errors) > 5:
            print(f"    ... and {len(errors)-5} more")

    metadata["elapsed_s"] = round(elapsed, 1)
    metadata["n_failed"] = len(failed_modes)
    metadata["scale_factor"] = None  # placeholder, filled after normalization

    output_path = None
    norm_scale = None
    if normalize_to_As:
        P_S_pivot = float(P_S_raw[pivot_idx])
        if np.isfinite(P_S_pivot) and P_S_pivot > 0:
            norm_scale = float(As / P_S_pivot)
        else:
            print(f"  WARNING: Pivot P_S={P_S_pivot} invalid, skipping normalization")
    scale_factor = norm_scale if norm_scale is not None else 1.0
    P_S = np.where(np.isfinite(P_S_raw), P_S_raw * scale_factor, np.nan)
    P_T = np.where(np.isfinite(P_T_raw), P_T_raw * scale_factor, np.nan)

    metadata["scale_factor"] = scale_factor
    metadata["norm_ok"] = norm_scale is not None
    metadata["n_completed"] = int(np.sum(np.isfinite(P_S_raw)))
    metadata["P_S_pivot_raw"] = float(P_S_raw[pivot_idx]) if np.isfinite(P_S_raw[pivot_idx]) else None
    metadata["P_S_min"] = float(np.nanmin(P_S))
    metadata["P_S_max"] = float(np.nanmax(P_S))
    metadata["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)
        safe_model = model.name.replace(" ", "_").replace("(", "").replace(")", "")
        try:
            m_str = f"_m{model.m:.3e}" if hasattr(model, 'm') and model.m else ""
        except Exception:
            m_str = ""
        try:
            xi_str = f"_xi{int(model.xi_val)}" if hasattr(model, 'xi_val') and model.xi_val else ""
        except Exception:
            xi_str = ""
        try:
            lam_str = f"_lam{model.lam:.3e}" if hasattr(model, 'lam') and model.lam else ""
        except Exception:
            lam_str = ""
        filename = (f"PS_{safe_model}{m_str}{xi_str}{lam_str}"
                    f"_phi{model.phi0:.2f}_y0{model.y0:.3f}"
                    f"_Nstar{metadata['N_star']:.0f}"
                    f"_{run_id}.json")
        output_path = os.path.join(output_dir, filename)

        def convert(val):
            if isinstance(val, np.floating):
                if np.isfinite(val):
                    return float(val)
                return None
            if isinstance(val, np.integer):
                return int(val)
            if isinstance(val, np.ndarray):
                return [None if np.isnan(x) else float(x) for x in val]
            if isinstance(val, list):
                return [None if (isinstance(x, float) and np.isnan(x)) else x for x in val]
            return val

        record = {
            "metadata": {k: convert(v) for k, v in metadata.items()},
            "spectrum": {
                "k_phys": convert(k_phys_grid),
                "k_code": convert(k_code_grid),
                "P_S": convert(P_S),
                "P_T": convert(P_T),
                "P_S_raw": convert(P_S_raw),
                "P_T_raw": convert(P_T_raw),
                "start_idx": convert(start_indices),
            },
            "errors": errors[:100] if errors else [],
        }
        with open(output_path, "w") as f:
            json.dump(record, f, indent=2)

        # Remove checkpoint file
        cp_path = os.path.join(output_dir, f"_checkpoint_{run_id}.json")
        if os.path.exists(cp_path):
            try:
                os.remove(cp_path)
            except OSError:
                pass

    print(f"  P_S range: [{np.nanmin(P_S):.4e}, {np.nanmax(P_S):.4e}]")
    if norm_scale is not None:
        print(f"  Normalized to As={As:.2e} at k_pivot (scale={norm_scale:.4e})")

    return {
        "status": "success" if n_ok > 0 else "error",
        "message": f"{n_ok}/{n_modes} modes completed" if n_ok > 0 else "All modes failed",
        "k_phys": k_phys_grid,
        "P_S": P_S,
        "P_T": P_T,
        "metadata": metadata,
        "output_file": output_path,
    }


def build_model(args):
    """Construct an InflationModel instance from parsed CLI arguments."""
    if args.model == "HiggsModel":
        return HiggsModel(lam=args.lam, xi=args.xi)
    if args.model == "FullHiggsModel":
        return FullHiggsModel(lam=args.lam, xi=args.xi, v_vev=args.v_vev)
    if args.model == "PunctuatedInflationModel":
        return PunctuatedInflationModel(m=args.m, lam=args.lam, phi0=args.phi0_inflection)
    raise ValueError(f"Unknown model: {args.model} — use HiggsModel, FullHiggsModel, or PunctuatedInflationModel")


def parse_args():
    """Parse CLI arguments for the P_S(k) pipeline."""
    parser = argparse.ArgumentParser(description="Compute P_S(k) across k grid for a model.")
    parser.add_argument("--model", default="HiggsModel", choices=[
        "HiggsModel", "FullHiggsModel", "PunctuatedInflationModel"
    ])
    parser.add_argument("--phi0", type=float, default=None)
    parser.add_argument("--phi0-inflection", type=float, default=None, dest="phi0_inflection",
                        help="Inflection point location (breaks perfect inflection when ≠ m/√λ)")
    parser.add_argument("--y0", type=float, default=None)
    parser.add_argument("--xi", type=float, default=15000.0)
    parser.add_argument("--lam", type=float, default=0.13)
    parser.add_argument("--m", type=float, default=1.1323e-7, help="mass parameter for PunctuatedInflationModel")
    parser.add_argument("--v-vev", type=float, default=0.0)

    parser.add_argument("--k-min", type=float, default=1e-5)
    parser.add_argument("--k-max", type=float, default=1.0)
    parser.add_argument("--num-k", type=int, default=80)
    parser.add_argument("--k-pivot-phys", type=float, default=k_pivot_phys)
    parser.add_argument("--N-star", type=float, default=N_star_default)
    parser.add_argument("--k-start-factor", type=float, default=100.0)

    parser.add_argument("--bg-steps", type=int, default=10000)
    parser.add_argument("--T-max", type=float, default=5000.0)
    parser.add_argument("--ms-steps", type=int, default=5000)
    parser.add_argument("--n-cores", type=int, default=None, dest="n_cores",
                        help="parallel workers (1 = serial)")
    parser.add_argument("--n-workers", type=int, default=None, dest="n_workers",
                        help=argparse.SUPPRESS)
    parser.add_argument("--use-weighted", action="store_true", help="dense sampling in USR zone 1e-4..1e-2")
    parser.add_argument("--n-dense", type=int, default=120, help="k-modes in dense zone")
    parser.add_argument("--n-outer", type=int, default=60, help="k-modes outside dense zone")
    parser.add_argument("--normalize-to-As", action="store_true")
    parser.add_argument("--As", type=float, default=As)
    parser.add_argument("--output-dir", default="outputs/simulations/pspectra")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main():
    """CLI entry point: parse args, build model, run pipeline, print result path."""
    args = parse_args()
    model = build_model(args)
    k_grid = None
    if args.use_weighted:
        k_grid = build_weighted_kgrid(
            args.k_min, args.k_max, args.k_pivot_phys,
            n_dense=args.n_dense, n_outer=args.n_outer,
        )
    n_workers = args.n_cores if args.n_cores is not None else (args.n_workers or 1)
    result = run_pspectrum_pipeline(
        model=model,
        phi0=args.phi0,
        y0=args.y0,
        k_min=args.k_min,
        k_max=args.k_max,
        num_k=args.num_k,
        k_pivot_phys=args.k_pivot_phys,
        N_star=args.N_star,
        k_start_factor=args.k_start_factor,
        T_span_bg=None,
        bg_steps=args.bg_steps,
        T_max=args.T_max,
        ms_steps=args.ms_steps,
        n_workers=n_workers,
        k_phys_grid=k_grid,
        normalize_to_As=args.normalize_to_As,
        As=args.As,
        output_dir=args.output_dir,
        save_outputs=not args.no_save,
    )

    if result["status"] != "success":
        print(f"  ERROR: {result['message']}")
        return
    print(f"  Saved: {result['output_file']}")
    meta = result["metadata"]
    print(f"  N_total={meta['N_total']:.1f}, N_pivot={meta['N_pivot']:.1f}, modes={meta['n_completed']}/{meta['num_k']}")


def load_pspectrum(path):
    """Load a cached P_S(k) JSON file into a dict of numpy arrays.

    Matches the return format expected by compute_cl_sw and analyse.
    """
    with open(path) as f:
        record = json.load(f)
    meta = record["metadata"]
    spec = record["spectrum"]
    return {
        "metadata": meta,
        "k_phys": np.array(spec["k_phys"]),
        "k_code": np.array(spec["k_code"]),
        "P_S": np.array(spec["P_S"]),
        "P_T": np.array(spec["P_T"]),
        "P_S_raw": np.array(spec["P_S_raw"]),
        "P_T_raw": np.array(spec["P_T_raw"]),
        "start_idx": np.array(spec["start_idx"]),
    }


if __name__ == "__main__":
    main()
