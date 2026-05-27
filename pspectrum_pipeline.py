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
- pivot : Large-scale pivot k_* = 0.002 Mpc^-1 at which P_R(k_*) = A_s
          (anchors near ell≈28 for low-ell anomaly; Planck default 0.05 is for high-ell)

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

from scripts.constants import As, k_pivot_phys, N_star_default
from scripts.plotting import make_filename

import inf_dyn_background as bg_solver
import inf_dyn_MS_full as ms_solver
from models import HiggsModel, FullHiggsModel, PunctuatedInflationModel, EzquiagaCHIModel, inflection_parameters
from numba_ms_solver import numba_run_ms, build_numba_splines

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


def find_end_of_inflation(epsH, window_frac=0.05):
    """
    Find index where epsilon_H permanently crosses 1 (true end of inflation).

    Strategy:
      1. Find the start of inflation (first eps < 1).
      2. Find all upward crossings (eps goes from <1 to >=1) after start.
      3. For each upward crossing, check if eps MEAN-stays >= 1 over the
         next `window` steps. The first crossing that satisfies this is
         the true end of inflation.
      4. If no permanent crossing is found, fall back to scanning
         backward from the last step above 1.

    This correctly handles:
      - SR starts (eps0 < 1): start_idx = 0, single upward crossing
      - USR/kinetic-dominance starts (eps0 > 1): start_idx after
        the initial transient decays below 1
      - USR transient spikes: eps briefly exceeds 1 and drops back;
        the mean over window remains < 1, so they are skipped.

    Returns -1 if inflation never begins or never ends within the window.
    The caller (run_pspectrum_pipeline) falls back to the last index if -1.
    """
    window = max(20, int(len(epsH) * window_frac))

    # 1. Find start of inflation (first eps < 1)
    start_idx = -1
    for idx, eps in enumerate(epsH):
        if eps < 1.0:
            start_idx = idx
            break
    if start_idx == -1:
        return -1

    # 2. Find all upward crossings after start
    candidates = []
    for i in range(start_idx + 1, len(epsH)):
        if epsH[i - 1] < 1.0 and epsH[i] >= 1.0:
            candidates.append(i)

    if not candidates:
        return -1

    # 3. For each candidate, check if eps STAYS above 1
    for idx in candidates:
        end = min(idx + window, len(epsH))
        if np.mean(epsH[idx:end]) >= 1.0:
            return idx

    # 4. Fallback: if eps is above 1 at the end but no permanent crossing
    #    was found, step backward to locate the first crossing.
    if epsH[-1] >= 1.0:
        for idx in range(len(epsH) - 1, start_idx - 1, -1):
            if epsH[idx] < 1.0:
                return idx + 1

    return -1


def ensure_k_pivot(k_grid, k_pivot_phys, rtol=1e-6):
    """Ensure the pivot scale is exactly present in k_grid. Adds it if missing."""
    if np.any(np.isclose(k_grid, k_pivot_phys, rtol=rtol, atol=0.0)):
        pivot_idx = int(np.where(np.isclose(k_grid, k_pivot_phys, rtol=rtol, atol=0.0))[0][0])
        return k_grid, pivot_idx
    k_grid = np.sort(np.append(k_grid, k_pivot_phys))
    pivot_idx = int(np.where(k_grid == k_pivot_phys)[0][0])
    return k_grid, pivot_idx


def build_weighted_kgrid(k_min, k_max, k_pivot_phys, dense_min=1e-4, dense_max=1e-2, n_dense=120, n_outer=60):
    """
    Build a k-grid with dense logarithmic sampling in a specified zone.

    Concentrates ~2/3 of modes in [dense_min, dense_max] for adequate resolution
    of spectral features (CMB USR dip at 1e-4..1e-2 for Higgs, PBH peak at 1e9..1e14 for Ezquiaga).
    """
    k_low = np.logspace(np.log10(k_min), np.log10(dense_min), n_outer // 2)
    k_dense = np.logspace(np.log10(dense_min), np.log10(dense_max), n_dense)
    k_high = np.logspace(np.log10(dense_max), np.log10(k_max), n_outer // 2)
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


def _compute_mode_batch(args):
    """Worker for batched parallel k-mode execution.

    Builds background interpolation AND Numba spline coefs ONCE per batch,
    then processes each mode sequentially. Dispatches to Numba or Python.
    Returns list of (idx, P_S, P_T, start_idx, error).
    """
    indices, k_codes, bg_sol, T_span_bg, end_idx, k_start_factor, ms_steps, model, use_numba, ms_method = args
    try:
        interp = build_bg_interpolators_fast(bg_sol, T_span_bg)
        bg_coefs = build_numba_splines(bg_sol, T_span_bg, model=model)
    except Exception as e:
        return [(idx, None, None, -1, f"setup failed: {e}") for idx in indices]

    results = []
    for idx, k_code in zip(indices, k_codes):
        try:
            xi, y0v, zi, ni, t_start, t_end, start_idx = extract_mode_initial_conditions(
                bg_sol, T_span_bg, end_idx, k_code, k_start_factor
            )
            T_ms = np.linspace(t_start, t_end, ms_steps)
            if use_numba:
                ms_sol = numba_run_ms(bg_sol, T_span_bg, T_ms, ni, k_code, model,
                                      bg_coefs=bg_coefs, method=ms_method)
            else:
                ms_sol = ms_solver.run_ms_simulation(interp, ni, T_ms, k_code, model)
            d = ms_solver.get_ms_derived_quantities_with_bg(ms_sol, interp, T_ms, model, k_code, ni)
            ps = float(d["P_S"][-1])
            pt = float(d["P_T"][-1])
            if np.isfinite(ps) and ps > 0:
                results.append((idx, ps, pt, start_idx, None))
            else:
                results.append((idx, None, None, start_idx, f"non-finite P_S={ps}"))
        except Exception as e:
            results.append((idx, None, None, -1, str(e)))
    return results


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
    bg_steps=None,
    T_max=None,
    ms_steps=5000,
    normalize_to_As=True,
    As=As,
    output_dir="outputs/simulations/pspectra",
    save_outputs=True,
    k_phys_grid=None,
    n_workers=1,
    use_numba=True,
    executor=None,
    ms_method='dp5',
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
    k_pivot_phys : float, large-scale pivot scale (default 0.002 Mpc^-1)
    N_star : float, e-folds before end where pivot exits
    k_start_factor : float, k/(aH) at which to start MS integration (default 100)
    T_span_bg : array or None, background time grid
    bg_steps, T_max : background integration parameters.
        If None, uses model.T_max and model.bg_steps (model-specific defaults).
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
        model.x0 = float(phi0)
    if y0 is not None:
        model.y0 = float(y0)

    # Resolve background integration parameters (use model defaults if not specified)
    _T_max = T_max if T_max is not None else getattr(model, 'T_max', 5000.0)
    _bg_steps = bg_steps if bg_steps is not None else getattr(model, 'bg_steps', 10000)

    if T_span_bg is None:
        T_span_bg = np.linspace(0.0, _T_max, _bg_steps)

    bg_sol = bg_solver.run_background_simulation(model, T_span_bg)
    derived_bg = bg_solver.get_derived_quantities(bg_sol, model)

    end_idx = find_end_of_inflation(derived_bg["epsH"])
    if end_idx == -1:
        # Fallback: use last index (field trapped at inflection, USR-only trajectory)
        end_idx = len(derived_bg["epsH"]) - 1

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
    checkpoint_interval = max(1, max(n_modes // 10, min(n_modes // 4, 25)))
    t_start_all = time.time()
    failed_modes = []
    errors = []

    # Build background interpolation and spline coefficients (shared)
    bg_interp = build_bg_interpolators_fast(bg_sol, T_span_bg)
    bg_coefs = build_numba_splines(bg_sol, T_span_bg, model=model)

    # Build metadata early (needs pivot info from above, which we have)
    run_id = str(uuid.uuid4())[:8]
    metadata = {
        "model": model.name,
        "x0": float(model.x0),
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
        "bg_steps": int(_bg_steps),
        "ms_steps": int(ms_steps),
        "T_max": float(_T_max),
        "run_id": run_id,
    }

    def save_checkpoint(iteration):
        if not save_outputs:
            return
        if iteration % checkpoint_interval != 0:
            return
        n_comp = int(np.sum(np.isfinite(P_S_raw)))
        partial = {
            "_type": "checkpoint",
            "format_version": 2,
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
            if use_numba:
                ms_sol = numba_run_ms(bg_sol, T_span_bg, T_ms, ni, k_code_val, model,
                                      bg_coefs=bg_coefs, method=ms_method)
            else:
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
        from concurrent.futures import             ProcessPoolExecutor, as_completed
        import multiprocessing
        n_actual = min(n_workers, n_modes, multiprocessing.cpu_count())
        print(f"  Computing {n_modes} k-modes on {n_actual} workers...")

        # Batch: one IPC round-trip per worker instead of per mode
        indices = list(range(n_modes))
        chunks = []
        for w in range(n_actual):
            ci = indices[w::n_actual]
            ck = [k_code_grid[i] for i in ci]
            chunks.append((ci, ck, bg_sol, T_span_bg, end_idx,
                           k_start_factor, ms_steps, model, use_numba, ms_method))

        done_count = 0
        pool = executor if executor is not None else ProcessPoolExecutor(max_workers=n_actual)
        try:
            futures = {pool.submit(_compute_mode_batch, c): c[0][0] for c in chunks}
            for future in as_completed(futures):
                for idx, ps, pt, si, err in future.result():
                    done_count += 1
                    record_result(idx, ps, pt, si, err)
                if done_count % checkpoint_interval == 0:
                    elapsed = time.time() - t_start_all
                    rate = done_count / max(elapsed, 0.01)
                    eta = (n_modes - done_count) / rate
                    print(f"    {done_count}/{n_modes} modes  "
                          f"[{elapsed:.0f}s<{eta:.0f}s, {rate:.1f}mode/s]",
                          flush=True)
                    save_checkpoint(done_count)
        finally:
            if executor is None:
                pool.shutdown()
    else:
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(k_code_grid), total=n_modes,
                            desc="  k-modes", unit="mode",
                            mininterval=1.0)  # 1s update for SSH
        except ImportError:
            iterator = enumerate(k_code_grid)
            print(f"  Computing {n_modes} k-modes (serial)...", flush=True)

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
        N_star = metadata["N_star"]
        filename = make_filename("ps", model.x0, model.y0, N_star, ".json")
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
            "_type": "result",
            "format_version": 2,
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
        "end_idx": end_idx,
        "bg_sol": bg_sol,
        "derived_bg": derived_bg,
    }


def model_from_config(config):
    """Construct an InflationModel instance from a config dict."""
    model_name = config["model"]
    params = config.get("model_params", {})
    ics = config.get("ics", {})
    inflection = config.get("inflection", {})

    if model_name == "HiggsModel":
        model = HiggsModel(lam=params.get("lam", 0.13), xi=params.get("xi", 15000.0))
    elif model_name == "FullHiggsModel":
        model = FullHiggsModel(lam=params.get("lam", 0.1), xi=params.get("xi", 1000.0), v_vev=params.get("v_vev", 0.0))
    elif model_name == "PunctuatedInflationModel":
        model = PunctuatedInflationModel(m=params.get("m", 1.1323e-7), lam=params.get("lam", 3.3299e-15), phi0=params.get("phi0_inflection", None))
    elif model_name == "EzquiagaCHIModel":
        model = EzquiagaCHIModel(
            lambda_0=params.get("lambda_0", 2.23e-7),
            b_lambda=params.get("b_lambda", 1.2e-6),
            xi_0=params.get("xi_0", 7.55),
            b_xi=params.get("b_xi", 11.5),
            c=params.get("c", 0.77),
            n_grid=params.get("n_grid", 5000),
        )
        x_c = inflection.get("x_c")
        if x_c is not None:
            beta = inflection.get("beta", 1e-5)
            a_new, b_new = inflection_parameters(x_c, params.get("c", 0.77), beta)
            model.a = a_new
            model.b = b_new
            model.v0 = model._V0 * model.a / (model.b * model.c)**2
            print(f"  Self-consistent inflection params: a={a_new:.4f}, b={b_new:.6f}")
        model.patch_background_solver()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if "x0" in ics:
        model.x0 = ics["x0"]
    if "y0" in ics:
        model.y0 = ics["y0"]
    if "T_max" in params:
        model.T_max = params["T_max"]
    if "bg_steps" in params:
        model.bg_steps = params["bg_steps"]
    return model


def build_model(args):
    """Construct an InflationModel instance from parsed CLI arguments (no-config fallback)."""
    cfg = {"model": args.model or "HiggsModel"}
    if args.model == "PunctuatedInflationModel":
        cfg["model_params"] = {"m": 1.1323e-7, "lam": 3.3299e-15}
    return model_from_config(cfg)


def parse_args():
    """Parse CLI arguments for the P_S(k) pipeline."""
    parser = argparse.ArgumentParser(description="Compute P_S(k) across k grid for a model.")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file with model params, ICs, and pipeline settings")
    parser.add_argument("--model", default=None, choices=[
        "HiggsModel", "FullHiggsModel", "PunctuatedInflationModel", "EzquiagaCHIModel"
    ])
    parser.add_argument("--phi0", type=float, default=None)
    parser.add_argument("--y0", type=float, default=None)

    parser.add_argument("--k-min", type=float, default=None)
    parser.add_argument("--k-max", type=float, default=None)
    parser.add_argument("--num-k", type=int, default=None)
    parser.add_argument("--k-pivot-phys", type=float, default=None)
    parser.add_argument("--N-star", type=float, default=None)
    parser.add_argument("--k-start-factor", type=float, default=None)

    parser.add_argument("--bg-steps", type=int, default=None)
    parser.add_argument("--T-max", type=float, default=None)
    parser.add_argument("--ms-steps", type=int, default=None)
    parser.add_argument("--n-cores", type=int, default=None, dest="n_cores",
                        help="parallel workers (1 = serial)")
    parser.add_argument("--n-workers", type=int, default=None, dest="n_workers",
                        help=argparse.SUPPRESS)
    parser.add_argument("--use-weighted", action="store_true", default=None,
                        help="dense sampling in USR zone 1e-4..1e-2")
    parser.add_argument("--n-dense", type=int, default=None, help="k-modes in dense zone")
    parser.add_argument("--n-outer", type=int, default=None, help="k-modes outside dense zone")
    parser.add_argument("--dense-min", type=float, default=None, help="Dense zone lower bound (Mpc^-1)")
    parser.add_argument("--dense-max", type=float, default=None, help="Dense zone upper bound (Mpc^-1)")
    parser.add_argument("--dense-min", type=float, default=None, help="dense zone start (Mpc^-1)")
    parser.add_argument("--dense-max", type=float, default=None, help="dense zone end (Mpc^-1)")
    parser.add_argument("--normalize-to-As", action="store_true", default=None)
    parser.add_argument("--As", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-plot", action="store_true", default=None,
                        help="Skip auto-plotting P_S(k)")
    return parser.parse_args()


def _resolve(key, cli_val, config, config_key=None):
    """CLI value overrides config value."""
    if cli_val is not None:
        return cli_val
    cfg = config.get("pipeline", {}) if config else {}
    return cfg.get(config_key or key, None)


def _resolve_bool(key, cli_val, config):
    """CLI boolean overrides config boolean. Respects False/None distinction."""
    if cli_val is not None:
        return cli_val
    cfg = config.get("pipeline", {}) if config else {}
    return cfg.get(key, False)


def main():
    """CLI entry point: parse args, build model, run pipeline, print result path."""
    args = parse_args()
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        model = model_from_config(config)
    else:
        model = build_model(args)

    # Resolve pipeline params: CLI overrides config
    pipe_cfg = config.get("pipeline", {}) if config else {}
    phi0 = args.phi0 if args.phi0 is not None else (config.get("ics", {}).get("phi0") if config else None)
    y0 = args.y0 if args.y0 is not None else (config.get("ics", {}).get("y0") if config else None)
    N_star = _resolve("N_star", args.N_star, config) or N_star_default
    k_min = _resolve("k_min", args.k_min, config) or 1e-5
    k_max = _resolve("k_max", args.k_max, config) or 1.0
    num_k = _resolve("num_k", args.num_k, config) or 80
    use_weighted = _resolve_bool("use_weighted", args.use_weighted, config)
    n_dense = _resolve("n_dense", args.n_dense, config) or 120
    n_outer = _resolve("n_outer", args.n_outer, config) or 60
    dense_min = _resolve("dense_min", args.dense_min, config) or 1e-4
    dense_max = _resolve("dense_max", args.dense_max, config) or 1e-2
    dense_min = _resolve("dense_min", args.dense_min, config)
    dense_max = _resolve("dense_max", args.dense_max, config)
    n_cores = _resolve("n_cores", args.n_cores, config)
    n_workers = n_cores if n_cores is not None else 1
    normalize_to_As = _resolve_bool("normalize_to_As", args.normalize_to_As, config)
    As_target = _resolve("As", args.As, config) or As
    bg_steps = _resolve("bg_steps", args.bg_steps, config)
    T_max = _resolve("T_max", args.T_max, config)
    ms_steps = _resolve("ms_steps", args.ms_steps, config) or 5000
    output_dir = _resolve("output_dir", args.output_dir, config) or "outputs/simulations/pspectra"
    k_start_factor = _resolve("k_start_factor", args.k_start_factor, config) or 100.0

    k_grid = None
    if use_weighted:
        _dm = dense_min or 1e-4
        _dM = dense_max or 1e-2
        k_grid = build_weighted_kgrid(
            k_min, k_max, k_pivot_phys or k_pivot_phys,
            dense_min=dense_min, dense_max=dense_max,
            n_dense=n_dense, n_outer=n_outer,
        )

    result = run_pspectrum_pipeline(
        model=model,
        phi0=phi0,
        y0=y0,
        k_min=k_min,
        k_max=k_max,
        num_k=num_k,
        k_pivot_phys=k_pivot_phys or k_pivot_phys,
        N_star=N_star,
        k_start_factor=k_start_factor,
        T_span_bg=None,
        bg_steps=bg_steps,
        T_max=T_max,
        ms_steps=ms_steps,
        n_workers=n_workers,
        k_phys_grid=k_grid,
        normalize_to_As=normalize_to_As,
        As=As_target,
        output_dir=output_dir,
        save_outputs=not args.no_save,
    )

    if result["status"] != "success":
        print(f"  ERROR: {result['message']}")
        return
    print(f"  Saved: {result['output_file']}")
    meta = result["metadata"]
    print(f"  N_total={meta['N_total']:.1f}, N_pivot={meta['N_pivot']:.1f}, modes={meta['n_completed']}/{meta['num_k']}")

    auto_plot = not _resolve_bool("no_plot", args.no_plot, config)
    if auto_plot:
        from scripts.plotting import plot_ps
        fname = make_filename("ps", float(meta["x0"]), float(meta["y0"]),
                              float(meta["N_star"]), ext="")
        plot_ps(result["k_phys"], result["P_S"],
                label=meta["model"],
                filename=fname,
                category="ps_plots",
                show_lcdm=True)

    if auto_plot and "Ezquiaga" in meta["model"]:
        from scripts.plotting import plot_ezquiaga_diagnostics
        plot_ezquiaga_diagnostics(
            model, result["bg_sol"], result["derived_bg"],
            result["end_idx"], float(meta["x0"]),
            N_star=float(meta["N_star"]),
        )


def load_pspectrum(path):
    """Load a cached P_S(k) JSON file into a dict of numpy arrays.

    Matches the return format expected by compute_cl_sw and analyse.
    """
    with open(path) as f:
        record = json.load(f)
    meta = record["metadata"]
    spec = record["spectrum"]
    def to_array(val):
        if val is None:
            return None
        return np.array([np.nan if x is None else x for x in val], dtype=float)

    return {
        "metadata": meta,
        "k_phys": to_array(spec.get("k_phys")),
        "k_code": to_array(spec.get("k_code")),
        "P_S": to_array(spec.get("P_S")),
        "P_T": to_array(spec.get("P_T")),
        "P_S_raw": to_array(spec.get("P_S_raw")),
        "P_T_raw": to_array(spec.get("P_T_raw")),
        "start_idx": to_array(spec.get("start_idx")),
    }


if __name__ == "__main__":
    main()
