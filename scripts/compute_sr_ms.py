"""
Standalone SR + MS computation for Ezquiaga CHI diagnostic plots.

Runs background evolution, computes SR-approximation P_S(k), then solves
Mukhanov-Sasaki for a PBH-weighted k-grid using the Fortran backend.
Caches results to JSON so plotting never runs the solver.

Usage (CLI):
    python -m scripts.compute_sr_ms --chi0 8.0 --y0 -1e-4

Imported from plotting.py (plot_ps_sr_ms_comparison), full_pbh_pipeline.py.
"""

import json
import os
import time

import numpy as np
from scipy.interpolate import CubicSpline

from scripts.constants import ROOT_DIR, S
from fortran_ms_solver import fortran_run_ms_grid, HAVE_FORTRAN
from inf_dyn_background import run_background_simulation, get_derived_quantities


# ── Exported helpers (moved from scripts/plotting) ─────────────────────────

def build_usr_weighted_kgrid(k_min=1e-8, k_max=1e18, n_dense=250, n_outer=20):
    """Build k-grid with ultra-dense zone covering the dip and peak (k=1e11 to 1e14)."""
    k_lo = np.logspace(np.log10(k_min), 9, n_outer)
    k_mid1 = np.logspace(9, 11, 30)
    k_dip = np.logspace(11, 14, n_dense)
    k_mid2 = np.logspace(14, 16, 50)
    k_hi = np.logspace(16, np.log10(k_max), n_outer)
    return np.unique(np.concatenate([k_lo, k_mid1, k_dip, k_mid2, k_hi]))


def compute_ps_sr(bg_sol, end_idx):
    """Compute SR-approximation P_S(k) from background solution."""
    x, y, z, n = bg_sol
    epsH = y**2 / (2 * z**2)
    k_phys = np.exp(n[:end_idx + 1]) * z[:end_idx + 1] * S
    e = epsH[:end_idx + 1]
    P_S = np.where(np.isfinite(e) & (e > 0),
                   (S * z[:end_idx + 1])**2 / (8 * np.pi**2 * e),
                   np.nan)
    return k_phys, P_S, e


# ── JSON I/O ───────────────────────────────────────────────────────────────

def _output_dir():
    p = os.path.join(ROOT_DIR, "outputs/simulations/pspectra")
    os.makedirs(p, exist_ok=True)
    return p


def _make_output_filename(model, chi0, y0):
    x0 = chi0 if chi0 is not None else model.x0
    return f"sr_ms_chi{x0:.2f}_y0{y0:+.3f}.json"


def save_sr_ms_output(data, model, chi0, y0):
    """Save SR+MS result dict to JSON. Returns file path."""
    fname = _make_output_filename(model, chi0, y0)
    path = os.path.join(_output_dir(), fname)

    def convert(val):
        if isinstance(val, (np.floating, float)):
            if np.isfinite(val):
                return float(val)
            return None
        if isinstance(val, (np.integer, int)):
            return int(val)
        if isinstance(val, np.ndarray):
            return [None if np.isnan(x) else float(x) for x in val]
        if isinstance(val, list):
            return [None if (isinstance(x, float) and np.isnan(x)) else x for x in val]
        return val

    record = {
        "_type": "sr_ms_result",
        "format_version": 1,
        "metadata": {k: convert(v) for k, v in data.get("metadata", {}).items()},
        "sr": {
            "k_phys": convert(data["k_phys_sr"]),
            "P_S": convert(data["P_S_sr"]),
        },
        "ms": {
            "k_phys": convert(data["k_phys_ms"]),
            "P_S": convert(data["P_S_ms"]),
        },
        "ns_local": data.get("n_s_local", {}),
    }

    with open(path, "w") as f:
        json.dump(record, f, indent=2)

    return path


def load_sr_ms_output(model, chi0, y0):
    """Load cached SR+MS JSON. Returns dict or None if not found."""
    fname = _make_output_filename(model, chi0, y0)
    path = os.path.join(_output_dir(), fname)
    if not os.path.exists(path):
        return None

    with open(path) as f:
        record = json.load(f)

    def to_array(val):
        if val is None:
            return None
        return np.array([np.nan if x is None else x for x in val], dtype=float)

    sr = record.get("sr", {})
    ms = record.get("ms", {})
    result: dict = {
        "k_phys_sr": to_array(sr.get("k_phys")),
        "P_S_sr": to_array(sr.get("P_S")),
        "k_phys_ms": to_array(ms.get("k_phys")),
        "P_S_ms": to_array(ms.get("P_S")),
        "n_s_local": record.get("ns_local", {}),
        "metadata": record.get("metadata", {}),
    }
    return result


# ── Core compute ───────────────────────────────────────────────────────────

def compute_sr_ms(
    model,
    chi0=None,
    y0=-1e-4,
    k_min=1e-8,
    k_max=1e18,
    n_dense=200,
    n_outer=20,
    n_workers=8,
    ms_method="dp5",
    k_start_factor=100.0,
    k_pivots=None,
) -> dict:
    """
    Run background + SR + Fortran MS for Ezquiaga CHI model.

    Parameters
    ----------
    model : EzquiagaCHIModel (already configured with inflection params)
    chi0 : float or None — sets model.x0 if given
    y0 : float — sets model.y0
    k_min, k_max : float — k-range in Mpc^-1
    n_dense, n_outer : int — PBH-weighted k-grid params
    n_workers : int — parallel workers (unused for Fortran grid solver)
    ms_method : str — MS integration method ('dp5')
    k_start_factor : float — how far before horizon exit to start
    k_pivots : list of (label, k_value) or None (defaults to pivots)

    Returns
    -------
    dict with keys: k_phys_sr, P_S_sr, k_phys_ms, P_S_ms, n_s_local, metadata
    """
    if chi0 is not None:
        model.x0 = chi0
    model.y0 = y0

    T = np.linspace(0, model.T_max, model.bg_steps)
    bg_sol = run_background_simulation(model, T)
    derived = get_derived_quantities(bg_sol, model)

    epsH = derived["epsH"]
    eps1_arr = np.where(np.isfinite(epsH) & (epsH >= 1.0))[0]
    end_idx = int(eps1_arr[0]) if len(eps1_arr) > 0 else len(epsH) - 1
    N_total = float(derived["N"][end_idx])
    print(f"  N_total = {N_total:.1f}, end_idx = {end_idx}")

    # SR
    k_sr, Ps_sr, _ = compute_ps_sr(bg_sol, end_idx)
    valid = np.where(np.isfinite(Ps_sr) & (Ps_sr > 0))[0]
    k_sr, Ps_sr = k_sr[valid], Ps_sr[valid]

    # MS via Fortran grid solver
    k_grid = build_usr_weighted_kgrid(k_sr.min(), k_sr.max(), n_dense=n_dense, n_outer=n_outer)
    n_k = len(k_grid)
    print(f"  Running MS for {n_k} modes (Fortran backend)...")

    t0 = time.time()
    k_codes = k_grid / S
    ps_arr, pt_arr, start_idx_arr = fortran_run_ms_grid(
        bg_sol, T, end_idx, k_codes, model,
        k_start_factor=k_start_factor, S=S,
    )
    elapsed = time.time() - t0

    ok = np.isfinite(ps_arr)
    n_ok = int(np.sum(ok))
    print(f"  MS done: {n_ok}/{n_k} OK in {elapsed:.0f}s")

    k_ms = k_grid if n_ok > 0 else None
    Ps_ms = ps_arr if n_ok > 0 else None

    # n_s local at pivot(s)
    n_s_results = {}
    h_width = 5
    for label, k_piv in (k_pivots or [("k=0.002", 0.002), ("k=0.05", 0.05)]):
        pi = int(np.argmin(np.abs(k_sr - k_piv)))
        if pi < h_width or pi > len(k_sr) - h_width - 1:
            continue
        lk = np.log(k_sr[pi - h_width:pi + h_width + 1])
        lp_sr = np.log(Ps_sr[pi - h_width:pi + h_width + 1])
        ns_sr = 1 + (lp_sr[-1] - lp_sr[0]) / (lk[-1] - lk[0])
        ns_ms_val = None
        if Ps_ms is not None and ok[pi]:
            interp_at_piv = np.interp(k_piv, k_grid, ps_arr)
            if np.isfinite(interp_at_piv) and interp_at_piv > 0:
                pi_ms = int(np.argmin(np.abs(k_grid - k_piv)))
                if pi_ms >= h_width and pi_ms <= len(k_grid) - h_width - 1:
                    lp_ms = np.log(np.maximum(ps_arr[pi_ms - h_width:pi_ms + h_width + 1], 1e-300))
                    lk_ms = np.log(k_grid[pi_ms - h_width:pi_ms + h_width + 1])
                    ns_ms_val = 1 + (lp_ms[-1] - lp_ms[0]) / (lk_ms[-1] - lk_ms[0])
        msg = f"  {label}: n_s_local(SR)={ns_sr:.4f}"
        if ns_ms_val is not None:
            msg += f", n_s_local(MS)={ns_ms_val:.4f}"
        print(msg)
        n_s_results[label] = {"SR_local": ns_sr, "MS_local": ns_ms_val}

    metadata = {
        "x0": float(model.x0),
        "y0": float(model.y0),
        "N_total": N_total,
        "end_idx": int(end_idx),
        "n_k": int(n_k),
        "n_ok": n_ok,
        "k_min": float(k_min),
        "k_max": float(k_max),
        "backend": "fortran",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    return {
        "k_phys_sr": k_sr,
        "P_S_sr": Ps_sr,
        "k_phys_ms": k_ms,
        "P_S_ms": Ps_ms,
        "n_s_local": n_s_results,
        "metadata": metadata,
        "N_total": N_total,
        "end_idx": end_idx,
    }


def compute_and_save(
    model,
    chi0=None,
    y0=-1e-4,
    k_min=1e-8,
    k_max=1e18,
    n_dense=200,
    n_outer=20,
    n_workers=8,
    ms_method="dp5",
    k_start_factor=100.0,
    k_pivots=None,
    force=False,
) -> dict:
    """
    Compute SR+MS, cache to JSON, return data dict.

    Skips computation if cached JSON exists (unless force=True).
    """
    cached = load_sr_ms_output(model, chi0, y0)
    if cached is not None and not force:
        print(f"  Using cached SR+MS data for chi={chi0 or model.x0}, y0={y0}")
        return cached

    data = compute_sr_ms(
        model,
        chi0=chi0,
        y0=y0,
        k_min=k_min,
        k_max=k_max,
        n_dense=n_dense,
        n_outer=n_outer,
        n_workers=n_workers,
        ms_method=ms_method,
        k_start_factor=k_start_factor,
        k_pivots=k_pivots,
    )
    out_path = save_sr_ms_output(data, model, chi0, y0)
    print(f"  Saved: {out_path}")
    return data


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from models.ezquiaga_chi import EzquiagaCHIModel, inflection_parameters

    parser = argparse.ArgumentParser(description="SR + MS computation for Ezquiaga CHI")
    parser.add_argument("--chi0", type=float, default=8.0)
    parser.add_argument("--y0", type=float, default=-1e-4)
    parser.add_argument("--xc", type=float, default=0.784)
    parser.add_argument("--c", type=float, default=0.77)
    parser.add_argument("--beta", type=float, default=1e-5)
    parser.add_argument("--k-min", type=float, default=1e-8)
    parser.add_argument("--k-max", type=float, default=1e18)
    parser.add_argument("--n-dense", type=int, default=200)
    parser.add_argument("--n-outer", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="Recompute even if cached")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plot")
    parser.add_argument("--plot-filename", type=str, default=None, help="Custom filename for the plot")
    args = parser.parse_args()

    a, b = inflection_parameters(args.xc, args.c, beta=args.beta)
    model = EzquiagaCHIModel(c=args.c)
    model.a, model.b = a, b

    data = compute_and_save(
        model,
        chi0=args.chi0,
        y0=args.y0,
        k_min=args.k_min,
        k_max=args.k_max,
        n_dense=args.n_dense,
        n_outer=args.n_outer,
        force=args.force,
    )

    meta = data.get("metadata") or {}
    n_ok = meta.get("n_ok", 0)
    n_total = data.get("N_total", meta.get("N_total", 0))
    print(f"Done: {n_ok} MS modes, N_total={n_total:.1f}")

    if args.plot:
        from scripts.plotting import plot_ps_sr_ms_comparison
        plot_fname = args.plot_filename or f"ps_sr_ms_comparison_ezquiaga_beta{args.beta:.1e}"
        print(f"Generating comparison plot as: {plot_fname}")
        plot_ps_sr_ms_comparison(
            model=model,
            chi0=args.chi0,
            y0=args.y0,
            filename=plot_fname,
            category="paper",
            dpi=300,
            data=data,
        )
