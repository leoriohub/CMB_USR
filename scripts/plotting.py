"""Plotting utilities + CLI for CMB anomaly analysis.

Importable module (plot_ps, plot_dell, etc.) + CLI subcommands:
    python -m scripts.plotting epsilon2        Fig 4: ε₂ evolution
    python -m scripts.plotting potential       Fig 1: Higgs potential
    python -m scripts.plotting usr-map          Fig 3: USR heatmap
    python -m scripts.plotting compare-configs  Multi-config D_ℓ/P_S comparison

All functions follow publication-ready conventions:
- Two-column format (~3.25-3.5in wide single, ~7in full)
- 300 DPI minimum
- Colorblind-friendly palette (Tol 2012)
- Font sizes set by PAPER_RCPARAMS (importable constant)
- Export PNG only
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline

from scripts.constants import As, k_pivot_phys, T_cmb, ROOT_DIR

TOL = {
    "blue": "#4477AA",
    "red": "#CC3311",
    "green": "#228833",
    "yellow": "#EE8866",
    "teal": "#44BB99",
    "purple": "#AA3377",
    "grey": "#666666",
    "dark": "#222222",
}
# Color convention across all plots:
#   TOL["red"]   — model/main data line
#   TOL["blue"]  — secondary/reference data (e.g. CAMB full)
#   TOL["dark"]  — Planck data points / LCDM baseline
#   TOL["grey"]  — reference lines, grid, annotations
#   TOL["green"] — feature markers (pivot, inflection)

COLORS = ["#CC3311", "#EE8866", "#44BB99", "#AA3377",
          "#4477AA", "#228833", "#DDCC77", "#88CCEE"]

PAPER_RCPARAMS = {
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "axes.grid": False,
}

MARKER_RCPARAMS = {
    "marker_size": 2.5,
    "marker_edge_width": 0.4,
    "line_width": 0.8,
    "reference_line_style": "--",
    "reference_line_width": 0.6,
}

# COLOR_MAP: standard role-based color assignment for comparison plots.
# Plot philosophy — data-first, minimal clutter:
#   No annotation boxes, reference lines, or decorations unless explicitly
#   requested. Let the data speak. Add elements only when they serve a
#   specific expository purpose.
COLOR_MAP = {
    "primary": TOL["red"],      # main model / SR approximation
    "secondary": TOL["blue"],   # reference / exact numerical (MS)
    "derived": TOL["purple"],   # residuals, derived quantities
    "reference": TOL["grey"],   # reference lines, zero lines
    "data": TOL["dark"],        # observational data (Planck)
    "feature": TOL["green"],    # pivot markers, inflection points
}

# ── Module-level workers (must be top-level for pickle) ─────────────────────

def _heatmap_worker(args):
    """Worker for usr-map multiprocessing."""
    idx, phi_val, y_val, xi, lam, bg_steps = args
    from scripts.usr_utils import run_bg_for_usr, measure_usr_duration
    N_arr, epsH_arr, eps2_arr = run_bg_for_usr(
        phi_val, y_val, xi=xi, lam=lam, bg_steps=bg_steps,
    )
    return idx, measure_usr_duration(N_arr, epsH_arr, eps2_arr, threshold=-5.5)

OUTPUT_DIRS = {
    "diagnostics": os.path.join(ROOT_DIR, "outputs/plots/diagnostics"),
    "powerloss": os.path.join(ROOT_DIR, "outputs/plots/powerloss"),
    "optimizer": os.path.join(ROOT_DIR, "outputs/plots/optimizer"),
    "paper": os.path.join(ROOT_DIR, "outputs/plots/paper"),
    "pspectra": os.path.join(ROOT_DIR, "outputs/simulations/pspectra"),
    "ps_plots": os.path.join(ROOT_DIR, "outputs/plots/pspectra"),
    "c_ell": os.path.join(ROOT_DIR, "outputs/simulations/c_ell"),
    "configs": os.path.join(ROOT_DIR, "outputs/simulations/configs"),
    "logs": os.path.join(ROOT_DIR, "outputs/simulations/logs"),
    "scans": os.path.join(ROOT_DIR, "outputs/simulations/scans"),
    "pbh": os.path.join(ROOT_DIR, "outputs/plots/pbh"),
}


def get_path(category, filename):
    """Return full path for an output file. Creates directory if needed."""
    out_dir = OUTPUT_DIRS.get(category)
    if out_dir is None:
        raise ValueError(
            f"Unknown output category: {category}. "
            f"Available: {list(OUTPUT_DIRS.keys())}"
        )
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


def save_fig(fig, filename, category="diagnostics", dpi=300):
    """Save a matplotlib figure to the correct output directory."""
    if not filename.endswith(".png"):
        filename += ".png"
    path = get_path(category, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def make_filename(name, phi0=None, y0=None, nstar=None, ext=".json"):
    """Generate standardized output filename.

    Pattern: {name}_phi{phi0:.2f}_y0{y0:+.3f}_nstar{nstar:.1f}{ext}
    If phi0 is None, returns {name}{ext} (for special files like camb_lcdm).

    Examples:
        make_filename("ps", 6.60, -0.736, 52.6)         → "ps_phi6.60_y0-0.736_nstar52.6.json"
        make_filename("camb", 6.60, -0.736, 52.6)        → "camb_phi6.60_y0-0.736_nstar52.6.json"
        make_filename("camb_lcdm")                        → "camb_lcdm.json"
        make_filename("planck", 6.60, -0.736, 52.6, ".png") → "planck_phi6.60_y0-0.736_nstar52.6.png"
    """
    if phi0 is not None:
        return f"{name}_phi{phi0:.2f}_y0{y0:+.3f}_nstar{nstar:.1f}{ext}"
    return f"{name}{ext}"


def find_ps(phi0, y0, nstar, tolerance=3.0):
    """Find cached P_S(k) JSON matching config params.

    Returns (path, metadata) or (None, None) if not found.
    Matches by phi0, y0, and N_star within tolerance.
    Tries new convention first, then legacy patterns.
    """
    import glob
    import json

    ps_dir = get_path("pspectra", "")

    # New convention
    pattern = os.path.join(ps_dir, f"ps_phi{phi0:.2f}_y0{y0:+.3f}_nstar*.json")
    matches = sorted(glob.glob(pattern))

    # Legacy patterns for backward compat during transition
    if not matches:
        for pat in [
            os.path.join(ps_dir, f"PS_Higgs*phi{phi0:.2f}_y0{y0:.3f}_*.json"),
            os.path.join(ps_dir, f"Higgs_Inflation*phi{phi0:.2f}_y0{y0:.3f}_*.json"),
        ]:
            matches = sorted(glob.glob(pat))
            if matches:
                break

    if not matches:
        return None, None

    # Score by N_star proximity
    scored = []
    for m in matches:
        try:
            with open(m) as f:
                rec = json.load(f)
            md = rec.get("metadata", {})
            ns = md.get("N_star", 0)
            if abs(ns - nstar) <= tolerance:
                scored.append((abs(ns - nstar), m, md))
        except Exception:
            continue

    if not scored:
        return None, None
    scored.sort(key=lambda x: x[0])
    return scored[0][1], scored[0][2]


# ── Ezquiaga CHI diagnostics & PS comparison ──────────────────────────────

S_CODE = 5e-5


def build_usr_weighted_kgrid(k_min, k_max, n_dense=200, n_outer=20):
    """Build k-grid with dense zone covering the PBH peak (k=1e9 to 1e13)."""
    k_lo = np.logspace(np.log10(k_min), 9, n_outer)
    k_dense = np.logspace(9, 13, n_dense)
    k_hi = np.logspace(13, np.log10(k_max), n_outer)
    return np.unique(np.concatenate([k_lo, k_dense, k_hi]))


def compute_ps_sr(bg_sol, end_idx):
    """Compute SR-approximation P_S(k) from background solution."""
    x, y, z, n = bg_sol
    epsH = y**2 / (2 * z**2)
    k_phys = np.exp(n[:end_idx+1]) * z[:end_idx+1] * S_CODE
    e = epsH[:end_idx+1]
    P_S = np.where(np.isfinite(e) & (e > 0),
                   (S_CODE * z[:end_idx+1])**2 / (8 * np.pi**2 * e),
                   np.nan)
    return k_phys, P_S, e


def plot_ezquiaga_diagnostics(model, bg_sol, derived, end_idx, chi0,
                               pivot_k=0.002, suffix="", N_star=None):
    """
    3-panel Ezquiaga diagnostics: N vs chi, V/V0 vs x, P_S vs N.

    Parameters
    ----------
    model : EzquiagaCHIModel instance
    bg_sol : tuple (x, y, z, n) from run_background_simulation
    derived : dict from get_derived_quantities
    end_idx : int, index where eps_H = 1
    chi0 : float, initial field value
    pivot_k : float, CMB pivot scale in Mpc^-1
    suffix : str, appended to filenames (e.g. "_chi8")
    N_star : float or None, e-folds before end for pivot exit
    """
    epsH = derived["epsH"]
    N = derived["N"]
    chi = bg_sol[0]
    z_bg = bg_sol[2]
    N_end_val = float(N[end_idx])
    N_std = N_end_val - N[:end_idx + 1]
    k_phys = np.exp(bg_sol[3][:end_idx + 1]) * z_bg[:end_idx + 1] * S_CODE
    pivot_idx = int(np.argmin(np.abs(np.log10(k_phys) - np.log10(pivot_k))))
    start_N = N_std[int(np.where(epsH[:end_idx + 1] > 1e-6)[0][0])] if np.any(epsH[:end_idx + 1] > 1e-6) else N_std[0]

    fig_n, ax_n = plt.subplots(figsize=(3.35, 2.6))
    ax_n.plot(chi[:end_idx + 1], N_std[:end_idx + 1], "-", color=TOL["red"], lw=1.3)
    ax_n.plot(chi[pivot_idx], N_std[pivot_idx], "o", color=TOL["green"], ms=4)
    ax_n.annotate("pivot", xy=(chi[pivot_idx], N_std[pivot_idx]),
                  fontsize=8, color=TOL["green"])
    ax_n.set_xlabel("chi", fontsize=14)
    ax_n.set_ylabel("N", fontsize=14)
    ax_n.set_xlim(0, chi0 + 0.5)
    ax_n.tick_params(labelsize=12)
    fig_n.tight_layout()
    fname_nchi = make_filename("ezquiaga_Nchi", model.x0, model.y0, N_star, ext="") if N_star is not None else f"ezquiaga_Nchi{suffix}"
    save_fig(fig_n, fname_nchi, "diagnostics")

    x_end = float(model._x_of_chi(float(chi[end_idx])))
    x_pivot = float(model._x_of_chi(float(chi[pivot_idx])))
    x_max = float(model._x_of_chi(float(chi[0])))
    x_plot_max = max(x_max * 1.15, 8.0)
    x_grid = np.linspace(0.05, x_plot_max, 3000)
    f_vals = np.array([model._V(float(x)) for x in x_grid])

    fig_v, ax_v = plt.subplots(figsize=(3.35, 2.6))
    ax_v.plot(x_grid, f_vals, "-", color=TOL["red"], lw=1.3)
    if np.isfinite(x_end):
        ax_v.axvline(x_end, color=TOL["grey"], ls="--", lw=0.8, alpha=0.5)
        ax_v.annotate("end", xy=(x_end, ax_v.get_ylim()[1]),
                      xytext=(5, 5), textcoords="offset points", fontsize=8, color=TOL["grey"])
    if np.isfinite(x_pivot):
        ax_v.axvline(x_pivot, color=TOL["green"], ls="-.", lw=0.8, alpha=0.5)
        ax_v.annotate("pivot", xy=(x_pivot, ax_v.get_ylim()[1]),
                      xytext=(5, 5), textcoords="offset points", fontsize=8, color=TOL["green"])
    ax_v.axvspan(0, x_max, alpha=0.08, color=TOL["red"])
    ax_v.set_xlim(0, x_plot_max)
    ax_v.set_xticks(np.linspace(0, x_plot_max, 7))
    ax_v.set_xlabel(r"$x = \phi/\mu$", fontsize=14)
    ax_v.set_ylabel(r"$V/V_0$", fontsize=14)
    ax_v.tick_params(labelsize=12)
    fig_v.tight_layout()
    fname_v = make_filename("ezquiaga_Vshape", model.x0, model.y0, N_star, ext="") if N_star is not None else f"ezquiaga_Vshape{suffix}"
    save_fig(fig_v, fname_v, "diagnostics")

    epsH_clip = np.clip(epsH[:end_idx + 1], 1e-30, None)
    P_S_sr = (S_CODE * z_bg[:end_idx + 1])**2 / (8 * np.pi**2 * epsH_clip)
    mask_ps = N_std <= start_N

    fig_ps, ax_ps = plt.subplots(figsize=(3.35, 2.6))
    ax_ps.semilogy(N_std[:end_idx + 1][mask_ps], P_S_sr[mask_ps], "-", color=TOL["red"], lw=1.3)
    ax_ps.axvline(N_std[pivot_idx], color=TOL["green"], ls="-.", lw=0.8, alpha=0.4)
    ax_ps.annotate("pivot", xy=(N_std[pivot_idx], ax_ps.get_ylim()[0]),
                   xytext=(5, 10), textcoords="offset points", fontsize=8, color=TOL["green"])
    ax_ps.set_xlabel("N", fontsize=14)
    ax_ps.set_ylabel(r"$P_{\mathcal{R}}$", fontsize=14)
    ax_ps.tick_params(labelsize=12)
    ax_ps.set_xlim(0, max(N_std[0] + 2, 70))
    fig_ps.tight_layout()
    fname_ps = make_filename("ezquiaga_PS_N", model.x0, model.y0, N_star, ext="") if N_star is not None else f"ezquiaga_PS_N{suffix}"
    save_fig(fig_ps, fname_ps, "diagnostics")


def plot_ps_sr_ms_comparison(
    model,
    chi0=None,
    y0=-1e-4,
    k_min=1e-8,
    k_max=1e18,
    n_dense=200,
    n_outer=20,
    n_workers=8,
    ms_enabled=True,
    k_pivots=None,
    filename=None,
    category="diagnostics",
    dpi=200,
):
    """
    Run background, compute SR + MS P_S(k), plot overlay with ratio panel.

    Parameters
    ----------
    model : EzquiagaCHIModel (already configured with inflection params)
    chi0 : float or None — sets model.x0 if given
    y0 : float — sets model.y0 (default -1e-4)
    k_min, k_max : float — k-range in Mpc^-1
    n_dense, n_outer : int — PBH-weighted k-grid params
    n_workers : int — parallel workers for MS solver
    ms_enabled : bool — if True, run MS solver
    k_pivots : list of (label, k_value) or None (defaults to [(k=0.002,), (k=0.05,)])
    filename : str or None — plot filename (default: "ps_sr_ms_chi{chi0}")
    category : str — output subdirectory
    dpi : int — figure resolution

    Returns
    -------
    dict with keys: k_phys_sr, P_S_sr, k_phys_ms, P_S_ms, n_s_local, N_total
    """
    from inf_dyn_background import run_background_simulation, get_derived_quantities

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

    k_sr, Ps_sr, _ = compute_ps_sr(bg_sol, end_idx)
    valid = np.where(np.isfinite(Ps_sr) & (Ps_sr > 0))[0]
    k_sr, Ps_sr = k_sr[valid], Ps_sr[valid]

    k_ms, Ps_ms = None, None
    n_s_results = {}
    ok_ms = np.array([], dtype=bool)
    k_ok = np.array([])

    if ms_enabled:
        k_grid = build_usr_weighted_kgrid(k_min, k_max, n_dense=n_dense, n_outer=n_outer)
        n_k = len(k_grid)
        print(f"  Running MS for {n_k} modes ({n_dense} dense + {n_outer*2} outer) on {n_workers} workers...")

        chunks = np.array_split(k_grid, n_workers)
        model_params = (model.a, model.b, model.v0, model.x0, model.y0)
        batch_args = [(ch, bg_sol, T, end_idx, model_params, 'dp5') for ch in chunks]

        Ps_ms_arr = np.full(n_k, np.nan)
        k_to_result = {kp: i for i, kp in enumerate(k_grid)}

        t0 = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_solve_ms_batch, ba) for ba in batch_args]
            done = 0
            for future in as_completed(futures):
                for kp, ps in future.result():
                    if kp in k_to_result and ps is not None:
                        Ps_ms_arr[k_to_result[kp]] = ps
                done += 1
                print(f"    batch {done}/{n_workers}", flush=True)

        elapsed = time.time() - t0
        n_ok = int(np.sum(np.isfinite(Ps_ms_arr)))
        print(f"  MS done: {n_ok}/{n_k} OK in {elapsed:.0f}s")
        k_ms, Ps_ms = k_grid, Ps_ms_arr
        ok_ms = np.isfinite(Ps_ms)
        k_ok = k_grid[ok_ms]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.loglog(k_sr, Ps_sr, "-", color=TOL["blue"], lw=1.5, label="SR approx")
    interp_ms_vals = None
    if ms_enabled and np.any(ok_ms):
        ps_ok = Ps_ms[ok_ms]
        k_line = np.logspace(np.log10(k_ok[0]), np.log10(k_ok[-1]), 500)
        if len(k_ok) >= 4:
            spl = CubicSpline(np.log(k_ok), np.log(np.maximum(ps_ok, 1e-300)),
                              bc_type='not-a-knot', extrapolate=False)
            ps_line = np.exp(spl(np.log(k_line)))
        else:
            ps_line = np.exp(np.interp(np.log(k_line), np.log(k_ok), np.log(np.maximum(ps_ok, 1e-300)),
                                       left=np.nan, right=np.nan))
        ax1.loglog(k_line, ps_line, "-", color=TOL["red"], lw=1.2,
                   alpha=0.7, label=f"MS solver ({len(ps_ok)}/{len(Ps_ms)})")
        ax1.loglog(k_ok, ps_ok, "o", color=TOL["red"], ms=2, alpha=0.4)
        interp_ms_vals = np.interp(k_sr, k_ok, ps_ok, left=np.nan, right=np.nan)

    for label, k_piv in (k_pivots or [("k=0.002", 0.002), ("k=0.05", 0.05)]):
        ax1.axvline(k_piv, color=TOL["grey"], ls=":", lw=0.8, alpha=0.5)
        ax1.annotate(label, xy=(k_piv, ax1.get_ylim()[0]), fontsize=8,
                     color=TOL["grey"], ha="center")

    ax1.legend(fontsize=11)
    ax1.set_ylabel(r"$P_{\mathcal{R}}(k)$", fontsize=14)
    ax1.tick_params(labelsize=12)
    ax1.grid(True, which="both", alpha=0.2)

    if interp_ms_vals is not None:
        ratio = interp_ms_vals / Ps_sr
        ax2.semilogx(k_sr, ratio, "-", color=TOL["blue"], lw=1.2)
        rmin, rmax = np.nanmin(ratio), np.nanmax(ratio)
        if not np.isfinite(rmin): rmin = 0.5
        if not np.isfinite(rmax): rmax = 1.5
        ax2.set_ylim(max(0, rmin - 0.2), min(5, rmax + 0.2))
    ax2.axhline(1.0, color=TOL["grey"], ls="--", lw=0.8)
    ax2.set_xlabel(r"$k$ [Mpc$^{-1}$]", fontsize=14)
    ax2.set_ylabel("MS / SR", fontsize=14)
    ax2.tick_params(labelsize=12)
    ax2.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    fname = filename or f"ps_sr_ms_chi{model.x0}"
    save_fig(fig, fname, category, dpi=dpi)

    for label, k_piv in (k_pivots or [("k=0.002", 0.002), ("k=0.05", 0.05)]):
        pi = int(np.argmin(np.abs(k_sr - k_piv)))
        if pi < 5 or pi > len(k_sr) - 6:
            continue
        d5 = 5
        lk = np.log(k_sr[pi-d5:pi+d5+1])
        lp_sr = np.log(Ps_sr[pi-d5:pi+d5+1])
        ns_sr = 1 + (lp_sr[-1] - lp_sr[0]) / (lk[-1] - lk[0])
        ns_ms_val = None
        if interp_ms_vals is not None and np.isfinite(interp_ms_vals[pi]):
            lp_ms = np.log(np.maximum(interp_ms_vals[pi-d5:pi+d5+1], 1e-300))
            ns_ms_val = 1 + (lp_ms[-1] - lp_ms[0]) / (lk[-1] - lk[0])
        msg = f"  {label}: n_s_local(SR)={ns_sr:.4f}"
        if ns_ms_val is not None:
            msg += f", n_s_local(MS)={ns_ms_val:.4f}"
        print(msg)
        n_s_results[label] = {"SR_local": ns_sr, "MS_local": ns_ms_val}

    return {
        "k_phys_sr": k_sr,
        "P_S_sr": Ps_sr,
        "k_phys_ms": k_ms,
        "P_S_ms": Ps_ms,
        "n_s_local": n_s_results,
        "N_total": N_total,
        "end_idx": end_idx,
    }


def _solve_ms_batch(args):
    """Worker for parallel MS solver — top-level for pickle."""
    k_phys_batch, bg_sol, T_span_bg, end_idx, model_params, ms_method = args
    from scipy.interpolate import CubicSpline
    import inf_dyn_MS_full as ms_solver
    from numba_ms_solver import numba_run_ms, build_numba_splines
    from models.ezquiaga_chi import EzquiagaCHIModel

    m = EzquiagaCHIModel()
    m.a, m.b, m.v0, m.x0, m.y0 = model_params
    m.patch_background_solver()

    interp = tuple(
        CubicSpline(T_span_bg, bg_sol[i], bc_type='not-a-knot', extrapolate=True)
        for i in range(4)
    )
    bg_coefs = build_numba_splines(bg_sol, T_span_bg, model=m)
    n_bg, z_bg = bg_sol[3], bg_sol[2]
    log_az = n_bg + np.log(np.maximum(z_bg, 1e-300))
    t_end = T_span_bg[end_idx]

    results = []
    for kp in k_phys_batch:
        k_code = kp / S_CODE
        si = int(np.argmin(np.abs(log_az[:end_idx] - np.log(k_code) + np.log(100.0))))
        si = max(si, 0)
        ni = bg_sol[3][si]
        T_ms = np.linspace(T_span_bg[si], t_end, 5000)
        try:
            sol = numba_run_ms(bg_sol, T_span_bg, T_ms, ni, k_code, m,
                               bg_coefs=bg_coefs, method=ms_method)
            d = ms_solver.get_ms_derived_quantities_with_bg(sol, interp, T_ms, m, k_code, ni)
            ps = float(d["P_S"][-1])
            if np.isfinite(ps) and ps > 0:
                results.append((kp, ps))
            else:
                results.append((kp, None))
        except Exception:
            results.append((kp, None))
    return results


def plot_ps(k_phys, P_S, label="Higgs USR", filename="ps", category="powerloss",
            show_lcdm=True, k_dip=None, k_min=None, k_max=None):
    """Plot primordial power spectrum with optional LCDM baseline.

    Parameters
    ----------
    k_phys : array
        Physical wavenumbers (Mpc^-1)
    P_S : array
        Primordial power spectrum values
    label : str
        Legend label
    filename : str
        Output filename (without extension)
    category : str
        Output directory category from OUTPUT_DIRS
    show_lcdm : bool
        Overlay LCDM power-law (ns=0.965)
    k_dip : float or None
        Mark a vertical line at this k value
    k_min, k_max : float or None
        Restrict plotted k-range (default: full data range)
    """
    mask = np.isfinite(P_S)
    if np.sum(mask) > 5:
        logk_interp = CubicSpline(np.log(k_phys[mask]), P_S[mask],
                                  bc_type='not-a-knot', extrapolate=True)
        k_dense = np.logspace(np.log10(k_phys[mask].min()), np.log10(k_phys[mask].max()), 1000)
        ps_dense = np.clip(logk_interp(np.log(k_dense)), 0, None)
    else:
        k_dense, ps_dense = k_phys, P_S

    k_min = k_min if k_min is not None else np.min(k_dense)
    k_max = k_max if k_max is not None else np.max(k_dense)
    in_range = (k_dense >= k_min) & (k_dense <= k_max)
    k_dense = k_dense[in_range]
    ps_dense = ps_dense[in_range]

    fig, ax = plt.subplots(figsize=(3.35, 2.6))

    ax.semilogx(k_dense, ps_dense, "-", color=TOL["red"], lw=1.3, label=label)

    if show_lcdm:
        ns_lcdm = 0.965
        ps_lcdm = As * (k_dense / k_pivot_phys) ** (ns_lcdm - 1.0)
        ax.semilogx(k_dense, ps_lcdm, "-", color=TOL["dark"], lw=1.1, alpha=0.6,
                  label=r"$\Lambda$CDM ($n_s=0.965$)")

    if k_dip is not None and k_dip > 0:
        ps_dip = float(np.interp(k_dip, k_phys, P_S))
        ax.axvline(k_dip, color=TOL["red"], ls=":", lw=1.5, alpha=0.5)
        ax.annotate(f"$k_{{dip}}={k_dip:.2e}$",
                    xy=(k_dip, ps_dip),
                    xytext=(0.55, 0.7), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=TOL["red"], lw=1.5),
                    fontsize=7, color=TOL["red"])

    ax.tick_params(labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.yaxis.get_offset_text().set_fontsize(10)
    ax.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$", fontsize=14)
    ax.set_ylabel(r"$\mathcal{P}_{\mathcal{R}}(k)$", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25, which="major")

    fig.tight_layout()
    save_fig(fig, filename, category)


def plot_dell(ells, D_ell_model, planck_ells=None, D_planck=None,
              D_err_lower=None, D_err_upper=None, D_ell_lcdm=None,
              ells_lcdm=None, model_label="Model", filename="dell",
              category="powerloss", ell_max=30):
    """Plot D_ell with Planck data and optional LCDM baseline."""
    fig, ax = plt.subplots(figsize=(3.7, 2.6))

    if planck_ells is not None and D_planck is not None:
        mask = planck_ells <= ell_max
        yerr = [D_err_lower[mask], D_err_upper[mask]] if D_err_upper is not None else None
        ax.errorbar(planck_ells[mask], D_planck[mask], yerr=yerr,
                    fmt="o", color=TOL["dark"], capsize=3, capthick=1,
                    markersize=4, elinewidth=1,
                    label=r"Planck 2018 low-$\ell$ TT")

    ell_dense = np.linspace(ells.min(), min(ells.max(), ell_max), 200)
    if len(ells) >= 4:
        D_interp = CubicSpline(ells, D_ell_model, bc_type='not-a-knot', extrapolate=True)(ell_dense)
    else:
        D_interp = np.interp(ell_dense, ells, D_ell_model)
    ax.plot(ell_dense, D_interp, "-", color=TOL["red"], lw=1.5, label=model_label)

    if ells_lcdm is not None and D_ell_lcdm is not None:
        mask = ells_lcdm <= ell_max
        ells_l_mask = ells_lcdm[mask]
        D_l_mask = D_ell_lcdm[mask]
        if len(ells_l_mask) >= 4:
            D_lcdm_interp = CubicSpline(ells_l_mask, D_l_mask, bc_type='not-a-knot', extrapolate=True)(ell_dense)
        else:
            D_lcdm_interp = np.interp(ell_dense, ells_l_mask, D_l_mask)
        ax.plot(ell_dense, D_lcdm_interp, "--", color=TOL["grey"], lw=1.2,
                alpha=0.6, label=r"$\Lambda$CDM")

    ax.set_xlabel(r"$\ell$", fontsize=14)
    ax.set_ylabel(r"$D_\ell^{TT}\ [\mu{\rm K}^2]$", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(1.5, ell_max + 0.5)

    fig.tight_layout()
    save_fig(fig, filename, category)


def plot_background(bg_sol, derived, filename="background", category="diagnostics"):
    """4-panel background trajectory dashboard."""
    x, y, z, n = bg_sol
    N = derived["N"]
    epsH = derived["epsH"]
    etaH = derived["etaH"]

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

    ax = axes[0, 0]
    ax.plot(N, x, color=TOL["blue"], lw=1.5)
    ax.set_xlabel(r"$N$ (e-folds)", fontsize=10)
    ax.set_ylabel(r"$\phi$ [$M_P$]", fontsize=10)
    ax.grid(True, alpha=0.25, which="both")

    ax = axes[0, 1]
    ax.semilogy(N, epsH, color=TOL["red"], lw=1.5)
    ax.axhline(1.0, color=TOL["grey"], ls="--", lw=1, alpha=0.5)
    ax.text(0.98, 0.95, r"$\epsilon_H = 1$", transform=ax.transAxes,
            color=TOL["grey"], ha="right", va="top", fontsize=9)
    ax.set_xlabel(r"$N$ (e-folds)", fontsize=10)
    ax.set_ylabel(r"$\epsilon_H$", fontsize=10)
    ax.grid(True, alpha=0.25, which="both")

    ax = axes[1, 0]
    ax.plot(N, etaH, color=TOL["green"], lw=1.5)
    ax.axhline(0.0, color=TOL["grey"], ls="--", lw=1, alpha=0.5)
    ax.set_xlabel(r"$N$ (e-folds)", fontsize=10)
    ax.set_ylabel(r"$\eta_H$", fontsize=10)
    ax.grid(True, alpha=0.25, which="both")

    ax = axes[1, 1]
    ax.plot(x, y, color=TOL["yellow"], lw=1.5)
    ax.set_xlabel(r"$\phi$ [$M_P$]", fontsize=10)
    ax.set_ylabel(r"$d\phi/dT$ [code units]", fontsize=10)
    ax.grid(True, alpha=0.25, which="both")

    fig.tight_layout()
    save_fig(fig, filename, category)


def plot_camb_comparison(camb_data, filename="camb_dell",
                         category="powerloss"):
    """CAMB D_ell vs LCDM + Planck at low ell."""
    ells = camb_data["ells"]
    D_camb = camb_data["D_camb"]
    D_pl = camb_data["D_pl"]
    p_ells = camb_data["planck_ells"]
    D_p = camb_data["D_planck"]
    D_err_lo = camb_data["D_err_lower"]
    D_err_hi = camb_data["D_err_upper"]

    low = ells <= 30
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.errorbar(p_ells, D_p, yerr=[D_err_lo, D_err_hi],
                fmt="o", color=TOL["dark"], capsize=3, capthick=1,
                markersize=4, elinewidth=1, label="Planck 2018", zorder=5)
    ax.plot(ells[low], D_camb[low], "-", color=TOL["blue"], lw=1.5,
            label="CAMB (full)", zorder=4)
    ax.plot(ells[low], D_pl[low], "--", color=TOL["grey"], lw=1.2,
            label=r"$\Lambda$CDM (CAMB)", zorder=2)

    ax.set_xlabel(r"$\ell$", fontsize=10)
    ax.set_ylabel(r"$D_\ell^{TT}\ [\mu{\rm K}^2]$", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(1.5, 31)

    fig.tight_layout()
    save_fig(fig, filename, category)


def plot_camb_fullsky(camb_data, filename="camb_fullsky", category="powerloss"):
    """
    Broken-axis full-sky D_ell plot with Planck 2018 data.

    Left panel: log x-scale (ell=2-30) for low-ell Commander data.
    Right panel: linear x-scale (ell=32-2500) for binned TT.
    """
    ells = camb_data["ells"]
    D_camb = camb_data["D_camb"]
    D_pl = camb_data["D_pl"]
    ells_lcdm = camb_data["ells_lcdm"]
    p_ells = camb_data["planck_ells"]
    D_p = camb_data["D_planck"]
    D_lo = camb_data["D_err_lower"]
    D_hi = camb_data["D_err_upper"]

    from scripts.chi2_analysis import load_planck_binned
    b_ells, b_D, b_lo, b_hi = load_planck_binned()

    fig = plt.figure(figsize=(7, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 4], wspace=0)
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1], sharey=ax_left)

    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.tick_params(left=False)

    ax_left.set_xscale("log")
    ax_left.set_xlim(1.8, 30)
    ax_left.set_xticks([2, 10, 30])
    ax_left.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_left.tick_params(axis="x", which="minor", bottom=False)

    ax_right.set_xlim(30, ells.max())
    ax_right.tick_params(labelleft=False, labelsize=8)

    ax_left.tick_params(labelsize=8)
    ax_left.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]", fontsize=9)
    ax_left.set_ylim(-100, 6500)

    for ax in [ax_left, ax_right]:
        ax.plot(ells, D_camb, "-", color=TOL["red"], lw=1.2,
                label="Higgs USR", zorder=4)
        ax.plot(ells_lcdm, D_pl, "--", color=TOL["dark"], lw=1.2,
                alpha=0.6, label=r"$\Lambda$CDM", zorder=3)

    ax_left.errorbar(p_ells, D_p, yerr=[D_lo, D_hi], fmt="o",
                     color=TOL["dark"], capsize=1.5, markersize=2,
                     elinewidth=0.4, label="Planck 2018", zorder=5)

    ax_right.errorbar(b_ells, b_D, yerr=[b_lo, b_hi], fmt="o",
                      color=TOL["dark"], capsize=1, markersize=1.5,
                      elinewidth=0.3, zorder=5)

    ax_left.set_xlabel(r"$\ell$", fontsize=9)
    ax_right.set_xlabel(r"$\ell$", fontsize=9)

    ax_left.grid(True, alpha=0.15, which="both")
    ax_right.grid(True, alpha=0.15, which="both")

    ax_right.axvline(x=30, color=TOL["grey"], ls="--", lw=1.5, zorder=0)
    ax_right.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    save_fig(fig, filename, category)


# ── CLI subcommands ──────────────────────────────────────────────────────────

def _cli_epsilon2(xi=15000.0, lam=0.13, phi0=5.43, y0=-0.07):
    """Fig 4: ε₂ evolution with USR and transition shading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.usr_utils import run_bg_for_usr, measure_usr_duration

    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 18,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12,
        "figure.dpi": 300, "font.family": "serif",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
    })

    T_max = max(100.0, xi / 5.0)
    N, epsH, eps2 = run_bg_for_usr(phi0, y0, xi=xi, lam=lam,
                                   T_max=T_max, bg_steps=10000)
    if np.any(epsH >= 1.0):
        end = int(np.argmax(epsH >= 1.0))
        cutoff = min(len(N), end + 20)
    else:
        cutoff = len(N)

    N_plot = N[:cutoff]
    eps2_plot = eps2[:cutoff]

    dur_usr = measure_usr_duration(N, epsH, eps2, threshold=-5.5)
    transition_mask = (eps2_plot > -5.5) & (eps2_plot < -1)
    dN = np.diff(N_plot, prepend=N_plot[0])
    dur_trans = float(np.sum(dN[transition_mask]))
    print(f"  Pure USR duration (ε₂ < -5.5):  {dur_usr:.4f} e-folds")
    print(f"  Transition duration (-5.5 < ε₂ < -1): {dur_trans:.4f} e-folds")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(N_plot, eps2_plot, linewidth=2, color=TOL["blue"],
            label=r"$\epsilon_2 \equiv d\ln\epsilon_1/dN$")
    ax.axhline(0, color="k", linewidth=0.8, alpha=0.3)
    ax.axhline(-6, color=TOL["red"], linestyle="--", linewidth=1.5, alpha=0.8)
    ax.fill_between(N_plot, eps2_plot, -6,
                    where=(eps2_plot < -1) & (eps2_plot > -8),
                    color="#DDCC33", alpha=0.3, label="Transition")
    ax.fill_between(N_plot, eps2_plot, -6,
                    where=(eps2_plot < -5.5) & (eps2_plot > -8),
                    color=TOL["red"], alpha=0.4, label="Pure USR")
    ax.text(0.1, -6.5, "USR", color=TOL["red"],
            fontweight="bold", alpha=0.8, fontsize=13)

    below_minus1 = eps2_plot < -1
    if np.any(below_minus1):
        last_below = int(np.where(below_minus1)[0][-1])
        x_max = float(N_plot[last_below]) + 0.5
    else:
        x_max = float(N_plot[-1])
    ax.set_xlabel("e-folds ($N$)")
    ax.set_ylabel(r"$\epsilon_2$")
    ax.set_xlim(0, x_max)
    ax.set_ylim(-8, 2)
    ax.legend(loc="upper right")

    save_fig(fig, "paper_fig4_eps2_evolution", "paper")


def _cli_potential(xi=15000.0, lam=0.13, phi0=5.7, nstar=60):
    """Fig 1: Higgs effective potential with CMB pivot marker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from models import HiggsModel
    from scripts.usr_utils import compute_xcmb

    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 18,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12,
        "figure.dpi": 300, "font.family": "serif",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
    })

    model = HiggsModel(lam=lam, xi=xi)
    v0 = model.v0
    x_cmb, N_total = compute_xcmb(x0=phi0, y0_sr=-1e-6, N_star=nstar,
                                  lam=lam, xi=xi)
    if x_cmb is None:
        x_cmb = 5.42
    print(f"  N_total = {N_total:.2f}, x_cmb = {x_cmb:.5f}")

    x_arr = np.linspace(0, 10, 500)
    V_arr = v0 * model.f(x_arr)

    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    ax.plot(x_arr, V_arr, color=TOL["red"], lw=1.3, label=r"Potential $U_E(x)$")
    V_cmb = v0 * model.f(x_cmb)
    ax.plot(x_cmb, V_cmb, "o", color=TOL["blue"], markersize=5,
            label=rf"CMB Pivot ($x_* = {x_cmb:.2f}$)")
    ax.axvline(x=x_cmb, linewidth=1.0, color=TOL["grey"], linestyle=":", alpha=0.6)
    ax.set_xlabel(r"Field $x$")
    ax.set_ylabel(r"$U(x)/M_P^4$")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, np.max(V_arr) * 1.05)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="lower right")
    save_fig(fig, "paper_fig1_higgs_potential", "paper")


def _cli_usr_map(phi_min=5.18, phi_max=5.93, n_phi=100,
                 y_min_abs=0.01, y_max_abs=0.20, n_y=100,
                 workers=12, xi=15000.0, lam=0.13,
                 bg_steps=30000, x_star=5.42):
    """Fig 3: USR duration heatmap over (φ₀, |y₀|) space."""
    import multiprocessing as mp
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 18,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12,
        "figure.dpi": 300, "font.family": "serif",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
    })

    phi_grid = np.linspace(phi_min, phi_max, n_phi)
    y_grid = np.linspace(-y_min_abs, -y_max_abs, n_y)
    n_total = n_phi * n_y

    print(f"  Grid: {n_phi}×{n_y} = {n_total} points, Workers: {workers}")
    t0 = time.time()

    tasks = [(i, phi_val, y_val, xi, lam, bg_steps)
             for i, (phi_val, y_val) in enumerate(
                 (p, y) for y in y_grid for p in phi_grid
             )]

    grid_dur = np.zeros((n_y, n_phi))
    done = 0
    with mp.Pool(workers) as pool:
        for idx, val in pool.imap_unordered(_heatmap_worker, tasks):
            j = idx % n_phi
            i = idx // n_phi
            grid_dur[i, j] = val
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{n_total} ({100*done/n_total:.0f}%)")

    print(f"  Done in {time.time()-t0:.0f}s")

    phi_edges = np.linspace(phi_min, phi_max, n_phi + 1)
    y_edges = np.linspace(y_min_abs, y_max_abs, n_y + 1)
    X_edges, Y_edges = np.meshgrid(phi_edges, y_edges)

    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    mesh = ax.pcolormesh(X_edges, Y_edges, grid_dur, shading="flat", cmap="magma")
    cbar = fig.colorbar(mesh, ax=ax, label="USR Duration [N]")
    cbar.ax.tick_params(labelsize=12)
    ax.axvline(x=x_star, color=TOL["blue"], linestyle="--", linewidth=1.0,
               label=rf"$x_* = {x_star}$")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$|y_0|$")
    save_fig(fig, "paper_fig3_usr_heatmap", "paper")


def _cli_compare_configs(phi0_str="", y0_str="", nstar_str="",
                         labels_str="", output_suffix="camb_top_configs",
                         from_log=None, n_configs=10,
                         xi=15000.0, lam=0.13, quick=False):
    """Multi-config P_S(k) and D_ℓ comparison plot."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.constants import As, k_pivot_phys, T_cmb
    from pspectrum_pipeline import run_pspectrum_pipeline
    from scripts.camb_wrapper import (
        compute_cl_full_camb, compute_cl_camb_powerlaw, compute_chi2_camb,
    )
    from scripts.planck_data import C_ell_to_d_ell, get_planck_data_asymmetric
    from models import HiggsModel

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 13,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 8, "figure.dpi": 150})

    def _load_or_run(cfg):
        path, _ = find_ps(cfg["phi0"], cfg["y0"], cfg["N_star"])
        if path is not None:
            with open(path) as f:
                rec = json.load(f)
            spec = rec["spectrum"]
            return {"k_phys": np.array(spec["k_phys"]),
                    "P_S": np.array(spec["P_S"])}
        print("  Running pipeline phi0={} y0={} N*={}".format(
            cfg["phi0"], cfg["y0"], cfg["N_star"]))
        model = HiggsModel(lam=lam, xi=xi)
        result = run_pspectrum_pipeline(
            model=model, phi0=cfg["phi0"], y0=cfg["y0"],
            k_min=1e-5, k_max=1.0, k_pivot_phys=k_pivot_phys,
            N_star=cfg["N_star"], normalize_to_As=True, As=As,
            num_k=80, n_workers=4, save_outputs=False,
        )
        if result["status"] != "success":
            print("  FAILED: {}".format(result.get("message", "")))
            return None
        return {"k_phys": result["k_phys"], "P_S": result["P_S"]}

    def _read_log_configs(path, n=10, phi0_list=None, y0_list=None, nstar_list=None):
        recs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        ok = [r for r in recs if r.get("status") == "ok" and "D_ell" in r]
        if not ok:
            print("WARNING: No entries with D_ell found in log.")
            return None
        if phi0_list is not None and y0_list is not None:
            selected = []
            for phi0, y0, ns in zip(phi0_list, y0_list, nstar_list or []):
                for r in ok:
                    if abs(r["phi0"] - phi0) < 0.01 and abs(r["y0"] - y0) < 0.001:
                        if nstar_list is None or abs(r.get("N_star", 0) - ns) < 0.5:
                            selected.append(r)
                            break
            if not selected:
                selected = sorted(ok, key=lambda r: r.get("chi2", 999))[:n]
        else:
            selected = sorted(ok, key=lambda r: r.get("chi2", 999))[:n]
        configs = []
        for r in selected:
            label = "phi{:.2f}_y0{:.3f}".format(r["phi0"], r["y0"])
            configs.append({
                "label": label, "phi0": r["phi0"], "y0": r["y0"],
                "N_star": r.get("N_star", 0), "chi2": r.get("chi2", 0),
                "D_ell": np.array(r["D_ell"]),
                "k_phys": np.array(r.get("k_phys", [])),
                "P_S": np.array(r.get("P_S", [])),
            })
        return configs

    def _build_configs():
        if from_log is not None:
            pl = [float(x) for x in phi0_str.split(",")] if phi0_str else None
            yl = [float(x) for x in y0_str.split(",")] if y0_str else None
            nl = [float(x) for x in nstar_str.split(",")] if nstar_str else None
            n = len(pl) if pl else n_configs
            return _read_log_configs(from_log, n=n,
                                     phi0_list=pl, y0_list=yl, nstar_list=nl)
        if phi0_str:
            phi0s = [float(x) for x in phi0_str.split(",")]
            y0s = [float(x) for x in y0_str.split(",")]
            nstars = [float(x) for x in nstar_str.split(",")]
            labels = ([x.strip() for x in labels_str.split(",")]
                      if labels_str else [])
            if not labels:
                labels = ["phi{:.2f}_y0{:.3f}_N*{:.1f}".format(p, y, n)
                          for p, y, n in zip(phi0s, y0s, nstars)]
            return [{"label": l, "phi0": p, "y0": y, "N_star": n}
                    for l, p, y, n in zip(labels, phi0s, y0s, nstars)]
        return [
            {"label": "best mild", "phi0": 6.70, "y0": -0.070, "N_star": 65.2},
            {"label": "mild+", "phi0": 6.30, "y0": -0.095, "N_star": 64.7},
            {"label": "low-ell dip", "phi0": 7.10, "y0": -0.170, "N_star": 62.7},
            {"label": "deep dip", "phi0": 6.55, "y0": -0.230, "N_star": 63.4},
            {"label": "golden ref", "phi0": 6.60, "y0": -0.736, "N_star": 52.6},
            {"label": "lowest D2", "phi0": 6.55, "y0": -0.340, "N_star": 63.2},
        ]

    configs = _build_configs()
    suffix = output_suffix

    print("LCDM baseline...")
    ells_l, C_l, _, _ = compute_cl_camb_powerlaw(ell_max=2500)
    D_lcdm = C_ell_to_d_ell(ells_l, C_l)
    p_ells, D_p, D_lo, D_hi = get_planck_data_asymmetric()

    results = []
    for cfg in configs:
        if cfg is None:
            continue
        if from_log is not None and "D_ell" in cfg:
            ells = np.arange(2, 2 + len(cfg["D_ell"]))
            results.append({**cfg, "ells": ells, "D": cfg["D_ell"],
                            "k_phys": cfg.get("k_phys", np.array([])),
                            "P_S": cfg.get("P_S", np.array([]))})
            print("  {}: chi2={}, D2={:.0f}".format(
                cfg["label"], cfg.get("chi2", "?"), cfg["D_ell"][0]))
        else:
            print("\n" + cfg["label"] + "...")
            ps = _load_or_run(cfg)
            if ps is None:
                continue
            ells, C_TT, _, _ = compute_cl_full_camb(ps, ell_max=2500)
            D = C_ell_to_d_ell(ells, C_TT)
            chi2, _, _ = compute_chi2_camb(ps, ell_max=29)
            results.append({**cfg, "ells": ells, "D": D,
                            "chi2": round(chi2, 2),
                            "k_phys": ps["k_phys"], "P_S": ps["P_S"]})

    if not results:
        print("No configs successfully processed. Exiting.")
        return

    print("\nGenerating plots...")

    # P_S(k) panel
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    k_plot = np.logspace(-5, 0, 500)
    ps_lcdm = As * (k_plot / k_pivot_phys) ** (0.965 - 1.0)
    ax.loglog(k_plot, ps_lcdm, "--", color=TOL["grey"], lw=1.5, alpha=0.6,
              label=r"$\Lambda$CDM")
    for i, r in enumerate(results):
        c = COLORS[i % len(COLORS)]
        lab = r"{} $\chi^2$={:.1f}".format(r["label"], r["chi2"])
        ax.loglog(r["k_phys"], r["P_S"], "-", color=c, lw=1.2, label=lab)
    ax.axvline(k_pivot_phys, color=TOL["grey"], ls=":", lw=0.8, alpha=0.4)
    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"$\mathcal{P}_{\mathcal{R}}(k)$")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(1e-5, 1)
    fig.tight_layout()
    save_fig(fig, f"ps_comparison_{suffix}", "diagnostics")

    # D_ell low-ell panel
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.errorbar(p_ells, D_p, yerr=[D_lo, D_hi], fmt="o", color=TOL["dark"],
                capsize=3, markersize=4, elinewidth=1, label="Planck 2018",
                zorder=5)
    low = ells_l <= 30
    ax.semilogy(ells_l[low], D_lcdm[low], "--", color=TOL["grey"], lw=1.5,
                alpha=0.6, label=r"$\Lambda$CDM", zorder=2)
    for i, r in enumerate(results):
        c = COLORS[i % len(COLORS)]
        lab = r"{} $\chi^2$={:.1f}".format(r["label"], r["chi2"])
        e = r["ells"]
        m = e <= 30
        ax.semilogy(e[m], r["D"][m], "-", color=c, lw=1.5, label=lab, zorder=3)
    ax.set_xlabel(r"$\ell$", fontsize=13)
    ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]", fontsize=13)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(1.5, 31)
    fig.tight_layout()
    save_fig(fig, f"dell_comparison_{suffix}", "diagnostics")


# ── PBH abundance plot ────────────────────────────────────────────────────


def plot_pbh_abundance(M, f_pbh, zeta_c=0.052, gamma=0.4,
                       model_label="PBH-CHI", filename="pbh_abundance",
                       category="pbh", use_old_bounds=False, smooth_bounds=False):
    """
    Plot Ω_PBH/Ω_DM vs M/M_⊙ (Figure 3 style).

    log-log, solid TOL line, paper-style legend.
    Observational constraint overlays included from bradkav/PBHbounds.
    """
    mask = np.isfinite(M) & np.isfinite(f_pbh) & (M > 0) & (f_pbh > 0)
    M, f_pbh = M[mask], f_pbh[mask]
    if len(M) < 3:
        print("  WARNING: too few valid points for PBH abundance plot")
        return

    with plt.rc_context(PAPER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        
        bounds_dir = os.path.join(ROOT_DIR, "data", "PBHbounds", "bounds")
        
        # Define a unique color for each constraint dataset to avoid repeats (Paul Tol colorblind-safe)
        colors_dict = {
            "EGB": "#EE7733",            # Orange
            "femto-lensing": "#EEDD88",  # Sand/Yellow
            "GRB-parallax": "#FFAABB",   # Rose
            "HSC": "#44BB99",            # Teal
            "Kepler": "#AA3377",         # Purple
            "MACHO": "#666666",          # Grey
            "EROS": "#4477AA",           # Blue
            "LIGO": "#33BBEE",           # Cyan
            "Eri-II": "#228833",         # Green
            "Planck": "#222222"          # Dark
        }
        
        if use_old_bounds:
            datasets = [
                # (label, filename, color, alpha, (x, y, rotation))
                ("EGB", "EGRB.txt", TOL["yellow"], 0.08, (2e-18, 2e-2, 65)),
                ("femto-lensing", "FL.txt", TOL["yellow"], 0.08, (3e-16, 2.5e-1, 0)),
                ("GRB-parallax", "GRB-parallax.txt", TOL["yellow"], 0.08, (6e-14, 7e-2, 0)),
                ("HSC", "old_bounds/HSC_original.txt", TOL["teal"], 0.08, (1e-10, 2e-2, 0)),
                ("Kepler", "K.txt", TOL["purple"], 0.08, (2e-8, 2e-1, 0)),
                ("MACHO", "M.txt", TOL["grey"], 0.08, (1e-5, 2e-1, 0)),
                ("EROS", "EROS.txt", TOL["blue"], 0.08, (1e-2, 5e-2, 0)),
                ("Eri-II", "UFdwarfs.txt", TOL["green"], 0.08, (3e3, 1.5e-1, -55)),
                ("Planck", "old_bounds/CMB_withoutDM.txt", TOL["dark"], 0.08, (1e3, 3e-3, -40))
            ]
        else:
            datasets = [
                # (label, filename, color, alpha, (x, y, rotation))
                ("EGB", "EGRB.txt", TOL["yellow"], 0.08, (2e-18, 2e-2, 65)),
                ("femto-lensing", "FL.txt", TOL["yellow"], 0.08, (3e-16, 2.5e-1, 0)),
                ("GRB-parallax", "GRB-parallax.txt", TOL["yellow"], 0.08, (6e-14, 7e-2, 0)),
                ("HSC", "HSC.txt", TOL["teal"], 0.08, (1e-10, 6e-3, 0)),
                ("Kepler", "K.txt", TOL["purple"], 0.08, (2e-8, 2e-1, 0)),
                ("MACHO", "M.txt", TOL["grey"], 0.08, (1e-5, 2e-1, 0)),
                ("EROS", "EROS.txt", TOL["blue"], 0.08, (1e-2, 5e-2, 0)),
                ("LIGO", "LIGO.txt", TOL["purple"], 0.08, (8.0, 8e-3, -55)),
                ("Eri-II", "UFdwarfs.txt", TOL["green"], 0.08, (1.5e3, 6e-2, -50)),
                ("Planck", "CMB.txt", TOL["dark"], 0.08, (4.5e1, 2e-2, -82))
            ]
        
        if os.path.exists(bounds_dir):
            for label, fname, color_default, alpha, pos in datasets:
                color = colors_dict.get(label, color_default)
                path = os.path.join(bounds_dir, fname)
                if os.path.exists(path):
                    try:
                        data = np.loadtxt(path, comments='#')
                        m_b, f_b = data[:, 0], data[:, 1]
                        
                        # Sort to ensure monotonic mass coordinates
                        sort_idx = np.argsort(m_b)
                        m_b, f_b = m_b[sort_idx], f_b[sort_idx]
                        
                        # Extend boundaries to prevent visual clipping/patchiness
                        if label == "EGB":
                            # Discard the first two points which are flat digitization artifacts along the bottom axis
                            m_b = m_b[2:]
                            f_b = f_b[2:]
                            if len(m_b) < 2:
                                continue
                            # Extrapolate to the left plot limit (1e-19) following the log-log slope of the first two points
                            log_m1, log_m2 = np.log10(m_b[0]), np.log10(m_b[1])
                            log_f1, log_f2 = np.log10(f_b[0]), np.log10(f_b[1])
                            slope = (log_f2 - log_f1) / (log_m2 - log_m1)
                            
                            log_m_left = np.log10(1e-19)
                            log_f_left = log_f1 + slope * (log_m_left - log_m1)
                            f_left = 10**log_f_left
                            
                            m_b = np.insert(m_b, 0, 1e-19)
                            f_b = np.insert(f_b, 0, f_left)
                        elif label == "Eri-II" and m_b[-1] < 1e8:
                            if len(m_b) < 2:
                                continue
                            # Extrapolate to the right plot limit (1e8) following the log-log slope of the last two points
                            log_m1, log_m2 = np.log10(m_b[-2]), np.log10(m_b[-1])
                            log_f1, log_f2 = np.log10(f_b[-2]), np.log10(f_b[-1])
                            slope = (log_f2 - log_f1) / (log_m2 - log_m1)
                            
                            log_m_right = np.log10(1e8)
                            log_f_right = log_f2 + slope * (log_m_right - log_m2)
                            f_right = 10**log_f_right
                            
                            m_b = np.append(m_b, 1e8)
                            f_b = np.append(f_b, f_right)

                        
                        # Apply shape-preserving interpolation if smoothing is requested
                        if smooth_bounds and len(m_b) > 3:
                            from scipy.interpolate import PchipInterpolator
                            log_m = np.log10(m_b)
                            log_f = np.log10(f_b)
                            log_m, unique_idx = np.unique(log_m, return_index=True)
                            log_f = log_f[unique_idx]
                            
                            if len(log_m) > 3:
                                interp = PchipInterpolator(log_m, log_f)
                                m_dense = np.logspace(log_m.min(), log_m.max(), 300)
                                f_dense = np.clip(10**interp(np.log10(m_dense)), 1e-30, 10.0)
                                m_b, f_b = m_dense, f_dense

                        ax.plot(m_b, f_b, color=color, lw=0.8)
                        ax.fill_between(m_b, f_b, 10.0, color=color, alpha=alpha, lw=0)
                        tx, ty, rot = pos
                        ax.text(tx, ty, label, color=color, fontsize=6, 
                                rotation=rot, ha='center', va='center', zorder=10)
                    except Exception as e:
                        print(f"  WARNING: Failed to load {fname}: {e}")
        else:
            print("  WARNING: PBHbounds data not found at", bounds_dir)

        # Plot our model's PBH abundance curve
        ax.loglog(M, f_pbh, "-.", color=TOL["red"], lw=1.5, label=model_label)
        
        ax.set_xlabel(r"$M / M_\odot$")
        ax.set_ylabel(r"$\Omega_{\mathrm{PBH}} / \Omega_{\mathrm{DM}}$")
        ax.set_xlim(1e-19, 1e8)
        ax.set_ylim(1e-4, 1.0)
        ax.legend(loc="lower right", fontsize=7)
        # Use a gray grid
        ax.grid(True, alpha=0.2, color='gray', which="both", ls='--')
        fig.tight_layout()
        save_fig(fig, filename, category)


# ── Inflection potential zoom plot (Ezquiaga CHI) ─────────────────────────


def _V_shape(x, a, b, c=0.77):
    x = np.asarray(x, dtype=float)
    lnx = np.log(np.maximum(x, 1e-30))
    num = (1 + a * lnx**2) * x**4
    den = (1 + c * (1 + b * lnx) * x**2)**2
    return num / den


def plot_inflection_zoom(parameter_sets, x_range=(0.7, 0.9),
                         layout="split", filename="inflection_zoom",
                         category="pbh"):
    """
    Two-panel zoom of V(x) near the inflection point, comparing paper vs exact.

    Parameters
    ----------
    parameter_sets : dict of label -> dict
        Each entry: {"a": float, "b": float, "c": float, "color": str, "ls": str}.
        Typically two entries: paper params and exact/self-consistent.
    x_range : (float, float)
        Linear x range for the zoom.
    layout : "split" or "separate"
        "split": single 2-panel figure (paper left, exact right).
        "separate": two individual figures with _paper, _exact suffixes.
    filename : str
        Base filename (no extension).
    category : str
        OUTPUT_DIRS key for save_fig.
    """
    x_grid = np.linspace(x_range[0], x_range[1], 2000)

    def _filter_params(params):
        return {k: params[k] for k in ("a", "b", "c") if k in params}

    labels = list(parameter_sets.keys())
    colors = [parameter_sets[l].get("color", TOL["red"]) for l in labels]
    lss = [parameter_sets[l].get("ls", "-") for l in labels]

    with plt.rc_context(PAPER_RCPARAMS):
        if layout == "split":
            fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
            for ax, label, color, ls in zip(axes, labels, colors, lss):
                fp = _filter_params(parameter_sets[label])
                ax.plot(x_grid, _V_shape(x_grid, **fp),
                        color=color, ls=ls, lw=1.2)
                ax.axvline(0.784, color=TOL["grey"], ls=":", lw=0.8, alpha=0.5)
                ax.set_xlabel(r"$x = \phi/\mu$")
                ax.set_ylabel(r"$\hat{V}(x)$")
                ax.set_xlim(*x_range)
                ax.grid(True, alpha=0.2)
            fig.tight_layout()
            save_fig(fig, filename, category)
            plt.close(fig)

        elif layout == "separate":
            for label in labels:
                fig, ax = plt.subplots(figsize=(3.5, 2.5))
                fp = _filter_params(parameter_sets[label])
                color = parameter_sets[label].get("color", TOL["red"])
                ls = parameter_sets[label].get("ls", "-")
                ax.plot(x_grid, _V_shape(x_grid, **fp),
                        color=color, ls=ls, lw=1.2)
                ax.axvline(0.784, color=TOL["grey"], ls=":", lw=0.8, alpha=0.5)
                ax.set_xlabel(r"$x = \phi/\mu$")
                ax.set_ylabel(r"$\hat{V}(x)$")
                ax.set_xlim(*x_range)
                ax.grid(True, alpha=0.2)
                fig.tight_layout()
                save_fig(fig, f"{filename}_{label.lower().replace(' ', '_')}",
                         category)
                plt.close(fig)


def plot_full_potential(a, b, c=0.77, x_range=(0.01, 10),
                        filename="full_potential", category="pbh",
                        color=None):
    """
    Single-curve linear plot of V(x) over a wide x range.

    Shows the overall potential shape from small x to plateau,
    with a vertical marker at x_c.
    """
    x_grid = np.logspace(np.log10(x_range[0]), np.log10(x_range[1]), 5000)

    with plt.rc_context(PAPER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.plot(x_grid, _V_shape(x_grid, a, b, c),
                color=color or TOL["blue"], lw=1.2)
        ax.axvline(0.784, color=TOL["grey"], ls=":", lw=0.8, alpha=0.7)
        ax.set_xlabel(r"$x = \phi/\mu$")
        ax.set_ylabel(r"$\hat{V}(x)$")
        ax.set_xlim(*x_range)
        fig.tight_layout()
        save_fig(fig, filename, category)
        plt.close(fig)


def main():
    """CLI entry point: python -m scripts.plotting <subcommand> [args]."""
    import argparse
    parser = argparse.ArgumentParser(
        description="CMB anomaly plotting utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("epsilon2", help="Fig 4: ε₂ evolution with USR shading")
    p.add_argument("--phi0", type=float, default=5.43)
    p.add_argument("--y0", type=float, default=-0.07)
    p.add_argument("--xi", type=float, default=15000.0)
    p.add_argument("--lam", type=float, default=0.13)

    p = sub.add_parser("potential", help="Fig 1: Higgs effective potential")
    p.add_argument("--phi0", type=float, default=5.7)
    p.add_argument("--nstar", type=float, default=60)
    p.add_argument("--xi", type=float, default=15000.0)
    p.add_argument("--lam", type=float, default=0.13)

    p = sub.add_parser("usr-map", help="Fig 3: USR duration heatmap")
    p.add_argument("--phi-min", type=float, default=5.18)
    p.add_argument("--phi-max", type=float, default=5.93)
    p.add_argument("--n-phi", type=int, default=100)
    p.add_argument("--y-min-abs", type=float, default=0.01)
    p.add_argument("--y-max-abs", type=float, default=0.20)
    p.add_argument("--n-y", type=int, default=100)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--xi", type=float, default=15000.0)
    p.add_argument("--lam", type=float, default=0.13)
    p.add_argument("--bg-steps", type=int, default=30000)
    p.add_argument("--x-star", type=float, default=5.42)

    p = sub.add_parser("compare-configs",
                       help="Multi-config D_ell/P_S comparison")
    p.add_argument("--phi0", type=str, default="",
                   help="Comma-separated phi0 values")
    p.add_argument("--y0", type=str, default="",
                   help="Comma-separated y0 values")
    p.add_argument("--nstar", type=str, default="",
                   help="Comma-separated N_star values")
    p.add_argument("--labels", type=str, default="",
                   help="Comma-separated labels")
    p.add_argument("--output-suffix", type=str, default="camb_top_configs")
    p.add_argument("--from-log", type=str, default=None,
                   help="JSONL scan log with D_ell data")
    p.add_argument("--n-configs", type=int, default=10)
    p.add_argument("--xi", type=float, default=15000.0)
    p.add_argument("--lam", type=float, default=0.13)
    p.add_argument("--quick", action="store_true")

    args = parser.parse_args()

    if args.command == "epsilon2":
        _cli_epsilon2(phi0=args.phi0, y0=args.y0, xi=args.xi, lam=args.lam)
    elif args.command == "potential":
        _cli_potential(phi0=args.phi0, nstar=args.nstar, xi=args.xi, lam=args.lam)
    elif args.command == "usr-map":
        _cli_usr_map(
            phi_min=args.phi_min, phi_max=args.phi_max, n_phi=args.n_phi,
            y_min_abs=args.y_min_abs, y_max_abs=args.y_max_abs, n_y=args.n_y,
            workers=args.workers, xi=args.xi, lam=args.lam,
            bg_steps=args.bg_steps, x_star=args.x_star,
        )
    elif args.command == "compare-configs":
        _cli_compare_configs(
            phi0_str=args.phi0, y0_str=args.y0, nstar_str=args.nstar,
            labels_str=args.labels, output_suffix=args.output_suffix,
            from_log=args.from_log, n_configs=args.n_configs,
            xi=args.xi, lam=args.lam, quick=args.quick,
        )


if __name__ == "__main__":
    main()
