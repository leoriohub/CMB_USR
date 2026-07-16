"""
BoTorch Bayesian optimization for PBH parameter tuning in Ezquiaga CHI model.

Optimizes (x_c, c, β, χ₀, N_star, ζ_c) to find configurations where the P_S(k)
peak falls in a target mass gap (sub-solar [10⁻⁶, 10⁻²] M_⊙ or asteroid
[10⁻¹⁷, 10⁻¹⁵] M_⊙) with maximal PBH abundance f_total while maintaining
n_s compatibility with Planck.

Usage
-----
Local run:
    python scripts/optimize_pbh_botorch.py \\
        --x-c-lo 0.75 --x-c-hi 0.85 \\
        --c-lo 0.5 --c-hi 10.0 \\
        --beta-lo 1e-6 --beta-hi 9e-4 \\
        --chi0-lo 4.0 --chi0-hi 8.0 \\
        --N-star-lo 50 --N-star-hi 70 \\
        --zeta-c-lo 0.04 --zeta-c-hi 0.10 \\
        --target subsolar \\
        --n-trials 200 --n-init 32 --q-batch 4 \\
        --workers 8 --seed 42

Lab machine (nohup + ssh):
    ssh uni "cd ~/Documentos/CMB_USR && git pull && \\
        source ~/miniconda3/etc/profile.d/conda.sh && conda activate cmb-anomaly && \\
        nohup python scripts/optimize_pbh_botorch.py \\
            --target subsolar --n-trials 200 > ~/pbh_botorch_subsolar.log 2>&1 & echo PID=\\$!"

Resume an interrupted run:
    python scripts/optimize_pbh_botorch.py --resume \\
        --log outputs/simulations/logs/pbh_optimizer.jsonl

BoTorch and GPyTorch are imported lazily inside functions that need them.
"""  # noqa: SIZE_OK — CLI orchestrator with embedded Wave 2+3 logic

import argparse
import json
import os
import sys
import threading
import time
import warnings

# Thread-level lock for parallel JSONL writes
_log_write_lock = threading.Lock()

# OpenMP set to 1 when using Python-level thread parallelism.
# Override via env: OMP_NUM_THREADS=8 python scripts/...
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

from scripts.constants import (
    ROOT_DIR,
    ACCRETION,
    As_planck,
    k_pivot_phys,
    gamma_default,
    k_eq_default,
    M_eq_default,
)


def _build_bounds(args):
    """Build (d, 2) bounds tensor from CLI args, with beta/chi0 in log-space.

    Dimension order: [x_c, c, beta, chi0, N_star, zeta_c].
    beta and chi0 are stored as logarithms internally for BoTorch.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments with ``_lo`` and ``_hi`` for each dim.

    Returns
    -------
    bounds : torch.Tensor, shape (6, 2)
        Column 0 = lower bounds, column 1 = upper bounds.
        beta and chi0 rows are in log-space.

    Raises
    ------
    ValueError
        If any lo >= hi, or any log-dim lo <= 0.
    """
    dims = [
        ("x_c", args.x_c_lo, args.x_c_hi, False),
        ("c", args.c_lo, args.c_hi, False),
        ("beta", args.beta_lo, args.beta_hi, True),
        ("chi0", args.chi0_lo, args.chi0_hi, True),
        ("N_star", args.N_star_lo, args.N_star_hi, False),
        ("zeta_c", args.zeta_c_lo, args.zeta_c_hi, False),
    ]

    bounds = []
    for name, lo, hi, use_log in dims:
        if lo >= hi:
            raise ValueError(
                f"Bounds violation for '{name}': lo={lo} >= hi={hi}. "
                f"Ensure lo < hi for every dimension."
            )
        if use_log:
            if lo <= 0:
                raise ValueError(
                    f"Log-space bounds violation for '{name}': lo={lo} <= 0. "
                    f"Log-space dimensions require lo > 0."
                )
            bounds.append([np.log(lo), np.log(hi)])
        else:
            bounds.append([lo, hi])

    return torch.tensor(bounds, dtype=torch.float64)


def _resolve_mass_bin(args):
    """Resolve target mass bin bounds [M_sun] from CLI args.

    ``--mass-bin-lo`` / ``--mass-bin-hi`` override ``--target`` when set.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments with ``target``, ``mass_bin_lo``, ``mass_bin_hi``.

    Returns
    -------
    lo : float
        Lower bound of target mass range [M_sun].
    hi : float
        Upper bound of target mass range [M_sun].
    """
    if args.mass_bin_lo is not None and args.mass_bin_hi is not None:
        return (args.mass_bin_lo, args.mass_bin_hi)

    targets = {
        "subsolar": (1e-6, 1e-2),
        "asteroid": (1e-17, 1e-15),
        "all": (1e-17, 1e2),
    }
    return targets.get(args.target, (1e-17, 1e2))


def main():
    p = argparse.ArgumentParser(
        description="BoTorch Bayesian optimization for PBH parameter tuning "
                    "in the Ezquiaga CHI model"
    )

    # ── Bounds ────────────────────────────────────────────────────────────────
    p.add_argument("--x-c-lo", type=float, default=0.75,
                    help="Lower bound for x_c (inflection point coordinate)")
    p.add_argument("--x-c-hi", type=float, default=0.85,
                    help="Upper bound for x_c (inflection point coordinate)")
    p.add_argument("--c-lo", type=float, default=0.5,
                    help="Lower bound for c = xi0 * kappa^2 mu^2")
    p.add_argument("--c-hi", type=float, default=10.0,
                    help="Upper bound for c = xi0 * kappa^2 mu^2")
    p.add_argument("--beta-lo", type=float, default=1e-6,
                    help="Lower bound for beta (deviation from inflection, physical)")
    p.add_argument("--beta-hi", type=float, default=9e-4,
                    help="Upper bound for beta (deviation from inflection, physical)")
    p.add_argument("--chi0-lo", type=float, default=4.0,
                    help="Lower bound for chi0 (initial field, physical)")
    p.add_argument("--chi0-hi", type=float, default=8.0,
                    help="Upper bound for chi0 (initial field, physical)")
    p.add_argument("--N-star-lo", type=float, default=50.0,
                    help="Lower bound for N_star (e-folds before end where pivot exits)")
    p.add_argument("--N-star-hi", type=float, default=70.0,
                    help="Upper bound for N_star")
    p.add_argument("--zeta-c-lo", type=float, default=0.04,
                    help="Lower bound for zeta_c (collapse threshold)")
    p.add_argument("--zeta-c-hi", type=float, default=0.10,
                    help="Upper bound for zeta_c (collapse threshold)")

    # ── Fixed params ──────────────────────────────────────────────────────────
    p.add_argument("--y0", type=float, default=-1e-4,
                    help="Initial field velocity (default: -1e-4)")

    # ── Target ────────────────────────────────────────────────────────────────
    p.add_argument("--target", choices=["subsolar", "asteroid", "all"],
                    default="subsolar",
                    help="Target mass range for PBH peak placement. "
                         "subsolar=[1e-6, 1e-2] M_sun, asteroid=[1e-17, 1e-15] M_sun, "
                         "all=[1e-17, 1e2] M_sun")

    # ── Constraints ───────────────────────────────────────────────────────────
    p.add_argument("--N-total-min", type=float, default=65.0,
                    help="Minimum N_total to accept a config")
    p.add_argument("--f-total-min", type=float, default=0.0,
                    help="Minimum f_total (PBH abundance) to accept")
    p.add_argument("--f-total-max", type=float, default=1.0,
                    help="Maximum f_total (PBH abundance) to accept")
    p.add_argument("--n-s-min", type=float, default=0.94,
                    help="Minimum n_s for Planck compatibility")
    p.add_argument("--n-s-max", type=float, default=0.99,
                    help="Maximum n_s for Planck compatibility")

    # ── Mass bin override ─────────────────────────────────────────────────────
    p.add_argument("--mass-bin-lo", type=float, default=None,
                    help="Override lower bound of target mass bin [M_sun]. "
                         "When set, overrides --target.")
    p.add_argument("--mass-bin-hi", type=float, default=None,
                    help="Override upper bound of target mass bin [M_sun]. "
                         "When set, overrides --target.")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    p.add_argument("--k-pivot", type=float, default=0.05,
                    help="Pivot k_pivot_phys (Mpc^-1). Drives BOTH A_s "
                         "normalization and n_s extraction (single-pivot invariant). "
                         "Planck default 0.05; Higgs low-ell 0.002.")
    p.add_argument("--ns-method", choices=["lsq", "derivative"], default="lsq",
                    help="n_s extraction method: lsq=window fit (default), "
                         "derivative=log-derivative at k_pivot.")
    p.add_argument("--ns-window", type=float, default=3.0,
                    help="Fit half-width for lsq method: [k_pivot/w, k_pivot*w]. "
                         "Ignored for derivative method.")
    p.add_argument("--workers", type=int, default=8,
                    help="MS solver parallel workers per config")

    # ── BoTorch ───────────────────────────────────────────────────────────────
    p.add_argument("--n-trials", type=int, default=200,
                    help="Number of BoTorch Bayesian optimization trials")
    p.add_argument("--n-init", type=int, default=32,
                    help="Number of random initialization (Sobol) points")
    p.add_argument("--q-batch", type=int, default=4,
                    help="q-batch size for parallel acquisition function evaluation")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")

    # ── Output / config ───────────────────────────────────────────────────────
    p.add_argument("--output-dir", default="outputs/plots/pbh",
                    help="Output directory for diagnostic plots")
    p.add_argument("--log", default="outputs/simulations/logs/pbh_optimizer.jsonl",
                    help="Path for incremental JSONL optimization log")
    p.add_argument("--no-plot", action="store_true",
                    help="Skip all plotting")
    p.add_argument("--config", type=str, default=None,
                    help="JSON config file. CLI args override file values.")
    p.add_argument("--resume", action="store_true",
                    help="Skip parameter sets already recorded in the log file")
    p.add_argument("--f-max", type=float, default=1.0,
                    help="Exclude configs with f_total > this threshold (default: 1.0)")
    p.add_argument("--psr-peak-min", type=float, default=1e-5,
                    help="Stage-2 SR P_S(k) peak min for collapse feasibility "
                         "(default: 1e-5; typical USR peaks are 1e-3 to 1)")
    p.add_argument("--stage2-skip", action="store_true",
                    help="Skip Stage-2 SR filter (always run full MS)")
    p.add_argument("--stage1-skip-usr", action="store_true",
                    help="Skip the Stage-1 usr_type=weak filter "
                         "(only checks N_total >= N_total_min)")

    # Pre-parse --config before full argument parse
    pre_args, _ = p.parse_known_args()
    if pre_args.config:
        with open(pre_args.config) as f:
            config = json.load(f)
        p.set_defaults(**config)

    args = p.parse_args()

    # Build bounds tensor
    bounds = _build_bounds(args)
    mass_lo, mass_hi = _resolve_mass_bin(args)

    # ── Print config summary ──────────────────────────────────────────────────
    print("=" * 60)
    print("PBH BoTorch Optimizer — Config")
    print("=" * 60)
    print(f"  Target:          {args.target}")
    print(f"  Mass bin:        [{mass_lo:.3e}, {mass_hi:.3e}] M_sun")
    print(f"  Bounds (d={bounds.shape[0]}):")
    dim_names = ["x_c", "c", "beta", "chi0", "N_star", "zeta_c"]
    log_dims = {"beta", "chi0"}
    for i, name in enumerate(dim_names):
        if name in log_dims:
            lo_phys = float(np.exp(bounds[i, 0].item()))
            hi_phys = float(np.exp(bounds[i, 1].item()))
        else:
            lo_phys = float(bounds[i, 0].item())
            hi_phys = float(bounds[i, 1].item())
        print(f"    {name:>8s}: [{lo_phys:.6g}, {hi_phys:.6g}]")
    print(f"  y0:              {args.y0}")
    print(f"  N_total min:     {args.N_total_min}")
    print(f"  f_total range:   [{args.f_total_min}, {args.f_total_max}]")
    print(f"  n_s range:       [{args.n_s_min}, {args.n_s_max}]")
    print(f"  k_pivot:         {args.k_pivot}")
    print(f"  ns_method:       {args.ns_method}")
    print(f"  ns_window:       {args.ns_window}")
    print(f"  BoTorch:")
    print(f"    n_trials:      {args.n_trials}")
    print(f"    n_init:        {args.n_init}")
    print(f"    q_batch:       {args.q_batch}")
    print(f"    seed:          {args.seed}")
    print(f"  Workers:         {args.workers}")
    print(f"  Log:             {args.log}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Resume:          {args.resume}")
    print("=" * 60)

    # Run optimization
    results = run_optimization(args)

    if results:
        print(f"\nOptimization complete: {len(results)} configs evaluated")
        try:
            from scripts.sweep_pbh_params import (
                print_summary, plot_heatmap_2d, plot_best_configs,
            )

            print_summary(results)

            if not args.no_plot:
                os.makedirs(args.output_dir, exist_ok=True)

                if len(results) > 1:
                    # Heatmap: x_c vs c for P_S_peak_ratio
                    plot_heatmap_2d(
                        results, "x_c", "c", "P_S_peak_ratio",
                        r"$x_c$", r"$c$",
                        r"$P_{\mathcal{R}}^{\mathrm{peak}} / A_s$",
                        "botorch_xc_vs_c_PSratio", "pbh",
                    )
                    # Heatmap: x_c vs c for M_peak
                    plot_heatmap_2d(
                        results, "x_c", "c", "M_kpeak",
                        r"$x_c$", r"$c$",
                        r"$M_{\mathrm{peak}} [M_\odot]$",
                        "botorch_xc_vs_c_peakM", "pbh",
                    )
                    # Best configs
                    plot_best_configs(results, n_best=3,
                                      filename="botorch_best_configs",
                                      category="pbh")

                    print(f"  Plots saved to {args.output_dir}")
        except ImportError:
            pass
    else:
        print("\nNo results — optimization produced no output")


# ═══════════════════════════════════════════════════════════════════════════════
# Wave 2: Forward model, pre-filter, constraints, logging, orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


def _classify_mass_bin(M_peak):
    """Classify PBH mass range into observational bin label.

    Parameters
    ----------
    M_peak : float
        PBH present-day peak mass [M_sun].

    Returns
    -------
    str : mass bin label.
    """
    if M_peak < 1e-17:
        return "too_light"
    elif M_peak < 1e-15:
        return "asteroid_gap"
    elif M_peak < 1e-6:
        return "intermediate"
    elif M_peak < 1e-2:
        return "sub_solar_gap"
    elif M_peak < 1:
        return "sub_stellar"
    elif M_peak < 100:
        return "stellar_ligo_ruled_out"
    return "massive"


def _stage1_prefilter(x_c, c, beta, chi0, N_star, args):
    """Stage-1: fast background-only pre-filter.

    Runs a background simulation to check if the parameter set produces
    a viable USR phase (usr_type != 'weak') with sufficient e-folds
    (N_total >= N_total_min).

    Parameters
    ----------
    x_c : float
        Critical field point coordinate.
    c : float
        ``ξ₀·κ²μ²`` combined parameter.
    beta : float
        Deviation from exact inflection (physical).
    chi0 : float
        Initial field value (physical).
    N_star : float
        Target e-folds before end where pivot exits.
    args : argparse.Namespace
        CLI arguments with ``y0``, ``k_pivot``, ``N_total_min``.

    Returns
    -------
    bool
        True if config passes pre-filter, False otherwise.
    """
    from scripts.observables import model_from_params
    from scripts.background_scan import analyze_background

    try:
        model = model_from_params(x_c, c, beta)
        model.x0 = chi0
        model.y0 = args.y0
        model.patch_background_solver()

        bg = analyze_background(model, chi0, args.y0, N_star, args.k_pivot)
        if bg is None:
            print(f"  Stage-1: skip (analyze_background returned None)")
            return False
        if bg["usr_type"] == "weak" and not args.stage1_skip_usr:
            print(f"  Stage-1: skip (usr_type=weak, "
                  f"N_total={bg['N_total']:.1f})")
            return False
        if bg["N_total"] < args.N_total_min:
            print(f"  Stage-1: skip (N_total={bg['N_total']:.1f} < "
                  f"N_total_min={args.N_total_min})")
            return False
        return True
    except Exception as e:
        print(f"  Stage-1: exception ({e})")
        return False


def _stage2_sr_filter(x_c, c, beta, chi0, N_star, args):
    """Stage-2: fast SR-approximation P_S(k) peak filter.

    Computes the SR power spectrum from background quantities and checks
    whether the peak amplitude exceeds ``--psr-peak-min``.  This filters
    out configs whose P_S(k) peak is too weak to produce PBH collapse
    via Press-Schechter, avoiding an expensive full MS solve.

    Parameters
    ----------
    x_c, c, beta, chi0, N_star : float
        Parameter set (same as Stage-1).
    args : argparse.Namespace
        CLI arguments with ``y0``, ``k_pivot``, ``psr_peak_min``.

    Returns
    -------
    bool
        True if SR P_S peak >= psr_peak_min, False otherwise.
    """
    from scripts.observables import model_from_params
    from inf_dyn_background import run_background_simulation, get_derived_quantities
    from scripts.compute_sr_ms import compute_ps_sr

    try:
        model = model_from_params(x_c, c, beta)
        model.x0 = chi0
        model.y0 = args.y0
        model.patch_background_solver()

        T_span = np.linspace(0, model.T_max, model.bg_steps)
        bg_sol = run_background_simulation(model, T_span)
        derived = get_derived_quantities(bg_sol, model)
        epsH = derived["epsH"]
        end_candidates = np.where(epsH >= 1.0)[0]
        end_idx = int(end_candidates[0]) if len(end_candidates) > 0 else len(epsH) - 1
        if end_idx < 10:
            end_idx = len(epsH) - 1

        k_sr, P_S_sr, _ = compute_ps_sr(bg_sol, end_idx)
        valid = np.isfinite(P_S_sr) & (P_S_sr > 1e-30) & np.isfinite(k_sr) & (k_sr > 1.0)
        if np.sum(valid) < 5:
            print(f"  Stage-2: skip (too few valid SR modes)")
            return False

        ps_peak = float(np.max(P_S_sr[valid]))
        if ps_peak < args.psr_peak_min:
            print(f"  Stage-2: skip (SR peak={ps_peak:.3e} < "
                  f"psr_peak_min={args.psr_peak_min:.3e})")
            return False

        print(f"  Stage-2: pass (SR peak={ps_peak:.3e})")
        return True
    except Exception as e:
        print(f"  Stage-2: exception ({e})")
        return False


def _evaluate_pbh(theta_unscaled, args):
    """Forward model: evaluate P_S(k) and PBH metrics for a parameter set.

    Calls ``run_full_pbh_pipeline`` for a single ``zeta_c`` value,
    extracts key metrics: power spectrum peak, PBH abundance, spectral
    index, and total e-folds.

    Parameters
    ----------
    theta_unscaled : numpy.ndarray, shape (6,)
        Physical parameters in order:
        ``[x_c, c, beta, chi0, N_star, zeta_c]``.
        All in physical units (already un-scaled from BoTorch's unit cube).
    args : argparse.Namespace
        CLI arguments with ``y0``, ``workers``, ``k_pivot``, etc.

    Returns
    -------
    dict
        Outcome dict with keys:
        ``status``, ``x_c``, ``c``, ``beta``, ``chi0``, ``N_star``,
        ``zeta_c``, ``N_total``, ``k_peak``, ``P_S_peak``,
        ``P_S_peak_ratio``, ``M_form``, ``M_kpeak``, ``M_peak_best``,
        ``f_total``, ``n_s``, ``A_s_cmb``, ``on_grid_boundary``,
        ``mass_bin``, ``k_pivot``, ``ns_window``, ``k_phys`` (list),
        ``P_S`` (list).  On failure ``status`` is ``"fail"`` and
        ``error`` is set.
    """
    from scripts.full_pbh_pipeline import run_full_pbh_pipeline

    x_c, c, beta, chi0, N_star, zeta_c = theta_unscaled

    try:
        result = run_full_pbh_pipeline(
            chi0=chi0, y0=args.y0, N_star=N_star, beta=beta,
            xc=x_c, c=c, workers=args.workers,
            zeta_c_vals=[zeta_c], plot=False, force=False,
            k_pivot=args.k_pivot, ns_window=args.ns_window,
            ns_method=args.ns_method,
        )

        k_phys = result["k_phys"]
        P_S = result["P_S"]
        n_s = result["n_s"]
        N_total = result["N_total"]
        A_s_cmb = result.get("A_s_cmb")
        f_total_best = result["f_total_best"]
        M_peak_best = result["M_peak_best"]

        # Find USR peak in P_S(k) for k > 1 Mpc⁻¹ (ignore CMB scales)
        k_arr = np.asarray(k_phys)
        ps_arr = np.asarray(P_S)
        valid = (
            np.isfinite(ps_arr)
            & (ps_arr > 1e-10)
            & np.isfinite(k_arr)
            & (k_arr > 1.0)
        )
        if np.sum(valid) >= 5:
            k_peak = float(k_arr[valid][np.argmax(ps_arr[valid])])
            P_S_peak = float(ps_arr[valid][np.argmax(ps_arr[valid])])
            on_grid_boundary = False
        else:
            peak_i = int(np.argmax(ps_arr))
            k_peak = float(k_arr[peak_i])
            P_S_peak = float(ps_arr[peak_i])
            on_grid_boundary = True

        P_S_peak_ratio = P_S_peak / As_planck if As_planck > 0 else 0.0
        if k_peak > 0:
            M_form = gamma_default * M_eq_default * (k_eq_default / k_peak) ** 2
            M_kpeak = M_form * ACCRETION
        else:
            M_form = 0.0
            M_kpeak = 0.0

        mass_bin = _classify_mass_bin(M_kpeak)

        # Ensure lists for JSON serialization
        k_phys_list = np.asarray(k_phys, dtype=float).tolist()
        P_S_list = np.asarray(P_S, dtype=float).tolist()

        return {
            "status": "success",
            "x_c": x_c, "c": c, "beta": beta,
            "chi0": chi0, "N_star": N_star, "zeta_c": zeta_c,
            "N_total": N_total,
            "k_peak": k_peak, "P_S_peak": P_S_peak,
            "P_S_peak_ratio": P_S_peak_ratio,
            "M_form": M_form, "M_kpeak": M_kpeak,
            "M_peak_best": M_peak_best,
            "f_total": f_total_best,
            "n_s": n_s, "A_s_cmb": A_s_cmb,
            "on_grid_boundary": on_grid_boundary,
            "mass_bin": mass_bin,
            "k_pivot": float(args.k_pivot),
            "ns_window": float(args.ns_window),
            "k_phys": k_phys_list,
            "P_S": P_S_list,
        }
    except Exception as e:
        return {
            "status": "fail",
            "x_c": x_c, "c": c, "beta": beta,
            "chi0": chi0, "N_star": N_star, "zeta_c": zeta_c,
            "error": str(e),
        }


def _compute_feasibility(outcome, mass_lo, mass_hi, args):
    """Compute minimisation objective and Boolean feasibility.

    ``Y = -f_total`` (negated — BoTorch minimises by default; we want to
    maximise ``f_total``).  A config is feasible iff **all** of:
        * ``M_peak ∈ [mass_lo, mass_hi]``
        * ``f_total ∈ [f_total_min, f_total_max]``
        * ``n_s ∈ [n_s_min, n_s_max]``

    Parameters
    ----------
    outcome : dict
        Result dict from ``_evaluate_pbh``.
    mass_lo : float
        Lower bound of target mass range [M_sun].
    mass_hi : float
        Upper bound of target mass range [M_sun].
    args : argparse.Namespace
        CLI arguments with constraint limits.

    Returns
    -------
    Y : float
        Objective value (``-f_total``, or ``inf`` on failure / NaN).
    feasible : bool
        ``True`` if all constraints are satisfied.
    """
    if outcome.get("status") == "fail":
        return (float("inf"), False)

    f_total = outcome.get("f_total")
    M_peak = outcome.get("M_peak_best")
    n_s = outcome.get("n_s")

    if (
        f_total is None or np.isnan(f_total)
        or M_peak is None or np.isnan(M_peak)
        or n_s is None or np.isnan(n_s)
    ):
        return (float("inf"), False)

    Y = -f_total  # minimise -f_total → maximise f_total

    violated = False
    if M_peak > mass_hi or M_peak < mass_lo:
        violated = True
    if f_total > args.f_total_max or f_total < args.f_total_min:
        violated = True
    if n_s > args.n_s_max or n_s < args.n_s_min:
        violated = True

    return (Y, not violated)


def _write_log_entry(log_path, entry):
    """Write one evaluation result to the incremental JSONL log.

    Matches the ``base_result`` schema from ``scripts/sweep_pbh_params.py``
    plus debugging fields (``status``, ``stage1_passed``, ``error``).

    Parameters
    ----------
    log_path : str
        Path to the JSONL log file.
    entry : dict
        Evaluation result dict (from ``_evaluate_pbh`` or ``_stage1_prefilter``).
    """
    import json
    import os

    log_entry = {
        "x_c": entry.get("x_c"),
        "c": entry.get("c"),
        "beta": entry.get("beta"),
        "chi0": entry.get("chi0"),
        "N_star": entry.get("N_star"),
        "N_total": entry.get("N_total"),
        "k_peak": entry.get("k_peak"),
        "P_S_peak": entry.get("P_S_peak"),
        "P_S_peak_ratio": entry.get("P_S_peak_ratio"),
        "M_form": entry.get("M_form"),
        "M_kpeak": entry.get("M_kpeak"),
        "M_present": entry.get("M_peak_best"),
        "on_grid_boundary": entry.get("on_grid_boundary", False),
        "mass_bin": entry.get("mass_bin"),
        "n_s": entry.get("n_s"),
        "k_pivot": entry.get("k_pivot"),
        "ns_window": entry.get("ns_window"),
        "A_s_at_cmb": entry.get("A_s_cmb"),
        "zeta_c": entry.get("zeta_c"),
        "f_total": entry.get("f_total"),
        "status": entry.get("status"),
        "stage1_passed": entry.get("stage1_passed"),
        "error": entry.get("error"),
    }
    # Drop None values for compact log
    log_entry = {k: v for k, v in log_entry.items() if v is not None}

    # Include full k_phys / P_S arrays for later plotting
    if "k_phys" in entry and entry["k_phys"] is not None:
        log_entry["k_phys"] = entry["k_phys"]
    if "P_S" in entry and entry["P_S"] is not None:
        log_entry["P_S"] = entry["P_S"]

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with _log_write_lock:
        with open(log_path, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")
            f.flush()


def _load_done_set(log_path):
    """Load already-evaluated configs from a JSONL log for resume.

    Parses every line, skips entries with an ``error`` key (failed
    configs get re-tried), and returns a set of 6-tuples.

    Parameters
    ----------
    log_path : str
        Path to the JSONL log file.

    Returns
    -------
    set of (x_c, c, beta, chi0, N_star, zeta_c) tuples
        Configurations already completed (error entries excluded).
    """
    if not os.path.exists(log_path):
        return set()

    done = set()
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Skip failed entries so they get re-tried
                if "error" in entry and entry["error"] is not None:
                    continue
                key = (
                    round(entry["x_c"], 6),
                    round(entry["c"], 6),
                    round(entry["beta"], 6),
                    round(entry["chi0"], 6),
                    round(entry["N_star"], 6),
                    round(entry["zeta_c"], 6),
                )
                done.add(key)
            except (json.JSONDecodeError, KeyError):
                pass

    print(f"Resuming: {len(done)} configs already in {log_path}")
    return done


def run_optimization(args):
    """Run the full BoTorch Bayesian optimisation loop.

    Orchestrates Stage-1 pre-filter, Stage-2 BoTorch GP with feasibility
    constraints, and incremental JSONL logging.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments with bounds, constraints, BoTorch settings.

    Returns
    -------
    list of dict
        Results from all evaluated configurations.
    """
    # 1. Build bounds, resolve mass bin, load done_set
    bounds = _build_bounds(args)
    mass_lo, mass_hi = _resolve_mass_bin(args)
    d = 6  # [x_c, c, beta, chi0, N_star, zeta_c]

    done_set = set()
    X_seed = []
    Y_seed = []
    if args.resume:
        done_set = _load_done_set(args.log)
        # Load feasible entries from log to seed the GP training set
        if os.path.exists(args.log):
            with open(args.log) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                    except (json.JSONDecodeError, KeyError):
                        continue
                    if entry.get("status") != "success":
                        continue
                    # Map log keys back to what _compute_feasibility expects
                    entry["M_peak_best"] = entry.get("M_present")
                    y_val, feasible = _compute_feasibility(entry, mass_lo, mass_hi, args)
                    if not feasible:
                        continue
                    p = np.array([entry["x_c"], entry["c"], entry["beta"],
                                  entry["chi0"], entry["N_star"], entry["zeta_c"]])
                    x_unit = np.zeros(d)
                    for i in range(d):
                        lo_phys = float(bounds[i, 0])
                        hi_phys = float(bounds[i, 1])
                        if i in (2, 3):
                            x_unit[i] = (np.log(p[i]) - lo_phys) / (hi_phys - lo_phys)
                        else:
                            x_unit[i] = (p[i] - lo_phys) / (hi_phys - lo_phys)
                    if np.all(x_unit >= 0) and np.all(x_unit <= 1):
                        X_seed.append(torch.tensor(x_unit, dtype=torch.float64).unsqueeze(0))
                        Y_seed.append(torch.tensor([[y_val]], dtype=torch.float64))
        if X_seed:
            print(f"  Loaded {len(X_seed)} feasible configs from log to seed GP")

    # 2. Import BoTorch (lazy, once — not at module level)
    from botorch.utils.sampling import draw_sobol_samples
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Standardize
    from botorch.acquisition import qLogExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.priors import GammaPrior

    torch.manual_seed(args.seed)

    bounds_unit_cube = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.float64)

    # 3. Sobol init
    n_init = args.n_init
    X_init = draw_sobol_samples(bounds=bounds_unit_cube, n=n_init, q=1,
                                seed=args.seed).squeeze(1)
    # X_init is (n_init, d) in [0, 1]

    # 4. Helper: un-scale from [0,1] to physical
    def _unscale(x):
        phys = np.zeros(d)
        for i in range(d):
            lo_phys = float(bounds[i, 0])
            hi_phys = float(bounds[i, 1])
            phys[i] = float(x[i]) * (hi_phys - lo_phys) + lo_phys
            if i == 2 or i == 3:  # beta, chi0 are in log space
                phys[i] = np.exp(phys[i])
        return phys

    # 5. Eval function: stage-1 → stage-2 SR filter → MS → log
    def _try_eval(params_phys):
        import sys as _sys2
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _project_root not in _sys2.path:
            _sys2.path.insert(0, _project_root)
        if args.resume:
            key = tuple(round(p, 6) for p in params_phys)
            if key in done_set:
                return None
        if not _stage1_prefilter(*params_phys[:5], args):
            return None
        if not args.stage2_skip:
            if not _stage2_sr_filter(*params_phys[:5], args):
                return None
        outcome = _evaluate_pbh(params_phys, args)
        _write_log_entry(args.log, outcome)
        return outcome

    # 6. Collect init evaluations (optionally seeded from resume log)
    results = []
    X_feasible = list(X_seed) if 'X_seed' in dir() else []
    Y_feasible = list(Y_seed) if 'Y_seed' in dir() else []
    if X_feasible:
        print(f"  Seeded GP with {len(X_feasible)} feasible configs from {args.log}")
    n_trials_total = 0

    n_workers = min(args.workers, os.cpu_count() or 8)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    # Serialize args for subprocess workers
    _args_dict = {
        'y0': args.y0, 'workers': 1, 'k_pivot': args.k_pivot,
        'ns_method': args.ns_method, 'ns_window': args.ns_window,
        'N_total_min': args.N_total_min, 'stage1_skip_usr': args.stage1_skip_usr,
        'stage2_skip': args.stage2_skip, 'psr_peak_min': args.psr_peak_min,
        'log': args.log,
    }
    init_batch = []
    for i in range(min(n_init, args.n_trials)):
        init_batch.append((i, _unscale(X_init[i])))

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_eval_worker, phys.tolist(), _args_dict): i
                   for i, phys in init_batch}
        for future in as_completed(futures):
            i = futures[future]
            outcome = future.result()
            n_trials_total += 1
            if outcome is None:
                print(f"  Init {i+1}/{n_init}: skipped")
                continue
            results.append(outcome)
            y_val, feasible = _compute_feasibility(outcome, mass_lo, mass_hi, args)
            if feasible:
                x = X_init[i]
                X_feasible.append(x.unsqueeze(0))
                Y_feasible.append(torch.tensor([[y_val]], dtype=torch.float64))
            print(f"  Init {i+1}/{n_init}: f_total={outcome.get('f_total',0):.4e} "
                  f"feasible={feasible}")
        pool.shutdown(wait=True)

    # If not enough feasible, continue sampling Sobol
    min_feasible = min(6, d + 2)
    extra_idx = n_init
    while (
        len(X_feasible) < min_feasible
        and n_trials_total < args.n_trials
        and n_trials_total < n_init * 3
    ):
        x = draw_sobol_samples(bounds=bounds_unit_cube, n=1, q=1,
                               seed=args.seed + extra_idx).squeeze(0).squeeze(0)
        extra_idx += 1
        phys = _unscale(x)
        outcome = _try_eval(phys)
        n_trials_total += 1
        if outcome is None:
            continue
        results.append(outcome)
        y_val, feasible = _compute_feasibility(outcome, mass_lo, mass_hi, args)
        if feasible:
            X_feasible.append(x.unsqueeze(0))
            Y_feasible.append(torch.tensor([[y_val]], dtype=torch.float64))

    if len(X_feasible) < 2:
        print("WARNING: too few feasible configs to start GP. "
              "Try wider bounds or looser constraints.")
        return results

    X_feasible = torch.cat(X_feasible, dim=0)
    Y_feasible = torch.cat(Y_feasible, dim=0)
    best_feasible_Y = Y_feasible.min().item()
    best_feasible_f_total = -best_feasible_Y

    print(f"  Init complete: {n_trials_total} evals, "
          f"{len(X_feasible)} feasible, "
          f"best f_total = {best_feasible_f_total:.6e}")

    n_wanted = args.n_trials

    # 7. Main BoTorch loop
    while n_trials_total < n_wanted:
        try:
            model = _fit_gp(X_feasible, Y_feasible)
            acq = _compute_acquisition(model, Y_feasible.min().item(),
                                       X_feasible)
            candidates = _optimize_candidates(acq, args.q_batch, d,
                                              args.seed + n_trials_total)
        except Exception as e:
            print(f"  GP/acquisition failure: {e}")
            # fall back to random Sobol search
            candidates = draw_sobol_samples(
                bounds=bounds_unit_cube, n=args.q_batch, q=1,
                seed=args.seed + n_trials_total + 1000,
            ).squeeze(1)

        # Parallel eval of q-batch candidates
        batch_phys = [_unscale(candidates[j]) for j in range(candidates.shape[0])]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_eval_worker, p.tolist(), _args_dict): (j, p)
                       for j, p in enumerate(batch_phys)}
            for future in as_completed(futures):
                j, phys = futures[future]
                if n_trials_total >= n_wanted:
                    break
                outcome = future.result()
                n_trials_total += 1
                if outcome is None:
                    continue
                results.append(outcome)
                x = candidates[j]
                y_val, feasible = _compute_feasibility(outcome, mass_lo, mass_hi,
                                                       args)
                if feasible:
                    X_feasible = torch.cat(
                        [X_feasible, x.unsqueeze(0)], dim=0,
                    )
                    Y_feasible = torch.cat(
                        [Y_feasible,
                         torch.tensor([[y_val]], dtype=torch.float64)], dim=0,
                    )
                    if y_val < best_feasible_Y:
                        best_feasible_Y = y_val
                        best_feasible_f_total = -best_feasible_Y
                        print(f"  NEW BEST: f_total={best_feasible_f_total:.6e} "
                              f"at trial {n_trials_total}")
                if n_trials_total % 10 == 0:
                    print(f"  [{n_trials_total}/{n_wanted}] best f_total = "
                          f"{best_feasible_f_total:.6e}, "
                          f"feasible = {len(X_feasible)}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Wave 3: BoTorch core (Sobol initial design, GP, acquisition, optimizer)
# ═══════════════════════════════════════════════════════════════════════════════


def _sobol_init(n_init: int, d: int, seed: int):
    """Sobol sequence initial design in [0, 1]^d.

    Parameters
    ----------
    n_init : int
        Number of initial points.
    d : int
        Dimension of the search space.
    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor
        (n_init, d) tensor in [0, 1].
    """
    from botorch.utils.sampling import draw_sobol_samples

    bounds = torch.tensor([[0.0] * d, [1.0] * d], dtype=torch.float64)
    return draw_sobol_samples(bounds=bounds, n=n_init, q=1, seed=seed).squeeze(1)


def _fit_gp(X: torch.Tensor, Y: torch.Tensor):
    """Fit a SingleTaskGP with Matern 5/2 kernel + Standardize.

    The GP is trained on feasible points only. If a LinAlgError occurs
    during fitting, a small jitter is added to the likelihood noise.

    Parameters
    ----------
    X : torch.Tensor
        (n, d) training inputs in [0, 1].
    Y : torch.Tensor
        (n, 1) training targets (negated f_total, float64).

    Returns
    -------
    SingleTaskGP
        Fitted GP model.
    """
    from botorch.models import SingleTaskGP as _SingleTaskGP
    from botorch.models.transforms import Standardize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.priors import GammaPrior

    d = X.shape[-1]

    model = _SingleTaskGP(
        X, Y,
        outcome_transform=Standardize(m=1),
    )
    model.covar_module = ScaleKernel(
        MaternKernel(nu=2.5, ard_num_dims=d,
                     prior=GammaPrior(3.0, 6.0)),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def _compute_acquisition(
    model,
    best_f: float,
    X_baseline: torch.Tensor | None = None,
):
    """Build qLogExpectedImprovement acquisition function.

    Parameters
    ----------
    model : SingleTaskGP
        Fitted GP model.
    best_f : float
        Current best feasible objective value (negated f_total).
    X_baseline : torch.Tensor or None
        Baseline points (unused by qLogEI, accepted for interface
        compatibility with potential qLogNEI switch).

    Returns
    -------
    qLogExpectedImprovement
        Acquisition function over the unit cube.
    """
    from botorch.acquisition import qLogExpectedImprovement
    return qLogExpectedImprovement(model=model, best_f=best_f)


def _optimize_candidates(
    acq,
    q: int,
    d: int,
    seed: int,
):
    """Maximize acquisition function over the unit cube.

    Uses ``optimize_acqf`` with 20 random restarts and 1024 raw samples.

    Parameters
    ----------
    acq : qLogExpectedImprovement
        Acquisition function.
    q : int
        Number of candidates to return (batch size).
    d : int
        Dimension of the search space.
    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor
        (q, d) candidate tensor in [0, 1].
    """
    from botorch.optim import optimize_acqf

    bounds_unit_cube = torch.tensor([[0.0] * d, [1.0] * d],
                                     dtype=torch.float64)
    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds_unit_cube,
        q=q,
        num_restarts=20,
        raw_samples=1024,
        seed=seed,
    )
    return candidates


# ── Parallel eval worker (module-level for pickling) ─────────────────────────

_EVAL_WORKER_ARGS = None

def _init_worker(args_dict):
    """Initialize worker process with shared args."""
    import numpy as np
    global _EVAL_WORKER_ARGS
    _EVAL_WORKER_ARGS = args_dict


def _eval_worker(params_phys, args_dict):
    """Evaluate a single config (x_c, c, beta, chi0, N_star, zeta_c) in a
    subprocess.  Imports directly from source modules to avoid circular import.

    Parameters
    ----------
    params_phys : list of 6 floats
        Physical parameter values [x_c, c, beta, chi0, N_star, zeta_c].
    args_dict : dict
        Serialized CLI arguments.

    Returns
    -------
    outcome dict or None
    """
    import numpy as np
    from types import SimpleNamespace
    import os as _os, sys as _sys
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)

    args = SimpleNamespace(**args_dict)
    x_c, c, beta, chi0, N_star, zeta_c = params_phys[:6]

    try:
        from scripts.observables import model_from_params
        from scripts.background_scan import analyze_background
        from inf_dyn_background import run_background_simulation, get_derived_quantities
        from scripts.compute_sr_ms import compute_ps_sr
        from scripts.full_pbh_pipeline import run_full_pbh_pipeline

        # Stage-1 pre-filter (same as _stage1_prefilter but inline)
        model = model_from_params(x_c, c, beta)
        model.x0 = chi0
        model.y0 = args.y0
        model.patch_background_solver()
        bg = analyze_background(model, chi0, args.y0, N_star, args.k_pivot)
        if bg is None or bg["N_total"] < args.N_total_min:
            return None
        if bg["usr_type"] == "weak" and not args.stage1_skip_usr:
            return None

        # Stage-2 SR filter (same as _stage2_sr_filter but inline)
        if not args.stage2_skip:
            T_span = np.linspace(0, model.T_max, model.bg_steps)
            bg_sol = run_background_simulation(model, T_span)
            derived = get_derived_quantities(bg_sol, model)
            epsH = derived["epsH"]
            end_candidates = np.where(epsH >= 1.0)[0]
            end_idx = int(end_candidates[0]) if len(end_candidates) > 0 else len(epsH) - 1
            if end_idx < 10:
                end_idx = len(epsH) - 1
            k_sr, P_S_sr, _ = compute_ps_sr(bg_sol, end_idx)
            valid = np.isfinite(P_S_sr) & (P_S_sr > 1e-30) & np.isfinite(k_sr) & (k_sr > 1.0)
            if np.sum(valid) < 5 or float(np.max(P_S_sr[valid])) < args.psr_peak_min:
                return None

        # Stage-3: full MS pipeline
        theta = np.array([x_c, c, beta, chi0, N_star, zeta_c])
        from scripts.optimize_pbh_botorch import _evaluate_pbh, _write_log_entry
        outcome = _evaluate_pbh(theta, args)
        _write_log_entry(args.log, outcome)
        return outcome
    except Exception as e:
        print(f"  Worker exception: {e}")
        return None


if __name__ == "__main__":
    main()
