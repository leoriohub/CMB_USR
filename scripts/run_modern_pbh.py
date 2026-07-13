r"""
Modern PBH abundance pipeline — config-driven CLI with pluggable accretion
and formation models.

Supports two formation formalisms:

- **press_schechter** (legacy): :math:`\beta_{\mathrm{f}} = \operatorname{erfc}
  \bigl(\zeta_{\mathrm{c}} / \sqrt{2\,\mathcal{P_S}(k)}\bigr)` with
  horizon-mass scaling :math:`M(k) = \gamma M_{\mathrm{eq}} (k_{\mathrm{eq}}/k)^2`.

- **compaction** (modern): profile-dependent compaction function approach
  (Escrivà + 2021/2022) with critical scaling for PBH mass and
  shape-dependent collapse threshold :math:`C_{\mathrm{c}}(\alpha)`.

And four accretion models:

- ``PR`` — Park-Ricotti with photo-ionisation radiative feedback (default).
- ``BHL`` — Bondi-Hoyle-Lyttleton (Ricotti+2007).
- ``Chisholm`` — Legacy constant-factor growth (3 × 10⁷).
- ``Eddington`` — Eddington-limited BHL with DM halo density enhancement.

Config format is identical to :mod:`scripts.full_pbh_pipeline` (canonical
nested or flat).  See ``configs/ezquiaga/fig3.json`` for an example.

Examples
--------
.. code:: bash

    # Modern defaults (PR accretion + compaction formation)
    python scripts/run_modern_pbh.py --config configs/ezquiaga/subsolar_pbh.json

    # Reproduce old results
    python scripts/run_modern_pbh.py --config configs/ezquiaga/subsolar_pbh.json \\
        --accretion Chisholm --formation press_schechter --zeta-c 0.077

    # Custom threshold sweep
    python scripts/run_modern_pbh.py --config configs/ezquiaga/subsolar_pbh.json \\
        --zeta-c 0.07 0.075 0.08 --k-pivot 0.05 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np

from scripts.accretion import (
    ChisholmAccretion, EddingtonAccretion, PBHAccretion, PR_Accretion,
)
from scripts.constants import (
    ROOT_DIR,
    gamma_default,
    k_eq_default,
    M_eq_default,
)
from scripts.observables import extract_ns, interpolate_As
from scripts.plotting import get_path, make_filename

# ---------------------------------------------------------------------------
# Default ζ_c sweep (Press-Schechter convention)
# ---------------------------------------------------------------------------

DEFAULT_ZETA_C_VALS: list[float] = [
    0.05, 0.055, 0.06, 0.065, 0.068, 0.07,
    0.072, 0.075, 0.077, 0.078, 0.08, 0.083,
    0.085, 0.1,
]

# ---------------------------------------------------------------------------
# k-grid targeting PBH scales + CMB n_s fitting
# ---------------------------------------------------------------------------


def _pbh_weighted_kgrid() -> np.ndarray:
    """k-grid covering PBH scales (10⁶–10²² Mpc⁻¹) plus CMB n_s pivot.

    Replicates the grid from ``full_pbh_pipeline._pbh_weighted_kgrid``.
    """
    k_cmb_ns = np.logspace(np.log10(0.005), np.log10(0.5), 15)
    k_pbh = np.logspace(6.0, 22.0, 350)
    return np.unique(np.concatenate([k_cmb_ns, k_pbh]))


# ---------------------------------------------------------------------------
# Accretion factory
# ---------------------------------------------------------------------------

_ACCRETION_FACTORY: dict[str, type[PBHAccretion]] = {
    "PR": PR_Accretion,
    "BHL": PBHAccretion,
    "Chisholm": ChisholmAccretion,
    "Eddington": EddingtonAccretion,
}


def _build_accretion(model_name: str) -> PBHAccretion:
    """Build an accretion model instance from its string name.

    Parameters
    ----------
    model_name : str
        One of ``'PR'``, ``'BHL'``, ``'Chisholm'``.

    Returns
    -------
    PBHAccretion
        Configured accretion instance.

    Raises
    ------
    ValueError
        If *model_name* is not recognised.
    """
    cls = _ACCRETION_FACTORY.get(model_name)
    if cls is None:
        msg = (
            f"Unknown accretion model '{model_name}'. "
            f"Choose from {sorted(_ACCRETION_FACTORY)}."
        )
        raise ValueError(msg)
    if model_name == "BHL":
        return cls(model="BHL", lambda_acc=0.1)
    if model_name == "Chisholm":
        return cls()
    return cls()


# ---------------------------------------------------------------------------
# Equality normalisation (shared by both formation models)
# ---------------------------------------------------------------------------


def _f_pbh_from_beta(beta_f: np.ndarray, k_phys: np.ndarray,
                     k_eq: float = k_eq_default) -> np.ndarray:
    """Convert formation fraction to present-day DM fraction
    via equality normalisation :math:`f = \\beta \\, k / k_{\\mathrm{eq}}`."""
    return np.asarray(beta_f, dtype=float) * np.asarray(k_phys, dtype=float) / k_eq


def _f_total(f_pbh: np.ndarray, M_present: np.ndarray) -> float:
    """Total PBH DM fraction :math:`f_{\\mathrm{PBH}} = \\int f \\, d\\ln M`."""
    if len(f_pbh) == 0 or len(M_present) == 0:
        return 0.0
    return float(np.trapezoid(f_pbh, np.log(np.maximum(M_present, 1e-300))))


# ---------------------------------------------------------------------------
# Config → model builder
# ---------------------------------------------------------------------------


def _model_from_config(config: dict[str, Any]) -> Any:
    """Construct an ``EzquiagaCHIModel`` from a nested config dict.

    Expects the canonical format with ``model_params``, ``inflection``,
    ``ics`` blocks (same as ``configs/ezquiaga/fig3.json``).
    """
    from models.ezquiaga_chi import EzquiagaCHIModel, inflection_parameters

    mp = config.get("model_params", {})
    c_val: float = mp.get("c", 0.77)

    model = EzquiagaCHIModel(
        lambda_0=mp.get("lambda_0", 2.23e-7),
        xi_0=mp.get("xi_0", 7.55),
        c=c_val,
    )

    inf = config.get("inflection", {})
    x_c: float = inf.get("x_c", 0.784)
    beta: float = inf.get("beta", 1e-5)
    a, b = inflection_parameters(x_c, c_val, beta=beta)
    model.a = a
    model.b = b
    model.v0 = model._V0 * model.a / (model.b * model.c) ** 2

    ics = config.get("ics", {})
    model.x0 = ics.get("x0", 8.0)
    model.y0 = ics.get("y0", -1e-4)

    model.patch_background_solver()
    return model


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_modern_pbh_pipeline(
    config_path: str,
    accretion_model: str | None = None,
    formation_model: str | None = None,
    zeta_c_values: list[float] | None = None,
    plot: bool = True,
    k_pivot: float = 0.05,
    workers: int = 8,
    force: bool = False,
    smooth_bounds: bool = False,
) -> dict[str, Any]:
    """Modern PBH pipeline from config to plot.

    Parameters
    ----------
    config_path : str
        Path to JSON config (same canonical nested format as
        :mod:`scripts.full_pbh_pipeline`).
    accretion_model : str or None
        Accretion model (``'PR'``, ``'BHL'``, ``'Chisholm'``,
        ``'Eddington'``).  If *None* (default), resolved
        from the config's ``"accretion"`` field, falling back to
        ``'PR'``.
    formation_model : str or None
        Formation model (``'compaction'`` or ``'press_schechter'``).
        If *None* (default), resolved from the config's ``"formation"``
        field, falling back to ``'compaction'``.
    zeta_c_values : list of float or None
        Collapse thresholds to sweep.  Defaults to
        :data:`DEFAULT_ZETA_C_VALS`.
    plot : bool
        Generate the abundance plot with observational constraints.
    k_pivot : float
        Pivot wavenumber [Mpc⁻¹] for A_s normalisation and n_s extraction.
    workers : int
        Parallel workers for the MS solver.
    force : bool
        Re-run MS solver even if a cached P_S(k) file exists.
    smooth_bounds : bool
        Passed to constraint plotting for PCHIP-interpolation of sparse
        bound datasets (unused by this module, forwarded for API compat).

    Returns
    -------
    dict with keys
        ``k_phys``, ``P_S``, ``n_s``, ``N_total``, ``M_best``,
        ``f_pbh_best``, ``zeta_c_best``, ``f_total_best``,
        ``M_peak_best``, ``all_results``, ``accretion_model``,
        ``formation_model``.
    """
    # ── Imports (lazy — only when called) ──────────────────────────────
    from inf_dyn_background import (
        get_derived_quantities,
        run_background_simulation,
    )
    from pspectrum_pipeline import run_pspectrum_pipeline

    if zeta_c_values is None:
        zeta_c_values = list(DEFAULT_ZETA_C_VALS)

    t0 = time.time()
    pivot_k = float(k_pivot)

    # ── 1. Load config & build model ───────────────────────────────────
    with open(config_path) as fh:
        config: dict[str, Any] = json.load(fh)

    model = _model_from_config(config)

    # Pipeline-level parameters (may override defaults)
    pipe = config.get("pipeline", {})
    N_star: float = float(pipe.get("N_star", 65))
    chi0 = float(model.x0)
    y0 = float(model.y0)
    inf = config.get("inflection", {})
    xc_val: float = inf.get("x_c", 0.784)
    beta_val: float = inf.get("beta", 1e-5)
    c_val: float = config.get("model_params", {}).get("c", 0.77)

    _config_formation: str | None = config.get("formation")
    _config_accretion: str | None = config.get("accretion")
    if formation_model is None:
        formation_model = _config_formation or "compaction"
    elif _config_formation is not None and _config_formation != formation_model:
        import warnings as _w
        _w.warn(
            f"CLI --formation={formation_model} overrides config "
            f"formation={_config_formation}. To use the config value, "
            f"omit the --formation flag.",
            stacklevel=2,
        )
    if accretion_model is None:
        accretion_model = _config_accretion or "PR"
    elif _config_accretion is not None and _config_accretion != accretion_model:
        import warnings as _w
        _w.warn(
            f"CLI --accretion={accretion_model} overrides config "
            f"accretion={_config_accretion}. To use the config value, "
            f"omit the --accretion flag.",
            stacklevel=2,
        )

    print(f"Model: a={model.a:.6f}, b={model.b:.6f}, v0={model.v0:.4e}")
    print(f"Model: χ₀={chi0:.4f}, y₀={y0:.4e}, N_*={N_star:.1f}")

    # ── 2. Background solver → N_total ─────────────────────────────────
    T = np.linspace(0.0, model.T_max, model.bg_steps)
    bg_sol = run_background_simulation(model, T)
    derived_bg = get_derived_quantities(bg_sol, model)
    epsH = derived_bg["epsH"]
    end_candidates = np.where(epsH >= 1.0)[0]
    end_idx = int(end_candidates[0]) if len(end_candidates) > 0 else len(epsH) - 1
    N_total = float(derived_bg["N"][end_idx])
    print(f"Background: N_total={N_total:.4f}, end_idx={end_idx}")

    # ── 3. P_S(k): cache check or MS solver ────────────────────────────
    n_tr = round(N_total, 1)
    ps_fname = make_filename("ps", chi0, y0, n_tr, ".json")
    ps_dir = os.path.join(ROOT_DIR, "outputs/simulations/pspectra")
    os.makedirs(ps_dir, exist_ok=True)
    ps_path = os.path.join(ps_dir, ps_fname)

    cached = False
    rec: dict[str, Any] = {}
    if os.path.exists(ps_path) and not force:
        with open(ps_path) as fh:
            rec = json.load(fh)
        k_phys = np.array(rec["spectrum"]["k_phys"], dtype=float)
        P_S = np.array(rec["spectrum"]["P_S"], dtype=float)
        cached = True
        print(f"Loaded cached P_S(k): {ps_path}")
    else:
        k_grid = _pbh_weighted_kgrid()
        print(f"MS solver: {len(k_grid)} k-modes, {workers} workers")

        result = run_pspectrum_pipeline(
            model=model,
            phi0=None,
            y0=None,
            k_phys_grid=k_grid,
            k_pivot_phys=pivot_k,
            N_star=N_star,
            k_start_factor=100.0,
            ms_steps=5000,
            normalize_to_As=False,
            save_outputs=True,
            n_workers=workers,
        )
        k_phys = np.array(result["k_phys"], dtype=float)
        P_S = np.array(result["P_S"], dtype=float)

        print(f"  k range: [{k_phys.min():.2e}, {k_phys.max():.2e}], "
              f"{len(k_phys)} modes")

    # ── 4. Save P_S(k) if not cached ───────────────────────────────────
    if cached:
        metadata = rec.get("metadata", {})
    else:
        metadata = {
            "x0": chi0,
            "chi0": chi0,
            "y0": y0,
            "N_total": N_total,
            "N_star": N_star,
            "x_c": xc_val,
            "c": c_val,
            "beta": beta_val,
            "k_pivot_Mpc": pivot_k,
            "n_modes": len(k_phys),
            "source": "MS",
        }
        save_rec = {
            "_type": "pspectrum",
            "format_version": 1,
            "metadata": metadata,
            "spectrum": {
                "k_phys": k_phys.tolist(),
                "P_S": P_S.tolist(),
            },
        }
        with open(ps_path, "w") as fh:
            json.dump(save_rec, fh, indent=2)
        print(f"Saved P_S(k): {ps_path}")

    # ── 5. n_s at k_pivot ──────────────────────────────────────────────
    ns_window = float(pipe.get("ns_window", 3.0))
    ns_method: str = pipe.get("ns_method", "lsq")
    window_arg: float | None = ns_window if ns_method == "lsq" else None
    n_s, ns_meta = extract_ns(
        k_phys, P_S, k_pivot=pivot_k,
        ns_window=window_arg,
        method=ns_method,
    )
    A_s_cmb = interpolate_As(k_phys, P_S, pivot_k)
    print(f"  n_s(k={pivot_k}) = {n_s}, A_s = {A_s_cmb}")

    # ── 6. Build accretion & formation models ──────────────────────────
    accretion: PBHAccretion = _build_accretion(accretion_model)
    print(f"Accretion: {accretion_model}  Formation: {formation_model}")

    # ── 7. ζ_c sweep ────────────────────────────────────────────────────
    print(f"\n--- PBH abundance sweep ({formation_model}) ---")
    all_results: list[dict[str, Any]] = []

    for zc in zeta_c_values:
        if formation_model == "press_schechter":
            from scripts.full_pbh_pipeline import compute_pbh_abundance
            pbh = compute_pbh_abundance(
                k_phys, P_S, gamma_default, zc, k_eq_default, M_eq_default,
            )
            M_form = np.asarray(pbh["M"], dtype=float)
            f_pbh = np.asarray(pbh["f_pbh"], dtype=float)

            # Apply accretion to each mode
            M_present = np.array([
                accretion.M_of_redshift(float(m), 0.0) for m in M_form
            ])
        else:
            from scripts.compaction import beta_f_compaction
            # mu = zeta_c — sets the profile normalisation for the
            # compaction function (peak height parameter)
            beta_f, M_pbh_form, meta_comp = beta_f_compaction(
                k_phys, P_S, gamma=gamma_default, mu=zc,
            )
            # Convert to equality-normalised DM fraction
            f_pbh = _f_pbh_from_beta(beta_f, k_phys)

            # Apply accretion to each mode that collapsed
            M_present = np.array([
                accretion.M_of_redshift(float(m), 0.0)
                if m > 1e-300 else 0.0
                for m in M_pbh_form
            ])

        # Filter finite / positive
        ok = np.isfinite(M_present) & np.isfinite(f_pbh) & (M_present > 0) & (f_pbh > 0)
        M_ok = M_present[ok]
        fp_ok = f_pbh[ok]

        if len(M_ok) == 0:
            all_results.append({
                "zeta_c": zc, "f_total": 0.0, "M_peak": np.nan,
                "M": np.array([]), "f_pbh": np.array([]),
            })
            print(f"  ζ_c={zc:.3f}: f_total=0 (no collapse)")
            continue

        f_total = _f_total(fp_ok, M_ok)
        peak_i = int(np.argmax(fp_ok))
        M_peak = float(M_ok[peak_i]) if len(M_ok) > 0 else np.nan

        all_results.append({
            "zeta_c": zc,
            "f_total": f_total,
            "M_peak": M_peak,
            "M": M_ok,
            "f_pbh": fp_ok,
        })
        print(f"  ζ_c={zc:.3f}: f_total={f_total:.4e}, M_peak={M_peak:.3e}")

    # ── 8. Pick best viable (f_total ≤ 1, highest) ─────────────────────
    viable = [r for r in all_results if 0 < r["f_total"] <= 1.0]
    if viable:
        best = max(viable, key=lambda r: r["f_total"])
        zeta_c_best: float | None = best["zeta_c"]
        M_best: np.ndarray = best["M"]
        f_pbh_best: np.ndarray = best["f_pbh"]
        f_total_best: float = best["f_total"]
        M_peak_best: float = best["M_peak"]
        print(f"\nBest ζ_c = {zeta_c_best:.3f} (f_total = {f_total_best:.4e})")
    else:
        zeta_c_best = None
        M_best = np.array([])
        f_pbh_best = np.array([])
        f_total_best = 0.0
        M_peak_best = np.nan
        print("\nNo viable ζ_c found (all f_total > 1 or = 0)")

    # ── 9. Plot ─────────────────────────────────────────────────────────
    if plot and zeta_c_best is not None and len(M_best) > 0:
        _make_abundance_plot(
            M_best, f_pbh_best, chi0, y0, n_tr,
            zeta_c_best, accretion, accretion_model, formation_model,
        )

    elapsed = time.time() - t0
    print(f"\nPipeline done in {elapsed:.1f}s" + (" (cached)" if cached else ""))

    return {
        "k_phys": k_phys,
        "P_S": P_S,
        "n_s": n_s,
        "ns_meta": ns_meta,
        "N_total": N_total,
        "M_best": M_best,
        "f_pbh_best": f_pbh_best,
        "zeta_c_best": zeta_c_best,
        "f_total_best": f_total_best,
        "M_peak_best": M_peak_best,
        "all_results": all_results,
        "accretion_model": accretion_model,
        "formation_model": formation_model,
    }


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------


def _make_abundance_plot(
    M_best: np.ndarray,
    f_pbh_best: np.ndarray,
    chi0: float,
    y0: float,
    n_tr: float,
    zeta_c: float,
    accretion: PBHAccretion,
    accretion_model: str,
    formation_model: str,
) -> None:
    """Create the abundance plot with modern observational constraints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts import constraint_mapping
    from scripts.plotting import make_pbh_filename

    fname = make_pbh_filename(
        "pbh", chi0, y0, n_tr,
        formation=formation_model, accretion=accretion_model,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(
        M_best, f_pbh_best, "-.", color="#CC3311",
        lw=1.5, label=f"CHI (ζ_c={zeta_c:.3f})",
    )

    constraint_mapping.plot_modern_constraints(
        ax, accretion_model=accretion,
    )
    ax.legend(fontsize=8, loc="lower left", ncol=2)

    path = get_path("pbh", fname)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modern PBH pipeline — config-driven accretion + formation.",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to JSON config (canonical nested format, "
             "same as full_pbh_pipeline.py)",
    )
    parser.add_argument(
        "--accretion", choices=["PR", "BHL", "Chisholm", "Eddington"], default=None,
        help="Accretion model (default from config, or PR)",
    )
    parser.add_argument(
        "--formation", choices=["compaction", "press_schechter"],
        default=None,
        help="Formation model (default from config, or compaction)",
    )
    parser.add_argument(
        "--zeta-c", type=float, nargs="+",
        default=DEFAULT_ZETA_C_VALS,
        help="Collapse thresholds to sweep (default: 0.05–0.1)",
    )
    parser.add_argument(
        "--k-pivot", type=float, default=0.05,
        help="Pivot wavenumber Mpc⁻¹ (default: 0.05)",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="MS solver parallel workers (default: 8)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip abundance plot generation",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run MS solver even if cached P_S(k) exists",
    )
    parser.add_argument(
        "--smooth-bounds", action="store_true",
        help="PCHIP-interpolate sparse bound datasets",
    )

    args = parser.parse_args()

    result = run_modern_pbh_pipeline(
        config_path=args.config,
        accretion_model=args.accretion,
        formation_model=args.formation,
        zeta_c_values=args.zeta_c,
        plot=not args.no_plot,
        k_pivot=args.k_pivot,
        workers=args.workers,
        force=args.force,
        smooth_bounds=args.smooth_bounds,
    )

    # Print summary
    print("\n── Summary ──")
    print(f"  n_s         = {result['n_s']}")
    print(f"  N_total     = {result['N_total']:.2f}")
    print(f"  ζ_c best    = {result['zeta_c_best']}")
    print(f"  f_total     = {result['f_total_best']:.4e}")
    print(f"  M_peak      = {result['M_peak_best']:.4e}")
    print(f"  Accretion   = {result['accretion_model']}")
    print(f"  Formation   = {result['formation_model']}")
