"""Full PBH abundance pipeline for Ezquiaga CHI — one function does it all.

  1. Build EzquiagaCHIModel with inflection parameters
  2. Run background solver → N_total (cached check before MS)
  3. MS solver on PBH-weighted k-grid (10⁶–10²² Mpc⁻¹) — OR load cached P_S(k)
  4. Save P_S(k) to JSON
  5. Compute n_s at CMB pivot k=0.05
  6. Sweep ζ_c ∈ [0.05, 0.1], pick best f_total ≤ 1
  7. Plot abundance with observational bounds

Returns everything needed for downstream use.
"""

import json
import os
import time

import numpy as np
try:
    from numpy import trapezoid as trapz
except ImportError:
    from numpy import trapz
from scipy.special import erfc

from scripts.constants import (
    ROOT_DIR,
    S,
    gamma_default,
    k_eq_default,
    M_eq_default,
    ACCRETION,
)
from scripts.plotting import make_filename
from scripts.observables import extract_ns, interpolate_As

# ── Default ζ_c sweep ─────────────────────────────────────────────────────

DEFAULT_ZETA_C_VALS = [
    0.05,
    0.055,
    0.06,
    0.065,
    0.068,
    0.07,
    0.072,
    0.075,
    0.077,
    0.078,
    0.08,
    0.083,
    0.085,
    0.1,
]


# ── k-grid ─────────────────────────────────────────────────────────────────


def _pbh_weighted_kgrid():
    """k-grid targeting PBH scales (10⁶–10²² Mpc⁻¹), plus CMB-scale n_s fitting."""
    k_cmb_ns = np.logspace(np.log10(0.005), np.log10(0.5), 15)
    k_pbh = np.logspace(6, 22, 350)
    return np.unique(np.concatenate([k_cmb_ns, k_pbh]))


# ── Press-Schechter helpers ────────────────────────────────────────────────


def mass_from_k(k_phys, gamma=gamma_default, k_eq=k_eq_default, M_eq=M_eq_default):
    k = np.asarray(k_phys, dtype=float)
    return gamma * M_eq * (k_eq / k) ** 2


def beta_f_press_schechter(P_S, zeta_c=gamma_default):
    ps = np.asarray(P_S, dtype=float)
    return erfc(zeta_c / np.sqrt(2 * np.clip(ps, 1e-300, None)))


def beta_equality(beta_f, k_phys, k_eq=k_eq_default):
    return np.asarray(beta_f) * np.asarray(k_phys) / k_eq


def compute_pbh_abundance(
    k_phys, P_S, gamma=gamma_default, zeta_c=0.052, k_eq=k_eq_default, M_eq=M_eq_default
):
    """Press-Schechter PBH abundance from P_S(k).

    Returns dict with M (formation mass), beta_f, f_pbh.
    """
    k = np.asarray(k_phys, dtype=float)
    M = mass_from_k(k, gamma, k_eq, M_eq)
    bf = beta_f_press_schechter(P_S, zeta_c)
    beq = beta_equality(bf, k, k_eq)

    ok = np.isfinite(M) & np.isfinite(beq) & (M > 0) & (beq >= 0)
    M_ok = M[ok]
    beq_ok = np.maximum(beq[ok], 0.0)
    bf_ok = bf[ok]

    order = np.argsort(M_ok)
    return {
        "M": M_ok[order],
        "beta_f": bf_ok[order],
        "f_pbh": beq_ok[order],
    }


# ── Main pipeline ──────────────────────────────────────────────────────────


def run_full_pbh_pipeline(
    chi0=8.0,
    y0=-1e-4,
    N_star=65,
    beta=1e-5,
    xc=0.784,
    c=0.77,
    workers=8,
    fast_sr=False,
    zeta_c_vals=None,
    plot=True,
    use_old_bounds=False,
    smooth_bounds=False,
    force=False,
    k_pivot=0.05,
    ns_window=3.0,
    ns_method="lsq",
    formation_model="press_schechter",
    accretion_model="Chisholm",
):
    """Run the full PBH pipeline and produce the abundance plot.

    Parameters
    ----------
    chi0 : float
        Initial χ-field value.
    y0 : float
        Initial velocity dx/dT.
    N_star : float
        Target e-folds before end where pivot exits.
    beta : float
        Deviation from exact inflection.
    xc : float
        Critical field point x_c.
    c : float
        ξ₀·κ²μ² combined parameter.
    workers : int
        Parallel workers for MS solver.
    fast_sr : bool
        Use SR approximation instead of MS.
    zeta_c_vals : list of float or None
        Collapse thresholds to sweep. Defaults to DEFAULT_ZETA_C_VALS.
    plot : bool
        Generate the abundance plot.
    use_old_bounds : bool
        Use 2017 paper observational bounds.
    smooth_bounds : bool
        PCHIP-interpolate sparse bound datasets.
    force : bool
        Re-run MS solver even if cached P_S(k) exists.

    Returns
    -------
    dict with keys:
        k_phys : ndarray — physical wavenumbers [Mpc⁻¹]
        P_S : ndarray — scalar power spectrum
        n_s : float or None — spectral index at CMB pivot
        A_s_cmb : float or None — amplitude at CMB pivot
        N_total : float — total e-folds
        metadata : dict — full parameter snapshot
        ps_path : str — path to saved P_S(k) JSON
        cached : bool — whether P_S(k) was loaded from cache
        M_best : ndarray — best ζ_c mass function (present-day M⊙)
        f_pbh_best : ndarray — best ζ_c abundance
        zeta_c_best : float — best ζ_c value
        f_total_best : float — total abundance at best ζ_c
        M_peak_best : float — peak mass at best ζ_c (present-day M⊙)
        all_results : list[dict] — results for every ζ_c
    """
    from inf_dyn_background import get_derived_quantities, run_background_simulation
    from models.ezquiaga_chi import EzquiagaCHIModel, inflection_parameters

    if zeta_c_vals is None:
        zeta_c_vals = DEFAULT_ZETA_C_VALS

    t0 = time.time()
    pivot_k = k_pivot  # single pivot feeds As normalization AND n_s extraction

    # ── 1. Build model ─────────────────────────────────────────────────
    model = EzquiagaCHIModel(c=c)
    a, b = inflection_parameters(xc, c, beta=beta)
    model.a = a
    model.b = b
    model.v0 = model._V0 * model.a / (model.b * model.c) ** 2
    model.x0 = chi0
    model.y0 = y0
    model.patch_background_solver()
    print(f"Model: a={model.a:.6f}, b={model.b:.6f}, v0={model.v0:.4e}")

    # ── 2. Background solver → N_total (fast, always run) ─────────────
    T = np.linspace(0, model.T_max, model.bg_steps)
    bg_sol = run_background_simulation(model, T)
    derived = get_derived_quantities(bg_sol, model)
    epsH = derived["epsH"]
    end_candidates = np.where(epsH >= 1.0)[0]
    end_idx = int(end_candidates[0]) if len(end_candidates) > 0 else len(epsH) - 1
    N_total = float(derived["N"][end_idx])
    print(f"Background: N_total={N_total:.4f}, end_idx={end_idx}")

    # ── 3. P_S(k): cache check or MS solver ────────────────────────────
    n_tr = round(N_total, 1)
    fname = make_filename("ps", chi0, y0, n_tr, ".json")
    ps_dir = os.path.join(ROOT_DIR, "outputs/simulations/pspectra")
    os.makedirs(ps_dir, exist_ok=True)
    ps_path = os.path.join(ps_dir, fname)

    cached = False
    rec: dict = {}
    if os.path.exists(ps_path) and not force:
        with open(ps_path) as f:
            rec = json.load(f)
        k_phys = np.array(rec["spectrum"]["k_phys"])
        P_S = np.array(rec["spectrum"]["P_S"])
        cached = True
        print(f"Loaded cached P_S(k): {ps_path}")
    else:
        k_grid = _pbh_weighted_kgrid()
        print(f"MS solver: {len(k_grid)} k-modes, {workers} workers")

        if fast_sr:
            from scripts.plotting import compute_ps_sr

            k_cu, P_S_sr, _ = compute_ps_sr(bg_sol, end_idx)
            valid = np.isfinite(P_S_sr) & (P_S_sr > 0) & np.isfinite(k_cu) & (k_cu > 0)
            k_cu, P_S_sr = k_cu[valid], P_S_sr[valid]
            N_arr = derived["N"]
            N_pivot = N_total - N_star
            z_bg = bg_sol[2]
            n_bg = bg_sol[3]
            zp = float(np.interp(N_pivot, N_arr, z_bg))
            ap = float(np.exp(np.interp(N_pivot, N_arr, n_bg)))
            k_piv = ap * zp * S
            k_phys = k_cu * (pivot_k / k_piv)
            N_sr = N_arr[: end_idx + 1][valid]
            n_be = N_total - N_sr
            mask = n_be < 40
            k_phys, P_S = k_phys[mask], P_S_sr[mask]
            print(f"  SR: {len(k_phys)} valid modes")
        else:
            from pspectrum_pipeline import run_pspectrum_pipeline

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
            k_phys = np.array(result["k_phys"])
            P_S = np.array(result["P_S"])

        print(
            f"  k range: [{k_phys.min():.2e}, {k_phys.max():.2e}], {len(k_phys)} modes"
        )

    # ── 4. Save P_S(k) if not cached ──────────────────────────────────
    if cached:
        metadata = rec.get("metadata", {})
    else:
        metadata = {
            "x0": chi0,
            "chi0": chi0,
            "y0": y0,
            "N_total": N_total,
            "N_star": N_star,
            "x_c": xc,
            "c": c,
            "beta": beta,
            "k_pivot_Mpc": pivot_k,
            "ns_window": None,
            "n_modes": len(k_phys),
            "source": "MS" if not fast_sr else "SR",
        }
        save_rec = {
            "_type": "pspectrum",
            "format_version": 1,
            "metadata": metadata,
            "spectrum": {"k_phys": k_phys.tolist(), "P_S": P_S.tolist()},
        }
        with open(ps_path, "w") as f:
            json.dump(save_rec, f, indent=2)
        print(f"Saved P_S(k): {ps_path}")

    # ── 5. Compute n_s via logarithmic derivative or LSQ at k_pivot ────
    n_s, ns_meta = extract_ns(k_phys, P_S, k_pivot=pivot_k,
                              ns_window=ns_window if ns_method == "lsq" else None,
                              method=ns_method)
    A_s_cmb = interpolate_As(k_phys, P_S, pivot_k)
    print(f"  n_s(k={pivot_k}) = {n_s}, A_s = {A_s_cmb}")

    # ── 6. Build accretion model ───────────────────────────────────────
    from scripts.accretion import (
        ChisholmAccretion, EddingtonAccretion, PBHAccretion, PR_Accretion,
    )

    _ACCRETION_FACTORY = {
        "PR": PR_Accretion,
        "BHL": PBHAccretion,
        "Chisholm": ChisholmAccretion,
        "Eddington": EddingtonAccretion,
    }

    if accretion_model not in _ACCRETION_FACTORY:
        raise ValueError(f"Unknown accretion model '{accretion_model}'. Choose from {sorted(_ACCRETION_FACTORY.keys())}")

    if accretion_model == "BHL":
        accretion = PBHAccretion(model="BHL", lambda_acc=0.1)
    else:
        accretion = _ACCRETION_FACTORY[accretion_model]()

    print(f"\n--- PBH abundance sweep ({formation_model}) ---")
    print(f"Accretion: {accretion_model}  Formation: {formation_model}")
    all_results = []

    for zc in zeta_c_vals:
        if formation_model == "press_schechter":
            pbh = compute_pbh_abundance(
                k_phys, P_S, gamma_default, zc, k_eq_default, M_eq_default
            )
            M_form = np.asarray(pbh["M"], dtype=float)
            f_pbh = np.asarray(pbh["f_pbh"], dtype=float)

            # Apply accretion
            if accretion_model == "Chisholm":
                M_present = M_form * 3e7
            else:
                M_present = np.array([
                    accretion.M_of_redshift(float(m), 0.0) for m in M_form
                ])
        elif formation_model == "compaction":
            from scripts.compaction import beta_f_compaction
            nw = max(1, workers)
            beta_f, M_pbh_form, meta_comp = beta_f_compaction(
                k_phys, P_S, gamma=gamma_default, mu=zc, n_workers=nw,
            )
            # Convert to equality-normalised DM fraction
            f_pbh = beta_equality(beta_f, k_phys, k_eq_default)

            # Only apply accretion to modes with non-zero formation fraction.
            M_pbh_ok = np.where(beta_f > 0, M_pbh_form, 0.0)
            M_present = np.array([
                accretion.M_of_redshift(float(m), 0.0)
                if m > 1e-300 else 0.0
                for m in M_pbh_ok
            ])
        else:
            raise ValueError(f"Unknown formation model '{formation_model}'")

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

        f_total = float(trapz(fp_ok, np.log(np.maximum(M_ok, 1e-300))))
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

    # Pick best viable (f_total ≤ 1, highest)
    viable = [r for r in all_results if 0 < r["f_total"] <= 1.0]
    if viable:
        best = max(viable, key=lambda r: r["f_total"])
        zeta_c_best = best["zeta_c"]
        M_best = best["M"]
        f_pbh_best = best["f_pbh"]
        f_total_best = best["f_total"]
        M_peak_best = best["M_peak"]
        print(f"\nBest ζ_c = {zeta_c_best:.3f} (f_total = {f_total_best:.4e})")
    else:
        zeta_c_best = None
        M_best = np.array([])
        f_pbh_best = np.array([])
        f_total_best = 0.0
        M_peak_best = np.nan
        print("\nNo viable ζ_c found (all f_total > 1 or = 0)")

    # ── 7. Plot ─────────────────────────────────────────────────────────
    if plot and zeta_c_best is not None and len(M_best) > 0:
        from scripts.plotting import make_pbh_filename

        if formation_model == "press_schechter" and accretion_model == "Chisholm":
            from scripts.plotting import plot_pbh_abundance
            plot_pbh_abundance(
                M_best,
                f_pbh_best,
                zeta_c=zeta_c_best,
                gamma=gamma_default,
                model_label=f"Ezquiaga CHI χ₀={chi0}, β={beta:.0e}, ζ_c={zeta_c_best}",
                filename=make_pbh_filename(
                    "pbh", chi0, y0, N_total,
                    formation="press_schechter", accretion="Chisholm",
                    beta=beta, zc=zeta_c_best,
                ),
                category="pbh",
                use_old_bounds=use_old_bounds,
                smooth_bounds=smooth_bounds,
            )
        else:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from scripts import constraint_mapping
            from scripts.plotting import get_path

            fname = make_pbh_filename(
                "pbh", chi0, y0, N_total,
                formation=formation_model, accretion=accretion_model,
                beta=beta, zc=zeta_c_best,
            )

            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.loglog(
                M_best, f_pbh_best, "-.", color="#CC3311",
                lw=1.5, label=f"CHI (zeta_c={zeta_c_best:.3f})",
            )
            constraint_mapping.plot_modern_constraints(
                ax, accretion_model=accretion,
            )
            ax.legend(fontsize=8, loc="lower left", ncol=2)

            path = get_path("pbh", fname)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved plot: {path}")

    elapsed = time.time() - t0
    print(f"\nPipeline done in {elapsed:.1f}s" + (" (cached)" if cached else ""))

    return {
        "k_phys": k_phys,
        "P_S": P_S,
        "n_s": n_s,
        "ns_meta": ns_meta,
        "k_pivot": pivot_k,
        "ns_window": ns_window,
        "A_s_cmb": A_s_cmb,
        "N_total": N_total,
        "end_idx": end_idx,
        "metadata": metadata,
        "ps_path": ps_path,
        "cached": cached,
        "M_best": M_best,
        "f_pbh_best": f_pbh_best,
        "zeta_c_best": zeta_c_best,
        "f_total_best": f_total_best,
        "M_peak_best": M_peak_best,
        "all_results": all_results,
        "accretion_model": accretion_model,
        "formation_model": formation_model,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="PBH abundance pipeline CLI")
    p.add_argument("--config", type=str, default=None,
                    help="JSON config file (CLI overrides file values)")
    p.add_argument("--chi0", type=float, default=8.0)
    p.add_argument("--y0", type=float, default=-1e-4)
    p.add_argument("--N-star", type=float, default=65.0)
    p.add_argument("--beta", type=float, default=1e-5)
    p.add_argument("--xc", type=float, default=0.784)
    p.add_argument("--c", type=float, default=0.77)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--zeta-c", type=float, nargs="+",
                    default=[0.05, 0.055, 0.06, 0.065, 0.068, 0.07, 0.072,
                             0.075, 0.077, 0.078, 0.08, 0.083, 0.085, 0.1])
    p.add_argument("--output-dir", type=str,
                    default="outputs/plots/pbh/top_configs")
    p.add_argument("--no-force", action="store_false", dest="force")
    p.set_defaults(force=True)
    p.add_argument("--tag", type=str, default=None,
                    help="Prefix for output filenames (e.g. 'rank07')")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument(
        "--k-pivot",
        type=float,
        default=0.05,
        help="Pivot k_pivot_phys (Mpc^-1) — drives BOTH As normalization and "
             "n_s extraction. Planck default 0.05; Higgs low-ell 0.002.",
    )
    p.add_argument(
        "--ns-window",
        type=float,
        default=3.0,
        help="Fit half-width for lsq method [k_pivot/w, k_pivot*w] (ignored for derivative)",
    )
    p.add_argument(
        "--ns-method",
        choices=["lsq", "derivative"],
        default="lsq",
        help="n_s extraction method: lsq=window fit (default), derivative=log-derivative at k_pivot",
    )
    p.add_argument(
        "--accretion",
        choices=["PR", "BHL", "Chisholm", "Eddington"],
        default=None,
        help="Accretion model (default: Chisholm)",
    )
    p.add_argument(
        "--formation",
        choices=["compaction", "press_schechter"],
        default=None,
        help="Formation model (default: press_schechter)",
    )

    pre, _ = p.parse_known_args()
    config_accretion = None
    config_formation = None
    if pre.config:
        from scripts.config_loader import load_config
        config_data = load_config(pre.config)
        p.set_defaults(**config_data)
        config_accretion = config_data.get("accretion")
        config_formation = config_data.get("formation")
    args = p.parse_args()

    accretion_val = args.accretion or config_accretion or "Chisholm"
    formation_val = args.formation or config_formation or "press_schechter"

    from scripts.constants import ROOT_DIR
    import os, json, shutil
    from datetime import datetime

    result = run_full_pbh_pipeline(
        chi0=args.chi0, y0=args.y0, N_star=args.N_star,
        beta=args.beta, xc=args.xc, c=args.c,
        workers=args.workers, zeta_c_vals=args.zeta_c,
        plot=not args.no_plot, force=args.force,
        k_pivot=args.k_pivot, ns_window=args.ns_window,
        ns_method=args.ns_method,
        formation_model=formation_val,
        accretion_model=accretion_val,
    )

    target_dir = os.path.join(ROOT_DIR, args.output_dir)
    os.makedirs(target_dir, exist_ok=True)

    ns = result["n_s"]
    ntot = result["N_total"]

    for zc_data in result["all_results"]:
        zc = zc_data["zeta_c"]
        ft = zc_data["f_total"]
        M = zc_data["M_peak"]

        print(f"zeta_c={zc:.4f}  f_total={ft:.4e}  M_peak={M:.4e}")

        if not args.no_plot and ft > 0:
            from scripts.plotting import make_pbh_filename
            tag = f"{args.tag}_" if args.tag else ""
            plot_name = args.tag if args.tag else "pbh"
            pid = make_pbh_filename(
                plot_name, args.chi0, args.y0, result["N_total"],
                formation=formation_val, accretion=accretion_val,
                beta=args.beta, zc=zc,
            )
            
            if formation_val == "press_schechter" and accretion_val == "Chisholm":
                from scripts.plotting import plot_pbh_abundance
                plot_pbh_abundance(
                    zc_data["M"], zc_data["f_pbh"],
                    zeta_c=zc, gamma=0.4,
                    model_label=f"beta={args.beta:.1e}, N*={args.N_star:.0f}, zeta_c={zc:.4f} (f={ft:.3e})",
                    filename=pid, category="pbh",
                )
            else:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from scripts import constraint_mapping
                
                fig, ax = plt.subplots(figsize=(6, 4.5))
                ax.loglog(
                    zc_data["M"], zc_data["f_pbh"], "-.", color="#CC3311",
                    lw=1.5, label=f"CHI (zeta_c={zc:.4f}, f={ft:.3e})",
                )
                
                from scripts.accretion import (
                    ChisholmAccretion, EddingtonAccretion, PBHAccretion, PR_Accretion,
                )
                _FACTORY = {
                    "PR": PR_Accretion,
                    "BHL": PBHAccretion,
                    "Chisholm": ChisholmAccretion,
                    "Eddington": EddingtonAccretion,
                }
                if accretion_val == "BHL":
                    accretion_inst = PBHAccretion(model="BHL", lambda_acc=0.1)
                else:
                    accretion_inst = _FACTORY[accretion_val]()
                    
                constraint_mapping.plot_modern_constraints(
                    ax, accretion_model=accretion_inst,
                )
                ax.legend(fontsize=8, loc="lower left", ncol=2)
                
                from scripts.plotting import get_path
                plot_path = get_path("pbh", pid)
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

            src = os.path.join(ROOT_DIR, "outputs/plots/pbh", pid)
            dst = os.path.join(target_dir, pid)
            if os.path.exists(src):
                shutil.move(src, dst)
                comp = {
                    "plot_file": pid,
                    "generated": datetime.now().isoformat(),
                    "pipeline": "full_pbh_pipeline CLI",
                    "config": {
                        "model": "EzquiagaCHIModel",
                        "x_c": args.xc, "c": args.c, "beta": args.beta,
                        "chi0": args.chi0, "y0": args.y0, "N_star": args.N_star,
                        "k_pivot": args.k_pivot, "ns_window": args.ns_window,
                        "formation": formation_val,
                        "accretion": accretion_val,
                    },
                    "background": {"N_total": ntot, "n_s": ns},
                    "pbh": {"zeta_c": zc, "f_total": ft, "M_present_Msun": M},
                }
                with open(dst.replace(".png", ".json"), "w") as f:
                    json.dump(comp, f, indent=2, default=str)
                print(f"  -> {dst}")
