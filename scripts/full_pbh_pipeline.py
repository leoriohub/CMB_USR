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

    # ── 5. Compute n_s via logarithmic derivative at k_pivot ──────────
    # n_s - 1 = d ln P_S / d ln k at k = k_pivot (theoretical definition).
    n_s, ns_meta = extract_ns(k_phys, P_S, k_pivot=pivot_k,
                              ns_window=args.ns_window if args.ns_method == "lsq" else None,
                              method=args.ns_method)
    A_s_cmb = interpolate_As(k_phys, P_S, pivot_k)
    print(f"  n_s(k={pivot_k}) = {n_s}, A_s = {A_s_cmb}")

    # ── 6. PBH abundance at multiple ζ_c ───────────────────────────────
    print("\n--- PBH abundance ---")
    all_results = []
    for zc in zeta_c_vals:
        pbh = compute_pbh_abundance(
            k_phys, P_S, gamma_default, zc, k_eq_default, M_eq_default
        )
        M_form, f_pbh = pbh["M"], pbh["f_pbh"]
        M_present = M_form * ACCRETION

        f_total = float(np.trapezoid(f_pbh, np.log(np.maximum(M_present, 1e-300))))
        peak_i = int(np.argmax(f_pbh))
        M_peak = float(M_present[peak_i]) if len(M_present) > 0 else np.nan
        all_results.append(
            {
                "zeta_c": zc,
                "f_total": f_total,
                "M_peak": M_peak,
                "M": M_present,
                "f_pbh": f_pbh,
            }
        )
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
        from scripts.plotting import plot_pbh_abundance

        plot_pbh_abundance(
            M_best,
            f_pbh_best,
            zeta_c=zeta_c_best,
            gamma=gamma_default,
            model_label=f"Ezquiaga CHI χ₀={chi0}, β={beta:.0e}, ζ_c={zeta_c_best}",
            filename=f"pbh_chi{chi0}_beta{beta:.0e}_zc{zeta_c_best:.3f}",
            category="pbh",
            use_old_bounds=use_old_bounds,
            smooth_bounds=smooth_bounds,
        )

    elapsed = time.time() - t0
    print(f"\nPipeline done in {elapsed:.1f}s" + (" (cached)" if cached else ""))

    return {
        "k_phys": k_phys,
        "P_S": P_S,
        "n_s": n_s,
        "ns_meta": ns_meta,
        "k_pivot": pivot_k,
        "ns_window": None,
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

    pre, _ = p.parse_known_args()
    if pre.config:
        from scripts.config_loader import load_config
        p.set_defaults(**load_config(pre.config))
    args = p.parse_args()

    from scripts.constants import ROOT_DIR
    import os, json, shutil
    from datetime import datetime

    result = run_full_pbh_pipeline(
        chi0=args.chi0, y0=args.y0, N_star=args.N_star,
        beta=args.beta, xc=args.xc, c=args.c,
        workers=args.workers, zeta_c_vals=args.zeta_c,
        plot=not args.no_plot, force=args.force,
        k_pivot=args.k_pivot,
    )

    target_dir = os.path.join(ROOT_DIR, args.output_dir)
    os.makedirs(target_dir, exist_ok=True)

    ns = result["n_s"]
    ntot = result["N_total"]

    for zc_data in result["all_results"]:
        zc = zc_data["zeta_c"]
        ft = zc_data["f_total"]
        M = zc_data["M_peak"]

        print(f"ζ_c={zc:.4f}  f_total={ft:.4e}  M_peak={M:.4e}")

        if not args.no_plot and ft > 0:
            from scripts.plotting import plot_pbh_abundance
            tag = f"{args.tag}_" if args.tag else ""
            pid = f"{tag}beta{args.beta:.1e}_Nstar{args.N_star:.0f}_zcz{zc:.4f}"
            plot_pbh_abundance(
                zc_data["M"], zc_data["f_pbh"],
                zeta_c=zc, gamma=0.4,
                model_label=f"β={args.beta:.1e}, N*={args.N_star:.0f}, ζ_c={zc:.4f} (f={ft:.3e})",
                filename=pid, category="pbh",
            )
            src = os.path.join(ROOT_DIR, "outputs/plots/pbh", pid + ".png")
            dst = os.path.join(target_dir, pid + ".png")
            if os.path.exists(src):
                shutil.move(src, dst)
                comp = {
                    "plot_file": pid + ".png",
                    "generated": datetime.now().isoformat(),
                    "pipeline": "full_pbh_pipeline CLI",
                    "config": {
                        "model": "EzquiagaCHIModel",
                        "x_c": args.xc, "c": args.c, "beta": args.beta,
                        "chi0": args.chi0, "y0": args.y0, "N_star": args.N_star,
                        "k_pivot": args.k_pivot, "ns_window": args.ns_window,
                    },
                    "background": {"N_total": ntot, "n_s": ns},
                    "pbh": {"zeta_c": zc, "f_total": ft, "M_present_Msun": M},
                }
                with open(dst.replace(".png", ".json"), "w") as f:
                    json.dump(comp, f, indent=2, default=str)
                print(f"  → {dst}")
