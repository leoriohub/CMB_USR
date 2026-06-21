"""Parameter sweep for PBH mass-range search in the Ezquiaga CHI model.

Sweeps (x_c, c, β, χ₀) to find parameter sets with P_S(k) peaks in target
observable mass gaps (sub-solar [10⁻⁶, 10⁻²] M_⊙, asteroid [10⁻¹⁷, 10⁻¹⁵] M_⊙).

Records n_s at CMB scales for Planck compatibility screening.
Uses MS solver (NOT SR — SR breaks during USR).
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from scipy.special import erfc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scripts.plotting import save_fig, TOL, PAPER_RCPARAMS

ACCRETION = 3e7

# Paper targets
TARGET_PS_RATIO = 2.3e4  # P_S_peak / As
TARGET_As = 2.1e-9
TARGET_M_PEAK = 11.0  # Msun (present-day)
TARGET_F_TOTAL = 0.42  # Omega_PBH / Omega_DM

# Fixed cosmology
K_EQ = 0.0104
M_EQ = 3.0e17
GAMMA = 0.4

MS_N_WORKERS = 8  # overridden by --workers CLI arg


def model_from_params(x_c, c, beta):
    from models.ezquiaga_chi import EzquiagaCHIModel, inflection_parameters

    m = EzquiagaCHIModel(c=c)
    a, b = inflection_parameters(x_c, c, beta=beta)
    m.a = a
    m.b = b
    m.v0 = m._V0 * m.a / (m.b * m.c) ** 2
    return m


def _pbh_weighted_kgrid(target="subsolar"):
    """k-grid targeting a specific PBH mass range, plus CMB-scale n_s fitting.

    Parameters
    ----------
    target : str
        "subsolar" — k ∈ [10⁶, 10¹³] (default, current behavior)
        "asteroid" — k ∈ [10¹², 10¹⁸]
        "all"      — k ∈ [10⁶, 10¹⁸]
    Returns
    -------
    grid : ndarray  sorted, unique k-values [Mpc⁻¹]
    """
    # CMB-scale points for n_s computation at k=0.05
    k_cmb_ns = np.logspace(np.log10(0.005), np.log10(0.5), 15)

    if target == "asteroid":
        k_pbh = np.logspace(12, 22, 350)
    elif target == "all":
        k_pbh = np.logspace(6, 22, 400)
    else:  # "subsolar"
        k_pbh = np.logspace(6, 13, 250)

    k_min_val = k_pbh[0] if len(k_pbh) > 0 else 1e6
    k_max_val = k_pbh[-1] if len(k_pbh) > 0 else 1e13
    grid = np.unique(np.concatenate([k_cmb_ns, k_pbh]))
    return grid, k_min_val, k_max_val


def compute_ns(
    k_phys, P_S, k_pivot=0.05, window=3.0
):  # always 0.05 (Planck) for Ezquiaga
    """Compute n_s and interpolated A_s at k_pivot from MS P_S(k).

    Fits ln(P_S) vs ln(k) over k ∈ [k_pivot/window, k_pivot*window].
    Returns (n_s, A_s_at_pivot) or (None, None) if too few points.
    """
    lo = k_pivot / window
    hi = k_pivot * window
    idx = (k_phys >= lo) & (k_phys <= hi)
    k_fit = k_phys[idx]
    ps_fit = P_S[idx]
    if len(k_fit) < 3:
        return None, None
    valid = np.isfinite(ps_fit) & (ps_fit > 1e-30)
    k_fit, ps_fit = k_fit[valid], ps_fit[valid]
    if len(k_fit) < 3:
        return None, None
    coeffs = np.polyfit(np.log(k_fit), np.log(ps_fit), 1)
    n_s = float(coeffs[0] + 1)
    A_s = float(np.exp(np.interp(np.log(k_pivot), np.log(k_fit), np.log(ps_fit))))
    return n_s, A_s


def classify_mass_range(M_peak):
    """Classify PBH peak present-day mass into an observable range bin."""
    if np.isnan(M_peak) or M_peak <= 0:
        return "unknown"
    if M_peak < 1e-17:
        return "too_light"
    if M_peak < 1e-15:
        return "asteroid_gap"
    if M_peak < 1e-6:
        return "intermediate"
    if M_peak < 1e-2:
        return "sub_solar_gap"
    if M_peak < 1:
        return "sub_stellar"
    if M_peak < 100:
        return "stellar_ligo_ruled_out"
    return "massive"


def run_ms(model, chi0=8.0, y0=-1e-4, target="subsolar", pivot_k=0.05, N_star=65):
    """Run the MS solver for a given model, return k_phys, P_S, N_total."""
    from pspectrum_pipeline import run_pspectrum_pipeline

    model.x0 = chi0
    model.y0 = y0
    model.patch_background_solver()

    k_grid, k_min_val, k_max_val = _pbh_weighted_kgrid(target)

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
        save_outputs=False,
        n_workers=MS_N_WORKERS,
    )

    k = np.array(result["k_phys"])
    ps = np.array(result["P_S"])
    N_total = float(result["metadata"]["N_total"])

    return k, ps, N_total, result, k_min_val, k_max_val


def find_pbh_peak(k_phys, P_S, k_min=None, k_max=None):
    """Find the USR peak in P_S(k), ignoring CMB-scale k < 1 Mpc⁻¹.

    Parameters
    ----------
    k_min, k_max : float or None
        Grid bounds for flagging peaks at the grid edge.
        Sets on_grid_boundary=True if k_peak is within 1% of a bound.
    """
    valid = np.isfinite(P_S) & (P_S > 1e-10) & np.isfinite(k_phys) & (k_phys > 1.0)
    k, ps = k_phys[valid], P_S[valid]
    if len(k) < 5:
        return None
    peak_i = int(np.argmax(ps))
    k_peak = float(k[peak_i])
    on_boundary = False
    if k_min is not None and k_max is not None:
        on_boundary = bool(k_peak <= k_min * 1.01 or k_peak >= k_max * 0.99)
    return {
        "k_peak": k_peak,
        "P_S_peak": float(ps[peak_i]),
        "on_grid_boundary": on_boundary,
    }


def compute_pbh_metrics(k_phys, P_S, zeta_c=0.052):
    """Compute PBH abundance for a given P_S(k) spectrum.

    f_total = Ω_PBH/Ω_DM can exceed 1 if zeta_c is too low for the P_S amplitude.
    """
    bf = erfc(zeta_c / np.sqrt(2 * np.clip(P_S, 1e-300, None)))
    beq = np.clip(bf * k_phys / K_EQ, 0.0, 1.0)
    M = GAMMA * M_EQ * (K_EQ / k_phys) ** 2 * ACCRETION

    ok = np.isfinite(M) & np.isfinite(beq) & (M > 0)
    M, beq = M[ok], beq[ok]
    order = np.argsort(M)
    M, beq = M[order], beq[order]

    f_total = float(np.trapezoid(beq, np.log(np.maximum(M, 1e-300))))
    peak_i = int(np.argmax(beq))
    M_peak = float(M[peak_i]) if len(M) > 0 else np.nan

    return {
        "f_total": float(f_total),
        "M_peak": M_peak,
        "M": M,
        "f_pbh": beq,
    }


def run_sweep(
    xc_vals,
    c_vals,
    beta_vals,
    chi0_vals,
    y0,
    zeta_c_vals,
    target="subsolar",
    N_total_min=65,
    pivot_k=0.002,
    N_star_vals=None,
    progress_fn=None,
    log_path=None,
    resume=False,
    f_max=1.0,
):
    """Run the full sweep over parameter grid.

    Parameters
    ----------
    target : str
        k-grid target ("subsolar", "asteroid", "all")
    N_total_min : float
        Minimum N_total to accept (configs with fewer e-folds are skipped)
    chi0_vals : array-like
        Initial field values to sweep
    N_star_vals : array-like or None
        N_star values to sweep (default: [65])
    """
    if N_star_vals is None:
        N_star_vals = [65]

    # Resume: load already-computed configs from existing log
    done_set = set()
    if resume and log_path and os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    k = (
                        entry["x_c"],
                        entry["c"],
                        entry["beta"],
                        entry["chi0"],
                        entry["N_star"],
                        entry.get("zeta_c", None),
                    )
                    done_set.add(k)
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"  Resuming: {len(done_set)} configs already in {log_path}")

    results = []
    n_unique = (
        len(xc_vals)
        * len(c_vals)
        * len(beta_vals)
        * len(chi0_vals)
        * len(N_star_vals)
    )
    total = n_unique * len(zeta_c_vals)

    def _write_log_entry(entry):
        """Write stripped entry to incremental JSONL log (opens/closes per write)."""
        if not log_path:
            return
        log_entry = {k: v for k, v in entry.items() if k not in ("k_phys", "P_S")}
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")
            f.flush()

    idx = 0
    skipped = 0
    for xc in xc_vals:
        for c in c_vals:
            for beta in beta_vals:
                for chi0 in chi0_vals:
                    for N_star in N_star_vals:
                        # MS solver runs ONCE per (xc, c, beta, chi0, N_star)
                        ms_done_key = (xc, c, beta, chi0, N_star)
                        ms_completed = any(
                            k[:5] == ms_done_key for k in done_set
                        )
                        if ms_completed:
                            # All zeta_c variants already done for this config
                            zeta_in_done = sum(
                                1 for k in done_set if k[:5] == ms_done_key
                            )
                            skipped += zeta_in_done
                            continue

                        idx += 1
                        n_remaining = total - len(done_set)
                        n_done = idx - skipped
                        if progress_fn:
                            progress_fn(
                                n_done, n_remaining, xc, c, beta, chi0, N_star, None
                            )

                        try:
                            m = model_from_params(xc, c, beta)
                            (
                                k_m,
                                ps_ms,
                                N_total,
                                pipe_result,
                                k_min_val,
                                k_max_val,
                            ) = run_ms(m, chi0, y0, target, pivot_k, N_star)
                            if N_total < N_total_min:
                                continue
                            peak = find_pbh_peak(k_m, ps_ms, k_min_val, k_max_val)
                            if peak is None:
                                continue
                            n_s, A_s_at_cmb = compute_ns(k_m, ps_ms)
                            M_kpeak = (
                                GAMMA
                                * M_EQ
                                * (K_EQ / peak["k_peak"]) ** 2
                                * ACCRETION
                            )
                            mass_bin = classify_mass_range(M_kpeak)
                            ps_ratio = peak["P_S_peak"] / TARGET_As
                            base_result = {
                                "x_c": xc,
                                "c": c,
                                "beta": beta,
                                "chi0": chi0,
                                "N_star": N_star,
                                "N_total": N_total,
                                "k_peak": peak["k_peak"],
                                "P_S_peak": peak["P_S_peak"],
                                "P_S_peak_ratio": ps_ratio,
                                "M_form": GAMMA
                                * M_EQ
                                * (K_EQ / peak["k_peak"]) ** 2
                                if peak["k_peak"] > 0
                                else 0,
                                "M_kpeak": M_kpeak,
                                "on_grid_boundary": peak.get(
                                    "on_grid_boundary", False
                                ),
                                "mass_bin": mass_bin,
                                "n_s": n_s,
                                "A_s_at_cmb": A_s_at_cmb,
                                "k_phys": k_m.tolist(),
                                "P_S": ps_ms.tolist(),
                            }

                            # zeta_c loop: only cheap PBH metrics
                            for zeta_c in zeta_c_vals:
                                zc_key = (xc, c, beta, chi0, N_star, zeta_c)
                                if zc_key in done_set:
                                    continue
                                pbh = compute_pbh_metrics(k_m, ps_ms, zeta_c)
                                result = dict(base_result)
                                result["zeta_c"] = zeta_c
                                result["f_total"] = pbh["f_total"]
                                result["M_present"] = pbh["M_peak"]
                                results.append(result)
                                _write_log_entry(result)

                        except Exception as e:
                            for zeta_c in zeta_c_vals:
                                zc_key = (xc, c, beta, chi0, N_star, zeta_c)
                                if zc_key in done_set:
                                    continue
                                error_entry = {
                                    "x_c": xc,
                                    "c": c,
                                    "beta": beta,
                                    "chi0": chi0,
                                    "N_star": N_star,
                                    "zeta_c": zeta_c,
                                    "error": str(e),
                                }
                                results.append(error_entry)
                                _write_log_entry(error_entry)

    if skipped > 0:
        print(f"  Skipped {skipped} already-computed configs")

    # Filter by f_max (but keep error entries)
    n_before = len(results)
    results = [r for r in results if "error" in r or r.get("f_total", 0) <= f_max]
    n_filtered = n_before - len(results)
    if n_filtered > 0:
        print(f"  Filtered {n_filtered} configs with f_total > {f_max}")

    return results


def plot_sweep_1d(results, param, fixed, param_label, filename, category):
    """Overlay P_S(k) or f_PBH(M) for 1D parameter variation."""
    # Filter to fixed values
    filtered = []
    for r in results:
        match = True
        for k, v in fixed.items():
            if abs(r.get(k, -1) - v) > 1e-6:
                match = False
                break
        if match:
            filtered.append(r)

    if len(filtered) < 2:
        print(f"  Not enough results for {param} sweep")
        return

    # Sort by param value
    filtered.sort(key=lambda r: r[param])

    with plt.rc_context(PAPER_RCPARAMS):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

        colors = plt.colormaps["viridis"](np.linspace(0.2, 0.9, len(filtered)))
        for i, r in enumerate(filtered):
            if "k_phys" not in r or "P_S" not in r:
                continue
            k = np.array(r["k_phys"])
            ps = np.array(r["P_S"])
            val = r[param]
            label_str = f"{param}={val:.4f}"
            ax1.loglog(k, ps, "-", color=colors[i], lw=1.2, label=label_str)

        ax1.axhline(
            TARGET_As * TARGET_PS_RATIO,
            color=TOL["grey"],
            ls="--",
            lw=0.8,
            label=f"paper peak ({TARGET_PS_RATIO:.1e})",
        )
        ax1.set_xlabel(r"$k$ [Mpc$^{-1}$]")
        ax1.set_ylabel(r"$\mathcal{P}_{\mathcal{R}}(k)$")
        ax1.legend(fontsize=6, loc="upper left")
        ax1.set_xlim(1e5, 1e14)
        ax1.set_ylim(1e-10, 1e-2)
        ax1.grid(True, alpha=0.2, which="both")

        # Legend for fixed params
        fixed_str = ", ".join(f"{k}={v}" for k, v in fixed.items())
        ax1.text(
            0.02,
            0.02,
            f"Fixed: {fixed_str}",
            transform=ax1.transAxes,
            fontsize=7,
            alpha=0.7,
        )

        ax2.axis("off")

        fig.tight_layout()
        save_fig(fig, filename, category)


def plot_heatmap_2d(
    results, x_param, y_param, z_param, x_label, y_label, z_label, filename, category
):
    """2D heatmap of a metric over two parameters."""
    x_vals = sorted({r[x_param] for r in results})
    y_vals = sorted({r[y_param] for r in results})
    z_grid = np.full((len(y_vals), len(x_vals)), np.nan)

    for r in results:
        xi = x_vals.index(r[x_param])
        yi = y_vals.index(r[y_param])
        z_grid[yi, xi] = r.get(z_param, np.nan)

    # Auto-detect norm: LogNorm only if all values are positive
    z_flat = z_grid[~np.isnan(z_grid)]
    use_log = len(z_flat) > 0 and np.all(z_flat > 0)
    norm = LogNorm() if use_log else None

    with plt.rc_context(PAPER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        im = ax.pcolormesh(
            x_vals, y_vals, z_grid, shading="auto", cmap="viridis", norm=norm
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(z_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.tight_layout()
        save_fig(fig, filename, category)


def plot_best_configs(results, n_best=3, filename="pbh_best_configs", category="pbh"):
    """Plot f_PBH(M) for the top-N configs in target mass bins."""
    target_bins = {"asteroid_gap", "sub_solar_gap"}
    filtered = [
        r for r in results if r.get("mass_bin") in target_bins and not r.get("error")
    ]
    filtered.sort(key=lambda r: r.get("f_total", 0), reverse=True)
    best = filtered[:n_best]
    if len(best) < 1:
        print("  No configs in target mass bins to plot")
        return

    with plt.rc_context(PAPER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        colors = [TOL["red"], TOL["blue"], TOL["green"]]
        for i, r in enumerate(best):
            k = np.array(r.get("k_phys", []))
            ps = np.array(r.get("P_S", []))
            if len(k) < 5 or len(ps) < 5:
                continue
            pbh = compute_pbh_metrics(k, ps, zeta_c=0.052)
            lbl = (
                f"x_c={r['x_c']:.4f}, c={r['c']:.3f}, "
                f"β={r['beta']:.1e}, {r['mass_bin']}"
            )
            if len(pbh["M"]) > 5:
                ax.loglog(
                    pbh["M"], pbh["f_pbh"], "-", color=colors[i % 3], lw=1.5, label=lbl
                )

        ax.axhline(1.0, color=TOL["grey"], ls=":", lw=0.8)
        ax.set_xlabel(r"$M / M_\odot$")
        ax.set_ylabel(r"$\Omega_{\mathrm{PBH}} / \Omega_{\mathrm{DM}}$")
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(True, alpha=0.2, which="both")
        fig.tight_layout()
        save_fig(fig, filename, category)


def print_summary(results, n_show=20):
    """Print a formatted summary table, grouped by target mass bins."""
    target_bins = {"asteroid_gap", "sub_solar_gap"}
    # Target bins first, sorted by f_total descending
    target = [
        r for r in results if r.get("mass_bin") in target_bins and not r.get("error")
    ]
    target.sort(key=lambda r: r.get("f_total", 0), reverse=True)
    # Other bins (for reference)
    other = [
        r
        for r in results
        if r.get("mass_bin") not in target_bins and not r.get("error")
    ]
    other.sort(key=lambda r: r.get("f_total", 0), reverse=True)
    combined = target[:n_show] + other[: max(0, n_show - len(target))]

    print(
        f"\n{'Rank':<5} {'x_c':<8} {'c':<8} {'beta':<9} "
        f"{'N_total':<8} {'k_peak':<10} {'P_S_peak':<10} "
        f"{'M_kpeak':<10} {'f_total':<9} {'flg':<4} {'mass_bin':<22} {'n_s':<8}"
    )
    print("-" * 115)
    for i, r in enumerate(combined[:n_show]):
        ns_str = f"{r.get('n_s', 'N/A'):.4f}" if r.get("n_s") is not None else "N/A"
        mk = r.get("M_kpeak", r.get("M_present", 0))
        flag = ""
        if r.get("on_grid_boundary"):
            flag = "B"
        print(
            f"{i + 1:<5} {r['x_c']:<8.4f} {r['c']:<8.4f} {r['beta']:<9.1e} "
            f"{r['N_total']:<8.1f} {r['k_peak']:<10.3e} {r['P_S_peak']:<10.4e} "
            f"{mk:<10.4e} {r['f_total']:<9.4e} "
            f"{flag:<4} {r['mass_bin']:<22} {ns_str:<8}"
        )


def main():
    p = argparse.ArgumentParser(description="PBH mass-range sweep for Ezquiaga CHI")
    p.add_argument("--x_c-lo", type=float, default=0.776)
    p.add_argument("--x_c-hi", type=float, default=0.792)
    p.add_argument("--n-xc", type=int, default=9)
    p.add_argument("--c-lo", type=float, default=0.77)
    p.add_argument("--c-hi", type=float, default=0.77)
    p.add_argument("--n-c", type=int, default=1)
    p.add_argument("--beta-vals", type=float, nargs="+", default=[1e-5, 3e-5, 1e-4])
    p.add_argument(
        "--chi0-vals",
        type=float,
        nargs="+",
        default=[8.0],
        help="Initial field values to sweep (e.g., 4.0 5.0 6.0)",
    )
    p.add_argument(
        "--N-star-vals",
        type=float,
        nargs="+",
        default=[65.0],
        help="N_star values to sweep (e.g., 50 60 70)",
    )
    p.add_argument("--y0", type=float, default=-1e-4)
    p.add_argument(
        "--zeta-c-vals",
        type=float,
        nargs="+",
        default=[0.077],
        help="Collapse threshold values to sweep (default: 0.077)",
    )
    p.add_argument(
        "--target",
        choices=["subsolar", "asteroid", "all"],
        default="subsolar",
        help="k-grid target mass range",
    )
    p.add_argument(
        "--N-total-min",
        type=float,
        default=65.0,
        help="Minimum N_total to accept a config",
    )
    p.add_argument(
        "--pivot-k",
        type=float,
        default=0.05,
        help="MS solver pivot k_pivot_phys (Planck default 0.05; Higgs low-ell 0.002)",
    )
    p.add_argument(
        "--workers", type=int, default=8, help="MS solver parallel workers per config"
    )
    p.add_argument("--output-dir", default="outputs/plots/pbh")
    p.add_argument("--log", default="outputs/simulations/logs/pbh_sweep.jsonl")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON config file (CLI args override file values)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip configs already in the log file",
    )
    p.add_argument(
        "--f-max",
        type=float,
        default=1.0,
        help="Exclude configs with f_total > this threshold (default: 1.0)",
    )
    # Pre-parse to check for --config (needed before full parse)
    pre_args, _ = p.parse_known_args()
    if pre_args.config:
        with open(pre_args.config) as f:
            p.set_defaults(**json.load(f))

    args = p.parse_args()

    global MS_N_WORKERS
    MS_N_WORKERS = args.workers

    xc_vals = np.linspace(args.x_c_lo, args.x_c_hi, args.n_xc)
    c_vals = np.linspace(args.c_lo, args.c_hi, args.n_c)
    beta_vals = args.beta_vals
    chi0_vals = args.chi0_vals
    N_star_vals = args.N_star_vals

    zeta_c_vals = args.zeta_c_vals
    total = (
        len(xc_vals)
        * len(c_vals)
        * len(beta_vals)
        * len(chi0_vals)
        * len(N_star_vals)
        * len(zeta_c_vals)
    )
    n_unique = (
        len(xc_vals)
        * len(c_vals)
        * len(beta_vals)
        * len(chi0_vals)
        * len(N_star_vals)
    )
    total_entries = n_unique * len(zeta_c_vals)
    print(
        f"Sweep: {len(xc_vals)} x_c × {len(c_vals)} c × "
        f"{len(beta_vals)} β × {len(chi0_vals)} χ₀ × "
        f"{len(N_star_vals)} N* × {len(zeta_c_vals)} ζ_c = "
        f"{n_unique} MS solves → {total_entries} entries  "
        f"target={args.target}  pivot_k={args.pivot_k}  "
        f"N_total_min={args.N_total_min}  workers={args.workers}"
    )

    t0 = time.time()

    def progress(i, n, xc, c, beta, chi0, N_star, zeta_c=None):
        elapsed = time.time() - t0
        rate = i / max(elapsed, 0.01)
        eta = (n - i) / rate
        zc_str = f"ζ_c={zeta_c:.4f}" if zeta_c is not None else "ζ_c=*"
        sys.stdout.write(
            f"\r  [{i}/{n}] x_c={xc:.4f} c={c:.3f} β={beta:.1e} χ₀={chi0:.1f} "
            f"N*={N_star:.0f} {zc_str}  "
            f"[{elapsed:.0f}s<{eta:.0f}s, {rate:.1f}cfg/s]  "
        )
        sys.stdout.flush()

    results = run_sweep(
        xc_vals,
        c_vals,
        beta_vals,
        chi0_vals,
        args.y0,
        zeta_c_vals=args.zeta_c_vals,
        target=args.target,
        N_total_min=args.N_total_min,
        pivot_k=args.pivot_k,
        N_star_vals=N_star_vals,
        progress_fn=progress,
        log_path=args.log,
        resume=args.resume,
        f_max=args.f_max,
    )
    print(f"\n  Done in {time.time() - t0:.1f}s")
    print(f"  Log: {args.log}")

    # Print summary
    print_summary(results)

    # Plots
    if not args.no_plot:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)

        plot_sweep_1d(
            results, "x_c", {"c": 0.77, "beta": 1e-5}, "x_c", "sweep_xc_PS", "pbh"
        )
        plot_sweep_1d(
            results, "c", {"x_c": 0.784, "beta": 1e-5}, "c", "sweep_c_PS", "pbh"
        )
        plot_sweep_1d(
            results, "beta", {"x_c": 0.784, "c": 0.77}, "β", "sweep_beta_PS", "pbh"
        )

        plot_heatmap_2d(
            results,
            "x_c",
            "c",
            "M_present",
            r"$x_c$",
            r"$c$",
            r"$M_{\mathrm{peak}} [M_\odot]$",
            "sweep_xc_vs_c_peakM",
            "pbh",
        )
        plot_heatmap_2d(
            results,
            "x_c",
            "c",
            "P_S_peak_ratio",
            r"$x_c$",
            r"$c$",
            r"$P_{\mathcal{R}}^{\mathrm{peak}} / A_s$",
            "sweep_xc_vs_c_PSratio",
            "pbh",
        )
        plot_heatmap_2d(
            results,
            "x_c",
            "c",
            "n_s",
            r"$x_c$",
            r"$c$",
            r"$n_s (k=0.05)$",
            "sweep_xc_vs_c_ns",
            "pbh",
        )

        plot_best_configs(results, n_best=3)


if __name__ == "__main__":
    main()
