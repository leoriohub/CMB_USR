"""Parameter sweep for PBH abundance in the Ezquiaga CHI model.

Sweeps (x_c, c, β) to find parameter sets matching the paper's Figure 3:
  μ ≈ 11 M_⊙, Ω_PBH^eq ≈ 0.42, P_S_peak/As ≈ 2.3×10⁴

Uses the Mukhanov-Sasaki (MS) solver (NOT SR — SR breaks during USR).
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from scipy.special import erfc

from scripts.constants import (
    k_pivot_phys, S, ROOT_DIR,
)
from scripts.plotting import (
    make_filename, save_fig, get_path, TOL, PAPER_RCPARAMS,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACCRETION = 3e7

# Paper targets
TARGET_PS_RATIO = 2.3e4   # P_S_peak / As
TARGET_As = 2.1e-9
TARGET_M_PEAK = 11.0       # Msun (present-day)
TARGET_F_TOTAL = 0.42      # Omega_PBH / Omega_DM

# Fixed cosmology
K_EQ = 0.0104
M_EQ = 3.0e17
GAMMA = 0.4

MS_N_WORKERS = 8


def model_from_params(x_c, c, beta):
    from models.ezquiaga_chi import EzquiagaCHIModel, inflection_parameters
    m = EzquiagaCHIModel(c=c)
    a, b = inflection_parameters(x_c, c, beta=beta)
    m.a = a
    m.b = b
    m.v0 = m._V0 * m.a / (m.b * m.c) ** 2
    return m


def _pbh_weighted_kgrid():
    """k-grid focused on PBH peak region, covering k = 10⁶ to 10¹³ Mpc⁻¹.
    250 modes total for adequate USR peak resolution."""
    k_cmb = np.array([1e-6, 0.002])
    k_pbh = np.logspace(6, 13, 250)
    grid = np.unique(np.concatenate([k_cmb, k_pbh]))
    return grid


def run_ms(model, chi0=8.0, y0=-1e-4):
    """Run the MS solver for a given model, return k_phys, P_S, N_total."""
    from pspectrum_pipeline import run_pspectrum_pipeline

    model.x0 = chi0
    model.y0 = y0
    model.patch_background_solver()

    k_grid = _pbh_weighted_kgrid()

    result = run_pspectrum_pipeline(
        model=model,
        phi0=None,
        y0=None,
        k_phys_grid=k_grid,
        k_pivot_phys=0.002,
        N_star=65,
        k_start_factor=100.0,
        ms_steps=5000,
        normalize_to_As=False,
        save_outputs=False,
        n_workers=MS_N_WORKERS,
    )

    k = np.array(result["k_phys"])
    ps = np.array(result["P_S"])
    N_total = float(result["metadata"]["N_total"])

    return k, ps, N_total, result


def find_ps_peak(k_phys, P_S):
    """Find the USR peak in P_S(k). The MS grid starts at k ≥ 1e5 Mpc⁻¹,
    so the only peak in this range is the USR peak."""
    valid = np.isfinite(P_S) & (P_S > 1e-10) & np.isfinite(k_phys) & (k_phys > 0)
    k, ps = k_phys[valid], P_S[valid]
    if len(k) < 5:
        return None
    peak_i = int(np.argmax(ps))
    return {
        "k_peak": float(k[peak_i]),
        "P_S_peak": float(ps[peak_i]),
    }


def compute_pbh_metrics(k_phys, P_S, zeta_c=0.052):
    """Compute PBH abundance for a given P_S(k) spectrum."""
    bf = erfc(zeta_c / np.sqrt(2 * np.clip(P_S, 1e-300, None)))
    beq = np.clip(bf * k_phys / K_EQ, 0.0, 1.0)
    M = GAMMA * M_EQ * (K_EQ / k_phys) ** 2 * ACCRETION

    ok = np.isfinite(M) & np.isfinite(beq) & (M > 0)
    M, beq = M[ok], beq[ok]
    order = np.argsort(M)
    M, beq = M[order], beq[order]

    f_total = np.trapezoid(beq, np.log(np.maximum(M, 1e-300)))
    peak_i = int(np.argmax(beq))
    M_peak = float(M[peak_i]) if len(M) > 0 else np.nan

    return {
        "f_total": float(f_total),
        "M_peak": M_peak,
        "M": M,
        "f_pbh": beq,
    }


def score_config(params, ps_peak, ps_ratio, m_peak, f_total):
    """Score a config by distance from paper targets. Lower is better."""
    w_ps = 1.0
    w_m = 1.0
    w_f = 1.0
    s_ps = (ps_ratio / TARGET_PS_RATIO - 1) ** 2 * w_ps if ps_ratio > 0 else 100
    s_m = (np.log10(max(m_peak, 1e-10) / TARGET_M_PEAK)) ** 2 * w_m
    s_f = (np.log10(max(f_total, 1e-10) / TARGET_F_TOTAL)) ** 2 * w_f
    return s_ps + s_m + s_f


def run_sweep(xc_vals, c_vals, beta_vals, chi0, y0, zeta_c, progress_fn=None):
    """Run the full sweep over parameter grid."""
    results = []
    total = len(xc_vals) * len(c_vals) * len(beta_vals)
    idx = 0
    for xc in xc_vals:
        for c in c_vals:
            for beta in beta_vals:
                idx += 1
                if progress_fn:
                    progress_fn(idx, total, xc, c, beta)
                try:
                    m = model_from_params(xc, c, beta)
                    k_m, ps_ms, N_total, pipe_result = run_ms(m, chi0, y0)
                    if N_total < 65:
                        continue
                    peak = find_ps_peak(k_m, ps_ms)
                    if peak is None:
                        continue
                    pbh = compute_pbh_metrics(k_m, ps_ms, zeta_c)
                    ps_ratio = peak["P_S_peak"] / TARGET_As
                    sc = score_config(
                        (xc, c, beta),
                        peak["P_S_peak"], ps_ratio,
                        pbh["M_peak"], pbh["f_total"],
                    )
                    result = {
                        "x_c": xc, "c": c, "beta": beta,
                        "N_total": N_total,
                        "k_peak": peak["k_peak"],
                        "P_S_peak": peak["P_S_peak"],
                        "P_S_peak_ratio": ps_ratio,
                        "M_form": GAMMA * M_EQ * (K_EQ / peak["k_peak"]) ** 2
                        if peak["k_peak"] > 0 else 0,
                        "M_present": pbh["M_peak"],
                        "f_total": pbh["f_total"],
                        "score": sc,
                        "k_phys": k_m.tolist(),
                        "P_S": ps_ms.tolist(),
                    }
                    results.append(result)
                except Exception as e:
                    results.append({
                        "x_c": xc, "c": c, "beta": beta,
                        "score": 1e6, "error": str(e),
                    })
    return results


def write_log(results, log_path):
    """Write results to a JSONL log."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        for r in results:
            # Strip large arrays for log
            entry = {k: v for k, v in r.items()
                     if k not in ("k_phys", "P_S")}
            json.dump(entry, f)
            f.write("\n")


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

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(filtered)))
        for i, r in enumerate(filtered):
            if "k_phys" not in r or "P_S" not in r:
                continue
            k = np.array(r["k_phys"])
            ps = np.array(r["P_S"])
            val = r[param]
            label_str = f"{param}={val:.4f}"
            ax1.loglog(k, ps, "-", color=colors[i], lw=1.2, label=label_str)

        ax1.axhline(TARGET_As * TARGET_PS_RATIO, color=TOL["grey"], ls="--", lw=0.8,
                    label=f"paper peak ({TARGET_PS_RATIO:.1e})")
        ax1.set_xlabel(r"$k$ [Mpc$^{-1}$]")
        ax1.set_ylabel(r"$\mathcal{P}_{\mathcal{R}}(k)$")
        ax1.legend(fontsize=6, loc="upper left")
        ax1.set_xlim(1e5, 1e14)
        ax1.set_ylim(1e-10, 1e-2)
        ax1.grid(True, alpha=0.2, which="both")

        # Legend for fixed params
        fixed_str = ", ".join(f"{k}={v}" for k, v in fixed.items())
        ax1.text(0.02, 0.02, f"Fixed: {fixed_str}",
                 transform=ax1.transAxes, fontsize=7, alpha=0.7)

        ax2.axis("off")

        fig.tight_layout()
        save_fig(fig, filename, category)


def plot_heatmap_2d(results, x_param, y_param, z_param,
                    x_label, y_label, z_label, filename, category):
    """2D heatmap of a metric over two parameters."""
    x_vals = sorted(set(r[x_param] for r in results))
    y_vals = sorted(set(r[y_param] for r in results))
    z_grid = np.full((len(y_vals), len(x_vals)), np.nan)

    for r in results:
        xi = x_vals.index(r[x_param])
        yi = y_vals.index(r[y_param])
        z_grid[yi, xi] = r.get(z_param, np.nan)

    with plt.rc_context(PAPER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        im = ax.pcolormesh(x_vals, y_vals, z_grid, shading="auto",
                           cmap="viridis", norm=matplotlib.colors.LogNorm())
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(z_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.tight_layout()
        save_fig(fig, filename, category)


def plot_best_configs(results, n_best=3, filename="pbh_best_configs", category="pbh"):
    """Plot f_PBH(M) for the top-N scoring configs."""
    scored = [r for r in results if r.get("score", 1e6) < 1e5]
    scored.sort(key=lambda r: r["score"])
    best = scored[:n_best]
    if len(best) < 1:
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
            lbl = (f"x_c={r['x_c']:.4f}, c={r['c']:.3f}, "
                   f"β={r['beta']:.1e}")
            if len(pbh["M"]) > 5:
                ax.loglog(pbh["M"], pbh["f_pbh"], "-", color=colors[i % 3],
                          lw=1.5, label=lbl)

        ax.axhline(1.0, color=TOL["grey"], ls=":", lw=0.8)
        ax.set_xlabel(r"$M / M_\odot$")
        ax.set_ylabel(r"$\Omega_{\mathrm{PBH}} / \Omega_{\mathrm{DM}}$")
        ax.set_xlim(1e-4, 1e4)
        ax.set_ylim(1e-4, 1.0)
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(True, alpha=0.2, which="both")
        fig.tight_layout()
        save_fig(fig, filename, category)


def print_summary(results, n_show=15):
    """Print a formatted summary table."""
    scored = [r for r in results if r.get("score", 1e6) < 1e5]
    scored.sort(key=lambda r: r["score"])

    print(f"\n{'Rank':<5} {'x_c':<8} {'c':<8} {'beta':<9} "
          f"{'N_total':<8} {'k_peak':<10} {'P_S_peak':<10} "
          f"{'PS_ratio':<10} {'M_pres':<8} {'f_total':<9} {'score':<8}")
    print("-" * 100)
    for i, r in enumerate(scored[:n_show]):
        print(f"{i + 1:<5} {r['x_c']:<8.4f} {r['c']:<8.4f} {r['beta']:<9.1e} "
              f"{r['N_total']:<8.1f} {r['k_peak']:<10.3e} {r['P_S_peak']:<10.4e} "
              f"{r['P_S_peak_ratio']:<10.2e} {r['M_present']:<8.2f} "
              f"{r['f_total']:<9.4e} {r['score']:<8.2f}")


def main():
    p = argparse.ArgumentParser(
        description="PBH parameter sweep for Ezquiaga CHI")
    p.add_argument("--x_c-lo", type=float, default=0.776)
    p.add_argument("--x_c-hi", type=float, default=0.792)
    p.add_argument("--n-xc", type=int, default=9)
    p.add_argument("--c-lo", type=float, default=0.77)
    p.add_argument("--c-hi", type=float, default=0.77)
    p.add_argument("--n-c", type=int, default=1)
    p.add_argument("--beta-vals", type=float, nargs="+",
                   default=[1e-5, 3e-5, 1e-4])
    p.add_argument("--chi0", type=float, default=8.0)
    p.add_argument("--y0", type=float, default=-1e-4)
    p.add_argument("--zeta-c", type=float, default=0.052)
    p.add_argument("--output-dir", default="outputs/plots/pbh")
    p.add_argument("--log", default="outputs/simulations/logs/pbh_sweep.jsonl")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    xc_vals = np.linspace(args.x_c_lo, args.x_c_hi, args.n_xc)
    c_vals = np.linspace(args.c_lo, args.c_hi, args.n_c)
    beta_vals = args.beta_vals

    total = len(xc_vals) * len(c_vals) * len(beta_vals)
    print(f"Sweep: {len(xc_vals)} x_c × {len(c_vals)} c × "
          f"{len(beta_vals)} β = {total} configs")

    t0 = time.time()

    def progress(i, n, xc, c, beta):
        elapsed = time.time() - t0
        rate = i / max(elapsed, 0.01)
        eta = (n - i) / rate
        sys.stdout.write(
            f"\r  [{i}/{n}] x_c={xc:.4f} c={c:.3f} β={beta:.1e}  "
            f"[{elapsed:.0f}s<{eta:.0f}s, {rate:.1f}cfg/s]  ")
        sys.stdout.flush()

    results = run_sweep(xc_vals, c_vals, beta_vals,
                        args.chi0, args.y0, args.zeta_c,
                        progress_fn=progress)
    print(f"\n  Done in {time.time() - t0:.1f}s")

    # Write log
    write_log(results, args.log)
    print(f"  Log: {args.log}")

    # Print summary
    print_summary(results)

    # Plots
    if not args.no_plot:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)

        plot_sweep_1d(results, "x_c", {"c": 0.77, "beta": 1e-5},
                       "x_c", "sweep_xc_PS", "pbh")
        plot_sweep_1d(results, "c", {"x_c": 0.784, "beta": 1e-5},
                       "c", "sweep_c_PS", "pbh")
        plot_sweep_1d(results, "beta", {"x_c": 0.784, "c": 0.77},
                       "β", "sweep_beta_PS", "pbh")

        plot_heatmap_2d(results, "x_c", "c", "M_present",
                        r"$x_c$", r"$c$", r"$M_{\mathrm{peak}} [M_\odot]$",
                        "sweep_xc_vs_c_peakM", "pbh")
        plot_heatmap_2d(results, "x_c", "c", "P_S_peak_ratio",
                        r"$x_c$", r"$c$", r"$P_{\mathcal{R}}^{\mathrm{peak}} / A_s$",
                        "sweep_xc_vs_c_PSratio", "pbh")

        plot_best_configs(results, n_best=3)

    # Print top-3 for MS validation
    scored = [r for r in results if r.get("score", 1e6) < 1e5]
    scored.sort(key=lambda r: r["score"])
    print("\nTop-3 configs for MS validation:")
    for i, r in enumerate(scored[:3]):
        print(f"  {i + 1}. pbh_run --x_c {r['x_c']} --c {r['c']} "
              f"--beta {r['beta']:.1e}  (score={r['score']:.2f})")


if __name__ == "__main__":
    main()
