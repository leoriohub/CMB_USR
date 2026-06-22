#!/usr/bin/env python3
"""Re-run top-ranked Ezquiaga configs through the full pipeline.

Reads the top configs from pbh_consolidated.json, calls run_full_pbh_pipeline()
for each, and produces:
  - Plots in outputs/plots/pbh/top_configs/ (rank-prefixed)
  - JSONL per config with P_S peak + boundary info
  - Batch summary JSON and printed table
"""

import argparse
import json
import os
import shutil
import time

import numpy as np

from scripts.constants import ROOT_DIR
from scripts.plotting import OUTPUT_DIRS
from scripts.full_pbh_pipeline import run_full_pbh_pipeline

PLOT_DIR = os.path.join(ROOT_DIR, "outputs/plots/pbh/top_configs")
LOGS_DIR = os.path.join(ROOT_DIR, "outputs/simulations/logs")


def find_ps_peak(k_phys, P_S, k_grid_lo=1e6, k_grid_hi=1e22):
    """Find the P_S peak at PBH scales (k > 1 Mpc⁻¹) and flag grid-boundary."""
    pbh_mask = k_phys > 1.0
    if not np.any(pbh_mask):
        return np.nan, np.nan, False

    ps_pbh = P_S[pbh_mask]
    k_pbh = k_phys[pbh_mask]
    peak_i = int(np.argmax(ps_pbh))
    k_peak = float(k_pbh[peak_i])
    ps_peak = float(ps_pbh[peak_i])
    on_boundary = bool(k_peak <= k_grid_lo * 1.01 or k_peak >= k_grid_hi * 0.99)
    return ps_peak, k_peak, on_boundary


def main():
    p = argparse.ArgumentParser(description="Batch PBH pipeline for top configs")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument(
        "--force", action="store_true", help="Re-run MS solver even if cached"
    )
    args = p.parse_args()

    # 1. Load consolidated ranking
    cons_path = os.path.join(
        ROOT_DIR, "outputs/simulations/pbh_results/pbh_consolidated.json"
    )
    with open(cons_path) as f:
        cons = json.load(f)

    top_configs = cons["physics_ranking"][: args.top_n]
    print(
        f"Loaded {len(cons['physics_ranking'])} ranked configs, "
        f"processing top {args.top_n}"
    )
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 2. Process each config via the pipeline
    summaries = []
    for rank, cfg in enumerate(top_configs, 1):
        chi0 = cfg["chi0"]
        beta = cfg["beta"]
        nstar = cfg["N_star"]
        xc = cfg["x_c"]
        c_val = cfg["c"]
        y0 = cfg.get("y0", -1e-4)

        print(f"\n{'=' * 60}")
        print(
            f"[{rank}/{args.top_n}] χ₀={chi0} β={beta:.0e} N*={nstar} "
            f"x_c={xc} c={c_val}"
        )

        t0 = time.time()

        try:
            pipe = run_full_pbh_pipeline(
                chi0=chi0,
                y0=y0,
                N_star=nstar,
                beta=beta,
                xc=xc,
                c=c_val,
                workers=args.workers,
                plot=not args.no_plot,
                force=args.force,
            )

            # P_S peak at PBH scales
            ps_peak, k_peak, on_boundary = find_ps_peak(pipe["k_phys"], pipe["P_S"])
            boundary_flag = " BOUNDARY" if on_boundary else ""
            print(f"  P_S_peak={ps_peak:.4e} at k={k_peak:.3e}{boundary_flag}")

            # JSONL per config
            log_path = os.path.join(
                LOGS_DIR,
                f"top_config_rank{rank:02d}_chi{chi0}_beta{beta:.0e}_Nstar{nstar:.0f}.jsonl",
            )
            os.makedirs(LOGS_DIR, exist_ok=True)
            with open(log_path, "w") as f:
                for r in pipe["all_results"]:
                    entry = {
                        "chi0": chi0,
                        "beta": beta,
                        "x_c": xc,
                        "c": c_val,
                        "N_star": nstar,
                        "N_total": pipe["N_total"],
                        "n_s": pipe["n_s"],
                        "A_s_at_cmb": pipe["A_s_cmb"],
                        "P_S_peak": ps_peak,
                        "k_peak": k_peak,
                        "on_grid_boundary": on_boundary,
                        "zeta_c": r["zeta_c"],
                        "f_total": r["f_total"],
                        "M_peak_Msun": r["M_peak"],
                        "source": "top_config_batch",
                    }
                    json.dump(entry, f)
                    f.write("\n")

            # Move plot to top_configs/ subdirectory
            if not args.no_plot and pipe["zeta_c_best"] is not None:
                plot_id = f"rank{rank:02d}_chi{chi0}_beta{beta:.0e}_nstar{nstar:.0f}"
                # The pipeline saves to outputs/plots/pbh/ as
                # pbh_chi{chi0}_beta{beta:.0e}_zc{zeta_c_best:.3f}.png
                src_name = (
                    f"pbh_chi{chi0}_beta{beta:.0e}_zc{pipe['zeta_c_best']:.3f}.png"
                )
                src = os.path.join(
                    OUTPUT_DIRS.get("pbh", os.path.join(ROOT_DIR, "outputs/plots/pbh")),
                    src_name,
                )
                dst = os.path.join(PLOT_DIR, plot_id + ".png")
                if os.path.exists(src):
                    shutil.move(src, dst)
                    print(f"  Plot: {dst}")
                else:
                    print(f"  WARNING: plot not found at {src}")

            elapsed = time.time() - t0
            summaries.append(
                {
                    "rank": rank,
                    "chi0": chi0,
                    "beta": beta,
                    "x_c": xc,
                    "c": c_val,
                    "N_star": nstar,
                    "N_total": pipe["N_total"],
                    "n_s": pipe["n_s"],
                    "A_s": pipe["A_s_cmb"],
                    "P_S_peak": ps_peak,
                    "k_peak": k_peak,
                    "on_grid_boundary": on_boundary,
                    "best_zeta_c": pipe["zeta_c_best"],
                    "f_total": pipe["f_total_best"],
                    "M_present": pipe["M_peak_best"],
                    "cached": pipe["cached"],
                    "elapsed_s": round(elapsed, 1),
                    "ps_path": pipe["ps_path"],
                    "log_path": log_path,
                }
            )
            print(f"  Done in {elapsed:.1f}s" + (" (cached)" if pipe["cached"] else ""))

        except Exception as e:
            import traceback

            traceback.print_exc()
            summaries.append(
                {
                    "rank": rank,
                    "chi0": chi0,
                    "beta": beta,
                    "x_c": xc,
                    "c": c_val,
                    "N_star": nstar,
                    "error": str(e),
                }
            )

    # 3. Batch summary JSON
    summary_path = os.path.join(LOGS_DIR, "top_configs_batch_summary.json")
    clean_out = list(summaries)
    with open(summary_path, "w") as f:
        json.dump(clean_out, f, indent=2, default=str)
    print(f"\nBatch summary: {summary_path}")

    # 4. Summary table
    print(f"\n{'=' * 100}")
    header = (
        f"{'Rank':<5} {'β':>10} {'N*':<5} {'Ntot':<7} {'n_s':<7} "
        f"{'P_S_peak':<12} {'M_present':<12} {'f_total':<10} "
        f"{'ζ_c':<6} {'time':<7} {'cache':<5}"
    )
    print(header)
    print("-" * 100)
    for s in summaries:
        if "error" in s:
            print(f"{s['rank']:<5} ERROR: {s['error']}")
        else:
            b_fmt = f"{s['beta']:.0e}"
            cache_mark = "✓" if s.get("cached") else " "
            print(
                f"{s['rank']:<5} {b_fmt:>10} {s['N_star']:<5.0f} "
                f"{s['N_total']:<7.1f} {s['n_s']:<7.4f} "
                f"{s['P_S_peak']:<12.4e} {s['M_present']:<12.4e} "
                f"{s['f_total']:<10.4e} {s['best_zeta_c']:<6.3f} "
                f"{s['elapsed_s']:<7.1f}s {cache_mark:<5}"
            )


if __name__ == "__main__":
    main()
