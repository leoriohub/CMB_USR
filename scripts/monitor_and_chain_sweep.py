#!/usr/bin/env python3
"""Monitor current PBH sweep, analyze results on completion, submit next sweep.

Usage on lab machine:
  nohup python scripts/monitor_and_chain_sweep.py --log outputs/simulations/logs/pbh_sweep_full.jsonl --pid-file ~/pbh_sweep_pid > ~/monitor_chain.log 2>&1 &
"""

import os
import sys
import json
import time
import argparse
import subprocess
import datetime
import numpy as np


def get_current_pid():
    """Get PID of running sweep_pbh_params job."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "sweep_pbh_params.*pbh_sweep_full"],
            stderr=subprocess.DEVNULL, timeout=10)
        pids = [int(p) for p in out.decode().strip().split('\n') if p.strip()]
        return pids[-1] if pids else None  # parent is highest PID
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def count_configs_in_log(log_path):
    """Count completed configs in JSONL log."""
    if not os.path.exists(log_path):
        return 0, []
    with open(log_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return len(lines), lines


def analyze_results(log_path):
    """Analyze completed sweep results."""
    if not os.path.exists(log_path):
        print("  NO LOG FILE YET")
        return None

    with open(log_path) as f:
        lines = [json.loads(ln) for ln in f if ln.strip()]

    if not lines:
        print("  EMPTY LOG")
        return None

    # Track counts
    total = len(lines)
    errors = [e for e in lines if 'error' in e and e.get('error')]
    ok = [o for o in lines if 'error' not in o]

    if not ok:
        print(f"  {total} entries, ALL ERRORS")
        return None

    # Find targets
    target_bins = {"asteroid_gap", "sub_solar_gap"}
    targets = [r for r in ok if r.get("mass_bin") in target_bins]
    targets.sort(key=lambda r: r.get("f_total", 0), reverse=True)

    # Range of covered params
    k_peaks = [r.get("k_peak", 0) for r in ok if r.get("k_peak", 0) > 0]
    masses = [r.get("M_kpeak", r.get("M_present", 0)) for r in ok
              if (r.get("M_kpeak", r.get("M_present", 0)) or 0) > 0]
    xc_range = (min(r["x_c"] for r in ok), max(r["x_c"] for r in ok))
    c_range = (min(r["c"] for r in ok), max(r["c"] for r in ok))

    return {
        "total": total,
        "errors": errors,
        "ok": len(ok),
        "targets_in_bins": len(targets),
        "best_targets": targets[:10],
        "k_peak_range": (min(k_peaks), max(k_peaks)) if k_peaks else (0, 0),
        "mass_range": (min(masses), max(masses)) if masses else (0, 0),
        "xc_range": xc_range,
        "c_range": c_range,
        "all_params": ok,
    }


def design_next_sweep(analysis, current_log):
    """Design next sweep parameters based on completed results.

    Goal: shift PBH mass to lower masses (higher k_peak).
    Strategy: extend toward higher x_c and higher c where trends indicate higher k.
    """
    if analysis is None or analysis["ok"] == 0:
        # No results yet — fallback: extend the current grid
        return {
            "x_c_lo": 0.80,
            "x_c_hi": 0.85,
            "n_xc": 6,
            "c_lo": 5.0,
            "c_hi": 10.0,
            "n_c": 3,
            "beta_vals": [1e-6, 1e-5, 3e-5, 1e-4, 3e-4],
            "chi0_vals": [6.0, 8.0],
            "N_star_vals": [50, 65],
        }

    params = analysis["all_params"]
    xc_max = max(r["x_c"] for r in params)
    c_max = max(r["c"] for r in params)
    k_range = analysis["k_peak_range"]
    mass_range = analysis["mass_range"]

    print(f"\n  Current k_peak range: [{k_range[0]:.2e}, {k_range[1]:.2e}] Mpc⁻¹")
    print(f"  Current mass range: [{mass_range[0]:.2e}, {mass_range[1]:.2e}] M_sun")
    print(f"  Current x_c range: up to {xc_max}, c range: up to {c_max}")

    # Check how many configs hit target bins
    targets = analysis["targets_in_bins"]
    print(f"  Configs in target mass bins: {targets}")

    # Find the highest k (lowest mass) configs with real peaks
    with_peaks = [r for r in params if r.get("k_peak", 0) > 1e6]
    with_peaks.sort(key=lambda r: r["k_peak"], reverse=True)

    print(f"  Configs with k_peak > 1e6: {len(with_peaks)}")

    highest_k = with_peaks[:5] if with_peaks else []
    for r in highest_k:
        print(f"    x_c={r['x_c']:.4f} c={r['c']:.4f} beta={r['beta']:.1e} "
              f"chi0={r['chi0']} N*={r['N_star']:.0f} "
              f"k_peak={r['k_peak']:.2e} M={r.get('M_kpeak',0):.2e} "
              f"n_s={r.get('n_s','?'):.4f}")

    # Check if best configs hit the boundary → need to extend
    extend_xc = xc_max >= 0.795
    extend_c = c_max >= 4.9

    next_sweep = {}

    if extend_xc:
        next_sweep["x_c_lo"] = xc_max
        next_sweep["x_c_hi"] = min(xc_max + 0.05, 0.88)
        next_sweep["n_xc"] = 5
    else:
        # Refine around best x_c
        best = targets[:3] if targets else with_peaks[:3]
        if best:
            best_xc = np.mean([r["x_c"] for r in best])
            next_sweep["x_c_lo"] = max(best_xc - 0.01, 0.78)
            next_sweep["x_c_hi"] = min(best_xc + 0.01, 0.80)
            next_sweep["n_xc"] = 5
        else:
            next_sweep["x_c_lo"] = 0.80
            next_sweep["x_c_hi"] = 0.85
            next_sweep["n_xc"] = 6

    if extend_c:
        next_sweep["c_lo"] = c_max
        next_sweep["c_hi"] = min(c_max * 2, 12.0)
        next_sweep["n_c"] = 3
    else:
        best = targets[:3] if targets else with_peaks[:3]
        if best:
            best_c = np.mean([r["c"] for r in best])
            next_sweep["c_lo"] = max(best_c - 0.5, 0.77)
            next_sweep["c_hi"] = best_c + 0.5
            next_sweep["n_c"] = 3
        else:
            next_sweep["c_lo"] = 5.0
            next_sweep["c_hi"] = 10.0
            next_sweep["n_c"] = 3

    # Always include very low beta for stronger USR
    next_sweep["beta_vals"] = [1e-6, 5e-6, 1e-5, 3e-5, 1e-4, 3e-4, 6e-4]
    next_sweep["chi0_vals"] = [6.0, 8.0]
    next_sweep["N_star_vals"] = [55, 65, 70]

    return next_sweep


def submit_sweep(params, log_name="pbh_sweep_v2.jsonl"):
    """Build and submit the next sweep command."""
    base = ("cd ~/Documentos/CMB_USR && "
            "git pull && "
            "source ~/miniconda3/etc/profile.d/conda.sh && "
            "conda activate cmb-anomaly && "
            "nohup python scripts/sweep_pbh_params.py "
            f"--x_c-lo {params['x_c_lo']} "
            f"--x_c-hi {params['x_c_hi']} "
            f"--n-xc {params['n_xc']} "
            f"--c-lo {params['c_lo']} "
            f"--c-hi {params['c_hi']} "
            f"--n-c {params['n_c']} "
            f"--beta-vals " + " ".join(f"{b:.1e}" for b in params['beta_vals']) + " "
            "--chi0-vals " + " ".join(f"{c:.1f}" for c in params['chi0_vals']) + " "
            "--N-star-vals " + " ".join(f"{n:.0f}" for n in params['N_star_vals']) + " "
            f"--target all "
            f"--pivot-k 0.05 "
            f"--N-total-min 65.0 "
            f"--workers 4 "
            f"--log outputs/simulations/logs/{log_name} "
            f"> ~/{log_name.replace('.jsonl', '.log')} 2>&1 & echo PID=\\$!")

    ssh_cmd = f"ssh uni -o ConnectTimeout=15 \"{base}\""
    print("\n=== SUBMITTING NEXT SWEEP ===")
    print(f"  x_c ∈ [{params['x_c_lo']:.4f}, {params['x_c_hi']:.4f}] × {params['n_xc']}")
    print(f"  c   ∈ [{params['c_lo']:.3f}, {params['c_hi']:.3f}] × {params['n_c']}")
    print(f"  β   ∈ {params['beta_vals']}")
    print(f"  χ₀  ∈ {params['chi0_vals']}")
    print(f"  N*  ∈ {params['N_star_vals']}")
    print(f"  Log: {log_name}")

    print("\n  SSH command (copy-paste to run):")
    print("  " + ssh_cmd.replace("\n", " "))

    return ssh_cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="outputs/simulations/logs/pbh_sweep_full.jsonl")
    p.add_argument("--poll-interval", type=int, default=180, help="Poll interval in seconds")
    p.add_argument("--pid-file", default=None, help="File with initial PID to monitor")
    args = p.parse_args()

    log_path = os.path.expanduser(args.log)
    work_dir = os.path.expanduser("~/Documentos/CMB_USR")
    os.chdir(work_dir)

    print(f"[{datetime.datetime.now().isoformat()}] Monitor started")
    print(f"  Sweep log: {log_path}")
    print(f"  Poll interval: {args.poll_interval}s")
    print(f"  Working dir: {work_dir}")
    sys.stdout.flush()

    # Read initial PID if provided
    initial_pid = None
    if args.pid_file:
        pid_file = os.path.expanduser(args.pid_file)
        if os.path.exists(pid_file):
            with open(pid_file) as f:
                initial_pid = int(f.read().strip())
            print(f"  Initial PID from file: {initial_pid}")

    # Wait loop
    print(f"\n[{datetime.datetime.now().isoformat()}] Entering wait loop...")
    sys.stdout.flush()

    while True:
        pid = get_current_pid()
        if pid is not None:
            print(f"[{datetime.datetime.now().isoformat()}] PID {pid} still running. "
                  f"Next check in {args.poll_interval}s")
            sys.stdout.flush()
            time.sleep(args.poll_interval)
            continue

        # Check if log file exists — job might have finished
        if os.path.exists(log_path):
            n_done, _ = count_configs_in_log(log_path)
            if n_done > 0:
                print(f"\n[{datetime.datetime.now().isoformat()}] "
                      f"Job finished! {n_done} configs in log.")
                sys.stdout.flush()
                break

        print(f"[{datetime.datetime.now().isoformat()}] No PID found and no log. "
              f"Next check in {args.poll_interval}s")
        sys.stdout.flush()
        time.sleep(args.poll_interval)

    # Analyze results
    print("\n=== ANALYZING RESULTS ===")
    sys.stdout.flush()
    analysis = analyze_results(log_path)

    if analysis is None:
        print("  No results to analyze. Cannot design next sweep.")
        sys.exit(1)

    # Print summary
    print("\n  Sweep summary:")
    print(f"  Total entries: {analysis['total']}")
    print(f"  OK configs: {analysis['ok']}")
    print(f"  Errors: {len(analysis['errors'])}")
    print(f"  Configs in target mass bins: {analysis['targets_in_bins']}")
    print(f"  k_peak range: [{analysis['k_peak_range'][0]:.2e}, {analysis['k_peak_range'][1]:.2e}]")
    print(f"  Mass range: [{analysis['mass_range'][0]:.2e}, {analysis['mass_range'][1]:.2e}] M_sun")

    if analysis['targets_in_bins'] > 0:
        print("\n  Top target-binned configs:")
        for i, r in enumerate(analysis['best_targets'][:5]):
            print(f"  #{i+1}: x_c={r['x_c']:.4f} c={r['c']:.4f} β={r['beta']:.1e} "
                  f"χ₀={r['chi0']} N*={r['N_star']:.0f} "
                  f"k_peak={r['k_peak']:.2e} M={r.get('M_kpeak',0):.2e} "
                  f"f_total={r.get('f_total',0):.2e} "
                  f"n_s={r.get('n_s','?'):.4f}")

    # Design next sweep
    print("\n=== DESIGNING NEXT SWEEP ===")
    sys.stdout.flush()
    next_params = design_next_sweep(analysis, log_path)

    # Submit next sweep
    submit_sweep(next_params, log_name="pbh_sweep_v2.jsonl")

    print(f"\n[{datetime.datetime.now().isoformat()}] Monitor done.")


if __name__ == "__main__":
    main()
