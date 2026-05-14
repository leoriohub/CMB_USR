#!/usr/bin/env python3
"""Analyze CAMB scan JSONL logs: stats, correlations, filtering, export."""
import argparse
import csv
import json
import sys

import numpy as np
from scipy import stats as sp_stats

FIELD_NAMES = {
    "phi0": "phi0", "y0": "y0", "N_star": "N_star", "Nstar": "N_star",
    "chi2": "chi2", "d2": "d2", "k_dip": "k_dip", "suppression": "suppression_pct",
    "N_total": "N_total", "status": "status",
}


def parse_args():
    p = argparse.ArgumentParser(description="Analyze CAMB scan JSONL logs")
    p.add_argument("logfiles", nargs="+", help="JSONL scan log file(s)")
    p.add_argument("--sort-by", default="d2",
                   choices=["d2", "chi2", "Nstar", "N_star", "suppression"],
                   help="Sort metric (default: d2)")
    p.add_argument("--top", type=int, default=10,
                   help="Show top N configs (default: 10)")
    p.add_argument("--filter", default=None,
                   help="Filter expression, e.g. 'chi2<30 and d2<600'")
    p.add_argument("--no-stats", action="store_true",
                   help="Hide distribution statistics")
    p.add_argument("--corr", action="store_true",
                   help="Show parameter correlations")
    p.add_argument("--compare", action="store_true",
                   help="Compare multiple files")
    p.add_argument("--export", choices=["json", "csv"], default=None,
                   help="Export filtered results")
    p.add_argument("--export-prefix", default="scan_results",
                   help="Prefix for export files (default: scan_results)")
    return p.parse_args()


def load_records(path):
    records = []
    errors = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("status") == "ok":
                    records.append(r)
            except json.JSONDecodeError:
                errors += 1
    if errors:
        print(f"  Warning: {errors} unparseable lines in {path}", file=sys.stderr)
    return records


def filter_records(records, expr):
    if not expr:
        return records
    normalized = expr
    for alias, field in FIELD_NAMES.items():
        if alias != field:
            normalized = normalized.replace(alias, field)
    filtered = []
    errors = 0
    for r in records:
        try:
            result = eval(normalized, {"__builtins__": {}}, dict(r))
            if result:
                filtered.append(r)
        except Exception:
            errors += 1
    if errors:
        print(f"  Warning: {errors} records skipped (eval error)", file=sys.stderr)
    if not filtered:
        print(f"  Warning: filter '{expr}' returned 0 results", file=sys.stderr)
    return filtered


def compute_stats(records):
    if not records:
        return {}
    return {
        "d2": np.array([r.get("d2", 9999) for r in records]),
        "chi2": np.array([r.get("chi2", 999) for r in records]),
        "N_star": np.array([r.get("N_star", 0) for r in records]),
        "suppression": np.array([r.get("suppression_pct", 0) for r in records]),
        "k_dip": np.array([r.get("k_dip", 0) for r in records]),
    }


def compute_correlations(records):
    if len(records) < 3:
        return {}
    metrics = ["d2", "chi2", "N_star", "k_dip", "suppression_pct"]
    data = {m: np.array([r.get(m, np.nan) for r in records]) for m in metrics}
    mask = np.ones(len(records), dtype=bool)
    for arr in data.values():
        mask &= ~np.isnan(arr)
    if np.sum(mask) < 3:
        return {}
    clean = {m: data[m][mask] for m in metrics}
    return clean


def print_summary(records, label):
    if not records:
        print(f"  {label}: No valid configs")
        return
    d2s = [r.get("d2", 9999) for r in records]
    chi2s = [r.get("chi2", 999) for r in records]
    nstars = [r.get("N_star", 0) for r in records]
    print(f"  File: {label}")
    print(f"    Total ok configs: {len(records)}")
    print(f"    D2 range:    {min(d2s):.1f} - {max(d2s):.1f}")
    print(f"    Chi2 range:  {min(chi2s):.1f} - {max(chi2s):.1f}")
    print(f"    N* range:    {min(nstars):.1f} - {max(nstars):.1f}")


def print_distribution(records):
    stats = compute_stats(records)
    if not stats:
        return
    print(f"\n  Distribution Statistics:")
    print(f"  {'Metric':<14} {'Mean':>9} {'Median':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    print(f"  {'-'*59}")
    for name, arr in [("D2", "d2"), ("Chi2", "chi2"), ("N*", "N_star"),
                      ("Suppression", "suppression"), ("k_dip", "k_dip")]:
        a = stats[arr]
        print(f"  {name:<14} {np.mean(a):>9.2f} {np.median(a):>9.2f} {np.std(a):>9.2f} {np.min(a):>9.2f} {np.max(a):>9.2f}")


def print_top(records, sort_by, top_n):
    sort_key = {"d2": "d2", "chi2": "chi2", "Nstar": "N_star", "N_star": "N_star",
                "suppression": "suppression_pct"}.get(sort_by, "d2")
    sorted_recs = sorted(records, key=lambda r: r.get(sort_key, 9999))
    if sort_key == "suppression_pct":
        sorted_recs = list(reversed(sorted_recs))
    shown = sorted_recs[:top_n]
    if not shown:
        return
    print(f"\n  Top {len(shown)} by {sort_by}:")
    print(f"  {'#':>3}  {'phi0':>6}  {'y0':>8}  {'N*':>6}  {'chi2':>7}  {'d2':>7}  {'k_dip':>10}  {'supp%':>7}")
    print(f"  {'-'*66}")
    for i, r in enumerate(shown, 1):
        print(f"  {i:3d}  {r.get('phi0', 0):6.2f}  {r.get('y0', 0):+8.3f}  "
              f"{r.get('N_star', 0):6.1f}  {r.get('chi2', 0):7.2f}  "
              f"{r.get('d2', 0):7.1f}  {r.get('k_dip', 0):10.2e}  "
              f"{r.get('suppression_pct', 0):7.1f}%")


def print_correlations(records):
    data = compute_correlations(records)
    if not data:
        print("  Not enough data for correlations (need >= 3 records)")
        return
    targets = ["d2", "chi2"]
    params = ["phi0", "y0", "N_star", "k_dip", "suppression_pct"]
    print(f"\n  Parameter Correlations (Pearson r):")
    print(f"  {'Target':<10} ", end="")
    for p in params:
        print(f"{p:>16}", end="")
    print()
    print(f"  {'-'*10} {'-'*16*len(params)}")
    for t in targets:
        t_arr = np.array([r.get(t, np.nan) for r in records])
        print(f"  {t:<10}", end="")
        for p in params:
            p_arr = np.array([r.get(p, np.nan) for r in records])
            mask = ~(np.isnan(t_arr) | np.isnan(p_arr))
            if np.sum(mask) < 3:
                print(f"   {'N/A':>14}", end="")
            else:
                r_val, _ = sp_stats.pearsonr(t_arr[mask], p_arr[mask])
                print(f"   {r_val:>+14.3f}", end="")
        print()


def print_comparison(file_stats):
    print("  (Task 6)")


def export_results(records, fmt, prefix):
    print("  (Task 7)")


def main():
    args = parse_args()
    for path in args.logfiles:
        records = load_records(path)
        records = filter_records(records, args.filter)
        print()
        print_summary(records, path)
        if not args.no_stats:
            print_distribution(records)
        print_top(records, args.sort_by, args.top)
        if args.corr:
            print_correlations(records)
        if args.export:
            export_results(records, args.export, args.export_prefix)

    if args.compare and len(args.logfiles) >= 2:
        file_stats = []
        for path in args.logfiles:
            recs = load_records(path)
            recs = filter_records(recs, args.filter)
            file_stats.append({"label": path, "n": len(recs),
                               "d2s": [r.get("d2", 9999) for r in recs],
                               "chi2s": [r.get("chi2", 999) for r in recs],
                               "nstars": [r.get("N_star", 0) for r in recs]})
        print_comparison(file_stats)


if __name__ == "__main__":
    main()
