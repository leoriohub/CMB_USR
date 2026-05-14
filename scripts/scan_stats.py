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
    print(f"  Filter expression not yet implemented: {expr}", file=sys.stderr)
    return records


def compute_stats(records):
    return {}


def compute_correlations(records):
    return {}


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
    print("  (Task 2)")


def print_top(records, sort_by, top_n):
    print("  (Task 3)")


def print_correlations(records):
    print("  (Task 5)")


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
