#!/usr/bin/env python3
"""Consolidate all PBH sweep data with ζ_c-aware ranking.

Groups by physics config (xc, c, β, χ₀, N*), shows f_total at each ζ_c,
ranks by maximum viable f_total ≤ 1.

Usage:
    python scripts/consolidate_pbh_data.py
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "outputs/simulations/pbh_results")
GAMMA, K_EQ, M_EQ, ACCR = 0.4, 0.0104, 3.0e17, 3e7

MASS_LABELS = [
    ("too_light", 0, 1e-17),
    ("asteroid_gap", 1e-17, 1e-15),
    ("intermediate", 1e-15, 1e-6),
    ("sub_solar_gap", 1e-6, 1e-2),
    ("sub_stellar", 1e-2, 1.0),
    ("stellar_ligo", 1.0, 100.0),
    ("massive", 100.0, float("inf")),
]

MASS_WEIGHT = {
    "sub_solar_gap": 3.0,
    "asteroid_gap": 2.5,
    "intermediate": 1.0,
    "sub_stellar": 0.5,
    "stellar_ligo": 0.3,
    "massive": 0.1,
    "too_light": 0.1,
    "unknown": 0.0,
}


def classify_mass(M):
    if M is None or M <= 0 or np.isnan(M):
        return "unknown"
    for label, lo, hi in MASS_LABELS:
        if lo <= M < hi:
            return label
    return "unknown"


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser(description="Consolidate PBH sweep data (ζ_c-aware)")
    p.add_argument("--min-f", type=float, default=1e-6, help="Min f_total to include")
    p.add_argument("--top-n", type=int, default=30, help="Configs per ζ_c slice")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.jsonl")))
    print(f"Files: {len(files)}")

    raw = []
    for fpath in files:
        fname = os.path.basename(fpath)
        for d in load_jsonl(fpath):
            d["_src"] = fname
            raw.append(d)

    # Separate errors and good
    errors = [d for d in raw if "error" in d]
    good = [d for d in raw if "error" not in d]
    print(f"Total: {len(raw)}  Errors: {len(errors)}  Good: {len(good)}")

    # Normalize: ensure zeta_c field exists, classify mass, flag grid boundary
    for d in good:
        d.setdefault("zeta_c", None)
        if "mass_bin" not in d or not d["mass_bin"]:
            mk = d.get("M_kpeak") or d.get("M_present")
            d["mass_bin"] = classify_mass(mk) if mk else "unknown"
        d["_grid_b"] = d.get("on_grid_boundary", False)

    # Group by physics config: (xc, c, beta, chi0, N_star)
    groups = defaultdict(list)
    for d in good:
        key = (d["x_c"], d["c"], d["beta"], d["chi0"], d["N_star"])
        groups[key].append(d)
    print(f"Unique physics configs: {len(groups)}")

    # For each group, collect f_total across ζ_c
    def fmt_beta(b):
        return f"{b:.0e}" if b < 1e-4 else f"{b:.1e}"

    def fmt_row(d):
        b = fmt_beta(d["beta"])
        mk = d.get("M_kpeak") or d.get("M_present") or 0
        ns = d.get("n_s", 0)
        gb = " B" if d["_grid_b"] else "  "
        zc = d.get("zeta_c")
        zc_str = f"{zc:.3f}" if zc is not None else " — "
        return f"  β={b:>7} N*={d['N_star']:4.0f} ζ_c={zc_str}  f={d['f_total']:9.4e}  M_k={mk:9.3e}  n_s={ns:.4f}{gb}  {d['mass_bin']:16s}  {d['_src']}"

    # Collect all ζ_c values across the dataset
    all_zeta = sorted({d["zeta_c"] for d in good if d["zeta_c"] is not None})
    print(f"\nζ_c values in data: {all_zeta}")

    # ── Per-ζ_c rankings ────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("RANKINGS BY ζ_c (top viable: 0 < f_total ≤ 1)")
    print("=" * 100)
    top_by_zc = {}
    for zc in all_zeta:
        # Entries at this ζ_c, non-boundary, f_total ≤ 1
        candidates = [
            d
            for d in good
            if d.get("zeta_c") == zc
            and not d["_grid_b"]
            and d["f_total"] > args.min_f
            and d["f_total"] <= 1.0
        ]
        candidates.sort(key=lambda d: d["f_total"], reverse=True)

        print(f"\n--- ζ_c={zc:.3f} ({len(candidates)} candidates) ---")
        if candidates:
            for d in candidates[: args.top_n]:
                print(fmt_row(d))
        top_by_zc[zc] = candidates[:5]

    # ── Best physics configs (f_total closest to 1 from below) ──────
    print("\n" + "=" * 100)
    print("BEST PHYSICS CONFIGS (max viable f_total ≤ 1)")
    print("=" * 100)

    physics_ranking = []
    for key, entries in groups.items():
        xc, c, beta, chi0, nstar = key
        # Only rank entries with explicit ζ_c (old sweeps used wrong default)
        with_zeta = [
            d
            for d in entries
            if d.get("zeta_c") is not None and d.get("f_total") is not None
        ]
        if not with_zeta:
            continue
        # Filter to viable: f_total ≤ 1, not grid-boundary, physical n_s
        viable = [
            d
            for d in with_zeta
            if d["f_total"] <= 1.0
            and not d["_grid_b"]
            and d.get("n_s") is not None
            and 0.7 < d["n_s"] < 1.3
        ]
        if not viable:
            continue
        # Best = highest viable f_total
        best = max(viable, key=lambda d: d["f_total"])
        # Score: f_total × mass_priority × n_s_bonus
        mass_prio = MASS_WEIGHT.get(best["mass_bin"], 0.5)
        f_score = best["f_total"]  # 0 to 1
        ns = best.get("n_s") or 0
        ns_bonus = max(0, 1.0 - abs(ns - 0.965) / 0.1)
        score = f_score * mass_prio * ns_bonus

        physics_ranking.append(
            {
                "key": key,
                "best_entry": best,
                "best_f": best["f_total"],
                "best_zeta": best["zeta_c"],
                "mass_bin": best["mass_bin"],
                "M_kpeak": best.get("M_kpeak") or best.get("M_present") or 0,
                "n_s": ns,
                "score": score,
                "all_zeta": {
                    d["zeta_c"]: d["f_total"]
                    for d in entries
                    if d["zeta_c"] is not None
                },
                "src": best["_src"],
            }
        )

    physics_ranking.sort(key=lambda r: r["score"], reverse=True)

    print(
        f"\n{'#':<4} {'β':>8} {'N*':<5} {'ζ_c':<6} {'f_total':<10} {'M_k':<12} "
        f"{'n_s':<8} {'bin':<16} {'score':<8} {'source'}"
    )
    print("-" * 100)
    for rank, r in enumerate(physics_ranking[: args.top_n * 2], 1):
        d = r["best_entry"]
        b_fmt = fmt_beta(d["beta"])
        zc_str = f"{r['best_zeta']:.3f}" if r["best_zeta"] is not None else " — "
        print(
            f"{rank:<4} {b_fmt:>8} {d['N_star']:<5.0f} {zc_str:<6} {r['best_f']:<10.4e} "
            f"{r['M_kpeak']:<12.3e} {r['n_s']:<8.4f} {r['mass_bin']:<16} "
            f"{r['score']:<8.4f} {r['src']}"
        )

    # ── Summary by mass bin ─────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY BY MASS BIN")
    print("=" * 100)
    by_bin = defaultdict(list)
    for r in physics_ranking:
        by_bin[r["mass_bin"]].append(r)
    for bin_name in ["sub_solar_gap", "asteroid_gap", "intermediate", "sub_stellar"]:
        entries = by_bin.get(bin_name, [])
        print(f"\n{bin_name} ({len(entries)} physics configs):")
        for r in entries[:5]:
            d = r["best_entry"]
            b_fmt = fmt_beta(d["beta"])
            zc_str = f"{r['best_zeta']:.3f}" if r["best_zeta"] is not None else " — "
            print(
                f"  β={b_fmt:>7} N*={d['N_star']:4.0f} ζ_c={zc_str}  "
                f"f={r['best_f']:9.4e}  M_k={r['M_kpeak']:9.3e}  n_s={r['n_s']:.4f}"
            )

    # ── Write consolidated JSON ──────────────────────────────────
    out_path = os.path.join(DATA_DIR, "pbh_consolidated.json")
    output = {
        "generated": datetime.now().isoformat(),
        "statistics": {
            "total_raw": len(raw),
            "errors": len(errors),
            "good": len(good),
            "unique_physics_configs": len(groups),
            "zeta_c_values": all_zeta,
            "files": [os.path.basename(f) for f in files],
        },
        "top_by_zeta_c": {
            f"{zc:.3f}": [
                {
                    k: v
                    for k, v in d.items()
                    if k not in ("k_phys", "P_S", "_src", "_grid_b")
                }
                for d in top_by_zc[zc]
            ]
            for zc in all_zeta
            if zc in top_by_zc
        },
        "physics_ranking": [],
        "by_mass_bin": {},
    }

    for rank, r in enumerate(physics_ranking[:200], 1):
        d = r["best_entry"]
        entry = {
            k: v for k, v in d.items() if k not in ("k_phys", "P_S", "_src", "_grid_b")
        }
        entry["rank"] = rank
        entry["score"] = r["score"]
        entry["best_zeta"] = r["best_zeta"]
        entry["all_zeta"] = r["all_zeta"]
        entry["source"] = r["src"]
        output["physics_ranking"].append(entry)

    for bin_name in [
        "sub_solar_gap",
        "asteroid_gap",
        "intermediate",
        "sub_stellar",
        "stellar_ligo",
        "massive",
    ]:
        bin_entries = [r for r in physics_ranking if r["mass_bin"] == bin_name]
        if bin_entries:
            output["by_mass_bin"][bin_name] = {
                "count": len(bin_entries),
                "top": [
                    {
                        k: v
                        for k, v in r["best_entry"].items()
                        if k not in ("k_phys", "P_S", "_src", "_grid_b")
                    }
                    for r in bin_entries[:5]
                ],
            }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWritten: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
