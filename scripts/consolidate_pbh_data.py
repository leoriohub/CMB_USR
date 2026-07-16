#!/usr/bin/env python3
"""Consolidate and rank PBH sweep data with quality filters, scoring, optional plots and report.

Groups by physics config (xc, c, β, χ₀, N*), applies quality filters,
scores by mass bin × n_s × P_S ratio, and produces rankings.

Usage:
    python scripts/consolidate_pbh_data.py                           # rankings only
    python scripts/consolidate_pbh_data.py --top-n 50                # more configs
    python scripts/consolidate_pbh_data.py --plots                   # + diagnostic plots
    python scripts/consolidate_pbh_data.py --report                  # + markdown report
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np

from scripts.constants import ROOT_DIR, NumpyJSONEncoder

# ── Paths ──────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = os.path.join(ROOT_DIR, "outputs/simulations/pbh_results")
PLOT_DIR = os.path.join(ROOT_DIR, "outputs/plots/pbh")
LOG_DIR = os.path.join(ROOT_DIR, "outputs/simulations/logs")

# ── Mass bins ──────────────────────────────────────────────────────────────
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

# Ranking targets
TARGET_NS = 0.965  # Planck 2018 TT+lowE
PLANCK_As = 2.1e-9
A_S_TOLERANCE = 1e8  # relaxed: high-c regime shifts A_s
REALISTIC_K_MIN = 1e5  # Mpc⁻¹ — below this is grid-edge


# ── Helpers ────────────────────────────────────────────────────────────────


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def classify_mass(M):
    if M is None or M <= 0 or np.isnan(M):
        return "unknown"
    for label, lo, hi in MASS_LABELS:
        if lo <= M < hi:
            return label
    return "unknown"


def fmt_beta(b):
    return f"{b:.0e}" if b < 1e-4 else f"{b:.1e}"


def is_grid_boundary(k_peak):
    """Detect k_peak at a round power of 10 (grid edge artifact)."""
    if k_peak is None or k_peak <= 0:
        return False
    logk = np.log10(k_peak)
    return abs(logk - round(logk)) < 0.01 and k_peak > 1e4


def quality_filter(entry):
    """Check if a sweep entry passes quality filters.

    Returns (passes: bool, reason: str or None).
    """
    if "error" in entry:
        return False, f"error: {entry['error']}"

    k_peak = entry.get("k_peak", 0)
    if k_peak <= 0:
        return False, "k_peak <= 0"
    if k_peak < REALISTIC_K_MIN:
        return False, f"k_peak={k_peak:.2e} below {REALISTIC_K_MIN:.0e}"

    # Grid-boundary peaks
    if is_grid_boundary(k_peak):
        return False, f"grid-boundary k_peak=10^{round(np.log10(k_peak)):.0f}"
    if entry.get("on_grid_boundary", False):
        return False, "flagged on_grid_boundary"

    # A_s sanity
    a_s = entry.get("A_s_at_cmb", None)
    if a_s is not None and a_s > 0:
        ratio = a_s / PLANCK_As
        if ratio > A_S_TOLERANCE or ratio < 1.0 / A_S_TOLERANCE:
            return False, f"A_s={a_s:.4e} outside factor {A_S_TOLERANCE}"

    # n_s sanity
    n_s = entry.get("n_s", None)
    if n_s is not None and (n_s < 0.5 or n_s > 2.0):
        return False, f"n_s={n_s:.4f} unphysical"

    return True, None


def compute_score(entry, target_bins=("sub_solar_gap", "asteroid_gap")):
    """Score a config for desirability.

    score = mass_bonus × ns_bonus × ratio_bonus

    mass_bonus: 2.0 for primary target bin, 1.5 for secondary, 0.5 otherwise
    ns_bonus:   Gaussian centered at Planck 0.965, sigma=0.04
    ratio_bonus: log10(P_S_peak_ratio) / 5, clamped to ~1
    """
    mb = entry.get("mass_bin", "unknown")
    mass_bonus = (
        2.0 if mb == "sub_solar_gap" else (1.5 if mb == "asteroid_gap" else 0.5)
    )

    ns = entry.get("n_s", None)
    ns_bonus = np.exp(-0.5 * ((ns - TARGET_NS) / 0.04) ** 2) if ns is not None else 0.5

    ratio = entry.get("P_S_peak_ratio", None)
    ratio_bonus = (
        np.log10(max(ratio, 1)) / 5.0 if ratio is not None and ratio > 0 else 0.5
    )

    return mass_bonus * ns_bonus * ratio_bonus


# ── Plots ──────────────────────────────────────────────────────────────────


def _plot_ranking_scatter(rankings, filename="ranking_overview"):
    """n_s vs M_kpeak scatter colored by mass bin, sized by ratio."""
    from scripts.plotting import save_fig, TOL, PAPER_RCPARAMS
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = []
    for r in rankings[:100]:
        entry = r["best_entry"]
        points.append(
            {
                "M": r["M_kpeak"],
                "ns": entry.get("n_s", 0) or 0,
                "ratio": entry.get("P_S_peak_ratio", 0) or 1,
                "mbin": entry.get("mass_bin", "unknown"),
            }
        )
    if not points:
        return

    color_map = {
        "asteroid_gap": TOL["teal"],
        "sub_solar_gap": TOL["green"],
        "sub_stellar": TOL["blue"],
        "stellar_ligo": TOL["red"],
        "massive": TOL["grey"],
        "intermediate": TOL["yellow"],
        "too_light": TOL["grey"],
    }

    with plt.rc_context(PAPER_RCPARAMS):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))
        for p in points:
            c = color_map.get(p["mbin"], TOL["grey"])
            sz = max(10, min(80, np.log10(max(p["ratio"], 1)) * 10))
            ax1.scatter(p["ns"], p["M"], c=c, s=sz, alpha=0.6, edgecolors="none")
        ax1.axvline(
            TARGET_NS,
            color=TOL["grey"],
            ls="--",
            lw=0.8,
            label=f"Planck n_s={TARGET_NS}",
        )
        ax1.axhline(1e-6, color=TOL["red"], ls=":", lw=0.6)
        ax1.axhline(1e-2, color=TOL["red"], ls=":", lw=0.6)
        ax1.set_xlabel(r"$n_s$ (MS)")
        ax1.set_ylabel(r"$M_{\mathrm{peak}} [M_\odot]$")
        ax1.set_yscale("log")
        ax1.set_xlim(0.85, 1.10)
        ax1.legend(fontsize=6, loc="upper right")
        ax1.grid(True, alpha=0.2)
        ax1.text(1.08, 5e-3, "sub-solar", fontsize=7, color=TOL["green"], ha="right")
        ax1.text(1.08, 5e-11, "asteroid", fontsize=7, color=TOL["teal"], ha="right")
        ax1.text(1.08, 50, "LIGO ruled", fontsize=7, color=TOL["red"], ha="right")

        bins = Counter(p["mbin"] for p in points)
        order = [
            "asteroid_gap",
            "sub_solar_gap",
            "sub_stellar",
            "stellar_ligo",
            "intermediate",
            "massive",
            "too_light",
        ]
        labels, counts, colors_bar = [], [], []
        for b in order:
            if b in bins:
                labels.append(b)
                counts.append(bins[b])
                colors_bar.append(color_map.get(b, TOL["grey"]))
        ax2.barh(range(len(labels)), counts, color=colors_bar, alpha=0.8)
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels, fontsize=7)
        ax2.set_xlabel("Configs")
        ax2.grid(True, alpha=0.2, axis="x")
        fig.tight_layout()
        save_fig(fig, filename, "pbh")


def _plot_ns_vs_beta(rankings, filename="ns_vs_beta"):
    """n_s vs β colored by c."""
    from scripts.plotting import save_fig, TOL, PAPER_RCPARAMS
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    points = [(r, r["best_entry"]) for r in rankings if r.get("n_s")]
    if not points:
        return
    with plt.rc_context(PAPER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(4.5, 3.0))
        c_vals = sorted({e["c"] for _, e in points})
        cmap = plt.colormaps["viridis"]
        norm = LogNorm(vmin=max(min(c_vals), 0.5), vmax=max(c_vals))
        for r, e in points:
            ax.scatter(e["beta"], r["n_s"], c=[cmap(norm(e["c"]))], s=20, alpha=0.7)
        ax.axhline(TARGET_NS, color=TOL["grey"], ls="--", lw=0.8)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$n_s$")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.2)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax).set_label(
            r"$c$", fontsize=8
        )
        fig.tight_layout()
        save_fig(fig, filename, "pbh")


def _plot_kpeak_vs_ns(rankings, filename="kpeak_vs_ns"):
    """k_peak vs n_s colored by c, sized by P_S ratio."""
    from scripts.plotting import save_fig, TOL, PAPER_RCPARAMS
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    points = [
        (r, r["best_entry"])
        for r in rankings
        if r.get("n_s") and r.get("M_kpeak", 0) > 0
    ]
    if not points:
        return
    with plt.rc_context(PAPER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        c_vals = sorted({e["c"] for _, e in points})
        norm = LogNorm(vmin=max(min(c_vals), 0.5), vmax=max(c_vals))
        cmap = plt.colormaps["viridis"]
        for r, e in points:
            ratio = max(e.get("P_S_peak_ratio", 1e3), 1e3)
            sz = max(15, min(100, np.log10(ratio) * 8))
            ax.scatter(r["n_s"], r["M_kpeak"], c=[cmap(norm(e["c"]))], s=sz, alpha=0.6)
        ax.axvline(TARGET_NS, color=TOL["grey"], ls="--", lw=0.8)
        ax.axhline(2e11, color=TOL["red"], ls=":", lw=0.6)
        ax.axhline(6e17, color=TOL["teal"], ls=":", lw=0.6)
        ax.set_xlabel(r"$n_s$")
        ax.set_ylabel(r"$M_{\mathrm{peak}} [M_\odot]$")
        ax.set_yscale("log")
        ax.set_xlim(0.90, 1.10)
        ax.grid(True, alpha=0.2)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax).set_label(
            r"$c$", fontsize=8
        )
        fig.tight_layout()
        save_fig(fig, filename, "pbh")


# ── Report ─────────────────────────────────────────────────────────────────


def _build_markdown_report(stats, rankings):
    lines = [
        "# PBH Sweep Consolidation Report",
        f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n",
        "## 1. Data Inventory\n",
        "| File | Total | Errors | Passed | Key Rejections | Mass Bins |",
        "|------|-------|--------|--------|----------------|-----------|",
    ]
    for fname, s in sorted(stats.items()):
        top_rej = "; ".join(
            f"{r}: {n}" for r, n in list(s.get("rejected", {}).items())[:2]
        )
        mb = "; ".join(f"{b}: {n}" for b, n in s.get("mass_bins", {}).items())
        lines.append(
            f"| {fname} | {s['total']} | {s['errors']} | {s['passed']} | {top_rej} | {mb} |"
        )

    lines.extend(
        [
            "\n## 2. Top-Ranked Configs\n",
            "| Rank | File | x_c | c | β | χ₀ | N* | k_peak | M [M_⊙] | n_s | Ratio | Mass Bin | Score |",
            "|------|------|-----|----|-------|----|----|--------|----------|------|-------|----------|-------|",
        ]
    )
    for rank, r in enumerate(rankings[:30], 1):
        d = r["best_entry"]
        lines.append(
            f"| {rank} | {r['src']} | {d['x_c']:.4f} | {d['c']:.4f} | {fmt_beta(d['beta'])} | "
            f"{d['chi0']} | {d['N_star']} | {r.get('M_kpeak', 0):.3e} | "
            f"{r.get('M_kpeak', 0):.3e} | {r.get('n_s', 0):.4f} | "
            f"{d.get('P_S_peak_ratio', 0):.1e} | {r['mass_bin']} | {r['score']:.3f} |"
        )

    lines.append("\n## 3. Known Issues\n")
    all_rejected = Counter()
    for _, s in stats.items():
        for reason, n in s.get("rejected", {}).items():
            all_rejected[reason] += n
    for reason, n in all_rejected.most_common(10):
        lines.append(f"- **{n}x** {reason}")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Consolidate and rank PBH sweep data")
    p.add_argument(
        "--data",
        type=str,
        default=None,
        help="Single JSONL file to consolidate (overrides --data-dir)",
    )
    p.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Sweep JSONL directory (used when --data is not given)",
    )
    p.add_argument("--min-f", type=float, default=1e-6, help="Min f_total to include")
    p.add_argument("--top-n", type=int, default=30, help="Configs per ζ_c slice")
    p.add_argument("--plots", action="store_true", help="Generate diagnostic plots")
    p.add_argument("--report", action="store_true", help="Generate markdown report")
    p.add_argument("--quick", action="store_true", help="Skip filters, raw ranking")
    args = p.parse_args()

    # ── Load data ──────────────────────────────────────────────────────
    if args.data:
        files = [args.data]
    else:
        files = sorted(glob.glob(os.path.join(args.data_dir, "*.jsonl")))
    print(f"Files: {len(files)}")

    raw = []
    for fpath in files:
        fname = os.path.basename(fpath)
        for d in load_jsonl(fpath):
            d["_src"] = fname
            raw.append(d)

    errors = [d for d in raw if "error" in d]
    good = [d for d in raw if "error" not in d]
    print(f"Total: {len(raw)}  Errors: {len(errors)}  Good: {len(good)}")

    # Normalize
    for d in good:
        d.setdefault("zeta_c", None)
        if "mass_bin" not in d or not d["mass_bin"]:
            mk = d.get("M_kpeak") or d.get("M_present")
            d["mass_bin"] = classify_mass(mk) if mk else "unknown"
        d["_grid_b"] = d.get("on_grid_boundary", False)

    # ── Quality filters ────────────────────────────────────────────────
    if not args.quick:
        rejected = Counter()
        filtered = []
        for d in good:
            ok, reason = quality_filter(d)
            if ok:
                filtered.append(d)
            else:
                rejected[reason] += 1
        if rejected:
            print("\nQuality filter rejections:")
            for reason, n in rejected.most_common(10):
                print(f"  {n:>5}x  {reason}")
        good = filtered
        print(f"After filters: {len(good)} configs\n")

    # ── Per-ζ_c rankings ───────────────────────────────────────────────
    all_zeta = sorted({d["zeta_c"] for d in good if d["zeta_c"] is not None})
    print(f"ζ_c values: {all_zeta}")

    groups = defaultdict(list)
    for d in good:
        key = (d["x_c"], d["c"], d["beta"], d["chi0"], d["N_star"])
        groups[key].append(d)
    print(f"Unique physics configs: {len(groups)}\n")

    print("=" * 100)
    print("RANKINGS BY ζ_c (top viable: 0 < f_total ≤ 1)")
    print("=" * 100)
    top_by_zc = {}
    for zc in all_zeta:
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
        for d in candidates[: args.top_n]:
            mk = d.get("M_kpeak") or d.get("M_present") or 0
            ns = d.get("n_s", 0)
            gb = " B" if d["_grid_b"] else "  "
            print(
                f"  β={fmt_beta(d['beta']):>7} N*={d['N_star']:4.0f}  "
                f"f={d['f_total']:9.4e}  M_k={mk:9.3e}  n_s={ns:.4f}{gb}  "
                f"{d['mass_bin']:16s}  {d['_src']}"
            )
        top_by_zc[zc] = [dict(d) for d in candidates[:5]]

    # ── Physics config ranking ─────────────────────────────────────────
    print("\n" + "=" * 100)
    print("BEST PHYSICS CONFIGS (composite score)")
    print("=" * 100)

    physics_ranking = []
    for key, entries in groups.items():
        with_zeta = [
            d
            for d in entries
            if d.get("zeta_c") is not None and d.get("f_total") is not None
        ]
        if not with_zeta:
            continue
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
        best = max(viable, key=lambda d: d["f_total"])
        score = compute_score(best)
        physics_ranking.append(
            {
                "key": key,
                "best_entry": best,
                "best_f": best["f_total"],
                "best_zeta": best["zeta_c"],
                "mass_bin": best["mass_bin"],
                "M_kpeak": best.get("M_kpeak") or best.get("M_present") or 0,
                "n_s": best.get("n_s") or 0,
                "score": score,
                "src": best["_src"],
                "all_zeta": {
                    d["zeta_c"]: d["f_total"]
                    for d in entries
                    if d["zeta_c"] is not None
                },
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
        zc_str = f"{r['best_zeta']:.3f}" if r["best_zeta"] is not None else " — "
        print(
            f"{rank:<4} {fmt_beta(d['beta']):>8} {d['N_star']:<5.0f} {zc_str:<6} "
            f"{r['best_f']:<10.4e} {r['M_kpeak']:<12.3e} {r['n_s']:<8.4f} "
            f"{r['mass_bin']:<16} {r['score']:<8.4f} {r['src']}"
        )

    # ── Summary by mass bin ────────────────────────────────────────────
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
            zc_str = f"{r['best_zeta']:.3f}" if r["best_zeta"] is not None else " — "
            print(
                f"  β={fmt_beta(d['beta']):>7} N*={d['N_star']:4.0f} ζ_c={zc_str}  "
                f"f={r['best_f']:9.4e}  M_k={r['M_kpeak']:9.3e}  n_s={r['n_s']:.4f}"
            )

    # ── Write consolidated JSON ────────────────────────────────────────
    out_path = os.path.join(args.data_dir, "pbh_consolidated.json")
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
                for d in lst
            ]
            for zc, lst in top_by_zc.items()
        },
        "physics_ranking": [],
        "by_mass_bin": {},
    }
    for rank, r in enumerate(physics_ranking[:200], 1):
        d = r["best_entry"]
        entry = {
            k: v for k, v in d.items() if k not in ("k_phys", "P_S", "_src", "_grid_b")
        }
        entry.update(
            rank=rank,
            score=r["score"],
            best_zeta=r["best_zeta"],
            all_zeta=r["all_zeta"],
            source=r["src"],
        )
        output["physics_ranking"].append(entry)
    for bn in [
        "sub_solar_gap",
        "asteroid_gap",
        "intermediate",
        "sub_stellar",
        "stellar_ligo",
        "massive",
    ]:
        be = [r for r in physics_ranking if r["mass_bin"] == bn]
        if be:
            output["by_mass_bin"][bn] = {
                "count": len(be),
                "top": [
                    {
                        k: v
                        for k, v in r["best_entry"].items()
                        if k not in ("k_phys", "P_S", "_src", "_grid_b")
                    }
                    for r in be[:5]
                ],
            }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyJSONEncoder)
    print(f"\nWritten: {out_path}")

    # ── Plots ──────────────────────────────────────────────────────────
    if args.plots:
        print("\nGenerating plots...")
        os.makedirs(PLOT_DIR, exist_ok=True)
        _plot_ranking_scatter(physics_ranking)
        _plot_ns_vs_beta(physics_ranking)
        _plot_kpeak_vs_ns(physics_ranking)

    # ── Report ─────────────────────────────────────────────────────────
    if args.report:
        # Recompute per-file stats for the report
        stats = {}
        for fpath in files:
            fname = os.path.basename(fpath)
            data = load_jsonl(fpath)
            errs = [d for d in data if "error" in d]
            gd = [d for d in data if "error" not in d]
            passed = sum(1 for d in gd if quality_filter(d)[0])
            reasons = Counter()
            for d in gd:
                ok, reason = quality_filter(d)
                if not ok:
                    reasons[reason] += 1
            mb = Counter()
            for d in gd:
                mk = d.get("M_kpeak") or d.get("M_present")
                mb[classify_mass(mk)] += 1
            stats[fname] = {
                "total": len(data),
                "errors": len(errs),
                "passed": passed,
                "rejected": dict(reasons.most_common(10)),
                "mass_bins": dict(mb),
            }
        report = _build_markdown_report(stats, physics_ranking)
        report_path = os.path.join(LOG_DIR, "pbh_consolidation_report.md")
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Report: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
