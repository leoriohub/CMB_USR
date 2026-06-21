"""Systematic analysis of Ezquiaga CHI sweep results.

Reproducible pipeline to:
1. Ingest all JSONL sweep logs and P_S(k) files
2. Apply quality filters (realistic k_peak, valid A_s, etc.)
3. Rank configs by PBH mass placement × n_s × P_S ratio
4. Produce structured summaries: JSON + Markdown report + diagnostic plots

Usage:
    python scripts/analyze_sweep_results.py                     # full analysis
    python scripts/analyze_sweep_results.py --quick             # summary only, no plots
    python scripts/analyze_sweep_results.py --show-top 20      # show top N configs

Re-run as new data arrives — results are deterministic.
"""

import os
import json
import argparse
from datetime import datetime
from collections import Counter

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scripts.plotting import get_path, save_fig, TOL, PAPER_RCPARAMS

# ── Constants ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "outputs/simulations/logs")
PS_DIR = os.path.join(ROOT, "outputs/simulations/pspectra")

# Mass gap boundaries [M_⊙]
MASS_BINS = {
    "too_light": (0, 1e-17),
    "asteroid_gap": (1e-17, 1e-15),
    "intermediate": (1e-15, 1e-6),
    "sub_solar_gap": (1e-6, 1e-2),
    "sub_stellar": (1e-2, 1.0),
    "stellar_ligo": (1.0, 100.0),
    "massive": (100.0, float("inf")),
}

# Planck pivot (default for CMB n_s)
PLANCK_PIVOT = 0.05
PLANCK_As = 2.1e-9
TARGET_NS = 0.965  # Planck 2018

# Quality thresholds
A_S_TOLERANCE = 1e8  # relaxed: high-c regime can shift A_s by 5+ orders of magnitude;
# A_s renormalization happens in CAMB, not a disqualifier for PBH work
MIN_PS_RATIO = 1000  # P_S_peak / As minimum for any peak
VALID_K_MIN = 1.0  # Mpc⁻¹ — below this is CMB only, no PBH peak
REALISTIC_K_MIN = 1e5  # Mpc⁻¹ — below this is probably grid-edge


# ── I/O ─────────────────────────────────────────────────────────────────────


def load_jsonl(path):
    """Load a JSONL file, return list of dicts."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_all_sweeps():
    """Load all Ezquiaga-relevant sweep logs."""
    sweeps = {}
    for fname in sorted(os.listdir(LOG_DIR)):
        if not fname.endswith(".jsonl"):
            continue
        # Skip Higgs-specific files
        if fname.startswith(("camb_", "overnight_", "top30")):
            continue
        if fname.startswith(("pbh_sweep", "chi0_sweep", "ms_sweep", "ms_regime")):
            data = load_jsonl(os.path.join(LOG_DIR, fname))
            if data:
                sweeps[fname] = data
    return sweeps


def load_pbh_json(path):
    """Load a PBH JSON file (full MS + PBH output), return dict or None."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── Quality filters ────────────────────────────────────────────────────────


def quality_filter(entry):
    """Check if a single sweep entry passes quality filters.

    Returns (passes: bool, reason: str or None).
    """
    # Error entries
    if "error" in entry:
        return False, f"error: {entry['error']}"

    # Must have parameters
    for k in ("x_c", "c", "beta"):
        if k not in entry:
            return False, f"missing {k}"

    # k_peak quality
    k_peak = entry.get("k_peak", 0)
    if k_peak <= 0:
        return False, f"k_peak={k_peak} <= 0"

    # Grid-boundary detection: k_peak at powers of 10 with round numbers
    # Typical grid edges: 1e6, 1e12, 1e18, 1e22
    logk = np.log10(k_peak)
    logk_frac = logk - round(logk)
    if abs(logk_frac) < 0.01 and k_peak > 1e4:
        # Check if it's at a suspicious round number
        return False, f"grid-boundary k_peak=10^{round(logk):.0f}"

    if k_peak < REALISTIC_K_MIN:
        return False, f"k_peak={k_peak:.2e} below realistic threshold"

    # A_s quality (if available)
    a_s = entry.get("A_s_at_cmb", None)
    if a_s is not None and a_s > 0:
        ratio = a_s / PLANCK_As
        if ratio > A_S_TOLERANCE or ratio < 1.0 / A_S_TOLERANCE:
            return (
                False,
                f"A_s={a_s:.4e} outside factor {A_S_TOLERANCE} of Planck ({PLANCK_As:.4e})",
            )

    # P_S ratio (if available)
    ratio = entry.get("P_S_peak_ratio", None)
    if ratio is not None and ratio < MIN_PS_RATIO:
        return False, f"P_S_peak_ratio={ratio:.1e} below {MIN_PS_RATIO}"

    # n_s validity (if available)
    n_s = entry.get("n_s", None)
    if n_s is not None and (n_s < 0.5 or n_s > 2.0):
        return False, f"n_s={n_s:.4f} outside physical range"

    return True, None


def classify_mass(M_kpeak):
    """Classify present-day mass in M_⊙ into target gap."""
    if M_kpeak <= 0 or np.isnan(M_kpeak):
        return "unknown"
    for label, (lo, hi) in MASS_BINS.items():
        if lo <= M_kpeak < hi:
            return label
    return "unknown"


# ── Ranking ─────────────────────────────────────────────────────────────────


def compute_score(entry, target_bins=("sub_solar_gap", "asteroid_gap")):
    """Score a config for desirability.

    Score = mass_bonus × ns_bonus × ratio_bonus

    mass_bonus: 2.0 for primary target bin, 1.5 for secondary, 1.0 otherwise
    ns_bonus: exp(-0.5 * ((n_s - 0.965) / 0.02)^2) — Gaussian centered at Planck
    ratio_bonus: log10(P_S_peak_ratio) / 5 — normalized to ~1 for typical viable configs
    """
    mb = entry.get("mass_bin", "unknown")
    if mb in target_bins:
        mass_bonus = 2.0 if mb == "sub_solar_gap" else 1.5
    else:
        mass_bonus = 0.5

    ns = entry.get("n_s", None)
    if ns is not None:
        ns_bonus = np.exp(-0.5 * ((ns - TARGET_NS) / 0.04) ** 2)
    else:
        ns_bonus = 0.5

    ratio = entry.get("P_S_peak_ratio", None)
    if ratio is not None and ratio > 0:
        ratio_bonus = np.log10(ratio) / 5.0
    else:
        ratio_bonus = 0.5

    return mass_bonus * ns_bonus * ratio_bonus


# ── Cross-referencing ───────────────────────────────────────────────────────


def bg_match_key(entry):
    """Create a match key for background-vs-MS comparison."""
    return (
        entry.get("x_c"),
        entry.get("c"),
        entry.get("beta"),
        entry.get("chi0"),
        entry.get("N_star"),
    )


def cross_reference_ms_vs_bg(ms_data, bg_data):
    """Compare MS n_s vs background SR n_s for matching configs."""
    bg_lookup = {}
    for b in bg_data:
        bg_lookup[bg_match_key(b)] = b

    comparisons = []
    for ms in ms_data:
        key = bg_match_key(ms)
        bg = bg_lookup.get(key)
        if bg and "n_s" in ms and "n_s_sr" in bg:
            comparisons.append(
                {
                    "x_c": ms["x_c"],
                    "c": ms["c"],
                    "beta": ms["beta"],
                    "chi0": ms["chi0"],
                    "N_star": ms["N_star"],
                    "n_s_ms": ms["n_s"],
                    "n_s_sr": bg["n_s_sr"],
                    "delta_ns": ms["n_s"] - bg["n_s_sr"],
                    "usr_type": bg.get("usr_type", "?"),
                }
            )
    return comparisons


# ── Sweep-by-sweep analysis ────────────────────────────────────────────────


def analyze_sweep_statistics(sweeps):
    """Compute statistics per sweep file."""
    stats = {}
    for fname, data in sweeps.items():
        total = len(data)
        errors = [d for d in data if "error" in d]
        error_types = Counter(d.get("error", "") for d in errors)

        good = [d for d in data if "error" not in d]
        filtered = []
        reasons = Counter()
        for d in good:
            ok, reason = quality_filter(d)
            if ok:
                filtered.append(d)
            else:
                reasons[reason] += 1

        # Mass bins among filtered
        mass_bins = Counter()
        ns_vals = []
        a_s_vals = []
        k_peaks = []
        for d in filtered:
            mb = d.get("mass_bin", classify_mass(d.get("M_kpeak", 0)))
            mass_bins[mb] += 1
            if "n_s" in d and d["n_s"] is not None:
                ns_vals.append(d["n_s"])
            if "A_s_at_cmb" in d and d["A_s_at_cmb"] is not None:
                a_s_vals.append(d["A_s_at_cmb"])
            if "k_peak" in d:
                k_peaks.append(d["k_peak"])

        stats[fname] = {
            "total": total,
            "errors": len(errors),
            "error_types": dict(error_types),
            "good": len(good),
            "passed_filters": len(filtered),
            "rejected": dict(reasons.most_common(10)),
            "mass_bins": dict(mass_bins),
            "n_s_range": (min(ns_vals), max(ns_vals)) if ns_vals else None,
            "a_s_range": (min(a_s_vals), max(a_s_vals)) if a_s_vals else None,
            "has_ms": any("k_peak" in d and d.get("k_peak", 0) > 1e6 for d in filtered),
        }
    return stats


# ── Report generation ───────────────────────────────────────────────────────


def build_markdown_report(stats, rankings, cross_refs):
    """Build a structured Markdown report."""
    lines = []
    lines.append("# Ezquiaga CHI Sweep Analysis Report")
    lines.append(f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")

    # Summary table
    lines.append("## 1. Data Inventory\n")
    lines.append("| File | Total | Errors | Passed | Key Rejections | Mass Bins |")
    lines.append("|------|-------|--------|--------|----------------|-----------|")
    for fname, s in sorted(stats.items()):
        top_rej = "; ".join(f"{r}: {n}" for r, n in list(s["rejected"].items())[:2])
        mb_str = "; ".join(f"{b}: {n}" for b, n in s["mass_bins"].items())
        _fname = fname
        lines.append(
            f"| {_fname} | {s['total']} | {s['errors']} | {s['passed_filters']} "
            f"| {top_rej} | {mb_str} |"
        )

    # Top rankings
    lines.append("\n## 2. Top-Ranked Configs\n")
    lines.append(
        "| Rank | File | x_c | c | β | χ₀ | N* | k_peak | M_k [M_⊙] | n_s | A_s | Ratio | Mass Bin | Score |"
    )
    lines.append(
        "|------|------|-----|----|-------|----|----|--------|----------|------|------|-------|----------|-------|"
    )

    for rank, (entry, fname) in enumerate(rankings[:30], 1):
        mbin = entry.get("mass_bin", classify_mass(entry.get("M_kpeak", 0)))
        ns_str = f"{entry.get('n_s', 0):.4f}" if entry.get("n_s") else "N/A"
        as_str = (
            f"{entry.get('A_s_at_cmb', 0):.3e}" if entry.get("A_s_at_cmb") else "N/A"
        )
        ratio = entry.get("P_S_peak_ratio", 0)
        score = compute_score(entry)
        lines.append(
            f"| {rank} | {fname} | {entry['x_c']:.4f} | {entry['c']:.4f} | "
            f"{entry['beta']:.1e} | {entry.get('chi0', '?')} | {entry.get('N_star', '?')} | "
            f"{entry.get('k_peak', 0):.3e} | {entry.get('M_kpeak', 0):.3e} | "
            f"{ns_str} | {as_str} | {ratio:.1e} | {mbin} | {score:.3f} |"
        )

    # Best by target mass
    lines.append("\n## 3. Best Configs by Target Mass Gap\n")
    for target in ["asteroid_gap", "sub_solar_gap"]:
        lines.append(f"\n### {target}\n")
        target_configs = [
            (e, f)
            for e, f in rankings
            if e.get("mass_bin", classify_mass(e.get("M_kpeak", 0))) == target
        ][:10]
        if not target_configs:
            lines.append("_None found._\n")
            continue
        lines.append(
            "| Rank | File | x_c | c | β | χ₀ | N* | k_peak | M_k [M_⊙] | n_s | A_s | Ratio | Score |"
        )
        lines.append(
            "|------|------|-----|----|-------|----|----|--------|----------|------|------|-------|-------|"
        )
        for rank, (e, f) in enumerate(target_configs, 1):
            score = compute_score(e)
            lines.append(
                f"| {rank} | {f} | {e['x_c']:.4f} | {e['c']:.4f} | {e['beta']:.1e} | "
                f"{e.get('chi0', '?')} | {e.get('N_star', '?')} | "
                f"{e.get('k_peak', 0):.3e} | {e.get('M_kpeak', 0):.3e} | "
                f"{e.get('n_s', 0):.4f} | {e.get('A_s_at_cmb', 0):.3e} | "
                f"{e.get('P_S_peak_ratio', 0):.1e} | {score:.3f} |"
            )

    # MS vs BG comparison
    if cross_refs:
        lines.append("\n## 4. MS vs Background n_s Comparison\n")
        lines.append("| x_c | c | β | χ₀ | N* | n_s (MS) | n_s (SR) | Δ | USR Type |")
        lines.append(
            "|-----|----|-------|----|----|----------|----------|----|----------|"
        )
        for c in cross_refs:
            lines.append(
                f"| {c['x_c']:.4f} | {c['c']:.4f} | {c['beta']:.1e} | "
                f"{c['chi0']} | {c['N_star']} | {c['n_s_ms']:.4f} | {c['n_s_sr']:.4f} | "
                f"{c['delta_ns']:+.5f} | {c['usr_type']} |"
            )

    # Key issues
    lines.append("\n## 5. Known Issues\n")
    all_rejected = Counter()
    for _, s in stats.items():
        for reason, n in s["rejected"].items():
            all_rejected[reason] += n
    for reason, n in all_rejected.most_common(10):
        lines.append(f"- **{n}x** {reason}")

    return "\n".join(lines)


# ── Diagnostic plots ────────────────────────────────────────────────────────


def plot_ranking_scatter(rankings, filename="ranking_overview", category="pbh"):
    """Scatter plot of n_s vs M_kpeak colored by score."""
    points = []
    for entry, _ in rankings[:100]:
        mbin = entry.get("mass_bin", classify_mass(entry.get("M_kpeak", 0)))
        points.append(
            {
                "M": entry.get("M_kpeak", entry.get("M_present", 0)),
                "ns": entry.get("n_s", 0),
                "ratio": entry.get("P_S_peak_ratio", 0),
                "mbin": mbin,
                "label": f"{entry.get('x_c', 0):.3f}_{entry.get('c', 0):.2f}_β{entry.get('beta', 0):.1e}",
            }
        )

    if not points:
        print("  No points to plot")
        return

    with plt.rc_context(PAPER_RCPARAMS):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

        # Left: n_s vs M
        color_map = {
            "asteroid_gap": TOL["teal"],
            "sub_solar_gap": TOL["green"],
            "sub_stellar": TOL["blue"],
            "stellar_ligo": TOL["red"],
            "massive": TOL["grey"],
            "intermediate": TOL["yellow"],
            "too_light": TOL["grey"],
        }
        for p in points:
            color = color_map.get(p["mbin"], TOL["grey"])
            size = max(10, min(80, np.log10(max(p["ratio"], 1)) * 10))
            ax1.scatter(p["ns"], p["M"], c=color, s=size, alpha=0.6, edgecolors="none")

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

        # Add mass gap annotations
        ax1.text(1.08, 5e-3, "sub-solar", fontsize=7, color=TOL["green"], ha="right")
        ax1.text(1.08, 5e-11, "asteroid", fontsize=7, color=TOL["teal"], ha="right")
        ax1.text(1.08, 50, "LIGO ruled", fontsize=7, color=TOL["red"], ha="right")

        # Right: histogram of mass bins
        bins = Counter(p["mbin"] for p in points)
        labels = []
        counts = []
        colors_bar = []
        order = [
            "asteroid_gap",
            "sub_solar_gap",
            "sub_stellar",
            "stellar_ligo",
            "intermediate",
            "massive",
            "too_light",
        ]
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
        save_fig(fig, filename, category)


def plot_ns_vs_beta(rankings, filename="ns_vs_beta", category="pbh"):
    """n_s vs β colored by c value for top configs."""
    points = [(e, f) for e, f in rankings if "n_s" in e and e["n_s"] is not None]
    if not points:
        return

    with plt.rc_context(PAPER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(4.5, 3.0))

        c_vals = sorted({e["c"] for e, _ in points})
        cmap = plt.colormaps["viridis"]
        norm = LogNorm(vmin=max(min(c_vals), 0.5), vmax=max(c_vals))

        for entry, _ in points:
            color = cmap(norm(entry["c"]))
            ax.scatter(
                entry["beta"],
                entry["n_s"],
                c=[color],
                s=20,
                alpha=0.7,
                edgecolors="none",
            )

        ax.axhline(TARGET_NS, color=TOL["grey"], ls="--", lw=0.8)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$n_s$")
        ax.set_xscale("log")
        ax.set_xlim(1e-6, 1e-3)
        ax.grid(True, alpha=0.2)

        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label(r"$c$", fontsize=8)

        fig.tight_layout()
        save_fig(fig, filename, category)


def plot_kpeak_vs_ns_grid(rankings, filename="kpeak_vs_ns", category="pbh"):
    """k_peak vs n_s colored by c, sized by P_S ratio."""
    points = [
        (e, f)
        for e, f in rankings
        if "n_s" in e and e["n_s"] is not None and e.get("k_peak", 0) > 0
    ]
    if not points:
        return

    with plt.rc_context(PAPER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(5.0, 3.5))

        c_vals = sorted({e["c"] for e, _ in points})
        norm = LogNorm(vmin=max(min(c_vals), 0.5), vmax=max(c_vals))
        cmap = plt.colormaps["viridis"]

        for entry, _ in points:
            color = cmap(norm(entry["c"]))
            ratio = max(entry.get("P_S_peak_ratio", 1e3), 1e3)
            size = max(15, min(100, np.log10(ratio) * 8))
            ax.scatter(
                entry["n_s"],
                entry["k_peak"],
                c=[color],
                s=size,
                alpha=0.6,
                edgecolors="none",
            )

        ax.axvline(TARGET_NS, color=TOL["grey"], ls="--", lw=0.8)
        ax.axhline(2e11, color=TOL["red"], ls=":", lw=0.6)  # sub-solar boundary
        ax.axhline(6e17, color=TOL["teal"], ls=":", lw=0.6)  # asteroid boundary
        ax.set_xlabel(r"$n_s$")
        ax.set_ylabel(r"$k_{\mathrm{peak}}$ [Mpc$^{-1}$]")
        ax.set_yscale("log")
        ax.set_xlim(0.90, 1.10)
        ax.grid(True, alpha=0.2)

        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label(r"$c$", fontsize=8)

        fig.tight_layout()
        save_fig(fig, filename, category)


# ── Main pipeline ───────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Systematic Ezquiaga CHI sweep analysis")
    p.add_argument(
        "--show-top", type=int, default=20, help="Number of top configs to display"
    )
    p.add_argument("--quick", action="store_true", help="Summary only, no plots")
    p.add_argument(
        "--output",
        default="ezquiaga_analysis_report.md",
        help="Output markdown filename (in logs dir)",
    )
    args = p.parse_args()

    print("=" * 70)
    print("Ezquiaga CHI Sweep Analysis — Systematic Pipeline")
    print("=" * 70)

    # Step 1: Load all data
    print("\n[1] Loading sweep data...")
    sweeps = load_all_sweeps()
    print(
        f"  Loaded {len(sweeps)} sweep files, {sum(len(v) for v in sweeps.values())} total entries"
    )

    # Step 2: Compute per-file statistics
    print("\n[2] Computing sweep statistics...")
    stats = analyze_sweep_statistics(sweeps)
    total_passed = sum(s["passed_filters"] for s in stats.values())
    total_rejected = sum(s["good"] - s["passed_filters"] for s in stats.values())
    total_errors = sum(s["errors"] for s in stats.values())
    print(
        f"  {total_passed} passed filters, {total_rejected} rejected, {total_errors} errors"
    )

    for fname, s in sorted(stats.items()):
        mb = ", ".join(f"{b}:{n}" for b, n in s["mass_bins"].items())
        print(
            f"  {fname:40s} {s['good']:>4} good → {s['passed_filters']:>4} passed [{mb}]"
        )

    # Step 3: Collect and rank all passing entries
    print("\n[3] Ranking configs...")
    all_good = []

    for fname, data in sweeps.items():
        for entry in data:
            ok, _ = quality_filter(entry)
            if ok:
                # Ensure mass_bin is set
                if "mass_bin" not in entry:
                    entry["mass_bin"] = classify_mass(
                        entry.get("M_kpeak", entry.get("M_present", 0))
                    )
                # Classify k_peak
                kp = entry.get("k_peak", 0)
                if kp > 1e17:
                    entry["k_type"] = "asteroid_scale"
                elif kp > 1e11:
                    entry["k_type"] = "subsolar_scale"
                elif kp > 1e6:
                    entry["k_type"] = "stellar_scale"
                else:
                    entry["k_type"] = "cmb_scale"

                all_good.append((entry, fname))

    # Sort by score descending
    all_good.sort(key=lambda x: compute_score(x[0]), reverse=True)

    print(f"  Total passing configs: {len(all_good)}")
    print(f"\n  Top {args.show_top}:")
    print(
        f"  {'Rank':<5} {'File':<25} {'Params':<35} {'M_k [M_⊙]':<14} {'n_s':<8} {'Score':<8}"
    )
    print(f"  {'-' * 5} {'-' * 25} {'-' * 35} {'-' * 14} {'-' * 8} {'-' * 8}")
    for i, (entry, fname) in enumerate(all_good[: args.show_top]):
        params = f"xc={entry['x_c']:.3f} c={entry['c']:.2f} β={entry['beta']:.1e}"
        if "chi0" in entry:
            params += f" χ₀={entry['chi0']}"
        if "N_star" in entry:
            params += f" N*={entry['N_star']}"
        mk = entry.get("M_kpeak", entry.get("M_present", 0))
        ns_str = f"{entry.get('n_s', 0):.4f}" if entry.get("n_s") else "N/A"
        score = compute_score(entry)
        print(
            f"  {i + 1:<5} {fname:<25} {params:<35} {mk:<14.3e} {ns_str:<8} {score:<8.3f}"
        )

    # Step 4: Cross-reference MS vs background
    print("\n[4] Cross-referencing MS vs background predictions...")
    ms_files = [f for f in sweeps if f.startswith(("ms_sweep", "ms_regime"))]
    bg_files = [f for f in sweeps if f.startswith("chi0_sweep")]
    all_cross_refs = []
    for msf in ms_files:
        for bgf in bg_files:
            cross = cross_reference_ms_vs_bg(sweeps[msf], sweeps[bgf])
            all_cross_refs.extend(cross)
    if all_cross_refs:
        deltas = [c["delta_ns"] for c in all_cross_refs]
        print(f"  {len(all_cross_refs)} matching pairs found")
        print(
            f"  MS - SR n_s: min={min(deltas):+.4f} max={max(deltas):+.4f} mean={np.mean(deltas):+.4f}"
        )
    else:
        print("  No matching MS/BG pairs found")

    # Step 5: Generate markdown report
    print("\n[5] Generating report...")
    report = build_markdown_report(stats, all_good, all_cross_refs)

    report_path = get_path("logs", args.output)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report: {report_path}")

    # Also write structured JSON
    json_data = {
        "generated": datetime.now().isoformat(),
        "statistics": {
            f: {k: v for k, v in s.items() if k != "rejected"} for f, s in stats.items()
        },
        "top_configs": [],
    }
    for entry, fname in all_good[:50]:
        json_data["top_configs"].append(
            {
                "source": fname,
                "x_c": entry["x_c"],
                "c": entry["c"],
                "beta": entry["beta"],
                "chi0": entry.get("chi0"),
                "N_star": entry.get("N_star"),
                "k_peak": entry.get("k_peak"),
                "M_kpeak": entry.get("M_kpeak", entry.get("M_present")),
                "n_s": entry.get("n_s"),
                "A_s_at_cmb": entry.get("A_s_at_cmb"),
                "P_S_peak_ratio": entry.get("P_S_peak_ratio"),
                "mass_bin": entry.get(
                    "mass_bin", classify_mass(entry.get("M_kpeak", 0))
                ),
                "score": compute_score(entry),
            }
        )

    json_path = os.path.join(os.path.dirname(report_path), "ezquiaga_top_configs.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  JSON: {json_path}")

    # Step 6: Diagnostic plots
    if not args.quick:
        print("\n[6] Generating plots...")
        plot_ranking_scatter(all_good)
        plot_ns_vs_beta(all_good)
        plot_kpeak_vs_ns_grid(all_good)

    # Step 7: Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Sweep files:   {len(sweeps)}")
    print(f"  Total entries: {sum(len(v) for v in sweeps.values())}")
    print(f"  Errors:        {total_errors}")
    print(f"  Passed filters: {total_passed}")
    print(f"  Top configs:   {len(all_good[:10])}")

    # Best by target
    for target in ["sub_solar_gap", "asteroid_gap"]:
        target_list = [
            (e, f)
            for e, f in all_good
            if e.get("mass_bin", classify_mass(e.get("M_kpeak", 0))) == target
        ]
        print(f"  {target}: {len(target_list)} viable configs")
        if target_list:
            best = target_list[0][0]
            print(
                f"    Best: xc={best['x_c']:.4f} c={best['c']:.4f} β={best['beta']:.1e} "
                f"M_k={best.get('M_kpeak', 0):.3e} n_s={best.get('n_s', 0):.4f}"
            )

    print(f"\n  Report: {report_path}")
    print("  To re-run: python scripts/analyze_sweep_results.py")


if __name__ == "__main__":
    main()
