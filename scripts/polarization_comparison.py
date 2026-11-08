"""
Planck EE/TE comparison for the USR power-suppression model.

This is the polarization counterpart of the TT analysis in the draft: the
same `P_S(k)` is injected into CAMB, the same ΛCDM baseline is computed, and
the resulting D_ell^EE and D_ell^TE are compared with Planck 2018 binned
polarization spectra — at the model's **own** amplitude, exactly as the
draft does for TT (no amplitude profiling, no τ scan).

Datasets used:

    TT : Commander ℓ=2–29 + binned PLIK ℓ≈47–2500
    EE : low-ℓ EE ℓ=2–29 + binned EE ℓ≈47–2500
    TE : binned TE ℓ≈47–2500   (low-ℓ TE is noise-dominated and not used)

Columns that are compared are the ones Planck actually constrains; the
χ² conventions (diagonal, residual-sign error bar) are the repo's existing
ones.

Outputs (flat, via scripts.plotting):
    outputs/plots/powerloss/planck_ee_te_phi…_y0…_nstar….png   3-panel overview
    outputs/plots/powerloss/planck_lowell_phi…_y0…_nstar….png  ℓ ≤ 40 zoom
    outputs/simulations/logs/polarization_…_….json + .csv      χ² table

Usage:
    python -m scripts.polarization_comparison \\
        --config configs/tau_profile/higgs_usr_good.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from scripts.chi2_analysis import (
    chi2_binned,
    chi2_commander,
    chi2_ee_binned,
    chi2_ee_low,
    chi2_te_binned,
)
from scripts.constants import CAMB_COSMOLOGY, ROOT_DIR
from scripts.planck_data import (
    C_ell_to_d_ell,
    load_planck_binned,
    load_planck_ee_binned,
    load_planck_ee_low,
    load_planck_te_binned,
    get_planck_data_asymmetric,
)
from scripts.plotting import PAPER_RCPARAMS, TOL, get_path, make_filename

#: name -> (chi2 function, "tt" | "ee" | "te")
DATASETS = (
    ("tt_commander", chi2_commander, "tt"),
    ("tt_binned", chi2_binned, "tt"),
    ("ee_low", chi2_ee_low, "ee"),
    ("ee_binned", chi2_ee_binned, "ee"),
    ("te_binned", chi2_te_binned, "te"),
)

PANEL_TEXT = {
    "tt": r"$D_\ell^{TT}\ [\mu\mathrm{K}^2]$",
    "ee": r"$D_\ell^{EE}\ [\mu\mathrm{K}^2]$",
    "te": r"$D_\ell^{TE}\ [\mu\mathrm{K}^2]$",
}


# ── config ─────────────────────────────────────────────────────────────────


def load_config(path):
    """Load a config that carries a ``configs`` list (the τ-profile schema)."""
    with open(path) as fh:
        cfg = json.load(fh)
    if not isinstance(cfg, dict) or "configs" not in cfg:
        raise ValueError("config must contain a 'configs' list")
    entries = cfg["configs"]
    if not entries:
        raise ValueError("config 'configs' must not be empty")
    for i, e in enumerate(entries):
        for key in ("label", "phi0", "y0", "n_star"):
            if key not in e:
                raise ValueError(f"configs[{i}] is missing '{key}'")
    cfg["_path"] = os.path.abspath(path)
    return cfg


def load_model_spectrum(entry):
    """Locate and load the cached P_S(k) for one config entry."""
    from scripts.plotting import find_ps

    path, _ = find_ps(entry["phi0"], entry["y0"], entry["n_star"])
    if path is None:
        raise FileNotFoundError(
            f"no cached P_S(k) for phi0={entry['phi0']}, y0={entry['y0']}, "
            f"n_star={entry['n_star']} (label '{entry['label']}')"
        )
    with open(path) as fh:
        raw = json.load(fh)
    spec = raw["spectrum"]
    return {"k_phys": spec["k_phys"], "P_S": spec["P_S"]}, path


# ── analysis ───────────────────────────────────────────────────────────────


def analyse(entry, ell_max=2500):
    """Compute model and ΛCDM TT/TE/EE and every dataset χ² for one config."""
    from scripts.camb_wrapper import compute_cl_camb_powerlaw, compute_cl_full_camb

    ps, ps_path = load_model_spectrum(entry)
    ells, C_TT, C_TE, C_EE = compute_cl_full_camb(ps, ell_max=ell_max)
    ells_l, L_TT, L_TE, L_EE = compute_cl_camb_powerlaw(ell_max=ell_max)

    model = {
        "tt": C_ell_to_d_ell(ells, C_TT),
        "te": C_ell_to_d_ell(ells, C_TE),
        "ee": C_ell_to_d_ell(ells, C_EE),
    }
    lcdm = {
        "tt": C_ell_to_d_ell(ells_l, L_TT),
        "te": C_ell_to_d_ell(ells_l, L_TE),
        "ee": C_ell_to_d_ell(ells_l, L_EE),
    }

    rows = []
    for name, fn, kind in DATASETS:
        cm, cl, n = fn(model[kind], ells, lcdm[kind], ells_l)
        rows.append({"dataset": name, "kind": kind, "n": n,
                     "chi2_model": cm, "chi2_lcdm": cl, "delta": cm - cl})

    return {
        "label": entry["label"],
        "phi0": float(entry["phi0"]),
        "y0": float(entry["y0"]),
        "n_star": float(entry["n_star"]),
        "p_s_file": os.path.basename(ps_path),
        "ells": ells,
        "model": model,
        "lcdm": lcdm,
        "chi2": rows,
        "tau": CAMB_COSMOLOGY["tau"],
    }


def summary_row(result):
    """Collapse the per-dataset χ² into one table row."""
    by = {r["dataset"]: r for r in result["chi2"]}
    tt = by["tt_commander"]["chi2_model"] + by["tt_binned"]["chi2_model"]
    tt_l = by["tt_commander"]["chi2_lcdm"] + by["tt_binned"]["chi2_lcdm"]
    ee = by["ee_low"]["chi2_model"] + by["ee_binned"]["chi2_model"]
    ee_l = by["ee_low"]["chi2_lcdm"] + by["ee_binned"]["chi2_lcdm"]
    te = by["te_binned"]["chi2_model"]
    te_l = by["te_binned"]["chi2_lcdm"]
    return {
        "label": result["label"],
        "phi0": result["phi0"], "y0": result["y0"], "n_star": result["n_star"],
        "chi2_tt": tt, "chi2_tt_lcdm": tt_l, "dchi2_tt": tt - tt_l,
        "chi2_ee": ee, "chi2_ee_lcdm": ee_l, "dchi2_ee": ee - ee_l,
        "chi2_te": te, "chi2_te_lcdm": te_l, "dchi2_te": te - te_l,
        "p_s_file": result["p_s_file"],
    }


# ── figures ────────────────────────────────────────────────────────────────


def _draw_panel(ax, kind, result, ell_max_data=2500):
    """One D_ell panel: Planck points, model curve, ΛCDM curve.

    Every series is masked to ``ell_max_data`` so a zoom panel stays a zoom.
    """
    if kind == "tt":
        e_lo, d_lo, lo_lo, hi_lo = get_planck_data_asymmetric()
        e_b, d_b, lo_b, hi_b = load_planck_binned()
    elif kind == "ee":
        e_lo, d_lo, lo_lo, hi_lo = load_planck_ee_low()
        e_b, d_b, lo_b, hi_b = load_planck_ee_binned()
    else:
        e_lo = d_lo = lo_lo = hi_lo = None
        e_b, d_b, lo_b, hi_b = load_planck_te_binned()

    if e_lo is not None:
        m = e_lo <= ell_max_data
        ax.errorbar(e_lo[m], d_lo[m], yerr=[lo_lo[m], hi_lo[m]], fmt="o",
                    ms=2.4, color=TOL["dark"], elinewidth=0.7, capsize=0,
                    label="Planck low-$\\ell$", zorder=3)
    m = e_b <= ell_max_data
    ax.errorbar(e_b[m], d_b[m], yerr=[lo_b[m], hi_b[m]], fmt="o", ms=2.4,
                color=TOL["grey"], elinewidth=0.7, capsize=0,
                label="Planck binned", zorder=3)

    ells = result["ells"]
    m = ells <= ell_max_data
    ax.plot(ells[m], result["model"][kind][m], color=TOL["blue"], lw=1.2,
            label="USR model", zorder=4)
    ax.plot(ells[m], result["lcdm"][kind][m], color=TOL["red"], lw=1.0,
            ls="--", label=r"$\Lambda$CDM", zorder=4)

    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(PANEL_TEXT[kind])
    ax.tick_params(direction="in", top=True, right=True)
    if kind == "te":
        ax.axhline(0.0, color=TOL["dark"], lw=0.5, alpha=0.5)


def make_figures(result, out_dir=get_path):
    """Overview (TT|EE|TE) and low-ℓ zoom figures for one config."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(PAPER_RCPARAMS)
    base = dict(phi0=result["phi0"], y0=result["y0"], nstar=result["n_star"])
    written = []

    # Overview: TT, EE, TE on the full multipole range.
    # EE/TE are truncated at the multipole where Planck's errors exceed half
    # the signal: beyond that the points are noise around zero and plotting
    # them against the physical curve is visually misleading.
    xmax = {"tt": 2500, "ee": 1500, "te": 1500}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    for ax, kind in zip(axes, ("tt", "ee", "te")):
        _draw_panel(ax, kind, result, ell_max_data=xmax[kind])
        ax.set_xlim(1.5, xmax[kind] + 0.5)
    axes[0].legend(frameon=False, fontsize=6, loc="upper right")
    fig.subplots_adjust(wspace=0.42)
    path = out_dir("powerloss",
                   make_filename("planck_ee_te", ext=".png", **base))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # Zoom: ℓ ≤ 40, where the suppression actually lives.
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.2))
    for ax, kind in zip(axes, ("tt", "ee")):
        _draw_panel(ax, kind, result, ell_max_data=40)
        ax.set_xlim(1.5, 40.5)
    axes[0].set_ylabel(PANEL_TEXT["tt"])
    axes[0].legend(frameon=False, fontsize=6, loc="upper right")
    fig.subplots_adjust(wspace=0.42)
    path = out_dir("powerloss",
                   make_filename("planck_lowell", ext=".png", **base))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    return written


# ── output ─────────────────────────────────────────────────────────────────


def write_table(results, cfg):
    """Per-config JSON + CSV χ² tables in outputs/simulations/logs/."""
    written = []
    for res, summ in results:
        base = dict(phi0=res["phi0"], y0=res["y0"], nstar=res["n_star"])
        payload = {
            "metadata": {
                "generated_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"),
                "config_path": cfg.get("_path"),
                "tau": res["tau"],
                "ell_max": int(res["ells"].max()),
                "assumptions": (
                    "diagonal uncorrelated errors; no covariance, no "
                    "foregrounds; model amplitude fixed at its own P_S "
                    "normalisation (same convention as the TT analysis)"
                ),
                "git": _git_metadata(),
            },
            "label": res["label"],
            "summary": summ,
            "datasets": res["chi2"],
        }
        path = get_path("logs", make_filename("polarization", ext=".json",
                                              **base))
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        written.append(path)

        path = get_path("logs", make_filename("polarization", ext=".csv",
                                              **base))
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "kind", "n", "chi2_model", "chi2_lcdm",
                        "delta"])
            for r in res["chi2"]:
                w.writerow([r["dataset"], r["kind"], r["n"],
                            f"{r['chi2_model']:.4f}", f"{r['chi2_lcdm']:.4f}",
                            f"{r['delta']:.4f}"])
        written.append(path)
    return written


def _git_metadata():
    try:
        sha = subprocess.check_output(
            ["git", "-C", ROOT_DIR, "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", ROOT_DIR, "status", "--porcelain"], text=True).strip())
        return {"commit": sha, "dirty": dirty}
    except Exception:  # pragma: no cover
        return {"commit": None, "dirty": None}


# ── CLI ────────────────────────────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare model EE/TE (and TT) with Planck at fixed A_s."
    )
    parser.add_argument("--config", required=True,
                        help="config JSON carrying a 'configs' list")
    parser.add_argument("--ell-max", type=int, default=2500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.dry_run:
        print(f"config: {cfg['_path']}")
        for e in cfg["configs"]:
            print(f"  {e['label']:<16} phi0={e['phi0']} y0={e['y0']} "
                  f"n_star={e['n_star']}")
        print(f"datasets: {[d[0] for d in DATASETS]}")
        print(f"ell_max : {args.ell_max}")
        return 0

    results = []
    print(f"  {'config':<17} {'χ²_TT':>9} {'ΔTT':>7} {'χ²_EE':>9} {'ΔEE':>7} "
          f"{'χ²_TE':>9} {'ΔTE':>7}")
    for entry in cfg["configs"]:
        res = analyse(entry, ell_max=args.ell_max)
        summ = summary_row(res)
        results.append((res, summ))
        print(f"  {summ['label']:<17} {summ['chi2_tt']:9.2f} "
              f"{summ['dchi2_tt']:+7.2f} {summ['chi2_ee']:9.2f} "
              f"{summ['dchi2_ee']:+7.2f} {summ['chi2_te']:9.2f} "
              f"{summ['dchi2_te']:+7.2f}")

    written = write_table(results, cfg)
    for res, _ in results:
        written += make_figures(res)
    print(f"\nwrote {len(written)} files")
    for p in written:
        print("  ", os.path.relpath(p, ROOT_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
