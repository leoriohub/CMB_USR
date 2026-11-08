"""
τ-profile scan of the USR low-ℓ power suppression.

For each configuration in a JSON config file this module computes, on a
common grid of the reionization optical depth τ,

    Δχ²(τ) = χ²_model(τ) − χ²_ΛCDM(τ)

for three dataset combinations:

    ``tt``           Planck TT only (commander ℓ≤29 + binned ℓ≈47–2500)
    ``tt_ee_te``     TT plus binned EE and TE (Planck PR3/PR3.02)
    ``tt_ee_te_bao`` the same, with the Gaussian BAO τ prior added

Both the model and the ΛCDM baseline are evaluated at the *same* τ, which
is what makes Δχ² sensitive to the shape of the reionization signature
rather than to an overall amplitude offset.

Physics target: the suppression mechanism lowers power at ℓ ≲ 30, exactly
where the reionization bump constrains τ. Profiling τ therefore answers
whether the dip *absorbs* the low-τ/high-τ tension or *fights* it.

Assumptions (all inherited from the TT-only analysis):
  * diagonal, uncorrelated errors — no covariance, no foreground nuisance
    parameters, no exact non-Gaussian low-ℓ likelihood;
  * A_s is fixed at the model's own P_S(k) normalisation, i.e. τ is
    profiled, not marginalised over an amplitude;
  * ΛCDM baseline = power law with ``As``/``ns`` from ``scripts.constants``.

Usage::

    python scripts/tau_profile.py --config configs/tau_profile/higgs_usr_good.json
    python scripts/tau_profile.py --config <cfg> --dry-run

Outputs (paths from ``scripts.plotting``, never hardcoded):
  * per-config JSON + CSV tables in ``outputs/simulations/logs/``
  * a summary JSON in ``outputs/simulations/scans/``
  * a publication PNG in ``outputs/plots/powerloss/``
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

from scripts.chi2_analysis import (
    chi2_binned,
    chi2_commander,
    chi2_ee_binned,
    chi2_ee_low,
    chi2_te_binned,
    chi2_unbinned,
)
from scripts.constants import CAMB_COSMOLOGY, ROOT_DIR
from scripts.planck_data import C_ell_to_d_ell
from scripts.plotting import (
    PAPER_RCPARAMS,
    TOL,
    get_path,
    make_filename,
)

#: Per-spectrum χ² functions. Signature: (D_model, ells, D_lcdm, ells_lcdm).
COMPONENT_FUNCS = {
    "commander": chi2_commander,
    "unbinned": chi2_unbinned,
    "binned": chi2_binned,
    "ee_low": chi2_ee_low,
    "ee_binned": chi2_ee_binned,
    "te_binned": chi2_te_binned,
}
COMBOS = ("tt", "tt_ee_te", "tt_ee_te_bao")
COMBO_LABELS = {
    "tt": r"TT",
    "tt_ee_te": r"TT+EE+TE",
    "tt_ee_te_bao": r"TT+EE+TE + BAO $\tau$",
}
COMBO_COLORS = {
    "tt": TOL["blue"],
    "tt_ee_te": TOL["green"],
    "tt_ee_te_bao": TOL["red"],
}

REQUIRED_TOP = ("tag", "tau_grid", "configs")
REQUIRED_ENTRY = ("label", "phi0", "y0", "n_star")


# ── Config handling ────────────────────────────────────────────────────────


def load_config(path):
    """Load and validate a τ-profile config, annotating its path and hash."""
    with open(path, "rb") as fh:
        raw = fh.read()
    cfg = json.loads(raw.decode("utf-8"))
    validate_config(cfg)
    cfg["_path"] = os.path.abspath(path)
    cfg["_sha256"] = hashlib.sha256(raw).hexdigest()
    return cfg


def validate_config(cfg):
    """Raise ValueError with a precise message when the config is unusable."""
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a JSON object")

    for key in REQUIRED_TOP:
        if key not in cfg:
            raise ValueError(f"config is missing required key '{key}'")

    tag = cfg["tag"]
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError("config 'tag' must be a non-empty string")

    tau_grid = cfg["tau_grid"]
    if not isinstance(tau_grid, dict):
        raise ValueError("config 'tau_grid' must be an object")
    for key in ("min", "max", "n"):
        if key not in tau_grid:
            raise ValueError(f"'tau_grid' is missing '{key}'")
    lo, hi = float(tau_grid["min"]), float(tau_grid["max"])
    n = int(tau_grid["n"])
    if n < 2:
        raise ValueError(f"'tau_grid.n' must be >= 2 (got {n})")
    if not lo < hi:
        raise ValueError(f"'tau_grid.min' ({lo}) must be < 'tau_grid.max' ({hi})")
    if lo < 0.0:
        raise ValueError(f"'tau_grid.min' ({lo}) must be >= 0")

    ell_max = cfg.get("ell_max", 2500)
    if int(ell_max) < 30:
        raise ValueError(f"'ell_max' must be >= 30 (got {ell_max})")

    components = cfg.get("tt_components", ["commander", "binned"])
    if not isinstance(components, list) or not components:
        raise ValueError("'tt_components' must be a non-empty list")
    unknown = [c for c in components if c not in COMPONENT_FUNCS]
    if unknown:
        raise ValueError(
            f"unknown TT component(s) {unknown}; "
            f"choose from {sorted(COMPONENT_FUNCS)}"
        )
    if "unbinned" in components and "commander" in components:
        raise ValueError(
            "'unbinned' already contains the ℓ≤29 Commander multipoles; "
            "pairing them double-counts — use ['commander', 'binned']"
        )

    pol = cfg.get("polarization_components",
                  ["ee_low", "ee_binned", "te_binned"])
    if not isinstance(pol, list) or not pol:
        raise ValueError("'polarization_components' must be a non-empty list")
    unknown_pol = [c for c in pol if c not in COMPONENT_FUNCS]
    if unknown_pol:
        raise ValueError(
            f"unknown polarization component(s) {unknown_pol}; "
            f"choose from {sorted(COMPONENT_FUNCS)}"
        )

    bao = cfg.get("bao_prior")
    if bao is not None:
        if not isinstance(bao, dict) or "tau" not in bao or "sigma" not in bao:
            raise ValueError("'bao_prior' needs both 'tau' and 'sigma'")
        if float(bao["sigma"]) <= 0.0:
            raise ValueError("'bao_prior.sigma' must be > 0")

    configs = cfg["configs"]
    if not isinstance(configs, list) or not configs:
        raise ValueError("config 'configs' must be a non-empty list")
    seen = set()
    for i, entry in enumerate(configs):
        if not isinstance(entry, dict):
            raise ValueError(f"configs[{i}] must be an object")
        for key in REQUIRED_ENTRY:
            if key not in entry:
                raise ValueError(f"configs[{i}] is missing '{key}'")
        label = entry["label"]
        if label in seen:
            raise ValueError(f"duplicate config label '{label}'")
        seen.add(label)
        for key in ("phi0", "y0", "n_star"):
            try:
                float(entry[key])
            except (TypeError, ValueError):
                raise ValueError(
                    f"configs[{i}]['{key}'] must be numeric (got {entry[key]!r})"
                )


def build_tau_grid(tau_grid):
    """Inclusive τ grid with exact endpoints."""
    lo, hi = float(tau_grid["min"]), float(tau_grid["max"])
    n = int(tau_grid["n"])
    if n < 2:
        raise ValueError(f"tau grid needs n >= 2 (got {n})")
    if not lo < hi:
        raise ValueError(f"tau grid needs min < max (got {lo}, {hi})")
    grid = np.linspace(lo, hi, n)
    grid[0] = lo
    grid[-1] = hi
    return grid


def bao_penalty(tau, prior):
    """Gaussian τ prior penalty χ²; 0 when no prior is configured."""
    if not prior:
        return 0.0
    return ((float(tau) - float(prior["tau"])) / float(prior["sigma"])) ** 2


# ── Inputs ─────────────────────────────────────────────────────────────────


def load_model_spectrum(entry):
    """Load the cached P_S(k) for one config entry via ``find_ps``.

    Returns (pspectrum_data, path). Raises FileNotFoundError naming the
    parameters when no cached spectrum matches.
    """
    from scripts.plotting import find_ps

    path, _ = find_ps(entry["phi0"], entry["y0"], entry["n_star"])
    if path is None:
        raise FileNotFoundError(
            "no cached P_S(k) for "
            f"phi0={entry['phi0']}, y0={entry['y0']}, n_star={entry['n_star']} "
            f"(label '{entry['label']}'); run pspectrum_pipeline for it first"
        )
    with open(path) as fh:
        raw = json.load(fh)
    spec = raw["spectrum"]
    return {"k_phys": spec["k_phys"], "P_S": spec["P_S"]}, path


def _spectra_to_d_ell(ells, C_TT, C_TE, C_EE):
    return (
        C_ell_to_d_ell(ells, C_TT),
        C_ell_to_d_ell(ells, C_TE),
        C_ell_to_d_ell(ells, C_EE),
    )


# ── χ² building blocks ─────────────────────────────────────────────────────


def _combo_chi2(scale, ells, ells_l, model_d, lcdm_d, names):
    """χ² for a component combination with the model amplitude scaled by *scale*.

    ``model_d`` / ``lcdm_d`` map a component name to its D_ell array. Scaling
    the whole D_ell array is equivalent to scaling the primordial amplitude
    (every C_ℓ is linear in A_s), so no extra CAMB calls are needed.
    """
    chi2_m = 0.0
    chi2_l = 0.0
    for name in names:
        key = {"ee_binned": "ee_binned", "ee_low": "ee_binned",
               "te_binned": "te_binned"}.get(name, "tt_spectrum")
        cm, cl, _ = COMPONENT_FUNCS[name](
            scale * model_d[key], ells, scale * lcdm_d[key], ells_l
        )
        chi2_m += cm
        chi2_l += cl
    return chi2_m, chi2_l


def profile_amplitude(ells, ells_l, model_d, lcdm_d, names, bounds=(0.3, 3.0)):
    """Profile a single multiplicative amplitude for model and ΛCDM separately.

    With A_s fixed by the model's own P_S normalisation the data determine τ
    almost completely (A_s and τ are degenerate at ℓ ≳ 30); a τ profile at
    fixed A_s therefore measures the amplitude mismatch, not the dip. Profiling
    the amplitude restores the degeneracy direction and makes τ meaningful.

    Returns (A_model, chi2_model_min, A_lcdm, chi2_lcdm_min).
    """
    from scipy.optimize import minimize_scalar

    def _model(a):
        return _combo_chi2(a, ells, ells_l, model_d, lcdm_d, names)[0]

    def _lcdm(a):
        return _combo_chi2(a, ells, ells_l, model_d, lcdm_d, names)[1]

    rm = minimize_scalar(_model, bounds=bounds, method="bounded",
                         options={"xatol": 1e-4})
    rl = minimize_scalar(_lcdm, bounds=bounds, method="bounded",
                         options={"xatol": 1e-4})
    return float(rm.x), float(rm.fun), float(rl.x), float(rl.fun)


def _parabolic_min(x, y, i):
    """Refine the minimum of sampled (x, y) with a parabola through 3 points."""
    if 0 < i < len(x) - 1:
        d = y[i - 1] - 2.0 * y[i] + y[i + 1]
        if d > 0.0:
            dx = 0.5 * (y[i - 1] - y[i + 1]) / d
            if abs(dx) <= 1.0:
                return float(x[i] + dx * (x[i + 1] - x[i])), float(
                    y[i] - 0.25 * (y[i - 1] - y[i + 1]) * dx
                )
    return float(x[i]), float(y[i])


# ── Scan ───────────────────────────────────────────────────────────────────


def run_scan(cfg, verbose=False, log=print):
    """Evaluate the τ grid for every config; returns (rows_by_label, metadata)."""
    from scripts.camb_wrapper import compute_cl_camb_powerlaw, compute_cl_full_camb

    tau_grid = build_tau_grid(cfg["tau_grid"])
    ell_max = int(cfg.get("ell_max", 2500))
    tt_components = list(cfg.get("tt_components", ["commander", "binned"]))
    pol_components = list(cfg.get("polarization_components",
                                  ["ee_low", "ee_binned", "te_binned"]))
    bao = cfg.get("bao_prior")

    # ΛCDM baseline is identical for every config: one CAMB call per τ.
    lcdm = {}
    t0 = time.time()
    for i, tau in enumerate(tau_grid, 1):
        start = time.time()
        ells, C_TT, C_TE, C_EE = compute_cl_camb_powerlaw(
            ell_max=ell_max, tau=float(tau)
        )
        lcdm[float(tau)] = (ells,) + _spectra_to_d_ell(ells, C_TT, C_TE, C_EE)
        if verbose:
            log(f"LC    [{i}/{len(tau_grid)}] tau={tau:.4f} "
                f"({time.time() - start:.1f}s)")

    log(f"LambdaCDM baseline ready in {time.time() - t0:.1f}s "
        f"({len(tau_grid)} CAMB calls)")

    rows_by_label = {}
    p_s_files = {}
    total = len(cfg["configs"]) * len(tau_grid)
    done = 0
    for entry in cfg["configs"]:
        ps, ps_path = load_model_spectrum(entry)
        label = entry["label"]
        p_s_files[label] = os.path.basename(ps_path)
        rows = []
        for tau in tau_grid:
            start = time.time()
            ells, C_TT, C_TE, C_EE = compute_cl_full_camb(
                ps, ell_max=ell_max, tau=float(tau)
            )
            d_tt, d_te, d_ee = _spectra_to_d_ell(ells, C_TT, C_TE, C_EE)
            model_d = {"tt_spectrum": d_tt, "ee_binned": d_ee, "te_binned": d_te}
            ells_l, l_tt, l_te, l_ee = lcdm[float(tau)]
            lcdm_d = {"tt_spectrum": l_tt, "ee_binned": l_ee, "te_binned": l_te}

            def _names(combo):
                return list(tt_components) if combo == "tt" else (
                    list(tt_components) + pol_components
                )

            row = {"tau": float(tau), "raw": {}, "profiled": {}}
            for combo in ("tt", "tt_ee_te"):
                names = _names(combo)
                # Per-component χ² at the model's own normalisation (A = 1).
                comps = {}
                for name in names:
                    key = {"ee_binned": "ee_binned", "ee_low": "ee_binned",
                           "te_binned": "te_binned"}.get(name, "tt_spectrum")
                    m_d, l_d = model_d[key], lcdm_d[key]
                    cm, cl, n = COMPONENT_FUNCS[name](m_d, ells, l_d, ells_l)
                    comps[name] = {"model": cm, "lcdm": cl, "n": n}
                m1 = sum(v["model"] for v in comps.values())
                l1 = sum(v["lcdm"] for v in comps.values())
                n1 = sum(v["n"] for v in comps.values())
                a_m, m_prof, a_l, l_prof = profile_amplitude(
                    ells, ells_l, model_d, lcdm_d, names
                )
                row["raw"][combo] = {"model": m1, "lcdm": l1,
                                     "delta": m1 - l1, "n": n1,
                                     "components": comps}
                row["profiled"][combo] = {
                    "model": m_prof, "lcdm": l_prof, "delta": m_prof - l_prof,
                    "n": n1, "a_model": a_m, "a_lcdm": a_l,
                }

            pen = bao_penalty(tau, bao)
            for key in ("raw", "profiled"):
                base = row[key]["tt_ee_te"]
                row[key]["tt_ee_te_bao"] = dict(base)
                row[key]["tt_ee_te_bao"]["delta"] = base["delta"] + pen
                row[key]["tt_ee_te_bao"]["model"] = base["model"] + pen
            row["bao_penalty"] = pen
            rows.append(row)

            done += 1
            pr = row["profiled"]["tt_ee_te"]
            log(f"{label:<16} [{done}/{total}] tau={tau:.4f} "
                f"A*={pr['a_model']:.3f} chi2*={pr['model']:8.1f} "
                f"Delta={pr['delta']:+6.1f} ({time.time() - start:.1f}s)")
        rows_by_label[label] = rows

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": cfg.get("_path"),
        "config_sha256": cfg.get("_sha256"),
        "tag": cfg["tag"],
        "ell_max": ell_max,
        "tau_grid": [float(x) for x in tau_grid],
        "tt_components": tt_components,
        "polarization_components": pol_components,
        "bao_prior": bao,
        "p_s_files": p_s_files,
        "tau_default_project": CAMB_COSMOLOGY["tau"],
        "amplitude_profiled": True,
        "camb_version": _camb_version(),
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "git": _git_metadata(),
        "assumptions": (
            "diagonal uncorrelated errors; no covariance, no foreground "
            "nuisance parameters, no exact non-Gaussian low-ell likelihood; "
            "the primordial amplitude A_s is profiled as a single "
            "multiplicative factor on D_ell (all C_ell are linear in A_s), "
            "separately for the model and for the LambdaCDM baseline; the "
            "'raw' columns are the A_s-fixed case, kept for comparison; "
            "LambdaCDM baseline uses As/ns from scripts.constants"
        ),
    }
    return rows_by_label, metadata


def _camb_version():
    try:
        import camb

        return camb.__version__
    except Exception:  # pragma: no cover - camb absent
        return None


def _git_metadata():
    try:
        sha = subprocess.check_output(
            ["git", "-C", ROOT_DIR, "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", ROOT_DIR, "status", "--porcelain"], text=True
            ).strip()
        )
        return {"commit": sha, "dirty": dirty}
    except Exception:  # pragma: no cover - git absent
        return {"commit": None, "dirty": None}


# ── Summary and outputs ────────────────────────────────────────────────────


def summarise(rows_by_label, cfg, metadata):
    """Per (config, combo) minima of the amplitude-profiled χ².

    Reports τ* for the model and for ΛCDM, the Δχ² between their best fits,
    and what imposing the BAO τ anchor costs each of them.
    """
    tau_ref = float(CAMB_COSMOLOGY["tau"])
    bao_tau = float(cfg["bao_prior"]["tau"]) if cfg.get("bao_prior") else None
    grid = np.asarray(metadata["tau_grid"], dtype=float)
    summary = {"metadata": metadata, "tau_ref": tau_ref, "entries": {}}

    for label, rows in rows_by_label.items():
        entry = {"combos": {}}
        for combo in COMBOS:
            y_m = np.array([r["profiled"][combo]["model"] for r in rows])
            y_l = np.array([r["profiled"][combo]["lcdm"] for r in rows])
            i_m = int(np.argmin(y_m))
            i_l = int(np.argmin(y_l))
            tau_m, chi_m = _parabolic_min(grid, y_m, i_m)
            tau_l, chi_l = _parabolic_min(grid, y_l, i_l)

            cost_m = cost_l = None
            if bao_tau is not None:
                # Interpolated χ² at the BAO anchor, minus each best fit.
                cost_m = float(np.interp(bao_tau, grid, y_m)) - chi_m
                cost_l = float(np.interp(bao_tau, grid, y_l)) - chi_l

            temps = np.array([r["profiled"][combo].get("a_model", np.nan)
                              for r in rows])
            entry["combos"][combo] = {
                "tau_star_model": tau_m,
                "chi2_model_min": chi_m,
                "tau_star_lcdm": tau_l,
                "chi2_lcdm_min": chi_l,
                "delta_best_fit": chi_m - chi_l,
                "n": int(rows[i_m]["profiled"][combo]["n"]),
                "chi2_red": float(chi_m / max(rows[i_m]["profiled"][combo]["n"] - 2, 1)),
                "a_model_at_min": float(temps[i_m]),
                "chi2_at_tau_ref": float(np.interp(tau_ref, grid, y_m)),
                "bao_cost_model": cost_m,
                "bao_cost_lcdm": cost_l,
                "tau_star_minus_bao": (
                    float(tau_m - bao_tau) if bao_tau is not None else None
                ),
            }
        summary["entries"][label] = entry
    return summary


def write_tables(rows_by_label, cfg, metadata, out_log):
    """Write one JSON + one CSV per config into outputs/simulations/logs/."""
    written = []
    for entry in cfg["configs"]:
        label = entry["label"]
        rows = rows_by_label[label]
        base = dict(phi0=float(entry["phi0"]), y0=float(entry["y0"]),
                    nstar=float(entry["n_star"]))

        json_name = make_filename("tau_profile", ext=".json", **base)
        json_path = out_log(json_name)
        with open(json_path, "w") as fh:
            json.dump(
                {"metadata": metadata, "label": label,
                 "config": {k: entry[k] for k in entry}, "rows": rows},
                fh, indent=2,
            )
        written.append(json_path)

        csv_name = make_filename("tau_profile", ext=".csv", **base)
        csv_path = out_log(csv_name)
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            header = ["tau"]
            for combo in COMBOS:
                header += [f"{combo}_{k}" for k in
                           ("raw_model", "raw_lcdm", "raw_delta",
                            "prof_model", "prof_lcdm", "prof_delta", "a_model")]
            writer.writerow(header)
            for r in rows:
                line = [f"{r['tau']:.6f}"]
                for combo in COMBOS:
                    raw = r["raw"][combo]
                    pr = r["profiled"][combo]
                    line += [f"{raw['model']:.6f}", f"{raw['lcdm']:.6f}",
                             f"{raw['delta']:.6f}", f"{pr['model']:.6f}",
                             f"{pr['lcdm']:.6f}", f"{pr['delta']:.6f}",
                             f"{pr['a_model']:.6f}"]
                writer.writerow(line)
        written.append(csv_path)
    return written


def make_plot(rows_by_label, cfg, summary, out_path):
    """Publication-quality χ²(τ) figure: model vs ΛCDM, amplitude-profiled.

    One panel per config. Solid: the USR model's TT+EE+TE χ²(τ) relative to
    its own minimum. Grey dashed: the same for the ΛCDM baseline. The dotted
    vertical line is the BAO τ anchor. If the two minima sit on top of each
    other, the primordial dip neither absorbs nor fights the τ measurement.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(PAPER_RCPARAMS)
    labels = [e["label"] for e in cfg["configs"]]
    fig, axes = plt.subplots(
        1, len(labels), figsize=(7.0, 2.2), sharey=True, squeeze=False
    )
    axes = axes[0]
    bao = cfg.get("bao_prior") or {}
    combo = "tt_ee_te"

    for ax, label in zip(axes, labels):
        rows = rows_by_label[label]
        tau = np.array([r["tau"] for r in rows])
        y_m = np.array([r["profiled"][combo]["model"] for r in rows])
        y_l = np.array([r["profiled"][combo]["lcdm"] for r in rows])
        y_m = y_m - y_m.min()
        y_l = y_l - y_l.min()

        ax.plot(tau, y_l, color=TOL["grey"], lw=1.0, ls="--",
                label=r"$\Lambda$CDM")
        ax.plot(tau, y_m, color=TOL["blue"], lw=1.2, label="USR model")
        i = int(np.argmin(y_m))
        ax.plot(tau[i], y_m[i], "o", ms=4, color=TOL["blue"], mfc="white",
                mew=0.9, zorder=5)
        j = int(np.argmin(y_l))
        ax.plot(tau[j], y_l[j], "s", ms=3.5, color=TOL["grey"], mfc="white",
                mew=0.9, zorder=5)
        if bao:
            ax.axvline(float(bao["tau"]), color=TOL["red"], ls=":", lw=0.9)
            ax.annotate("BAO", xy=(float(bao["tau"]), 0.02),
                        xycoords=("data", "axes fraction"),
                        xytext=(3, 0), textcoords="offset points",
                        ha="left", va="bottom", fontsize=6,
                        color=TOL["red"])
        ax.set_title(label, fontsize=8)
        ax.set_xlabel(r"$\tau$")
        ax.tick_params(direction="in", top=True, right=True)

    axes[0].set_ylabel(r"$\chi^2_{\rm prof}(\tau) - \chi^2_{\rm min}$")
    axes[0].legend(frameon=False, loc="upper center", fontsize=6)
    fig.subplots_adjust(wspace=0.05)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def load_tables(cfg, out_log):
    """Re-read the per-config tables written by a previous run."""
    rows_by_label = {}
    for entry in cfg["configs"]:
        base = dict(phi0=float(entry["phi0"]), y0=float(entry["y0"]),
                    nstar=float(entry["n_star"]))
        path = out_log(make_filename("tau_profile", ext=".json", **base))
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no table for '{entry['label']}' at {path}; run without "
                "--replot first"
            )
        with open(path) as fh:
            rec = json.load(fh)
        rows_by_label[entry["label"]] = rec["rows"]
        metadata = rec["metadata"]
    return rows_by_label, metadata


def dry_run(cfg):
    """Print the resolved plan and output filenames without importing CAMB."""
    grid = build_tau_grid(cfg["tau_grid"])
    tt_components = list(cfg.get("tt_components", ["commander", "binned"]))
    n_calls = len(grid) + len(grid) * len(cfg["configs"])
    print(f"tag              : {cfg['tag']}")
    print(f"config           : {cfg.get('_path')}")
    print(f"sha256           : {cfg.get('_sha256')}")
    print(f"ell_max          : {cfg.get('ell_max', 2500)}")
    print(f"tau grid         : {len(grid)} points [{grid[0]:.4f}, {grid[-1]:.4f}]")
    print(f"TT components    : {tt_components}")
    print(f"BAO prior        : {cfg.get('bao_prior')}")
    print(f"CAMB calls       : {n_calls} (1 LambdaCDM + 1 model per config per tau)")
    print("work items:")
    for entry in cfg["configs"]:
        for tau in grid:
            print(f"  {entry['label']:<16} tau={tau:.4f}")
    print("outputs:")
    for entry in cfg["configs"]:
        base = dict(phi0=float(entry["phi0"]), y0=float(entry["y0"]),
                    nstar=float(entry["n_star"]))
        for ext in (".json", ".csv"):
            print(f"  logs/{make_filename('tau_profile', ext=ext, **base)}")
    print(f"  scans/tau_profile_summary_{cfg['tag']}.json")
    print(f"  powerloss/tau_profile_{cfg['tag']}.png")


# ── CLI ────────────────────────────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="τ-profile scan: Δχ²(τ) for TT, TT+EE+TE and TT+EE+TE+BAO."
    )
    parser.add_argument("--config", required=True,
                        help="path to a τ-profile config JSON")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the config and print the plan; no CAMB")
    parser.add_argument("--output-tag", default=None,
                        help="override the config tag in output filenames")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--replot", action="store_true",
                        help="re-plot from existing tables instead of "
                             "recomputing the scan")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.output_tag:
        cfg["tag"] = args.output_tag

    if args.dry_run:
        dry_run(cfg)
        return 0

    if args.replot:
        rows_by_label, metadata = load_tables(cfg, get_path_log)
    else:
        rows_by_label, metadata = run_scan(cfg, verbose=args.verbose)
    summary = summarise(rows_by_label, cfg, metadata)

    written = write_tables(rows_by_label, cfg, metadata, get_path_log)
    summary_path = get_path("scans", f"tau_profile_summary_{cfg['tag']}.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    plot_path = get_path("powerloss", f"tau_profile_{cfg['tag']}.png")
    make_plot(rows_by_label, cfg, summary, plot_path)

    print("\nAmplitude-profiled minima (A_s profiled per model at every τ):")
    print(f"  {'config':<16} {'combo':<13} {'τ*(model)':>9} {'χ²_min':>8} "
          f"{'χ²/dof':>7} {'Δχ²_best':>9} {'τ*(ΛCDM)':>9} {'BAO cost':>9}")
    for label, entry in summary["entries"].items():
        for combo in COMBOS:
            c = entry["combos"][combo]
            cost = c["bao_cost_model"]
            cost_s = f"{cost:+9.2f}" if cost is not None else "        —"
            print(f"  {label:<16} {combo:<13} {c['tau_star_model']:9.4f} "
                  f"{c['chi2_model_min']:8.2f} {c['chi2_red']:7.3f} "
                  f"{c['delta_best_fit']:+9.2f} {c['tau_star_lcdm']:9.4f} "
                  f"{cost_s}")
    if cfg.get("bao_prior"):
        print("\n  'BAO cost' = χ²(τ_BAO) − χ²_min, i.e. what anchoring τ to "
              f"{cfg['bao_prior']['tau']} costs the model.")
    print(f"\nwrote: {len(written)} tables, {summary_path}, {plot_path}")
    return 0


def get_path_log(filename):
    """Indirection so tests can capture log paths without touching disk."""
    return get_path("logs", filename)


if __name__ == "__main__":
    raise SystemExit(main())
