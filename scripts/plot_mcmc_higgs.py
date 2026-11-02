"""Reproducible Higgs posterior plots; no sampler or physics implementation.

Bands describe parameter uncertainty in the model mean D_ell, not a posterior
predictive interval including cosmic variance. The inference statistic is the
existing approximate diagonal TT chi-square, not the official Planck likelihood.
Run expensive posterior evaluations on uni; ``--self-test`` uses mocks only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from scripts.planck_data import (
    C_ell_to_d_ell, get_planck_data_asymmetric, load_planck_binned,
)
from scripts.plotting import PAPER_RCPARAMS, TOL, get_path, make_filename, save_fig
from scripts.mcmc_higgs import (
    build_evaluation_args, config_hash, k_grid_from_config, load_archive,
    posterior_samples, source_manifest, validate_config,
)

PARAMETERS = ("phi0", "y0", "N_star")
LABELS = (r"$\phi_0\ [M_{\rm P}]$", r"$y_0$", r"$N_\star$")
QUANTILES = [16, 50, 84]


def sha256_file(path):
    """Hash bytes, including archives too large to read in one allocation."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance():
    """Record both committed identity and dirty-tree/source provenance."""
    root = Path(__file__).resolve().parents[1]

    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, stderr=subprocess.DEVNULL,
            ).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    versions = {"python": platform.python_version()}
    for name in ("numpy", "matplotlib", "scipy", "camb", "emcee", "corner"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    sources = ("scripts/plot_mcmc_higgs.py", "scripts/mcmc_higgs.py",
               "scripts/camb_scan.py", "scripts/plotting.py",
               "scripts/planck_data.py", "scripts/chi2_analysis.py")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "source_sha256": {name: sha256_file(root / name)
                          for name in sources if (root / name).is_file()},
        "packages": versions, "host": platform.node(),
    }


def select_posterior(chain, log_prob, burnin, thin):
    """Return finite posterior rows and exact original (step, walker) indices.

    Burn-in and thinning act on steps BEFORE C-order step/walker flattening.
    Nonfinite coordinates or log probabilities are excluded, never replaced.
    """
    chain = np.asarray(chain, dtype=float)
    log_prob = np.asarray(log_prob, dtype=float)
    if chain.ndim != 3 or chain.shape[2] != 3:
        raise ValueError("chain must have shape (nsteps, nwalkers, 3)")
    if log_prob.shape != chain.shape[:2]:
        raise ValueError("log_prob shape does not match chain")
    if not 0 <= burnin < len(chain) or thin < 1:
        raise ValueError("require 0 <= burnin < nsteps and thin >= 1")
    steps = np.arange(burnin, len(chain), thin)
    indices = np.column_stack((np.repeat(steps, chain.shape[1]),
                               np.tile(np.arange(chain.shape[1]), len(steps))))
    samples = posterior_samples(chain, burnin, thin)
    finite = np.isfinite(samples).all(axis=1) & np.isfinite(log_prob[burnin::thin].ravel())
    if not finite.any():
        raise ValueError("no finite posterior samples remain")
    return samples[finite], indices[finite], int((~finite).sum())


def fixed_band_indices(n_samples, n_band):
    """Evenly spaced, unique positions in the retained posterior, no RNG."""
    if n_samples < 1 or n_band < 1:
        raise ValueError("n_samples and n_band must be positive")
    return np.linspace(0, n_samples - 1, min(n_samples, n_band), dtype=int)


def evaluate_band(samples, args, k_grid, evaluator):
    """Reuse evaluate_config verbatim; fail rather than drop failed draws."""
    curves, chi2 = [], []
    ells = np.arange(2, args.ell_max + 1)
    for index, (phi0, y0, nstar) in enumerate(samples):
        try:
            result = evaluator(float(phi0), float(y0), float(nstar), args,
                               k_phys_grid=k_grid)
        except Exception as error:
            raise RuntimeError(f"posterior evaluation {index} raised: {error}") from error
        if result.get("status") != "ok":
            raise RuntimeError(f"posterior evaluation {index} failed: {result}")
        curve = np.asarray(result["D_ell"], dtype=float)
        if curve.shape != ells.shape or not np.isfinite(curve).all():
            raise ValueError(f"posterior evaluation {index}: invalid D_ell grid")
        value = float(result["chi2"])
        if not np.isfinite(value):
            raise ValueError(f"posterior evaluation {index}: nonfinite chi2")
        curves.append(curve)
        chi2.append(value)
    curves = np.asarray(curves)
    return ells, curves, np.percentile(curves, QUANTILES, axis=0), np.asarray(chi2)


def plot_corner(samples, filename):
    """Triangular marginal plot using matplotlib, without optional corner deps."""
    cmap = LinearSegmentedColormap.from_list("tol_posterior", ["white", TOL["blue"]])
    with plt.rc_context(PAPER_RCPARAMS):
        fig, axes = plt.subplots(3, 3, figsize=(7, 6.4))
        quantiles = np.percentile(samples, QUANTILES, axis=0)
        for row in range(3):
            for col in range(3):
                ax = axes[row, col]
                if col > row:
                    ax.set_visible(False)
                    continue
                if row == col:
                    ax.hist(samples[:, col], bins=35, density=True,
                            histtype="step", color=TOL["blue"], linewidth=1)
                    for value, style in zip(quantiles[:, col], (":", "-", ":")):
                        ax.axvline(value, color=TOL["red"], ls=style, lw=0.8)
                    median = quantiles[1, col]
                    ax.set_title(f"{median:.4g} (+{quantiles[2, col]-median:.3g}, "
                                 f"−{median-quantiles[0, col]:.3g})", fontsize=8)
                    ax.set_yticks([])
                else:
                    ax.hist2d(samples[:, col], samples[:, row], bins=35, cmap=cmap)
                if row == 2:
                    ax.set_xlabel(LABELS[col])
                else:
                    ax.tick_params(labelbottom=False)
                if col == 0 and row > 0:
                    ax.set_ylabel(LABELS[row])
                elif row != col:
                    ax.tick_params(labelleft=False)
                ax.tick_params(axis="x", labelrotation=25)
                ax.locator_params(axis="x", nbins=4)
        fig.subplots_adjust(wspace=0.12, hspace=0.16)
        return save_fig(fig, filename, category="paper", dpi=300)


def plot_band(ells, quantiles, lcdm_ells, lcdm_dell, filename):
    """Show low-ell detail plus full spectrum with shared Planck loaders."""
    low_data = get_planck_data_asymmetric()
    high_data = load_planck_binned() if ells[-1] > 30 else None
    with plt.rc_context(PAPER_RCPARAMS):
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8),
                                 gridspec_kw={"width_ratios": [1, 1.65]})
        for ax, low_panel in zip(axes, (True, False)):
            ax.fill_between(ells, quantiles[0], quantiles[2],
                            color=TOL["blue"], alpha=0.3, label="16–84% model band")
            ax.plot(ells, quantiles[1], color=TOL["blue"], lw=1.1,
                    label="Posterior median")
            ax.plot(lcdm_ells, lcdm_dell, color=TOL["dark"], ls="--", lw=0.9,
                    label=r"$\Lambda$CDM")
            for data, label in ((low_data, "Planck Commander"),
                                (high_data, "Planck binned TT")):
                if data is None or (low_panel and data is high_data):
                    continue
                ell, dell, lower, upper = map(np.asarray, data)
                mask = (ell >= 2) & (ell <= ells[-1])
                ax.errorbar(ell[mask], dell[mask], yerr=[lower[mask], upper[mask]],
                            fmt="o", ms=1.8, elinewidth=0.5, capsize=0,
                            color=TOL["grey"], label=label, zorder=1)
            ax.set_xlim(2, min(30, ells[-1]) if low_panel else ells[-1])
            ax.set_xlabel(r"Multipole $\ell$")
            ax.set_ylim(bottom=0)
        axes[0].set_ylabel(r"$D_\ell^{TT}\ [\mu{\rm K}^2]$")
        axes[1].legend(loc="best", frameon=False)
        fig.tight_layout(pad=0.5)
        return save_fig(fig, filename, category="paper", dpi=300)


def write_products(chain, log_prob, config, evaluation_args, k_grid, *, burnin,
                   thin, n_band, evaluator, baseline, input_metadata):
    """Importable postprocessor; callables permit solver-free synthetic tests."""
    started = datetime.now(timezone.utc).isoformat()
    tag = str(config["tag"])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("tag may contain only letters, digits, underscore, hyphen")
    samples, indices, dropped = select_posterior(chain, log_prob, burnin, thin)
    positions = fixed_band_indices(len(samples), n_band)
    try:
        ells, curves, quantiles, chi2 = evaluate_band(
            samples[positions], evaluation_args, k_grid, evaluator,
        )
    except Exception as error:
        # Never make a biased band by dropping/replacing a failed fixed draw.
        failure_path = get_path("scans", make_filename(f"failure_mcmc_{tag}", ext=".json"))
        failure = {
            "status": "band_rejected", "error": str(error),
            "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
            "config": config, "config_hash": config_hash(config),
            "input": input_metadata, "provenance": provenance(),
            "burnin_steps": burnin, "thin_steps": thin,
            "posterior_positions": positions.tolist(),
            "chain_indices": indices[positions].tolist(),
            "band_parameters": samples[positions].tolist(),
        }
        Path(failure_path).write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")
        raise RuntimeError(f"band rejected; failure recorded at {failure_path}: {error}") from error
    lcdm_ells, lcdm_cl = baseline(evaluation_args.ell_max)
    lcdm_dell = C_ell_to_d_ell(np.asarray(lcdm_ells), np.asarray(lcdm_cl))
    stem = f"mcmc_{tag}"
    band_path = get_path("c_ell", make_filename(f"dell_{stem}", ext=".npz"))
    np.savez_compressed(
        band_path, ell=ells, D_ell_samples=curves, percentiles=QUANTILES,
        D_ell_quantiles=quantiles, chi2=chi2, posterior_positions=positions,
        chain_indices=indices[positions], parameters=samples[positions],
        lcdm_ell=lcdm_ells, lcdm_D_ell=lcdm_dell,
    )
    corner_path = plot_corner(samples, make_filename(f"corner_{stem}", ext=".png"))
    figure_path = plot_band(ells, quantiles, np.asarray(lcdm_ells), lcdm_dell,
                            make_filename(f"dell_{stem}", ext=".png"))
    summary = {
        "schema_version": 1, "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": list(PARAMETERS), "config": config,
        "config_hash": config_hash(config),
        "input": input_metadata, "provenance": provenance(),
        "burnin_steps": burnin, "thin_steps": thin,
        "flatten_order": "step-major, walker-minor (C)",
        "chain_shape": list(chain.shape), "n_posterior": len(samples),
        "excluded_nonfinite": dropped,
        "selection": "linspace(0, n_posterior-1, min(n_posterior,n_band), dtype=int)",
        "n_band_requested": n_band, "n_band_actual": len(positions),
        "posterior_positions": positions.tolist(),
        "chain_indices": indices[positions].tolist(),
        "band_parameters": samples[positions].tolist(),
        "percentiles": QUANTILES, "percentile_method": "numpy linear",
        "parameter_quantiles": np.percentile(samples, QUANTILES, axis=0).tolist(),
        "ell_range": [int(ells[0]), int(ells[-1])],
        "band_interpretation": "pointwise parameter uncertainty in model mean; no cosmic variance",
        "statistic": "approximate diagonal TT chi2; not official Planck likelihood",
        "dpi": 300,
        "outputs": {"corner": corner_path, "band_plot": figure_path,
                    "band_data": band_path},
        "output_sha256": {str(path): sha256_file(path)
                          for path in (corner_path, figure_path, band_path)},
    }
    summary_path = get_path("scans", make_filename(f"posterior_{stem}", ext=".json"))
    Path(summary_path).write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary_path, summary


def postprocess(chain_path, config_path, *, burnin, thin, n_band=200,
                evaluator=None, baseline=None):
    """Verify archive provenance and reevaluate its recorded effective recipe."""
    archive = load_archive(chain_path)
    metadata = archive["metadata"]
    if metadata.get("source_hashes") != source_manifest():
        raise ValueError("current physics/source identity differs from archive (or is missing)")
    requested = json.loads(Path(config_path).read_text())
    config = validate_config(metadata["effective_config"])
    # Smoke archives differ legitimately only when the exact request is recorded.
    permitted = (config_hash(config),
                 config_hash(metadata.get("requested_config", config)))
    if config_hash(requested) not in permitted:
        raise ValueError("supplied config differs from archive effective/requested config")
    if metadata.get("parameters", list(PARAMETERS)) != list(PARAMETERS):
        raise ValueError("unsupported archive parameter ordering")
    if not config["full_chi2"]:
        warnings.warn("Diagnostic low-ell chain: NOT the full ell=2–2508 posterior",
                      stacklevel=2)
    elif config["ell_max"] < 2508:
        raise ValueError("full TT target ends at ell=2508, not plan example ell_max=2500")
    if evaluator is None or baseline is None:
        from scripts.camb_scan import evaluate_config, lcdm_baseline
        evaluator = evaluate_config if evaluator is None else evaluator
        baseline = lcdm_baseline if baseline is None else baseline
    return write_products(
        archive["chain"], archive["log_prob"], config,
        build_evaluation_args(config), k_grid_from_config(config),
        burnin=burnin, thin=thin, n_band=n_band, evaluator=evaluator,
        baseline=baseline,
        input_metadata={
            "chain_path": str(Path(chain_path).resolve()),
            "chain_sha256": sha256_file(chain_path),
            "config_path": str(Path(config_path).resolve()),
            "config_file_sha256": sha256_file(config_path),
            "supplied_config_hash": config_hash(requested),
            "archive_metadata": metadata,
        },
    )


def self_test():
    """Exercise actual archive/config/plots with no import of camb_scan/solver."""
    from PIL import Image
    from scripts.mcmc_higgs import save_archive
    from scripts.plotting import OUTPUT_DIRS

    config = dict(
        tag="synthetic_plot_selftest", seed=123, nwalkers=8, nsteps=12,
        workers=1, num_k=10, ell_max=2508, full_chi2=True,
        xi=15000.0, lam=0.13, k_min=1e-5, k_max=1.0,
        prior={"phi0": [6.3, 6.9], "y0": [-0.85, -0.5], "N_star": [56, 62]},
        init={"phi0": 6.6, "y0": -0.675, "N_star": 59,
              "scatter": [0.02, 0.02, 0.2]},
    )
    rng = np.random.default_rng(123)
    chain = rng.normal([6.6, -0.675, 59], [0.02, 0.02, 0.2], (12, 8, 3))
    log_prob = np.zeros((12, 8))
    log_prob[2, 1] = -np.inf
    calls = []

    def mock_evaluate(phi0, y0, nstar, args, k_phys_grid):
        calls.append((phi0, y0, nstar))
        assert args.workers == 1 and len(k_phys_grid) == config["num_k"]
        ell = np.arange(2, args.ell_max + 1)
        return {"status": "ok", "chi2": phi0 * 10,
                "D_ell": 1000 + ell * 0.1 + phi0 * 2 + y0 + nstar}

    def mock_baseline(ell_max):
        ell = np.arange(2, ell_max + 1)
        from scripts.planck_data import d_ell_to_C_ell
        return ell, d_ell_to_C_ell(ell, np.full(len(ell), 1100.0))

    with tempfile.TemporaryDirectory(prefix="higgs_plot_test_") as directory:
        root = Path(directory)
        paths_before = OUTPUT_DIRS.copy()
        try:
            for category in ("paper", "c_ell", "scans"):
                OUTPUT_DIRS[category] = str(root / category)
            chain_path, config_path = root / "chain.npz", root / "config.json"
            config_path.write_text(json.dumps(config))
            metadata = dict(effective_config=config, requested_config=config,
                            config_hash=config_hash(config), parameters=list(PARAMETERS),
                            source_hashes=source_manifest(), synthetic=True)
            save_archive(chain_path, chain, log_prob, np.zeros(8), 123, metadata)
            summary_path, summary = postprocess(
                chain_path, config_path, burnin=2, thin=2, n_band=7,
                evaluator=mock_evaluate, baseline=mock_baseline,
            )
            samples, indices, excluded = select_posterior(chain, log_prob, 2, 2)
            positions = fixed_band_indices(len(samples), 7)
            assert excluded == 1 and len(samples) == 39
            assert np.array_equal(summary["chain_indices"], indices[positions])
            assert np.array_equal(calls, samples[positions])
            with np.load(summary["outputs"]["band_data"]) as band:
                expected = (1000 + band["ell"][None, :] * 0.1
                            + samples[positions, 0, None] * 2
                            + samples[positions, 1, None] + samples[positions, 2, None])
                np.testing.assert_allclose(band["D_ell_samples"], expected)
                np.testing.assert_allclose(band["D_ell_quantiles"],
                                           np.percentile(expected, QUANTILES, axis=0))
                assert band["ell"][0] == 2 and band["ell"][-1] == 2508
                first_curves = band["D_ell_samples"].copy()
            for name in ("corner", "band_plot"):
                with Image.open(summary["outputs"][name]) as image:
                    assert image.format == "PNG"
                    assert all(dpi >= 299.9 for dpi in image.info["dpi"])
            assert json.loads(Path(summary_path).read_text())["config_hash"] == config_hash(config)
            assert "numpy" in summary["provenance"]["packages"]
            assert "git_commit" in summary["provenance"]
            _, second = postprocess(chain_path, config_path, burnin=2, thin=2, n_band=7,
                                    evaluator=mock_evaluate, baseline=mock_baseline)
            assert second["chain_indices"] == summary["chain_indices"]
            with np.load(second["outputs"]["band_data"]) as band:
                np.testing.assert_array_equal(band["D_ell_samples"], first_curves)
            assert len(fixed_band_indices(3, 200)) == 3
            for burnin, thin in ((12, 1), (-1, 1), (0, 0)):
                try:
                    select_posterior(chain, log_prob, burnin, thin)
                except ValueError:
                    pass
                else:
                    raise AssertionError("invalid burnin/thin accepted")
            try:
                evaluate_band(samples[:1], build_evaluation_args(config),
                              k_grid_from_config(config),
                              lambda *a, **kw: {"status": "mock_failure"})
            except RuntimeError:
                pass
            else:
                raise AssertionError("failed posterior draw silently omitted")
            try:
                postprocess(chain_path, config_path, burnin=2, thin=2, n_band=7,
                            evaluator=lambda *a, **kw: {"status": "mock_failure"},
                            baseline=mock_baseline)
            except RuntimeError:
                failure_path = get_path("scans", make_filename(
                    "failure_mcmc_synthetic_plot_selftest", ext=".json"))
                failure = json.loads(Path(failure_path).read_text())
                assert failure["status"] == "band_rejected"
                assert failure["chain_indices"] == summary["chain_indices"]
            else:
                raise AssertionError("failed band not rejected and recorded")
            assert "scripts.camb_scan" not in sys.modules, "self-test imported solver path"
            metadata["source_hashes"] = {}
            save_archive(chain_path, chain, log_prob, np.zeros(8), 123, metadata)
            try:
                postprocess(chain_path, config_path, burnin=2, thin=2,
                            evaluator=mock_evaluate, baseline=mock_baseline)
            except ValueError:
                pass
            else:
                raise AssertionError("source identity mismatch accepted")
            metadata["source_hashes"] = source_manifest()
            save_archive(chain_path, chain, log_prob, np.zeros(8), 123, metadata)
            mismatch = dict(config, xi=14000.0)
            config_path.write_text(json.dumps(mismatch))
            try:
                postprocess(chain_path, config_path, burnin=2, thin=2,
                            evaluator=mock_evaluate, baseline=mock_baseline)
            except ValueError:
                pass
            else:
                raise AssertionError("config mismatch accepted")
            print("SELF-TEST PASS: 39 retained, 1 nonfinite excluded, 7 fixed draws; "
                  "two identical bands; 16/50/84 verified; PNG 300 DPI; "
                  "provenance, mismatch and failure checks passed; NO solver calls")
        finally:
            OUTPUT_DIRS.clear()
            OUTPUT_DIRS.update(paths_before)


def _audit_curve(ell, dell, label):
    ell, dell = np.asarray(ell, dtype=float), np.asarray(dell, dtype=float)
    if (ell.ndim != 1 or len(ell) < 2 or dell.shape != ell.shape
            or not np.isfinite(ell).all() or not np.isfinite(dell).all()
            or not np.all(np.diff(ell) > 0) or ell[0] != 2
            or not np.array_equal(ell, np.arange(2, int(ell[-1]) + 1))):
        raise ValueError(f"{label}: expected contiguous finite ell=2..ell_max / D_ell arrays")
    return ell, dell


def _audit_metrics(ell, dell, lcdm_ell, lcdm_dell):
    """All chi-square totals/contributions come from existing shared helpers."""
    from scripts.chi2_analysis import chi2_binned, chi2_unbinned, chi2_model_lcdm
    from scripts.planck_data import load_planck_unbinned

    total_u, lcdm_u, n_u = chi2_unbinned(dell, ell, lcdm_dell, lcdm_ell)
    total_b, lcdm_b, n_b = chi2_binned(dell, ell, lcdm_dell, lcdm_ell)
    datasets = {"unbinned": load_planck_unbinned(), "binned": load_planck_binned()}
    contributions = {}
    for name, data in datasets.items():
        data_ell, data_dell, lower, upper = map(np.asarray, data)
        if name == "binned":
            # chi2_binned uses the mean asymmetric error for BOTH residual signs.
            lower = upper = (lower + upper) / 2
        contributions[name] = []
        for lo, hi in ((2, 29), (30, 199), (200, 799), (800, 1499), (1500, 2508)):
            mask = (data_ell >= lo) & (data_ell <= hi)
            value, reference = chi2_model_lcdm(
                dell, ell, planck_ells=data_ell[mask], D_planck=data_dell[mask],
                D_err_lower=lower[mask], D_err_upper=upper[mask],
                D_lcdm=lcdm_dell, ells_lcdm=lcdm_ell, ell_max=hi,
            )
            contributions[name].append(dict(ell_range=[lo, hi], n=int(mask.sum()),
                                             chi2=float(value), chi2_lcdm=float(reference)))
    low_chi2, low_lcdm = chi2_model_lcdm(dell, ell, D_lcdm=lcdm_dell,
                                          ells_lcdm=lcdm_ell, ell_max=29)
    return dict(chi2_low=float(low_chi2), chi2_lcdm_low=float(low_lcdm),
                chi2_unbinned=float(total_u), chi2_lcdm_unbinned=float(lcdm_u),
                n_unbinned=int(n_u), chi2_binned=float(total_b),
                chi2_lcdm_binned=float(lcdm_b), n_binned=int(n_b),
                contributions=contributions,
                endpoint_clamping_above_model=bool(ell[-1] < 2508))


def plot_chi2_audit(archive_path, current_path, tag="chi2_audit"):
    """Compare saved factorial audit and dry-run JSON without solver/CAMB calls.

    Binned standardized residuals are for display only: (D_model-D_data) /
    mean(err_lower,err_upper), exactly the error convention of chi2_binned.
    The shared chi-square helpers, never residual squaring here, own statistics.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("audit tag must be a safe flat filename label")
    audit = json.loads(Path(archive_path).read_text())
    current = json.loads(Path(current_path).read_text())
    if audit.get("event") != "audit_archive" or not audit.get("success"):
        raise ValueError("--audit-chi2 requires a successful core audit_archive JSON")
    if current.get("event") != "dry_run" or current["result"].get("status") != "ok":
        raise ValueError("--current-dryrun requires a successful pure dry_run JSON")
    if audit["fixed_theta"] != current["theta"]:
        raise ValueError("audit and current dry-run use different initial parameters")
    current_config = current["metadata"]["effective_config"]
    if not current_config["full_chi2"] or current_config["ell_max"] != 2508:
        raise ValueError("current diagnostic must cover the full ell=2–2508 target")
    labels = {"correct_pivot_grid": "Archived, correctly paired",
              "legacy_truncation_emulation": "Legacy truncation emulation",
              "replay_correct": "Historical solver replay, paired"}
    colors = {"correct_pivot_grid": TOL["blue"],
              "legacy_truncation_emulation": TOL["purple"],
              "replay_correct": TOL["green"], "current": TOL["red"]}
    cases = audit["cases"]
    identities = [(case["pairing"], case["ell_max"]) for case in cases]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate pairing/ell_max audit cells")
    full_cases = [case for case in cases if case["ell_max"] == 2508]
    required = {"correct_pivot_grid", "legacy_truncation_emulation"}
    if not required.issubset({case["pairing"] for case in full_cases}):
        raise ValueError("audit requires correct/legacy pairing cells at ell_max=2508")
    baseline_case = next(case for case in full_cases if case["pairing"] == "correct_pivot_grid")
    lcdm_ell, lcdm_dell = _audit_curve(baseline_case["lcdm_ell"],
                                       baseline_case["D_ell_lcdm"], "LCDM")
    if lcdm_ell[-1] != 2508 or np.any(lcdm_dell <= 0):
        raise ValueError("LCDM must be positive and cover ell=2–2508")
    metrics, curves = [], []
    for case in cases:
        pairing = case["pairing"]
        if pairing not in labels or case["ell_max"] not in (2500, 2508):
            raise ValueError("unknown audit pairing or ell_max")
        ell, dell = _audit_curve(case["ell"], case["D_ell"], pairing)
        ref_ell, ref_dell = _audit_curve(case["lcdm_ell"], case["D_ell_lcdm"], pairing + " LCDM")
        if ell[-1] != case["ell_max"] or ref_ell[-1] != case["ell_max"]:
            raise ValueError("audit cell array endpoint differs from labelled ell_max")
        # Different reference cosmologies would invalidate the common residual plot.
        if case["ell_max"] == 2508 and not np.allclose(
                ref_dell, np.interp(ref_ell, lcdm_ell, lcdm_dell), rtol=1e-6, atol=1e-6):
            raise ValueError("factorial audit cells do not share the same LCDM reference")
        measured = _audit_metrics(ell, dell, ref_ell, ref_dell)
        measured.update(pairing=pairing, ell_max=case["ell_max"],
                        recorded_metrics={key: case[key] for key in case if key.startswith("chi2_")})
        metrics.append(measured)
        if case["ell_max"] == 2508:
            curves.append(dict(pairing=pairing, label=labels[pairing], ell=ell,
                               D_ell=dell, metrics=measured))
    ell = np.arange(2, current_config["ell_max"] + 1)
    ell, dell = _audit_curve(ell, current["result"]["D_ell"], "current dry-run")
    measured = _audit_metrics(ell, dell, lcdm_ell, lcdm_dell)
    measured.update(pairing="current", ell_max=2508,
                    recorded_metrics={key: value for key, value in current["result"].items()
                                      if key.startswith("chi2") or key == "dchi2"})
    metrics.append(measured)
    curves.append(dict(pairing="current", label=f"Current ({current_config['num_k']} modes requested)",
                       ell=ell, D_ell=dell, metrics=measured))
    low = tuple(map(np.asarray, get_planck_data_asymmetric()))
    binned = tuple(map(np.asarray, load_planck_binned()))
    bin_ell, bin_dell, lower, upper = binned
    sigma = (lower + upper) / 2
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("invalid Planck binned errors")
    plotted = []
    with plt.rc_context(PAPER_RCPARAMS):
        fig, axes = plt.subplots(2, 2, figsize=(7, 5.6))
        low_ax, high_ax, frac_ax, residual_ax = axes.ravel()
        for item in curves:
            color = colors[item["pairing"]]
            style = {"legacy_truncation_emulation": ":", "replay_correct": "--"}.get(item["pairing"], "-")
            ell, dell = item["ell"], item["D_ell"]
            label = item["label"]
            unbinned_label = f"{label}: {item['metrics']['chi2_unbinned']:.1f}"
            for ax in (low_ax, high_ax):
                ax.plot(ell, dell, color=color, ls=style, lw=1, label=unbinned_label)
            fraction = (dell - np.interp(ell, lcdm_ell, lcdm_dell)) / np.interp(ell, lcdm_ell, lcdm_dell)
            standardized = (np.interp(bin_ell, ell, dell) - bin_dell) / sigma
            frac_ax.plot(ell, 100 * fraction, color=color, ls=style, lw=0.9)
            residual_ax.plot(bin_ell, standardized, color=color, ls=style, lw=0.9,
                             label=f"{label}: {item['metrics']['chi2_binned']:.1f}")
            plotted.append(dict(pairing=item["pairing"], label=label,
                                ell=ell.tolist(), D_ell=dell.tolist(),
                                fractional_residual_lcdm=fraction.tolist(),
                                binned_standardized_residual=standardized.tolist()))
        for ax in (low_ax, high_ax):
            ax.plot(lcdm_ell, lcdm_dell, color=TOL["dark"], ls="--", lw=0.9, label=r"$\Lambda$CDM")
        low_ax.errorbar(low[0], low[1], yerr=[low[2], low[3]], fmt="o", ms=2,
                        elinewidth=0.5, color=TOL["grey"], label="Planck Commander")
        high_ax.errorbar(bin_ell, bin_dell, yerr=[lower, upper], fmt="o", ms=1.6,
                         elinewidth=0.5, color=TOL["grey"], label="Planck binned TT")
        lcdm_standardized = (np.interp(bin_ell, lcdm_ell, lcdm_dell) - bin_dell) / sigma
        residual_ax.plot(bin_ell, lcdm_standardized, color=TOL["dark"], ls="--", lw=0.8,
                         label=f"LCDM: {measured['chi2_lcdm_binned']:.1f}")
        frac_ax.axhline(0, color=TOL["grey"], lw=0.6)
        residual_ax.axhline(0, color=TOL["grey"], lw=0.6)
        residual_ax.axhspan(-1, 1, color=TOL["grey"], alpha=0.12)
        low_ax.set_xlim(2, 30)
        high_ax.set_xlim(30, 2508)
        low_ax.set_ylim(bottom=0)
        high_ax.set_ylim(bottom=0)
        frac_ax.set_xlim(30, 2508)
        residual_ax.set_xlim(30, 2508)
        high_fraction = np.concatenate([
            100 * np.asarray(item["fractional_residual_lcdm"])[np.asarray(item["ell"]) >= 30]
            for item in plotted
        ])
        lo, hi = min(0.0, float(high_fraction.min())), max(0.0, float(high_fraction.max()))
        padding = max((hi - lo) * 0.12, 0.1)
        frac_ax.set_ylim(lo - padding, hi + padding)
        # Reserve clear legend space rather than obscuring the large negative residuals.
        residual_lo, residual_hi = residual_ax.get_ylim()
        residual_ax.set_ylim(residual_lo, residual_hi + (residual_hi - residual_lo) * 0.55)
        for ax in axes.ravel():
            ax.set_xlabel(r"Multipole $\ell$")
        for ax in (low_ax, high_ax):
            ax.set_ylabel(r"$D_\ell^{TT}\ [\mu{\rm K}^2]$")
        frac_ax.set_ylabel(r"$(D_\ell-D_\ell^{\Lambda\rm CDM})/D_\ell^{\Lambda\rm CDM}$ [%]")
        residual_ax.set_ylabel(r"$(D_{\rm model}-D_{\rm Planck})/\bar\sigma_{\rm bin}$")
        low_ax.set_title(r"Low multipoles; legend lists $\chi^2_{\rm unbin}$")
        high_ax.set_title("High multipoles (all curves end at 2508)")
        frac_ax.set_title(r"Fractional difference from LCDM ($\ell\geq30$)")
        residual_ax.set_title(r"Binned residuals; legend lists $\chi^2_{\rm bin}$")
        low_ax.legend(frameon=False, loc="best")
        residual_ax.legend(frameon=False, loc="upper right")
        fig.tight_layout(pad=0.5)
        image_path = save_fig(fig, make_filename(f"dell_{tag}", ext=".png"), "paper", dpi=300)
    summary = dict(
        schema_version=1, event="plot_chi2_audit", diagnostic_only=True,
        created_utc=datetime.now(timezone.utc).isoformat(), fixed_theta=audit["fixed_theta"],
        inputs={"audit": dict(path=str(Path(archive_path).resolve()), sha256=sha256_file(archive_path)),
                "current": dict(path=str(Path(current_path).resolve()), sha256=sha256_file(current_path))},
        input_metadata={"audit": audit["metadata"], "current": current["metadata"]},
        provenance=provenance(), plotted_curves=plotted, metrics=metrics,
        archived_scalar_metrics=audit.get("archived_scalar_metrics", {}),
        replay_comparison=audit.get("replay_comparison"),
        constants_comparison=audit.get("constants_comparison"),
        factorial_note="ell=2500 cases retained in metrics; unbinned ell=2501–2508 clamps endpoint diagnostically only",
        binned_data=dict(ell=bin_ell.tolist(), D_ell=bin_dell.tolist(), sigma_mean=sigma.tolist(),
                         lcdm_standardized_residual=lcdm_standardized.tolist()),
        lcdm=dict(ell=lcdm_ell.tolist(), D_ell=lcdm_dell.tolist()),
        residual_conventions={"fractional_lcdm": "(model-LCDM)/LCDM; plot displays percent",
                              "binned": "(interp(model, bin_ell)-Planck)/((err_lower+err_upper)/2)",
                              "unbinned_chi2": "existing shared helper; sign-selected asymmetric errors",
                              "binned_chi2": "existing shared helper; mean lower/upper errors"},
        interpretation="approximate diagonal TT chi2, not official Planck likelihood; legacy writer execution unproven",
        outputs=dict(figure=image_path, figure_sha256=sha256_file(image_path), dpi=300),
    )
    summary_path = get_path("scans", make_filename(f"summary_{tag}", ext=".json"))
    Path(summary_path).write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Audit summary: {summary_path}")
    return summary_path, summary


def self_test_audit():
    """Synthetic saved spectra exercise diagnostics; never import solver modules."""
    from PIL import Image
    from scripts.chi2_analysis import chi2_binned
    from scripts.plotting import OUTPUT_DIRS

    bin_ell, bin_dell, lower, upper = map(np.asarray, load_planck_binned())
    ell = np.arange(2, 2509)
    reference = np.interp(ell, bin_ell, bin_dell)
    theta = [6.6, -0.675, 59.0]
    cases = []
    factors = {"correct_pivot_grid": 1.01, "legacy_truncation_emulation": 1.0,
               "replay_correct": 1.015}
    for pairing, factor in factors.items():
        for endpoint in (2500, 2508):
            size = endpoint - 1
            cases.append(dict(pairing=pairing, ell_max=endpoint,
                              ell=ell[:size].tolist(), D_ell=(reference[:size] * factor).tolist(),
                              lcdm_ell=ell[:size].tolist(), D_ell_lcdm=reference[:size].tolist(),
                              diagnostic_only=True))
    audit = dict(event="audit_archive", success=True, fixed_theta=theta,
                 cases=cases, metadata={"synthetic": True}, archived_scalar_metrics={})
    current_dell = reference * (1 + 0.12 * ell / 2508)
    current = dict(event="dry_run", theta=theta,
                   result=dict(status="ok", D_ell=current_dell.tolist()),
                   metadata=dict(synthetic=True, effective_config=dict(
                       full_chi2=True, ell_max=2508, num_k=320)))
    previous_dirs = OUTPUT_DIRS.copy()
    with tempfile.TemporaryDirectory(prefix="higgs_chi2_audit_test_") as directory:
        root = Path(directory)
        try:
            for category in ("paper", "scans"):
                OUTPUT_DIRS[category] = str(root / category)
            archive_path, current_path = root / "audit.json", root / "current.json"
            archive_path.write_text(json.dumps(audit))
            current_path.write_text(json.dumps(current))
            summary_path, summary = plot_chi2_audit(archive_path, current_path,
                                                    tag="synthetic_chi2_audit")
            assert len(summary["plotted_curves"]) == 4 and len(summary["metrics"]) == 7
            current_curve = next(c for c in summary["plotted_curves"] if c["pairing"] == "current")
            expected = (np.interp(bin_ell, ell, current_dell) - bin_dell) / ((lower + upper) / 2)
            np.testing.assert_allclose(current_curve["binned_standardized_residual"], expected)
            np.testing.assert_allclose(current_curve["fractional_residual_lcdm"],
                                       (current_dell-reference)/reference)
            for item in summary["metrics"]:
                for dataset in ("binned", "unbinned"):
                    np.testing.assert_allclose(
                        sum(part["chi2"] for part in item["contributions"][dataset]),
                        item[f"chi2_{dataset}"], rtol=1e-12)
            current_metric = next(m for m in summary["metrics"] if m["pairing"] == "current")
            direct, _, _ = chi2_binned(current_dell, ell, reference, ell)
            np.testing.assert_allclose(current_metric["chi2_binned"], direct)
            assert all(m["endpoint_clamping_above_model"] == (m["ell_max"] == 2500)
                       for m in summary["metrics"])
            with Image.open(summary["outputs"]["figure"]) as image:
                assert image.format == "PNG" and all(dpi >= 299.9 for dpi in image.info["dpi"])
            assert Path(summary_path).is_file()
            assert summary["inputs"]["audit"]["sha256"] == sha256_file(archive_path)
            for kind in ("different_theta", "short_current", "missing_pairing"):
                invalid_current = json.loads(json.dumps(current))
                invalid_audit = json.loads(json.dumps(audit))
                if kind == "different_theta":
                    invalid_current["theta"][0] += 0.1
                elif kind == "short_current":
                    invalid_current["result"]["D_ell"].pop()
                else:
                    invalid_audit["cases"] = [case for case in cases
                                               if case["pairing"] != "correct_pivot_grid"]
                archive_path.write_text(json.dumps(invalid_audit))
                current_path.write_text(json.dumps(invalid_current))
                try:
                    plot_chi2_audit(archive_path, current_path, tag="invalid")
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"invalid audit accepted: {kind}")
            assert "scripts.camb_scan" not in sys.modules
            print("AUDIT SELF-TEST PASS: six factorial cells + current; four plotted curves; "
                  "shared-helper totals/range sums; exact standardized/fractional residuals; "
                  "PNG300DPI, input hashes, invalid-input rejection; NO solver imports/calls")
        finally:
            OUTPUT_DIRS.clear()
            OUTPUT_DIRS.update(previous_dirs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--burnin", type=int, help="discard this many initial steps (required)")
    parser.add_argument("--thin", type=int, help="retain every thin-th step after burn-in (required)")
    parser.add_argument("--n-band", type=int, default=200)
    parser.add_argument("--self-test", action="store_true", help="synthetic archive + mocked physics only")
    parser.add_argument("--audit-chi2", type=Path, help="saved core audit_archive JSON; no solver calls")
    parser.add_argument("--current-dryrun", type=Path, help="pure core dry_run JSON to compare with audit")
    parser.add_argument("--audit-tag", default="chi2_audit", help="flat audit output label")
    parser.add_argument("--self-test-audit", action="store_true", help="solver-free synthetic chi2 audit plot test")
    args = parser.parse_args(argv)
    if args.self_test_audit:
        self_test_audit()
        return 0
    if args.audit_chi2 is not None or args.current_dryrun is not None:
        if args.audit_chi2 is None or args.current_dryrun is None:
            parser.error("--audit-chi2 and --current-dryrun must be supplied together")
        plot_chi2_audit(args.audit_chi2, args.current_dryrun, args.audit_tag)
        return 0
    if args.self_test:
        self_test()
        return 0
    if any(value is None for value in (args.chain, args.config, args.burnin, args.thin)):
        parser.error("--chain, --config, --burnin and --thin are required")
    if args.n_band < 1:
        parser.error("--n-band must be positive")
    summary_path, _ = postprocess(args.chain, args.config, burnin=args.burnin,
                                 thin=args.thin, n_band=args.n_band)
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
