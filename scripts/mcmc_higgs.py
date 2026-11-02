"""Seeded Higgs inference and reproducible archive diagnostics.

All physics and the (two-decimal) diagonal TT chi-square come exclusively from
scripts.camb_scan.evaluate_config, not the official Planck likelihood code.
Run as ``python -m scripts.mcmc_higgs --help`` or the documented script entry.
"""
import os

# Force, rather than default: every spawned likelihood process has one thread.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import argparse
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
import multiprocessing as mp
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np

PARAMETERS = ("phi0", "y0", "N_star")
SCHEMA_VERSION = 1


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def validate_config(config):
    """Validate the effective, not merely requested, sampler recipe."""
    for key in ("seed", "nwalkers", "nsteps", "workers", "num_k", "ell_max"):
        if isinstance(config[key], bool) or not isinstance(config[key], int):
            raise ValueError(f"{key} must be an integer")
    if not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must fit NumPy RandomState")
    if config["nwalkers"] < 8 or config["nwalkers"] % 2:
        raise ValueError("nwalkers must be even and >=8")
    if min(config[k] for k in ("nsteps", "workers", "num_k")) < 1:
        raise ValueError("steps, workers and num_k must be positive")
    if config["num_k"] < 10 or config["ell_max"] < 29:
        raise ValueError("num_k>=10 and ell_max>=29 required")
    spec = config.get("k_grid")
    if spec is not None:
        if not isinstance(spec, dict) or spec.get("mode") != "weighted":
            raise ValueError("k_grid must be {'mode': 'weighted', ...}")
        if int(spec.get("n_dense", 150)) < 10 or int(spec.get("n_outer", 40)) < 4:
            raise ValueError("weighted k_grid needs n_dense>=10 and n_outer>=4")
    if not isinstance(config["full_chi2"], bool):
        raise ValueError("full_chi2 must be boolean")
    if config["full_chi2"] and config["ell_max"] < 2508:
        raise ValueError("full unbinned TT requires ell_max>=2508; no endpoint clamping")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config["tag"]):
        raise ValueError("tag must be a safe flat filename label")
    for key in ("xi", "lam", "k_min", "k_max"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if config["k_min"] >= config["k_max"]:
        raise ValueError("k_min must be below k_max")
    scatter = np.asarray(config["init"]["scatter"], dtype=float)
    if scatter.shape != (3,) or not np.all(np.isfinite(scatter) & (scatter > 0)):
        raise ValueError("init.scatter must contain three finite positive scales")
    for name in PARAMETERS:
        bounds = np.asarray(config["prior"][name], dtype=float)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
            raise ValueError(f"invalid prior for {name}")
        if not bounds[0] < config["init"][name] < bounds[1]:
            raise ValueError(f"init {name} must lie strictly within prior")
    interval = config.get("checkpoint_interval", 10)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("checkpoint_interval must be a positive integer")
    config_hash(config)
    return config


def effective_config(config, smoke=False):
    config = deepcopy(config)
    if smoke:
        config.update(nwalkers=8, nsteps=20, ell_max=30, full_chi2=False,
                      tag=config["tag"] + "_smoke")
    return validate_config(config)


def build_evaluation_args(config):
    return argparse.Namespace(**{k: config[k] for k in
        ("lam", "xi", "k_min", "k_max", "num_k", "ell_max", "full_chi2")}, workers=1)


def k_grid_from_config(config):
    """Uniform geometric grid, or a dense-in-dip weighted grid when requested.

    The suppression feature is ~0.05 dex wide, so a uniform grid at 160 modes
    puts only ~2 points inside it and the chi-square becomes noisy in N_star.
    ``k_grid`` in the config can request the project's existing weighted grid.
    """
    spec = config.get("k_grid") or {}
    if spec.get("mode") == "weighted":
        from pspectrum_pipeline import build_weighted_kgrid
        from scripts.constants import k_pivot_phys
        return build_weighted_kgrid(
            config["k_min"], config["k_max"], k_pivot_phys,
            dense_min=float(spec.get("dense_min", 1e-4)),
            dense_max=float(spec.get("dense_max", 1e-2)),
            n_dense=int(spec.get("n_dense", 150)),
            n_outer=int(spec.get("n_outer", 40)))
    return np.geomspace(config["k_min"], config["k_max"], config["num_k"])


def evaluate_point(theta, config):
    from scripts.camb_scan import evaluate_config
    return evaluate_config(*map(float, theta), build_evaluation_args(config),
                           k_phys_grid=k_grid_from_config(config))


def log_prior(theta, config):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (3,) or not np.all(np.isfinite(theta)):
        return -np.inf
    return 0.0 if all(config["prior"][p][0] < x < config["prior"][p][1]
                      for p, x in zip(PARAMETERS, theta)) else -np.inf


def log_probability(theta, config):
    if not np.isfinite(log_prior(theta, config)):
        return -np.inf
    result = evaluate_point(theta, config)
    chi2 = result.get("chi2", np.nan)
    if result.get("status") != "ok" or not np.isfinite(chi2):
        return -np.inf
    return -0.5 * float(chi2)


def initial_positions(config):
    rng = np.random.default_rng(config["seed"])
    center = np.array([config["init"][p] for p in PARAMETERS])
    positions = []
    for _ in range(100000):
        candidate = center + rng.normal(size=3) * config["init"]["scatter"]
        if np.isfinite(log_prior(candidate, config)):
            positions.append(candidate)
        if len(positions) == config["nwalkers"]:
            return np.asarray(positions)
    raise ValueError("Could not initialize walkers inside prior")


def _utcnow():
    return datetime.now(timezone.utc).isoformat()


def source_manifest():
    """Hash inference sources, core solvers, model factories and input data.

    Explicit discovery includes this new entry point even before its first commit.
    Git revision alone cannot identify a dirty or pre-commit smoke implementation.
    """
    root = Path(__file__).resolve().parents[1]
    files = {Path(__file__).resolve()}
    for pattern in ("*.py", "models/*.py", "fortran/*.f90", "fortran/*.so", "*.so", "data/Planck/*.csv", "data/Planck/*.txt"):
        files.update(root.glob(pattern))
    for name in ("camb_scan", "camb_wrapper", "chi2_analysis", "planck_data",
                 "constants", "optimizer_utils", "plotting", "observables",
                 "config_loader", "usr_utils"):
        files.add(root / "scripts" / (name + ".py"))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files) if path.is_file()}


def runtime_metadata(config, requested_config=None):
    from scripts.constants import As, k_pivot_phys, CAMB_COSMOLOGY
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    versions = {}
    for name in ("numpy", "scipy", "emcee", "h5py", "camb", "numba"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    return dict(schema_version=SCHEMA_VERSION, effective_config=config,
                requested_config=requested_config or config, config_hash=config_hash(config),
                git_commit=git("rev-parse", "HEAD"), git_status=git("status", "--porcelain"),
                versions=versions, source_hashes=source_manifest(), python=sys.version, host=platform.node(),
                platform=platform.platform(), start_time=_utcnow(),
                parameters=list(PARAMETERS), thread_settings={k: os.environ[k] for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")},
                likelihood="evaluate_config rounded chi2; diagonal asymmetric unbinned TT" if
                config["full_chi2"] else "evaluate_config rounded chi2; low-ell TT diagnostic",
                chi2_rounding_decimals=2, k_grid=k_grid_from_config(config).tolist(),
                evaluator_constants=dict(As=As, k_pivot_phys=k_pivot_phys,
                                         camb_cosmology=CAMB_COSMOLOGY),
                likelihood_ell_range=[2, 2508] if config["full_chi2"] else [2, 29])


def output_paths(config):
    from scripts.plotting import OUTPUT_DIRS, get_path, make_filename
    name = "higgs_mcmc_" + config["tag"]
    # plotting has no chains category; derive sibling from its canonical logs.
    chains = Path(OUTPUT_DIRS["logs"]).parent / "chains"
    chains.mkdir(parents=True, exist_ok=True)
    return {"archive": chains / make_filename(name, ext=".npz"),
            "checkpoint": chains / make_filename(name, ext=".h5"),
            "log": Path(get_path("logs", make_filename(name, ext=".jsonl")))}


def append_event(path, event):
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(time=_utcnow(), **event), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def save_archive(path, chain, log_prob, accepted, seed, metadata):
    temp = Path(str(path) + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, chain=chain, log_prob=log_prob,
                            accepted=accepted, seed=np.int64(seed),
                            metadata_json=np.array(json.dumps(metadata, allow_nan=False)))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def load_archive(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key].copy() for key in data.files}
    if result["chain"].ndim != 3 or result["chain"].shape[2] != 3:
        raise ValueError("chain must have shape (steps,walkers,3)")
    if result["log_prob"].shape != result["chain"].shape[:2]:
        raise ValueError("log_prob shape mismatch")
    result["metadata"] = json.loads(str(result.pop("metadata_json")))
    if result["metadata"]["config_hash"] != config_hash(result["metadata"]["effective_config"]):
        raise ValueError("archive config hash mismatch")
    return result


def posterior_samples(chain, burnin, thin):
    if burnin < 0 or thin < 1 or burnin >= len(chain):
        raise ValueError("require 0 <= burnin < steps and thin >= 1")
    return chain[burnin::thin].reshape(-1, 3)


def summarize_archive(path, burnin, thin):
    data = load_archive(path)
    samples = posterior_samples(data["chain"], burnin, thin)
    return dict(archive=str(path), burnin=burnin, thin=thin, n_samples=len(samples),
                quantile_levels=[0.16, 0.5, 0.84],
                quantiles={p: np.quantile(samples[:, i], [.16, .5, .84]).tolist()
                           for i, p in enumerate(PARAMETERS)},
                acceptance_fraction=(data["accepted"] / len(data["chain"])).tolist(),
                mean_acceptance_fraction=float(np.mean(data["accepted"] / len(data["chain"]))),
                nonfinite_log_prob_count=int(np.count_nonzero(~np.isfinite(data["log_prob"]))),
                retained_nonfinite_log_prob_count=int(np.count_nonzero(
                    ~np.isfinite(data["log_prob"][burnin::thin]))),
                diagnostics=data["metadata"].get("diagnostics"))


def compare_archives(first, second, atol=0.0):
    a, b = load_archive(first), load_archive(second)
    checks = {}
    for key in ("chain", "log_prob", "accepted", "seed"):
        checks[key] = bool(a[key].shape == b[key].shape and
                          np.allclose(a[key], b[key], rtol=0, atol=atol, equal_nan=False))
    checks["effective_config"] = a["metadata"]["config_hash"] == b["metadata"]["config_hash"]
    checks["source_hashes"] = a["metadata"].get("source_hashes") == b["metadata"].get("source_hashes")
    return dict(equal=all(checks.values()), atol=atol, checks=checks)


def validate_archive(path, burnin, thin):
    data = load_archive(path)
    samples = posterior_samples(data["chain"], burnin, thin)
    logs = data["log_prob"][burnin::thin].reshape(-1)
    if len(samples) < 20:
        raise ValueError("validation needs at least 20 retained samples")
    indices = np.linspace(0, len(samples) - 1, 20, dtype=int)
    config = data["metadata"]["effective_config"]
    records = []
    for index in indices:
        result = evaluate_point(samples[index], config)
        chi2 = result.get("chi2", np.nan)
        delta = float(chi2 + 2 * logs[index])
        ok = result.get("status") == "ok" and np.isfinite(delta) and delta == 0.0
        records.append(dict(index=int(index), theta=samples[index].tolist(),
                            status=result.get("status"), chi2=float(chi2) if np.isfinite(chi2) else None,
                            chi2_difference=delta if np.isfinite(delta) else None, equal=bool(ok)))
    return dict(archive=str(path), burnin=burnin, thin=thin, n_validation=20,
                equal=all(r["equal"] for r in records), records=records)


def run_sampler(config, requested_config=None, resume=False):
    import emcee
    paths = output_paths(config)
    identity_path = paths["checkpoint"].with_suffix(".json")
    metadata = runtime_metadata(config, requested_config)
    if resume:
        if not identity_path.exists() or not paths["checkpoint"].exists():
            raise ValueError("resume requires checkpoint and identity metadata")
        previous = json.loads(identity_path.read_text())
        if previous["config_hash"] != metadata["config_hash"]:
            raise ValueError("resume config hash mismatch (nsteps is total target)")
        if previous.get("source_hashes") != metadata["source_hashes"]:
            raise ValueError("resume source identity mismatch")
        invocation = {key: metadata[key] for key in
                      ("start_time", "git_commit", "git_status", "versions", "host", "source_hashes")}
        metadata = previous
        metadata.setdefault("resume_invocations", []).append(invocation)
    elif any(p.exists() for p in (*paths.values(), identity_path)):
        raise FileExistsError("outputs already exist; use --resume or a new config tag")
    identity_path.write_text(json.dumps(metadata, indent=2) + "\n")
    backend = emcee.backends.HDFBackend(str(paths["checkpoint"]))
    if not resume:
        backend.reset(config["nwalkers"], 3)
    context = mp.get_context("spawn").Pool(config["workers"]) if config["workers"] > 1 else nullcontext(None)
    started = time.monotonic()
    append_event(paths["log"], dict(event="resume" if resume else "start", metadata=metadata))
    with context as pool:
        sampler = emcee.EnsembleSampler(config["nwalkers"], 3, log_probability,
                                       args=(config,), pool=pool, backend=backend)
        state = None if resume else initial_positions(config)
        if not resume:
            sampler.random_state = np.random.RandomState(config["seed"]).get_state()
        remaining = config["nsteps"] - backend.iteration
        if remaining < 0:
            raise ValueError("checkpoint exceeds configured nsteps")
        while remaining:
            count = min(remaining, config.get("checkpoint_interval", 50))
            state = sampler.run_mcmc(state, count, progress=False)
            remaining -= count
            event = dict(event="progress", steps=int(backend.iteration),
                         target_steps=config["nsteps"], elapsed_seconds=time.monotonic()-started,
                         acceptance_fraction=sampler.acceptance_fraction.tolist())
            append_event(paths["log"], event)
            print(json.dumps(event), flush=True)
        try:
            tau = sampler.get_autocorr_time(tol=0)
            diagnostics = dict(autocorrelation=tau.tolist(),
                               autocorrelation_reliable=bool(np.all(backend.iteration >= 50*tau))) if np.all(np.isfinite(tau) & (tau > 0)) else dict(autocorrelation=None, autocorrelation_reliable=False)
        except (emcee.autocorr.AutocorrError, ValueError):
            diagnostics = dict(autocorrelation=None, autocorrelation_reliable=False)
        diagnostics["nonfinite_log_prob_count"] = int(np.count_nonzero(
            ~np.isfinite(sampler.get_log_prob())))
        metadata.update(end_time=_utcnow(), nsteps_completed=int(backend.iteration),
                        acceptance_fraction=sampler.acceptance_fraction.tolist(),
                        mean_acceptance_fraction=float(np.mean(sampler.acceptance_fraction)),
                        completed_posterior=diagnostics["nonfinite_log_prob_count"] == 0,
                        diagnostics=diagnostics)
        save_archive(paths["archive"], sampler.get_chain(), sampler.get_log_prob(),
                     backend.accepted, config["seed"], metadata)
    failed = diagnostics["nonfinite_log_prob_count"] > 0
    append_event(paths["log"], dict(event="failed_nonfinite_log_prob" if failed else "complete",
                                    metadata=metadata, archive=str(paths["archive"])))
    if failed:
        raise RuntimeError("Nonfinite stored log probabilities; archive preserved for diagnosis, not a completed posterior")
    return paths


def audit_init(config):
    """Evaluate fixed init at 1x/2x/4x requested grid resolution.

    Diagnostic only: no optimization, backend changes or historical reconstruction.
    Physics stays inside evaluate_point/evaluate_config. Redirect pipeline chatter
    to stderr so stdout is a single machine-readable official audit record.
    """
    point = [config["init"][p] for p in PARAMETERS]
    cases = []
    resolutions = [config["num_k"], 2 * config["num_k"], 4 * config["num_k"]]
    for resolution in resolutions:
        multiplier = resolution / config["num_k"]
        recipe = deepcopy(config)
        recipe["num_k"] = resolution
        validate_config(recipe)
        metadata = runtime_metadata(recipe, config)
        started = time.monotonic()
        with redirect_stdout(sys.stderr):
            evaluation = evaluate_point(point, recipe)
        scalars = {}
        for key, value in evaluation.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (str, bool, int, float, type(None))):
                scalars[key] = None if isinstance(value, float) and not np.isfinite(value) else value
        chi2 = scalars.get("chi2")
        ok = scalars.get("status") == "ok" and chi2 is not None and np.isfinite(chi2)
        metadata["end_time"] = _utcnow()
        cases.append(dict(multiplier=multiplier, num_k=recipe["num_k"], theta=point,
                          success=bool(ok), result=scalars, metadata=metadata,
                          elapsed_seconds=time.monotonic() - started))
    baseline = cases[0]["result"]
    for case in cases:
        case["difference_from_base"] = {
            key: float(case["result"][key] - baseline[key])
            for key in ("chi2", "d2", "k_dip", "N_total")
            if isinstance(case["result"].get(key), (int, float))
            and isinstance(baseline.get(key), (int, float))}
    return dict(event="audit_init", fixed_theta=point, num_k_resolutions=resolutions,
                success=all(case["success"] for case in cases), cases=cases,
                interpretation="grid-resolution diagnostic only; no convergence threshold asserted")


def _read_audit_record(path, theta):
    """Stream a JSONL input, hash exact bytes and require one exact seed match."""
    digest = hashlib.sha256()
    headers, matches = [], []
    with Path(path).open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            record = json.loads(raw)
            if record.get("_type") == "header":
                headers.append(record)
            elif all(record.get(p) == x for p, x in zip(PARAMETERS, theta)):
                matches.append((line_number, record))
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one exact init match, found {len(matches)}")
    line_number, record = matches[0]
    return record, headers, dict(path=str(path), sha256=digest.hexdigest(),
                                 bytes=Path(path).stat().st_size, physical_line=line_number)


def _spectrum_difference(ells, values, archived):
    """Array comparison only; archived TT arrays in this schema start at ell=2."""
    archived = np.asarray(archived, dtype=float)
    if archived.ndim != 1 or not len(archived) or not np.all(np.isfinite(archived)):
        raise ValueError("invalid archived D_ell array")
    historical_ells = np.arange(2, 2 + len(archived))
    common, current_idx, historical_idx = np.intersect1d(
        ells, historical_ells, return_indices=True)
    if not len(common):
        raise ValueError("no overlapping multipoles for comparison")
    delta = values[current_idx] - archived[historical_idx]
    worst = int(np.argmax(np.abs(delta)))
    return dict(n_compared=len(common), ell_min=int(common[0]), ell_max=int(common[-1]),
                reference_length=len(archived), reference_ell_start=2,
                full_reference_coverage=len(common) == len(archived),
                max_absolute_difference=float(np.max(np.abs(delta))),
                rms_difference=float(np.sqrt(np.mean(delta**2))),
                worst_ell=int(common[worst]), d2_difference=float(delta[0]),
                exactly_equal=bool(np.array_equal(values[current_idx], archived[historical_idx])))


def _camb_spectrum_diagnostic(grid, ps, ell_max):
    """Shared saved-spectrum evaluation: delegate every physics/chi2 operation."""
    from scripts.camb_scan import compute_camb_curves, lcdm_baseline, chi2_vs_planck
    from scripts.planck_data import C_ell_to_d_ell
    from scripts.chi2_analysis import chi2_unbinned, chi2_binned
    from scripts.optimizer_utils import find_k_dip
    _, ells, dell, _, _, _ = compute_camb_curves(dict(k_phys=grid, P_S=ps), ell_max)
    ells_lcdm, cl_lcdm = lcdm_baseline(ell_max)
    dell_lcdm = C_ell_to_d_ell(ells_lcdm, cl_lcdm)
    unbinned, lcdm_u, n_u = chi2_unbinned(dell, ells, dell_lcdm, ells_lcdm)
    binned, lcdm_b, n_b = chi2_binned(dell, ells, dell_lcdm, ells_lcdm)
    d2_indices = np.flatnonzero(np.asarray(ells) == 2)
    if len(d2_indices) != 1:
        raise ValueError("CAMB output must contain ell=2 exactly once")
    return dict(diagnostic_only=True, ell_max=ell_max,
                ell=np.asarray(ells).tolist(), D_ell=np.asarray(dell).tolist(),
                lcdm_ell=np.asarray(ells_lcdm).tolist(), D_ell_lcdm=np.asarray(dell_lcdm).tolist(),
                endpoint_clamping_above_model=ell_max < 2508,
                n_modes=len(grid), k_grid=grid.tolist(),
                chi2_unbinned=float(unbinned), chi2_lcdm_unbinned=float(lcdm_u),
                n_unbinned=int(n_u), chi2_binned=float(binned),
                chi2_lcdm_binned=float(lcdm_b), n_binned=int(n_b),
                chi2_low=float(chi2_vs_planck(ells, dell, 29)),
                d2=float(dell[d2_indices[0]]), k_dip=float(find_k_dip(grid, ps)))


def _array_difference(values, reference):
    delta = values - reference
    nonzero = reference != 0
    return dict(exactly_equal=bool(np.array_equal(values, reference)), n_compared=len(values),
                max_absolute_difference=float(np.max(np.abs(delta))),
                max_relative_difference=float(np.max(np.abs(delta[nonzero] / reference[nonzero])))
                if np.any(nonzero) else None,
                zero_reference_count=int(np.count_nonzero(~nonzero)))


def audit_saved_spectra(config, paths, declared_label=None):
    """Audit arbitrary v2 saved spectra without rerunning inflation/MS.

    A declared label is external provenance, never proof of the solver backend.
    Saved primordial arrays are passed unchanged; no normalization override.
    """
    from scripts.constants import As, k_pivot_phys
    metadata = runtime_metadata(config)
    cases = []
    for path in paths:
        raw = Path(path).read_bytes()
        saved = json.loads(raw)
        saved_metadata = saved["metadata"]
        spectrum = saved["spectrum"]
        grid, ps = (np.asarray(spectrum[key], dtype=float) for key in ("k_phys", "P_S"))
        if grid.ndim != 1 or len(grid) < 10 or ps.shape != grid.shape or not np.all(
                np.isfinite(grid) & (grid > 0)) or not np.all(np.diff(grid) > 0) or not np.all(
                np.isfinite(ps) & (ps > 0)):
            raise ValueError("saved spectrum requires >=10 sorted positive finite k/P pairs")
        with redirect_stdout(sys.stderr):
            case = _camb_spectrum_diagnostic(grid, ps, config["ell_max"])
        case.update(input=dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
                    saved_metadata=saved_metadata, declared_label=declared_label,
                    declared_label_verified=False, k_phys=grid.tolist(), P_S=ps.tolist(),
                    comparison_to_requested=dict(
                        theta_equal=all(saved_metadata.get(key) == config["init"][param]
                                        for key, param in zip(("x0", "y0", "N_star"), PARAMETERS)),
                        pivot_equal=saved_metadata.get("k_pivot_phys") == k_pivot_phys,
                        As_equal=saved_metadata.get("As_target") == As),
                    spectrum_passed_unchanged=True)
        raw_ps = spectrum.get("P_S_raw")
        if raw_ps is not None:
            raw_ps = np.asarray(raw_ps, dtype=float)
            if raw_ps.shape == ps.shape and np.all(np.isfinite(raw_ps)):
                case["P_S_raw"] = raw_ps.tolist()
        cases.append(case)
    comparisons = []
    for reference_index, index in ((r, i) for i in range(1, len(cases)) for r in range(i)):
        first, case = cases[reference_index], cases[index]
        a, b = np.asarray(first["k_phys"]), np.asarray(case["k_phys"])
        same_grid = a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-12, atol=0))
        comparison = dict(reference_case=reference_index, case=index, grid_compatible=same_grid,
                          chi2_difference=float(case["chi2_unbinned"] - first["chi2_unbinned"]),
                          background_mapping={key: dict(reference=first["saved_metadata"].get(key),
                                                        current=case["saved_metadata"].get(key))
                                              for key in ("N_total", "N_pivot", "k_pivot_code", "scale_factor", "P_S_pivot_raw")})
        mapping = {}
        for key in ("N_total", "N_pivot", "k_pivot_code", "scale_factor"):
            reference_value = first["saved_metadata"].get(key)
            current_value = case["saved_metadata"].get(key)
            if isinstance(reference_value, (int, float)) and isinstance(current_value, (int, float)):
                mapping[key + "_difference"] = float(current_value - reference_value)
                if reference_value != 0:
                    mapping[key + "_ratio"] = float(current_value / reference_value)
        current_total = case["saved_metadata"].get("N_total")
        reference_pivot = first["saved_metadata"].get("N_pivot")
        if isinstance(current_total, (int, float)) and isinstance(reference_pivot, (int, float)):
            mapping["derived_N_star_to_match_reference_N_pivot"] = float(current_total - reference_pivot)
            mapping["derivation"] = "current N_total - reference N_pivot; single diagnostic control, not fitted"
        comparison["pivot_mapping_comparison"] = mapping
        if same_grid:
            comparison["P_S"] = _array_difference(np.asarray(case["P_S"]), np.asarray(first["P_S"]))
            if "P_S_raw" in case and "P_S_raw" in first:
                comparison["P_S_raw"] = _array_difference(np.asarray(case["P_S_raw"]), np.asarray(first["P_S_raw"]))
        comparisons.append(comparison)
    metadata["end_time"] = _utcnow()
    return dict(event="audit_spectrum", success=True, metadata=metadata, cases=cases,
                comparisons=comparisons, interpretation="unchanged saved P_S evaluated by current shared CAMB/chi2; labels externally declared")


def audit_archive(config, archive_log, reference_log=None, replay_spectrum=None):
    """Factorial archived-P_S audit; never calls the inflation/MS solver.

    ``legacy_truncation_emulation`` explicitly replicates P[:len(header k)]
    from historical a3cbee8:scripts/camb_recompute_full_chi2.py:89-94.
    Existence of that source defect does NOT prove it wrote the reference log.
    Neither emulation nor ell=2500 endpoint-clamping is a production recipe.
    """
    from pspectrum_pipeline import ensure_k_pivot
    from scripts.constants import As, k_pivot_phys

    theta = [config["init"][p] for p in PARAMETERS]
    record, headers, provenance = _read_audit_record(archive_log, theta)
    if len(headers) != 1:
        raise ValueError("archived spectrum input must contain exactly one header")
    header = headers[0]
    if record.get("status") != "ok":
        raise ValueError("archived seed status must be ok")
    header_grid = np.asarray(header["k_phys"], dtype=float)
    spectrum = np.asarray(record["P_S"], dtype=float)
    if header_grid.ndim != 1 or len(header_grid) < 10 or not np.all(
            np.isfinite(header_grid) & (header_grid > 0)) or not np.all(np.diff(header_grid) > 0):
        raise ValueError("invalid historical header grid")
    if spectrum.ndim != 1 or not np.all(np.isfinite(spectrum) & (spectrum > 0)):
        raise ValueError("invalid historical primordial spectrum")
    correct_grid, pivot_index = ensure_k_pivot(header_grid, header["k_pivot_phys"])
    if len(spectrum) != len(correct_grid) or len(correct_grid) != len(header_grid) + 1:
        raise ValueError("audit expects one omitted pivot: header n, archived P_S n+1")
    reference = None
    inputs = {"archive": provenance}
    if reference_log is not None:
        reference, _, inputs["reference"] = _read_audit_record(reference_log, theta)
        if reference.get("status") != "ok":
            raise ValueError("reference seed status must be ok")
        if "D_ell_full" not in reference:
            raise ValueError("reference requires D_ell_full")
    constants = {"As": dict(header=float(header["As"]), current=float(As)),
                 "k_pivot_phys": dict(header=float(header["k_pivot_phys"]), current=float(k_pivot_phys))}
    for comparison in constants.values():
        comparison["equal"] = comparison["header"] == comparison["current"]
    metadata = runtime_metadata(config)
    metadata["archive_inputs"] = inputs
    metadata["archived_header"] = {key: value for key, value in header.items()
                                   if key not in ("k_phys", "ells")}
    archived_scalars = {key: value for key, value in record.items()
                       if isinstance(value, (str, bool, int, float, type(None)))}
    pairings = [("correct_pivot_grid", correct_grid, spectrum),
                ("legacy_truncation_emulation", header_grid, spectrum[:len(header_grid)])]
    replay_comparison = None
    if replay_spectrum is not None:
        raw = Path(replay_spectrum).read_bytes()
        replay = json.loads(raw)
        replay_metadata = replay["metadata"]
        if any(replay_metadata.get(key) != value for key, value in
               zip(("x0", "y0", "N_star"), theta)):
            raise ValueError("replay metadata initial coordinate mismatch")
        replay_grid = np.asarray(replay["spectrum"]["k_phys"], dtype=float)
        replay_ps = np.asarray(replay["spectrum"]["P_S"], dtype=float)
        if replay_grid.shape != correct_grid.shape or replay_ps.shape != spectrum.shape:
            raise ValueError("replay spectrum shape mismatch against archived pivot grid")
        if not np.all(np.isfinite(replay_grid) & (replay_grid > 0)) or not np.all(
                np.diff(replay_grid) > 0) or not np.all(np.isfinite(replay_ps) & (replay_ps > 0)):
            raise ValueError("invalid replay spectrum")
        if not np.allclose(replay_grid, correct_grid, rtol=1e-12, atol=0):
            raise ValueError("replay k-grid differs; pointwise spectrum comparison not defined")
        metadata["archive_inputs"]["replay"] = dict(path=str(replay_spectrum), bytes=len(raw),
                                                    sha256=hashlib.sha256(raw).hexdigest())
        metadata["replay_metadata"] = replay_metadata
        replay_comparison = dict(k_phys=_array_difference(replay_grid, correct_grid),
                                 P_S=_array_difference(replay_ps, spectrum))
        if isinstance(replay_metadata.get("N_total"), (int, float)) and isinstance(record.get("N_total"), (int, float)):
            replay_comparison["N_total_difference_from_rounded_archive"] = float(
                replay_metadata["N_total"] - record["N_total"])
        pairings.append(("replay_correct", replay_grid, replay_ps))
    cases = []
    with redirect_stdout(sys.stderr):
        for pairing, grid, ps in pairings:
            for ell_max in (2500, 2508):
                case = _camb_spectrum_diagnostic(grid, ps, ell_max)
                case["pairing"] = pairing
                ells, dell = np.asarray(case["ell"]), np.asarray(case["D_ell"])
                if "D_ell" in record:
                    case["archived_low_ell_comparison"] = _spectrum_difference(ells, dell, record["D_ell"])
                if reference is not None:
                    case["reference_spectrum_comparison"] = _spectrum_difference(ells, dell, reference["D_ell_full"])
                    case["reference_chi2_difference"] = float(case["chi2_unbinned"] - reference["chi2_unbinned"])
                cases.append(case)
    metadata["end_time"] = _utcnow()
    return dict(event="audit_archive", fixed_theta=theta, success=True, metadata=metadata,
                constants_comparison=constants, constants_match=all(x["equal"] for x in constants.values()),
                header_grid_length=len(header_grid), archived_spectrum_length=len(spectrum),
                inserted_pivot_index=int(pivot_index), archived_scalar_metrics=archived_scalars,
                cases=cases, replay_comparison=replay_comparison,
                interpretation="controlled archived-spectrum diagnostics; historical writer execution not proven",
                legacy_source_evidence="a3cbee8:scripts/camb_recompute_full_chi2.py:89-94; P[:len(header k)]")


def audit_background(config):
    """Fixed 2x2 background sampling diagnostic through the existing evaluator.

    Uses current algorithms/backend throughout; changes only fresh-model T_max
    and bg_steps. This is not a historical-algorithm replay or a sampling recipe.
    """
    from models import HiggsModel
    from scripts.camb_scan import evaluate_config
    from pspectrum_pipeline import ensure_k_pivot
    from scripts.constants import k_pivot_phys

    theta = [config["init"][p] for p in PARAMETERS]
    cases = []
    for t_max in (500.0, 2000.0):
        for bg_steps in (1000, 50000):
            recipe = deepcopy(config)
            recipe["num_k"] = 320
            recipe["audit_background"] = dict(T_max=t_max, bg_steps=bg_steps)
            validate_config(recipe)
            model = HiggsModel(lam=recipe["lam"], xi=recipe["xi"])
            model.T_max = t_max
            model.bg_steps = bg_steps
            model.x0, model.y0 = theta[:2]
            grid = k_grid_from_config(recipe)
            exact_grid, _ = ensure_k_pivot(grid, k_pivot_phys)
            metadata = runtime_metadata(recipe, config)
            started = time.monotonic()
            with redirect_stdout(sys.stderr):
                evaluation = evaluate_config(*theta, build_evaluation_args(recipe),
                                             k_phys_grid=grid, model=model)
            chi2 = evaluation.get("chi2", np.nan)
            success = evaluation.get("status") == "ok" and np.isfinite(chi2)
            metadata["end_time"] = _utcnow()
            cases.append(dict(theta=theta, T_max=t_max, bg_steps=bg_steps,
                              num_k=320, diagnostic_only=True, success=bool(success),
                              actual_model_attributes=dict(T_max=float(model.T_max),
                                  bg_steps=int(model.bg_steps), x0=float(model.x0), y0=float(model.y0),
                                  lam=float(model.lam), xi=float(model.xi_val)),
                              metadata=metadata, result=evaluation,
                              k_phys=exact_grid.tolist(),
                              elapsed_seconds=time.monotonic() - started))
    return dict(event="audit_background", fixed_theta=theta, num_k=320,
                success=all(case["success"] for case in cases), cases=cases,
                interpretation="fixed 2x2 T_max/background sampling audit under current evaluator; no algorithm/backend change")


def _explore_worker(payload):
    """Evaluate one grid/random point with the production recipe (picklable)."""
    theta, config = payload
    started = time.monotonic()
    result = evaluate_point(theta, config)
    chi2 = result.get("chi2", np.nan)
    ok = result.get("status") == "ok" and np.isfinite(chi2)
    return dict(theta=[float(x) for x in theta], status=result.get("status"),
                ok=bool(ok),
                chi2=float(chi2) if ok else None,
                chi2_lcdm=result.get("chi2_lcdm"),
                dchi2=result.get("dchi2"),
                d2=result.get("d2"), d2_lcdm=result.get("d2_lcdm"),
                chi2_low=result.get("chi2_low"),
                chi2_binned=result.get("chi2_binned_model"),
                chi2_binned_lcdm=result.get("chi2_binned_lcdm"),
                k_dip=result.get("k_dip"),
                suppression_pct=result.get("suppression_pct"),
                N_total=result.get("N_total"),
                error=result.get("error"),
                elapsed_seconds=time.monotonic() - started)


def _explore_points(args):
    """Deterministic grid, uniform random points, or replay of saved thetas."""
    if getattr(args, "points_from", None):
        thetas = []
        with Path(args.points_from).open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("_type") == "data" and record.get("theta"):
                    thetas.append(record["theta"])
        if not thetas:
            raise ValueError(f"no data records with theta in {args.points_from}")
        raw = np.asarray(thetas, dtype=float)
        return np.unique(np.column_stack([np.round(raw[:, 0], 4),
                                          np.round(raw[:, 1], 4),
                                          np.round(raw[:, 2], 2)]), axis=0)
    lo_p, hi_p = args.phi0_range
    lo_y, hi_y = args.y0_range
    lo_n, hi_n = args.nstar_range
    if args.n_random:
        rng = np.random.default_rng(args.seed)
        raw = np.column_stack([
            rng.uniform(lo_p, hi_p, args.n_random),
            rng.uniform(lo_y, hi_y, args.n_random),
            rng.uniform(lo_n, hi_n, args.n_random)])
    else:
        grid = np.meshgrid(np.linspace(lo_p, hi_p, args.n_phi0),
                           np.linspace(lo_y, hi_y, args.n_y0),
                           np.linspace(lo_n, hi_n, args.n_nstar), indexing="ij")
        raw = np.column_stack([g.ravel() for g in grid])
    # Round so duplicate requests collapse predictably (matches pipeline naming).
    rounded = np.column_stack([np.round(raw[:, 0], 4), np.round(raw[:, 1], 4),
                               np.round(raw[:, 2], 2)])
    return np.unique(rounded, axis=0)


def explore_configs(config, args):
    """Scan the production likelihood over a box; one JSONL record per point.

    Uses the same ``evaluate_point``/``evaluate_config`` path as the sampler, so
    the reported chi-square is the production unbinned statistic, not the
    coarse screening grid used by ``scripts/camb_scan.py --quick``.
    """
    from concurrent.futures import ProcessPoolExecutor
    if getattr(args, "explore_num_k", None):
        config = dict(config, num_k=int(args.explore_num_k))
        validate_config(config)
    points = _explore_points(args)
    out = Path(args.out) if args.out else Path(
        get_path("scans", f"higgs_explore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new --out")
    metadata = runtime_metadata(config, config)
    metadata["explore"] = dict(phi0_range=list(args.phi0_range), y0_range=list(args.y0_range),
                               nstar_range=list(args.nstar_range), n_requested=int(len(points)),
                               n_random=args.n_random, grid=[args.n_phi0, args.n_y0, args.n_nstar],
                               seed=args.seed, workers=args.workers)
    started = time.monotonic()
    done = 0
    with out.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(_type="header", metadata=metadata,
                                     config_hash=config_hash(config))) + "\n")
        stream.flush()
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            for record in pool.map(_explore_worker,
                                   [(point, config) for point in points], chunksize=8):
                stream.write(json.dumps(dict(_type="data", **record), allow_nan=False) + "\n")
                stream.flush()
                done += 1
                if done % 200 == 0 or done == len(points):
                    elapsed = time.monotonic() - started
                    eta = elapsed / done * (len(points) - done)
                    print(f"  [{done}/{len(points)}] {elapsed:.0f}s ETA {eta:.0f}s",
                          file=sys.stderr, flush=True)
    rows = []
    with out.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("_type") == "data" and record.get("ok"):
                rows.append(record)
    rows.sort(key=lambda r: r["chi2"])
    summary = dict(event="explore", out=str(out), n_points=len(points), n_ok=len(rows),
                   best_chi2=rows[0] if rows else None,
                   best_chi2_with_d2_below_lcdm=next(
                       (r for r in rows if r["d2"] is not None and r["d2"] < 1028.7), None),
                   metadata=metadata, elapsed_seconds=time.monotonic() - started)
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def pareto_from_explore(path, chi2_caps=(0.0, 5.0, 10.0, 20.0, 40.0, 80.0), top=10,
                        verify_num_k=None, config=None):
    """Pareto frontier of quadrupole suppression versus full-chi2 cost.

    Reads an ``--explore`` JSONL and, for each Delta-chi2 cap above the global
    minimum, reports the configuration with the smallest D2. Every reported
    point is a real evaluated record, not an interpolation.
    """
    raw = Path(path).read_bytes()
    rows, header = [], None
    for line in raw.decode().splitlines():
        record = json.loads(line)
        if record.get("_type") == "header":
            header = record
        elif record.get("_type") == "data" and record.get("ok"):
            rows.append(record)
    if not rows:
        raise ValueError(f"no successful records in {path}")
    rows.sort(key=lambda r: r["chi2"])
    chi2_min = rows[0]["chi2"]
    ladder = []
    for cap in chi2_caps:
        eligible = [r for r in rows if r["chi2"] <= chi2_min + cap]
        best = min(eligible, key=lambda r: r["d2"])
        ladder.append(dict(delta_chi2_cap=float(cap), n_eligible=len(eligible),
                           theta=best["theta"], chi2=best["chi2"],
                           dchi2_vs_global_min=best["chi2"] - chi2_min,
                           d2=best["d2"], k_dip=best["k_dip"],
                           suppression_pct=best["suppression_pct"]))
    unique = {tuple(point["theta"]): point for point in ladder}
    for record in rows[:top]:
        unique.setdefault(tuple(record["theta"]), dict(
            theta=record["theta"], chi2=record["chi2"],
            dchi2_vs_global_min=record["chi2"] - chi2_min, d2=record["d2"],
            k_dip=record["k_dip"], suppression_pct=record["suppression_pct"]))
    result = dict(event="pareto", input=str(path),
                input_sha256=hashlib.sha256(raw).hexdigest(),
                input_metadata=header.get("metadata") if header else None,
                n_records=len(rows), chi2_min=chi2_min,
                best_chi2=rows[0], ladder=ladder,
                configs=sorted(unique.values(), key=lambda p: p["d2"]))
    if verify_num_k:
        if config is None:
            raise ValueError("verify_num_k requires the production config")
        recipe = dict(config, num_k=int(verify_num_k))
        validate_config(recipe)
        result["verification"] = dict(num_k=int(verify_num_k),
                                      config_hash=config_hash(recipe),
                                      entries=[])
        with redirect_stdout(sys.stderr):
            for point in result["configs"]:
                evaluation = evaluate_point(point["theta"], recipe)
                ok = evaluation.get("status") == "ok" and np.isfinite(evaluation.get("chi2", np.nan))
                result["verification"]["entries"].append(dict(
                    theta=point["theta"], sweep_chi2=point["chi2"], sweep_d2=point["d2"],
                    verified_ok=bool(ok), verified_chi2=evaluation.get("chi2"),
                    verified_d2=evaluation.get("d2"), verified_k_dip=evaluation.get("k_dip"),
                    verified_suppression_pct=evaluation.get("suppression_pct"),
                    chi2_shift=(evaluation.get("chi2") - point["chi2"]) if ok else None))
    return result


def audit_end(config, t_max=500.0, bg_steps=1000):
    """Compare the historical and current end-of-inflation index rules.

    One background solve, no MS/CAMB. The historical rule (a7b45c4) scans
    backward for the LAST epsH >= 1; the current rule takes the first upward
    crossing that mean-stays above 1. Both feed the same fractional-crossing
    N_total formula and the same existing get_k_pivot_code helper.
    """
    import inf_dyn_background as bg_solver
    from models import HiggsModel
    from pspectrum_pipeline import find_end_of_inflation, get_k_pivot_code

    theta = [config["init"][p] for p in PARAMETERS]
    recipe = deepcopy(config)
    recipe["audit_end"] = dict(T_max=float(t_max), bg_steps=int(bg_steps))
    validate_config(recipe)
    model = HiggsModel(lam=recipe["lam"], xi=recipe["xi"])
    model.T_max = float(t_max)
    model.bg_steps = int(bg_steps)
    model.x0, model.y0 = theta[:2]
    T_span = np.linspace(0.0, float(t_max), int(bg_steps))
    with redirect_stdout(sys.stderr):
        bg_sol = bg_solver.run_background_simulation(model, T_span)
        derived = bg_solver.get_derived_quantities(bg_sol, model)
    eps = np.asarray(derived["epsH"], dtype=float)
    N = np.asarray(derived["N"], dtype=float)

    def fractional_N(index):
        value = float(N[index])
        if 0 < index < len(eps) and eps[index] > 1.0 >= eps[index - 1]:
            f = (1.0 - eps[index - 1]) / (eps[index] - eps[index - 1])
            if 0 <= f <= 1:
                value = float(N[index - 1] + f * (N[index] - N[index - 1]))
        return value

    def upward_crossings():
        start = next((i for i, e in enumerate(eps) if e < 1.0), None)
        if start is None:
            return None, []
        return start, [i for i in range(start + 1, len(eps))
                       if eps[i - 1] < 1.0 <= eps[i]]

    start_idx, crossings = upward_crossings()
    current_idx = int(find_end_of_inflation(eps))
    if current_idx == -1:
        current_idx = len(eps) - 1
    legacy_idx = -1
    if start_idx is not None:
        for i in range(len(eps) - 1, start_idx - 1, -1):
            if eps[i] >= 1.0:
                legacy_idx = i
                break

    def describe(index, label):
        if index < 0:
            return dict(rule=label, end_index=None, N_total=None,
                        N_pivot=None, k_pivot_code=None, eps_at_end=None)
        n_total = fractional_N(index)
        k_code, pivot_idx, _ = get_k_pivot_code(
            bg_sol, derived, index, theta[2], N_total=n_total)
        return dict(rule=label, end_index=int(index), N_total=n_total,
                    N_pivot=float(n_total - theta[2]) if n_total >= theta[2] else None,
                    k_pivot_code=None if k_code is None else float(k_code),
                    pivot_bg_index=None if pivot_idx is None else int(pivot_idx),
                    eps_at_end=float(eps[index]))

    current = describe(current_idx, "current_first_persistent_crossing")
    legacy = describe(legacy_idx, "legacy_last_eps_above_one")
    delta = None
    if current["N_total"] is not None and legacy["N_total"] is not None:
        delta = float(legacy["N_total"] - current["N_total"])
    return dict(event="audit_end", fixed_theta=theta, T_max=float(t_max),
                bg_steps=int(bg_steps), diagnostic_only=True,
                start_index=start_idx, n_upward_crossings=len(crossings),
                upward_crossing_indices=crossings,
                upward_crossing_N=[float(N[i]) for i in crossings],
                upward_crossing_eps=[float(eps[i]) for i in crossings],
                current_rule=current, legacy_rule=legacy,
                legacy_minus_current_N_total=delta,
                metadata=runtime_metadata(recipe, config),
                interpretation="identical trajectory; isolates the end-index rule only")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--audit-init", action="store_true",
                      help="fixed init, num_k at 1x/2x/4x; pure JSON diagnostic")
    mode.add_argument("--audit-spectrum", nargs="+", type=Path,
                      help="one or more saved v2 spectra, shared CAMB/chi2 audit")
    parser.add_argument("--spectrum-label", help="externally declared label; never backend verification")
    mode.add_argument("--audit-background", action="store_true",
                      help="fixed init and num_k320, T_max500/2000 x bg_steps1000/50000")
    mode.add_argument("--audit-end", action="store_true",
                      help="one trajectory; compare historical vs current end-index rule")
    mode.add_argument("--pareto", type=Path,
                      help="Pareto frontier from an --explore JSONL")
    mode.add_argument("--explore", action="store_true",
                      help="production-likelihood scan over a phi0/y0/N_star box")
    parser.add_argument("--phi0-range", nargs=2, type=float, default=[5.6, 6.9])
    parser.add_argument("--y0-range", nargs=2, type=float, default=[-0.9, -0.05])
    parser.add_argument("--nstar-range", nargs=2, type=float, default=[40.0, 62.0])
    parser.add_argument("--n-phi0", type=int, default=9)
    parser.add_argument("--n-y0", type=int, default=7)
    parser.add_argument("--n-nstar", type=int, default=11)
    parser.add_argument("--n-random", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scan-workers", type=int, default=8)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--points-from", type=Path,
                        help="re-evaluate theta values from an earlier explore JSONL")
    parser.add_argument("--pareto-verify-num-k", type=int,
                        help="re-evaluate frontier configs at this num_k")
    parser.add_argument("--explore-num-k", type=int,
                        help="override num_k for this exploration (convergence)")
    parser.add_argument("--audit-tmax", type=float, default=500.0)
    parser.add_argument("--audit-bg-steps", type=int, default=1000)
    mode.add_argument("--audit-archive", action="store_true",
                      help="archived P_S correct-grid versus legacy-truncation factorial")
    parser.add_argument("--archive-log", type=Path)
    parser.add_argument("--reference-log", type=Path)
    parser.add_argument("--replay-spectrum", type=Path,
                        help="optional saved historical-pipeline result JSON; no MS solve")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--compare", nargs=2, type=Path, metavar=("FIRST", "SECOND"))
    mode.add_argument("--summary", type=Path)
    mode.add_argument("--validate", type=Path)
    parser.add_argument("--burnin", type=int)
    parser.add_argument("--thin", type=int)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.compare:
        if args.atol < 0 or not np.isfinite(args.atol):
            parser.error("atol must be finite and nonnegative")
        result = compare_archives(*args.compare, atol=args.atol)
    elif args.pareto:
        base = None
        if args.pareto_verify_num_k:
            if args.config is None:
                parser.error("--pareto-verify-num-k requires --config")
            base = effective_config(json.loads(args.config.read_text()))
        result = pareto_from_explore(args.pareto,
                                     verify_num_k=args.pareto_verify_num_k,
                                     config=base)
    elif args.summary or args.validate:
        if args.burnin is None or args.thin is None:
            parser.error("postrun operations require explicit --burnin and --thin")
        result = (summarize_archive if args.summary else validate_archive)(
            args.summary or args.validate, args.burnin, args.thin)
    else:
        if args.config is None:
            parser.error("--config required for dry-run/smoke/sampling")
        requested = json.loads(args.config.read_text())
        config = effective_config(requested, args.smoke)
        if args.audit_spectrum:
            with redirect_stdout(sys.stderr):
                result = audit_saved_spectra(config, args.audit_spectrum, args.spectrum_label)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["success"] else 1
        if args.audit_end:
            with redirect_stdout(sys.stderr):
                result = audit_end(config, args.audit_tmax, args.audit_bg_steps)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0
        if args.audit_background:
            with redirect_stdout(sys.stderr):
                result = audit_background(config)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["success"] else 1
        if args.audit_archive:
            if args.archive_log is None:
                parser.error("--audit-archive requires --archive-log")
            with redirect_stdout(sys.stderr):
                result = audit_archive(config, args.archive_log, args.reference_log, args.replay_spectrum)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["success"] else 1
        if args.audit_init:
            result = audit_init(config)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["success"] else 1
        if args.explore:
            args.workers = args.scan_workers
            result = explore_configs(config, args)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0
        if args.dry_run:
            point = [config["init"][p] for p in PARAMETERS]
            with redirect_stdout(sys.stderr):
                evaluation = evaluate_point(point, config)
            result = dict(event="dry_run", theta=point, result=evaluation,
                          metadata=runtime_metadata(config, requested))
            print(json.dumps(result, allow_nan=False))
            return 0 if evaluation.get("status") == "ok" and np.isfinite(evaluation.get("chi2", np.nan)) else 1
        result = {k: str(v) for k, v in run_sampler(config, requested, args.resume).items()}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get("equal", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
