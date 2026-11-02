"""Rank existing full-chi2 Higgs scan records without running any solvers.

The filters are inclusive. Records sort by ascending full (unbinned) delta
chi2, then binned delta chi2 and initial conditions. No coordinate rounding
or deduplication is applied: each result retains its source line.
"""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import hashlib
import json
import math
from pathlib import Path
import subprocess

from scripts.camb_scan import D2_LCDM
from scripts.plotting import get_path, make_filename


REPO_ROOT = Path(__file__).resolve().parents[1]
SORT_KEYS = ("dchi2_total", "dchi2_high", "phi0", "y0", "N_star",
             "source_file", "source_line")

# Archived full-chi2 scans produced with the pre-479fdbb end-of-inflation rule
# (last eps_H >= 1 instead of the first permanent crossing) and, for the
# recomputation path, an 80-vs-81 spectrum/grid mispairing. Their N_total is
# ~0.9 e-folds too large, which displaces the pivot by 2.45x in k_pivot_code.
LEGACY_INPUTS = {
    "8af4e27fc79cc70e88e6d8efae0147e63d3f82098de3216c6e9659512db36cb1":
        "legacy end-of-inflation rule; N_total overshoots by ~0.9 e-folds",
}


def sha256_file(path):
    """Hash bytes incrementally, including large scan logs."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected finite number")
    if not math.isfinite(value):
        raise ValueError("expected finite number")
    return value


def record_metrics(record):
    """Extract full-log metrics; never substitute rounded d2 or low-ell chi2."""
    metrics = {key: _finite(record[key]) for key in
               ("phi0", "y0", "N_star", "k_dip", "chi2_binned",
                "chi2_lcdm_binned", "chi2_unbinned", "chi2_lcdm_unbinned")}
    spectrum = record["D_ell_full"]
    if not isinstance(spectrum, list) or not spectrum:
        raise ValueError("D_ell_full must be a nonempty list starting at ell=2")
    metrics["d2"] = _finite(spectrum[0])
    if metrics["d2"] < 0:
        raise ValueError("negative quadrupole")
    metrics["d2_suppression"] = 100 * (1 - metrics["d2"] / D2_LCDM)
    metrics["dchi2_high"] = metrics["chi2_binned"] - metrics["chi2_lcdm_binned"]
    metrics["dchi2_total"] = metrics["chi2_unbinned"] - metrics["chi2_lcdm_unbinned"]
    return metrics


def expand_logs(patterns):
    """Accept shell-expanded paths or quoted glob patterns; reject missing inputs."""
    paths = set()
    for pattern in patterns:
        matches = glob.glob(str(pattern))
        if not matches:
            raise ValueError(f"No logs match: {pattern}")
        for match in matches:
            path = Path(match).resolve()
            if not path.is_file():
                raise ValueError(f"Not a log file: {path}")
            paths.add(path)
    if not paths:
        raise ValueError("At least one input log is required")
    return sorted(paths)


def _source_name(path):
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def rank_seeds(logs, *, min_d2_suppression=15.0, max_dchi2_high=10.0,
               max_dchi2_total=15.0, allow_legacy=False):
    """Return deterministic provenance, diagnostics and sorted passing records.

    Headers, failed records and incomplete/nonfinite records are counted and
    skipped. Malformed JSON fails with its file/line instead of silently losing
    data. Hashes are checked again after parsing to detect concurrent appends.
    """
    filters = {"min_d2_suppression": _finite(min_d2_suppression),
               "max_dchi2_high": _finite(max_dchi2_high),
               "max_dchi2_total": _finite(max_dchi2_total)}
    counts = Counter({key: 0 for key in
                      ("records", "headers", "failed", "invalid", "filtered", "selected")})
    inputs, seeds = [], []
    legacy = []
    for path in expand_logs(logs):
        source = _source_name(path)
        digest = sha256_file(path)
        if digest in LEGACY_INPUTS:
            legacy.append({"path": source, "sha256": digest,
                           "reason": LEGACY_INPUTS[digest]})
            if not allow_legacy:
                raise ValueError(
                    f"{source} is LEGACY data: {LEGACY_INPUTS[digest]}. "
                    "Pass --allow-legacy to rank it anyway (never for production seeds).")
        inputs.append({"path": source, "sha256": digest, "size_bytes": path.stat().st_size})
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON at {source}:{number}: {exc.msg}") from exc
                counts["records"] += 1
                if not isinstance(record, dict):
                    counts["invalid"] += 1
                    continue
                if record.get("_type") == "header":
                    counts["headers"] += 1
                    continue
                if record.get("status") != "ok":
                    counts["failed"] += 1
                    continue
                try:
                    metrics = record_metrics(record)
                except (KeyError, TypeError, ValueError, OverflowError):
                    counts["invalid"] += 1
                    continue
                if (metrics["d2_suppression"] < min_d2_suppression or
                        metrics["dchi2_high"] > max_dchi2_high or
                        metrics["dchi2_total"] > max_dchi2_total):
                    counts["filtered"] += 1
                    continue
                metrics.update(source_file=source, source_line=number)
                seeds.append(metrics)
                counts["selected"] += 1
        if sha256_file(path) != digest:
            raise ValueError(f"Input changed while ranking: {source}")
    seeds.sort(key=lambda seed: tuple(seed[key] for key in SORT_KEYS))
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"schema_version": 1, "filters": filters, "git_commit": commit,
            "ranker_sha256": sha256_file(__file__), "D2_LCDM": D2_LCDM,
            "sort_keys": list(SORT_KEYS), "inputs": inputs,
            "data_status": "legacy" if legacy else "current",
            "legacy_inputs": legacy,
            "counts": dict(counts), "seeds": seeds}


def format_table(result):
    """Render all selected seeds with round-trip float precision."""
    columns = ("phi0", "y0", "N_star", "k_dip", "d2", "d2_suppression",
               "dchi2_high", "dchi2_total")
    lines = [" | ".join(columns)]
    lines.extend(" | ".join(str(seed[key]) for key in columns) for seed in result["seeds"])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-d2-suppression", type=float, default=15.0)
    parser.add_argument("--max-dchi2-high", type=float, default=10.0)
    parser.add_argument("--max-dchi2-total", type=float, default=15.0)
    parser.add_argument("--allow-legacy", action="store_true",
                        help="Acknowledge ranking archived pre-fix data; not for production seeds")
    args = parser.parse_args(argv)
    try:
        result = rank_seeds(args.logs, min_d2_suppression=args.min_d2_suppression,
                            max_dchi2_high=args.max_dchi2_high,
                            max_dchi2_total=args.max_dchi2_total,
                            allow_legacy=args.allow_legacy)
        out = args.out or Path(get_path("scans", make_filename("higgs_seed_ranking")))
        if out.resolve() in expand_logs(args.logs):
            raise ValueError("Output must not overwrite an input log")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result["counts"], sort_keys=True))
    print(format_table(result))
    print(f"Wrote {out}; sha256={sha256_file(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
