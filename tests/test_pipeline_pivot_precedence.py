"""Pivot precedence and truthful run-metadata regressions (issue #36).

Drives the actual ``full_pbh_pipeline`` CLI in an isolated sandbox with a small
explicit grid, then checks the saved spectrum metadata and compares the
observables the CLI *printed* against the observables recomputed from the
saved P_S(k) grid at the effective pivot. The printed pivot, n_s and A_s are
parsed from real stdout, so a wrong pivot computation is caught even when the
metadata field looks correct.
"""
from __future__ import annotations

import json
import re
import subprocess
from typing import Any

import numpy as np
import pytest

from scripts.observables import extract_ns, interpolate_As


_WINDOW = 3.0
_K_MIN = 1e-4
_K_MAX = 1e1
_NUM_K = 41

_OBSERVABLES_RE = re.compile(
    r"n_s\(k=(?P<pivot>[^)]+)\)\s*=\s*(?P<n_s>\S+),\s*A_s\s*=\s*(?P<As>\S+)"
)


def _config(pivot, N_star=65.0):
    return {
        "model": "EzquiagaCHIModel",
        "description": "sandbox pivot-precedence fixture",
        "model_params": {"lambda_0": 2.23e-7, "xi_0": 7.55, "c": 0.77},
        "inflection": {"x_c": 0.784, "beta": 1e-5},
        "ics": {"x0": 8.0, "y0": -1e-4},
        "pipeline": {
            "N_star": N_star,
            "k_min": _K_MIN,
            "k_max": _K_MAX,
            "num_k": _NUM_K,
            "k_pivot_phys": pivot,
            "ns_window": _WINDOW,
            "no_plot": True,
            "n_cores": 1,
            "normalize_to_As": False,
            "zeta_c": [0.05],
            "output_dir": "outputs/plots/pbh/top_configs",
        },
    }


def _run(sandbox, *argv):
    proc = sandbox.run_cli(*argv)
    assert proc.returncode == 0, f"CLI failed:\n{proc.stdout}\n{proc.stderr}"
    return proc


def _only_spectrum(sandbox):
    records = sandbox.spectra()
    assert len(records) == 1, f"expected one cached spectrum, got {len(records)}"
    return records[0]


def _printed_observables(proc: subprocess.CompletedProcess[str]) -> tuple[float, float, float]:
    """Parse the CLI's ``n_s(k=<pivot>) = <n_s>, A_s = <A_s>`` stdout line."""
    match = _OBSERVABLES_RE.search(proc.stdout)
    assert match is not None, f"no observables line in stdout:\n{proc.stdout}"
    ns_token = match.group("n_s")
    as_token = match.group("As")
    assert ns_token != "None" and as_token != "None", (
        f"run printed non-finite observables: n_s={ns_token}, A_s={as_token}"
    )
    return float(match.group("pivot")), float(ns_token), float(as_token)


def _assert_printed_matches_saved_spectrum(
    record: dict[str, Any],
    proc: subprocess.CompletedProcess[str],
    effective_pivot: float,
) -> None:
    """The CLI's printed observables must be recomputable from its saved grid."""
    k = np.asarray(record["spectrum"]["k_phys"], dtype=float)
    P_S = np.asarray(record["spectrum"]["P_S"], dtype=float)
    assert np.all(np.isfinite(P_S)) and np.all(P_S > 0)

    # The saved grid must bracket the fit window at the effective pivot.
    assert k.min() <= effective_pivot / _WINDOW
    assert k.max() >= effective_pivot * _WINDOW

    observed_pivot, observed_ns, observed_As = _printed_observables(proc)
    assert observed_pivot == pytest.approx(effective_pivot)

    expected_ns, _ = extract_ns(
        k, P_S, k_pivot=effective_pivot, ns_window=_WINDOW, method="lsq"
    )
    assert expected_ns is not None and np.isfinite(expected_ns)
    assert observed_ns == pytest.approx(expected_ns, rel=1e-9)

    expected_As = interpolate_As(k, P_S, effective_pivot)
    assert expected_As is not None and np.isfinite(expected_As)
    assert observed_As == pytest.approx(expected_As, rel=1e-9)


@pytest.mark.slow
def test_config_pivot_reaches_run_and_metadata(pbh_pipeline_sandbox):
    config = pbh_pipeline_sandbox.root / "cfg_pivot_0002.json"
    config.write_text(json.dumps(_config(0.002)))

    proc = _run(pbh_pipeline_sandbox, "--config", config, "--no-force",
                "--no-plot", "--workers", "1", "--accretion", "Chisholm1",
                "--zeta-c", "0.05")

    record = _only_spectrum(pbh_pipeline_sandbox)
    assert record["metadata"]["k_pivot_Mpc"] == pytest.approx(0.002)
    assert record["metadata"]["source"] == "MS"
    _assert_printed_matches_saved_spectrum(record, proc, 0.002)


@pytest.mark.slow
def test_explicit_cli_pivot_overrides_config(pbh_pipeline_sandbox):
    config = pbh_pipeline_sandbox.root / "cfg_pivot_0002.json"
    config.write_text(json.dumps(_config(0.002)))

    proc = _run(pbh_pipeline_sandbox, "--config", config, "--k-pivot", "0.02",
                "--no-force", "--no-plot", "--workers", "1",
                "--accretion", "Chisholm1", "--zeta-c", "0.05")

    record = _only_spectrum(pbh_pipeline_sandbox)
    assert record["metadata"]["k_pivot_Mpc"] == pytest.approx(0.02)
    _assert_printed_matches_saved_spectrum(record, proc, 0.02)
