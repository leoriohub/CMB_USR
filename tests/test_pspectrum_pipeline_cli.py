"""Regression tests for the pspectrum_pipeline CLI entry point.

`main()` is the documented way to run a config:

    python pspectrum_pipeline.py --config <cfg.json>

It resolved the pivot with

    k_pivot_phys = _piv_cfg if _piv_cfg is not None else k_pivot_phys

which assigns to the module-level `k_pivot_phys` name. That makes the name
local to the whole of `main()`, so the fallback read on the right-hand side
raises UnboundLocalError before any physics runs. Every CLI invocation failed.
The library entry point `run_pspectrum_pipeline` was unaffected, which is why
the existing tests (which call it directly) did not catch this.

These tests drive the real CLI as a subprocess, with a deliberately tiny
background grid so the run is quick.
"""

import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.fast

_TINY = {
    "model": "HiggsModel",
    "model_params": {"lam": 0.13, "xi": 15000.0,
                     "T_max": 200.0, "bg_steps": 2000},
    "ics": {"x0": 5.7, "y0": -0.17},
    "pipeline": {"N_star": 40.0, "k_pivot_phys": 0.002,
                 "ns_window": 4.0, "backend": "numba"},
}


def _run(tmp_path, cfg, extra=()):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    proc = subprocess.run(
        [sys.executable, "pspectrum_pipeline.py",
         "--config", str(cfg_path), "--num-k", "3", *extra],
        capture_output=True, text=True, timeout=600,
    )
    return proc


def test_cli_runs_to_completion(tmp_path):
    proc = _run(tmp_path, _TINY)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "UnboundLocalError" not in proc.stderr


def test_cli_reports_resolved_pivot_and_source(tmp_path):
    proc = _run(tmp_path, _TINY)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "k_pivot_phys=0.002 Mpc^-1 (from config)" in proc.stdout


def test_cli_pivot_falls_back_to_package_default(tmp_path):
    cfg = json.loads(json.dumps(_TINY))
    cfg["pipeline"].pop("k_pivot_phys")
    proc = _run(tmp_path, cfg)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "(from package default)" in proc.stdout


def test_cli_pivot_reports_cli_override_source(tmp_path):
    proc = _run(tmp_path, _TINY, extra=("--k-pivot-phys", "0.004"))
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "k_pivot_phys=0.004 Mpc^-1 (from --k-pivot-phys)" in proc.stdout
