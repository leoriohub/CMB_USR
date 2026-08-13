"""Parallelism tests for the config-scan scripts.

Cover the fix that made ``--workers`` real in ``fine_tune_fraction.py`` and
``camb_scan.py`` (issue #30):

- fast: unit tests for the new module-level pool workers (``_scan_point``,
  ``_scan_config``) with the heavy solver/CAMB stubbed out, plus a
  subprocess-isolated check that the modules set the thread caps at import.
- slow: end-to-end identity check that ``--workers 2`` produces the same scan
  results as ``--workers 1`` (real pipeline, not run in the pre-commit gate).
"""
import json
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

# Import under test — pulls the module's module-top thread caps, which is
# exactly what the subprocess env test below asserts on (isolated).
pytest.importorskip("numpy")
import scripts.fine_tune_fraction as ft  # noqa: E402
import scripts.camb_scan as cs  # noqa: E402


# ---------------------------------------------------------------------------
# _scan_point (fine_tune_fraction)
# ---------------------------------------------------------------------------

def _stub_evaluate(results):
    """Return a stub evaluate_point + the list of N* it was called with.

    ``results`` is a list of (status, d2). The stub returns the i-th result
    for the i-th call.
    """
    calls = []

    def stub(x0, y0, ns, k_grid):
        calls.append((float(x0), float(y0), float(ns)))
        status, d2 = results[len(calls) - 1]
        if status == "error":
            return {"status": "error", "message": "boom"}
        return {"status": "success", "N_total": 60.0, "d2": float(d2),
                "d2_lcdm": 1029.0}

    return stub, calls


K_GRID = np.logspace(-4, 0, 10)


@pytest.mark.fast
def test_scan_point_early_exit_on_suppressed(monkeypatch):
    """First N* suppressed -> exactly one record, suppressed=True."""
    stub, _ = _stub_evaluate([("success", 100.0)])  # 100 <= 823 -> suppressed
    monkeypatch.setattr(ft, "evaluate_point", stub)
    records, suppressed = ft._scan_point(
        5.70, -0.170, K_GRID, n_total=65.0, threshold=823.0,
        d2_lcdm=1029.0, t_over_v=0.5, completed=set())
    assert suppressed is True
    assert len(records) == 1
    assert records[0]["N_star"] == 55.0
    assert records[0]["suppressed"] is True


@pytest.mark.fast
def test_scan_point_ns_guard_breaks_on_n_total(monkeypatch):
    """ns >= n_total stops the sweep (pipeline errors on N_star >= N_total)."""
    stub, calls = _stub_evaluate([("success", 900.0)])  # never suppressed
    monkeypatch.setattr(ft, "evaluate_point", stub)
    records, suppressed = ft._scan_point(
        5.70, -0.170, K_GRID, n_total=56.0, threshold=823.0,
        d2_lcdm=1029.0, t_over_v=1.0, completed=set())
    assert suppressed is False
    assert len(records) == 1          # ns=55 only; 56 halted
    assert [c[2] for c in calls] == [55.0]


@pytest.mark.fast
def test_scan_point_scans_full_sweep_when_never_suppressed(monkeypatch):
    """No suppression -> every valid N* in [55, min(70, n_total-1)] is run."""
    okay = [("success", 900.0) for _ in ft.N_STAR_SWEEP]
    stub, calls = _stub_evaluate(okay)
    monkeypatch.setattr(ft, "evaluate_point", stub)
    records, suppressed = ft._scan_point(
        5.70, -0.170, K_GRID, n_total=99.0, threshold=823.0,
        d2_lcdm=1029.0, t_over_v=1.0, completed=set())
    assert suppressed is False
    assert len(records) == len(ft.N_STAR_SWEEP)          # 55..70 inclusive
    assert all(not r["suppressed"] for r in records)


@pytest.mark.fast
def test_scan_point_skips_completed_resume_keys(monkeypatch):
    """Already-completed (resume) triples are not re-evaluated."""
    stub, calls = _stub_evaluate([("success", 900.0)])
    monkeypatch.setattr(ft, "evaluate_point", stub)
    records, suppressed = ft._scan_point(
        5.70, -0.170, K_GRID, n_total=57.0, threshold=823.0,
        d2_lcdm=1029.0, t_over_v=1.0,
        completed={(5.70, -0.170, 55.0)})   # ns=55 skipped
    assert suppressed is False
    assert [c[2] for c in calls] == [56.0]  # 55 skipped, 56 evaluated, 57 halted
    assert len(records) == 1


@pytest.mark.fast
def test_scan_point_handles_error_records(monkeypatch):
    """A failing evaluate_point yields an error record, not suppressed."""
    stub, _ = _stub_evaluate([("error", None), ("error", None)])
    monkeypatch.setattr(ft, "evaluate_point", stub)
    records, suppressed = ft._scan_point(
        5.70, -0.170, K_GRID, n_total=57.0, threshold=823.0,
        d2_lcdm=1029.0, t_over_v=1.0, completed=set())
    assert suppressed is False
    assert [r["N_star"] for r in records] == [55.0, 56.0]  # 57 halted
    assert all(r["status"] == "error" for r in records)
    assert all(r["suppressed"] is False for r in records)
    assert all(r["d2"] is None for r in records)


# ---------------------------------------------------------------------------
# _scan_config (camb_scan)
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_scan_config_rebuilds_args_and_calls_evaluate(monkeypatch):
    """_scan_config unpacks the dict into a Namespace + passes model=None."""
    received = {}

    def fake_evaluate(phi0, y0, N_star, args, **kwargs):
        received.update({
            "phi0": phi0, "y0": y0, "N_star": N_star,
            "lam": args.lam, "xi": args.xi, "model": kwargs.get("model"),
            "executor": kwargs.get("executor"),
            "k_phys_grid": kwargs.get("k_phys_grid"),
        })
        return {"status": "ok", "marker": 1}

    monkeypatch.setattr(cs, "evaluate_config", fake_evaluate)
    k = np.array([1.0, 2.0])
    out = cs._scan_config({
        "phi0": 5.75, "y0": -0.17, "N_star": 55.0,
        "args_dict": {"lam": 0.13, "xi": 15000.0},
        "k_phys_grid": k,
    })
    assert out["marker"] == 1
    assert received["phi0"] == 5.75 and received["y0"] == -0.17
    assert received["lam"] == 0.13 and received["xi"] == 15000.0
    assert received["model"] is None              # reconstructed, not pickled
    assert received["executor"] is None           # no nested pool use
    assert received["k_phys_grid"] is k


@pytest.mark.fast
def test_phase2_ok_counter(monkeypatch, tmp_path, capsys):
    """Progress 'ok=' must equal the number of ok evals (counter, not log re-read)."""
    state = {"calls": 0}
    def fake_eval(phi0, y0, N_star, args, **kw):
        state["calls"] += 1
        return ({"status": "ok", "chi2": 20.0} if state["calls"] % 2
                else {"status": "pipe_error", "error": "stub"})
    monkeypatch.setattr(cs, "evaluate_config", fake_eval)
    monkeypatch.setattr(cs, "get_path", lambda cat, name: str(tmp_path / name))
    args = argparse.Namespace(
        quick=True, k_min=1e-5, k_max=1.0, ell_max=30, num_k=50,
        phi0_fine_window=0.05, y0_fine_window=0.02, nstar_fine_window=1.0,
        n_phi0_fine=2, n_y0_fine=1, n_nstar_fine=2,
        xi=15000.0, lam=0.13,
    )
    cs.run_phase2(args, set(), [(6.4, -0.5, 55)])
    assert "ok=2" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Module thread-cap contract (subprocess-isolated — no global mutation)
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_fine_tune_sets_thread_caps_at_import():
    """Importing the module defaults OMP/NUMBA threads to 1 (unless preset)."""
    code = ("import os; "
            "import scripts.fine_tune_fraction, scripts.camb_scan; "
            "print(os.environ.get('OMP_NUM_THREADS'), "
            "os.environ.get('NUMBA_NUM_THREADS'))")
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS")}
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "1 1"


# ---------------------------------------------------------------------------
# End-to-end: --workers N must not change results (real pipeline, slow)
# ---------------------------------------------------------------------------

def _run_scan(workers):
    """Run fine_tune scan on a 3x3 grid in a subprocess; return (records, log)."""
    cmd = [
        sys.executable, "scripts/fine_tune_fraction.py", "scan",
        "--n-phi0", "3", "--n-y0", "3",
        "--x0-min", "5.5", "--x0-max", "6.0",
        "--y0-min", "-0.25", "--y0-max", "-0.10",
        "--workers", str(workers),
    ]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                         timeout=1200)
    assert out.returncode == 0, out.stderr + out.stdout
    log = None
    for line in out.stdout.splitlines():
        if line.startswith("Log:"):
            log = line.split("Log:", 1)[1].strip()
    assert log is not None and Path(log).exists(), out.stdout
    records = []
    with open(log) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("_type") != "header":
                records.append(rec)
    return records, log

# The scan must produce byte-identical records regardless of worker count.
@pytest.mark.slow
def test_scan_workers_2_matches_workers_1():
    """Parallel (workers=2) scan must produce identical records to serial."""
    rec1, log1 = _run_scan(1)
    rec2, log2 = _run_scan(2)
    try:
        # Full-record identity (dict == dict), order-independent.
        key = lambda r: (r["x0"], r["y0"], r.get("N_star", -1))
        data1 = sorted(rec1, key=key)
        data2 = sorted(rec2, key=key)
        # Must have actually evaluated >=1 point for this to be meaningful,
        # and the counts (incl. not_viable/not_relevant) must match.
        assert any(r.get("status") == "success" for r in data1), (
            "grid produced no evaluated points; widen the test window")
        assert len(data1) == len(data2), f"{len(data1)} vs {len(data2)} records"
        for a, b in zip(data1, data2):
            assert a == b, f"record mismatch: {a} vs {b}"
    finally:
        for log in (log1, log2):
            if log and Path(log).exists():
                Path(log).unlink()
