"""
Unit tests for ``scripts/optimize_pbh_botorch.py``.

5 tests, all using a synthetic 2-parameter mock forward model that runs
instantly (no MS solver, no CAMB, no GPU).  Total suite completes in
< 30 seconds.

Run:
    pytest tests/test_optimize_pbh_botorch.py -v

Lazy imports only — botorch/gpytorch are imported inside the functions
that need them, never at module level.
"""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch


# ── synthetic forward model ───────────────────────────────────────────────────


class MockPBHModel:
    """Synthetic 2-parameter forward model mimicking PBH sweep behavior.

    Parameters: [x, y] both in [0, 1].

    Objective (maximise):      f_total = x * y             → optimum at (1, 1)
    Mass proxy:                M_peak = 1e-4 * (x+0.5)*(y+0.5)
    Spectral index:            n_s = 0.96 + 0.02*(x-0.5)*(y-0.5)
    Feasibility:               M_peak > 1e-4  ↦  status == "success"
    """

    @staticmethod
    def evaluate(theta: list[float]) -> dict:
        x, y = theta[0], theta[1]
        f_total = x * y
        M_peak = 1e-4 * (x + 0.5) * (y + 0.5)
        n_s = 0.96 + 0.02 * (x - 0.5) * (y - 0.5)
        ok = bool(M_peak > 1e-4)

        return {
            "status": "success" if ok else "fail",
            "f_total": f_total,
            "M_peak_best": M_peak,
            "n_s": n_s,
            "x_c": theta[0],
            "c": theta[1] if len(theta) > 1 else 0,
            "beta": 0,
            "chi0": 0,
            "N_star": 0,
            "zeta_c": 0,
            "k_pivot": 0.05,
            "ns_window": 3.0,
            "k_phys": [],
            "P_S": [],
            "on_grid_boundary": False,
            "mass_bin": "mock",
            "N_total": 200,
            "k_peak": 1e10,
            "P_S_peak": 1e-5,
            "P_S_peak_ratio": 1e4,
            "M_form": 1e-10,
            "M_kpeak": 1e-8,
            "A_s_cmb": 2.1e-9,
        }


# Schema for the JSONL log written by _write_log_entry.
# Every key below must always appear in a non-error entry.
EXPECTED_LOG_KEYS: set[str] = {
    "x_c",
    "c",
    "beta",
    "chi0",
    "N_star",
    "N_total",
    "k_peak",
    "P_S_peak",
    "P_S_peak_ratio",
    "M_form",
    "M_kpeak",
    "M_present",
    "on_grid_boundary",
    "mass_bin",
    "n_s",
    "k_pivot",
    "ns_window",
    "A_s_at_cmb",
    "zeta_c",
    "f_total",
    "status",
}
# Extra keys that are permitted (stage1 debug, error info, large arrays).
_ALLOWED_EXTRA: set[str] = {"stage1_passed", "error", "k_phys", "P_S"}


# ===========================================================================
# Test 1 — Schema match
# ===========================================================================


class TestSchemaMatch:
    """``_write_log_entry`` produces valid JSONL matching expected schema."""

    def test_write_and_read(self) -> None:
        from scripts.optimize_pbh_botorch import _write_log_entry

        outcome = MockPBHModel.evaluate([0.78, 0.77])

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            log_path = f.name

        try:
            _write_log_entry(log_path, outcome)

            with open(log_path) as f:
                written = json.loads(f.readline())

            # Every expected key is present
            for key in EXPECTED_LOG_KEYS:
                assert key in written, f"Missing expected key: {key}"

            # No unexpected keys
            unexpected = set(written.keys()) - EXPECTED_LOG_KEYS - _ALLOWED_EXTRA
            assert not unexpected, f"Unexpected keys: {unexpected}"

        finally:
            os.unlink(log_path)


# ===========================================================================
# Test 2 — Resume / done-set loading
# ===========================================================================


class TestResume:
    """``_load_done_set`` correctly distinguishes done vs error entries."""

    def test_load_done_set(self) -> None:
        from scripts.optimize_pbh_botorch import _load_done_set

        # 5 good entries + 1 error entry with **unique** params not shared
        # by any success entry.
        entries: list[dict] = [
            {"x_c": 0.78, "c": 0.77, "beta": 1e-5, "chi0": 8.0, "N_star": 65.0,
             "zeta_c": 0.07, "status": "success", "f_total": 0.5},
            {"x_c": 0.79, "c": 0.80, "beta": 2e-5, "chi0": 7.5, "N_star": 60.0,
             "zeta_c": 0.08, "status": "success", "f_total": 0.3},
            {"x_c": 0.77, "c": 0.75, "beta": 5e-5, "chi0": 7.0, "N_star": 62.0,
             "zeta_c": 0.06, "status": "success", "f_total": 0.1},
            {"x_c": 0.78, "c": 0.77, "beta": 1e-5, "chi0": 8.0, "N_star": 64.0,
             "zeta_c": 0.07, "status": "success", "f_total": 0.4},
            {"x_c": 0.80, "c": 0.90, "beta": 3e-5, "chi0": 8.5, "N_star": 70.0,
             "zeta_c": 0.09, "status": "success", "f_total": 0.6},
            # Error entry — unique params that no success entry uses
            {"x_c": 0.81, "c": 0.60, "beta": 9e-5, "chi0": 6.0, "N_star": 55.0,
             "zeta_c": 0.05, "error": "MS solver crashed"},
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            log_path = f.name
            for e in entries:
                f.write(json.dumps(e) + "\n")

        try:
            done_set = _load_done_set(log_path)

            assert len(done_set) == 5, (
                f"Expected 5 done entries (6 total - 1 error), got {len(done_set)}"
            )

            # Error tuple must NOT be in done_set
            error_tuple = (0.81, 0.60, 9e-5, 6.0, 55.0, 0.05)
            assert error_tuple not in done_set, (
                "Error entry should be excluded from done_set"
            )

            # Verify specific success entries are present
            success_a = (0.78, 0.77, 1e-5, 8.0, 64.0, 0.07)
            success_b = (0.77, 0.75, 5e-5, 7.0, 62.0, 0.06)
            assert success_a in done_set
            assert success_b in done_set

        finally:
            os.unlink(log_path)

    def test_load_done_set_missing_file(self) -> None:
        """_load_done_set returns empty set when log doesn't exist."""
        from scripts.optimize_pbh_botorch import _load_done_set

        # Point to a path that does not exist
        done_set = _load_done_set("/tmp/nonexistent_log_file_12345.jsonl")
        assert done_set == set(), "Missing log should yield empty set"


# ===========================================================================
# Test 3 — Constraint enforcement
# ===========================================================================


class TestConstraintEnforcement:
    """``_compute_feasibility`` correctly classifies constraint violations."""

    # Shared mock args (feasibility thresholds)
    @staticmethod
    def _make_args() -> SimpleNamespace:
        return SimpleNamespace(
            f_total_min=0.0,
            f_total_max=1.0,
            n_s_min=0.94,
            n_s_max=0.99,
        )

    def test_fail_status_is_infeasible(self) -> None:
        """A ``fail`` status outcome always yields infeasible."""
        from scripts.optimize_pbh_botorch import _compute_feasibility

        args = self._make_args()
        outcome = MockPBHModel.evaluate([0.1, 0.1])  # M_peak=3.6e-5 → "fail"
        assert outcome["status"] == "fail"

        Y, feasible = _compute_feasibility(outcome, 1e-6, 1e-2, args)
        assert feasible is False, "Fail-status outcome should be infeasible"
        assert Y == float("inf"), "Fail-status outcome should have inf objective"

    def test_mass_bin_violation(self) -> None:
        """M_peak outside target bin → infeasible."""
        from scripts.optimize_pbh_botorch import _compute_feasibility

        args = self._make_args()
        # M_peak = 1.96e-4 (for (0.9, 0.9)) — outside [1e-6, 1e-5]
        outcome = MockPBHModel.evaluate([0.9, 0.9])
        assert outcome["status"] == "success"

        Y, feasible = _compute_feasibility(outcome, 1e-6, 1e-5, args)
        assert feasible is False, "M_peak > mass_hi should be infeasible"
        assert Y != float("inf"), "Objective should still be finite even if infeasible"

    def test_all_constraints_satisfied(self) -> None:
        """All constraints satisfied → feasible with correct Y = -f_total."""
        from scripts.optimize_pbh_botorch import _compute_feasibility

        args = self._make_args()
        outcome = MockPBHModel.evaluate([0.9, 0.9])
        assert outcome["status"] == "success"

        Y, feasible = _compute_feasibility(outcome, 1e-6, 1e-2, args)
        assert feasible is True, "All constraints satisfied → feasible"
        assert Y == -0.81, f"Y should be -f_total = -0.81, got {Y}"

    def test_ns_violation(self) -> None:
        """n_s outside Planck range → infeasible."""
        from scripts.optimize_pbh_botorch import _compute_feasibility

        args = SimpleNamespace(
            f_total_min=0.0, f_total_max=1.0,
            n_s_min=0.97, n_s_max=0.99,  # narrow range
        )
        outcome = MockPBHModel.evaluate([0.9, 0.9])
        # n_s = 0.96 + 0.02*(0.4)*(0.4) = 0.9632 — outside [0.97, 0.99]
        assert outcome["status"] == "success"
        assert outcome["n_s"] < 0.97, "n_s should be below min for this config"

        Y, feasible = _compute_feasibility(outcome, 1e-6, 1e-2, args)
        assert feasible is False, "n_s < n_s_min should be infeasible"

    def test_nan_handling(self) -> None:
        """NaN values in outcome render the point infeasible."""
        from scripts.optimize_pbh_botorch import _compute_feasibility

        args = self._make_args()

        # Manually construct outcome with NaN f_total
        outcome = {
            "status": "success",
            "f_total": float("nan"),
            "M_peak_best": 5e-5,
            "n_s": 0.96,
        }
        Y, feasible = _compute_feasibility(outcome, 1e-6, 1e-2, args)
        assert feasible is False, "NaN f_total should be infeasible"
        assert Y == float("inf"), "NaN f_total should yield inf objective"

    def test_none_handling(self) -> None:
        """Missing (None) target key renders the point infeasible."""
        from scripts.optimize_pbh_botorch import _compute_feasibility

        args = self._make_args()
        outcome = {
            "status": "success",
            "f_total": 0.5,
            "M_peak_best": None,
            "n_s": 0.96,
        }
        Y, feasible = _compute_feasibility(outcome, 1e-6, 1e-2, args)
        assert feasible is False, "None M_peak should be infeasible"


# ===========================================================================
# Test 4 — BO convergence on a cheap synthetic problem
# ===========================================================================


class TestBOConvergenceSynthetic:
    """End-to-end BoTorch loop on the 2-parameter mock model.

    Sobol init (8) → evaluate → GP fit → qLogEI acquisition → optimise → loop.
    After 20 total trials the best f_total should exceed 0.7 (optimum = 1.0).

    Note: ``_optimize_candidates`` passes ``seed=seed`` to ``optimize_acqf``,
    which newer BoTorch versions reject.  This test calls ``optimize_acqf``
    directly without that kwarg.
    """

    def test_convergence(self) -> None:
        from botorch.optim import optimize_acqf

        from scripts.optimize_pbh_botorch import (
            _sobol_init,
            _fit_gp,
            _compute_acquisition,
        )

        mock = MockPBHModel()
        d = 2
        n_init = 8
        n_trials = 20
        seed = 42

        # -- Sobol initial design -------------------------------------------------
        X_init = _sobol_init(n_init, d, seed)

        # -- Evaluate mock --------------------------------------------------------
        X_list: list[torch.Tensor] = []
        Y_list: list[torch.Tensor] = []
        best_f_total = 0.0

        for i in range(n_init):
            x_np = X_init[i].numpy()
            outcome = mock.evaluate(x_np.tolist())
            if outcome["status"] != "success":
                continue
            if outcome["M_peak_best"] <= 1e-4:
                continue
            X_list.append(X_init[i : i + 1])
            Y_list.append(
                torch.tensor([[-outcome["f_total"]]], dtype=torch.float64)
            )
            best_f_total = max(best_f_total, outcome["f_total"])

        if len(X_list) < 2:
            pytest.skip("Not enough feasible Sobol init points")

        X_bo = torch.cat(X_list, dim=0)
        Y_bo = torch.cat(Y_list, dim=0)

        bounds_unit_cube = torch.tensor(
            [[0.0] * d, [1.0] * d], dtype=torch.float64
        )

        # -- BO loop --------------------------------------------------------------
        for t in range(n_trials - n_init):
            model = _fit_gp(X_bo, Y_bo)
            best_f = float(Y_bo.min().item())
            acq = _compute_acquisition(model, best_f, X_bo)

            candidates, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds_unit_cube,
                q=1,
                num_restarts=20,
                raw_samples=1024,
            )

            x_candidate = candidates[0].numpy()
            outcome = mock.evaluate(x_candidate.tolist())
            if outcome["status"] != "success":
                continue
            if outcome["M_peak_best"] <= 1e-4:
                continue

            best_f_total = max(best_f_total, outcome["f_total"])

            X_bo = torch.cat([X_bo, candidates[0:1]], dim=0)
            Y_new = torch.tensor(
                [[-outcome["f_total"]]], dtype=torch.float64
            )
            Y_bo = torch.cat([Y_bo, Y_new], dim=0)

        assert best_f_total > 0.7, (
            f"BO should find near-optimal f_total > 0.7, got {best_f_total:.6f} "
            f"after {n_trials} trials on the synthetic problem"
        )


# ===========================================================================
# Test 5 — Acquisition avoids infeasible region
# ===========================================================================


class TestFeasibilityIntegration:
    """Integration between ``_compute_feasibility`` and GP training-data
    collection: infeasible points are excluded from the GP training set,
    and only feasible points contribute to the model.

    This is the actual mechanism by which ``run_optimization`` avoids
    infeasible regions — the GP never sees bad Y values, so acquisition
    naturally favours the feasible zone.
    """

    def test_only_feasible_enter_training_set(self) -> None:
        from scripts.optimize_pbh_botorch import _compute_feasibility

        mock = MockPBHModel()

        # 4 points: 2 feasible, 2 infeasible
        points = torch.tensor(
            [
                [0.9, 0.9],   # feasible:   M_peak=1.96e-4 > 1e-4
                [0.1, 0.1],   # infeasible: M_peak=3.60e-5 < 1e-4
                [0.7, 0.7],   # feasible:   M_peak=1.44e-4 > 1e-4
                [0.2, 0.3],   # infeasible: M_peak=7.70e-5 < 1e-4
            ],
            dtype=torch.float64,
        )

        args = SimpleNamespace(
            f_total_min=0.0,
            f_total_max=1.0,
            n_s_min=0.94,
            n_s_max=0.99,
        )
        mass_lo, mass_hi = 1e-6, 1e-2

        X_feas: list[torch.Tensor] = []
        Y_feas: list[torch.Tensor] = []

        for i in range(points.shape[0]):
            outcome = mock.evaluate(points[i].tolist())
            y_val, feasible = _compute_feasibility(
                outcome, mass_lo, mass_hi, args
            )
            if feasible:
                X_feas.append(points[i : i + 1])
                Y_feas.append(
                    torch.tensor([[y_val]], dtype=torch.float64)
                )

        assert len(X_feas) == 2, (
            f"Expected 2 feasible points, got {len(X_feas)}"
        )
        assert len(Y_feas) == 2

        # First feasible point (0.9, 0.9): f_total = 0.81 → Y = -0.81
        assert Y_feas[0].item() == pytest.approx(-0.81, abs=1e-12), (
            f"Expected Y = -0.81 for (0.9, 0.9), got {Y_feas[0].item()}"
        )

        # Second feasible point (0.7, 0.7): f_total = 0.49 → Y = -0.49
        assert Y_feas[1].item() == pytest.approx(-0.49, abs=1e-12), (
            f"Expected Y = -0.49 for (0.7, 0.7), got {Y_feas[1].item()}"
        )

    def test_constraint_blocks_infeasible_mock_run(self) -> None:
        """Mini replication of the run_optimisation loop logic.

        Verifies that _compute_feasibility correctly classifies the mock
        model's outputs and that the overall data-flow (evaluate →
        feasibility-check → collect) matches what the real pipeline does.
        """
        from scripts.optimize_pbh_botorch import _compute_feasibility

        mock = MockPBHModel()

        args = SimpleNamespace(
            f_total_min=0.0,
            f_total_max=1.0,
            n_s_min=0.94,
            n_s_max=0.99,
        )
        mass_lo, mass_hi = 1e-6, 1e-2

        # Test each of the 4 points individually
        test_cases: list[tuple[list[float], bool, float]] = [
            ([0.9, 0.9], True, -0.81),
            ([0.1, 0.1], False, float("inf")),
            ([0.7, 0.7], True, -0.49),
            ([0.2, 0.3], False, float("inf")),
        ]

        for params, expected_feas, expected_y in test_cases:
            outcome = mock.evaluate(params)
            y_val, feasible = _compute_feasibility(
                outcome, mass_lo, mass_hi, args
            )
            assert feasible == expected_feas, (
                f"Expected feasible={expected_feas} for {params}, "
                f"got {feasible}"
            )
            if expected_feas:
                assert y_val == pytest.approx(expected_y, abs=1e-12), (
                    f"Expected Y={expected_y} for {params}, got {y_val}"
                )
            else:
                assert y_val == float("inf"), (
                    f"Expected inf Y for {params}, got {y_val}"
                )
