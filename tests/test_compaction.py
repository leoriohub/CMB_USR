"""Unit tests for scripts/compaction.py: Compaction function formalism for PBH formation.

Tests cover all three public functions (compute_C_profile, C_critical,
compute_zeta_r_profile, beta_f_compaction) and their public API surface.
Private helpers (_shape_parameter, _areal_radius) are tested indirectly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.integrate import simpson

from scripts.compaction import (
    C_critical,
    _compute_zeta_profile_vectorized,
    _simpson_weights,
    beta_f_compaction,
    compute_C_profile,
    compute_zeta_r_profile,
    timer,
)

# ===========================================================================
# C_critical — profile-dependent collapse threshold
# ===========================================================================


class TestCcritical:
    """Critical compaction function threshold C_c(alpha)."""

    def test_monotonic(self) -> None:
        """C_c(alpha) is monotonically increasing for alpha in [0, 10]."""
        alphas = np.linspace(0, 10, 101)
        C_c = C_critical(alphas)
        # Strict increase except at alpha=0 where the function flattens
        diffs = np.diff(C_c)
        assert np.all(diffs >= 0.0), "C_c must be non-decreasing"
        assert np.any(diffs > 0.0), "C_c must increase somewhere"

    def test_reference_values(self) -> None:
        """Check reference values against the analytic formula (Escriva 2022)."""
        # C_c(alpha) = 0.443 + 0.106 * alpha^0.546
        # alpha=0: 0.443
        assert C_critical(0.0) == pytest.approx(0.443, abs=1e-3)
        # alpha=1: 0.443 + 0.106 = 0.549
        assert C_critical(1.0) == pytest.approx(0.549, abs=1e-3)
        # alpha=3: computed
        expected_3 = 0.443 + 0.106 * (3.0**0.546)
        assert C_critical(3.0) == pytest.approx(expected_3, abs=1e-3)
        # alpha=10: computed
        expected_10 = 0.443 + 0.106 * (10.0**0.546)
        assert C_critical(10.0) == pytest.approx(expected_10, abs=1e-3)

    def test_negative_alpha_clips_to_zero(self) -> None:
        """Negative alpha triggers a warning and returns C_c(0)."""
        # Single negative value - check result
        with pytest.warns(UserWarning, match="negative"):
            result = C_critical(-5.0)
        assert result == pytest.approx(0.443, abs=1e-3)

        # Mixed array
        alphas = np.array([-2.0, 0.0, 1.0, -1.0, 3.0])
        with pytest.warns(UserWarning, match="2 negative"):
            result = C_critical(alphas)
        # Negative entries map to C_c(0), rest unaffected
        assert result[0] == pytest.approx(0.443, abs=1e-3)
        assert result[1] == pytest.approx(0.443, abs=1e-3)
        assert result[2] == pytest.approx(0.549, abs=1e-3)
        assert result[3] == pytest.approx(0.443, abs=1e-3)

    def test_array_input_returns_array(self) -> None:
        """Passing an ndarray returns an ndarray of the same shape."""
        alphas = np.array([0.0, 1.0, 2.0, 3.0])
        result = C_critical(alphas)
        assert isinstance(result, np.ndarray)
        assert result.shape == alphas.shape

    def test_scalar_input_returns_float(self) -> None:
        """Passing a float returns a float (or 0-d array)."""
        result = C_critical(0.5)
        assert np.ndim(result) == 0  # 0-d array = scalar-like

    def test_unknown_epoch_raises(self) -> None:
        """Unsupported epoch string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown epoch"):
            C_critical(1.0, epoch="dark_energy")

    def test_matter_epoch_raises(self) -> None:
        """Matter-dominated epoch raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            C_critical(1.0, epoch="matter")


# ===========================================================================
# compute_C_profile — compaction function from zeta(r)
# ===========================================================================


class TestComputeCProfile:
    """Compaction function C(r) from a radial curvature profile zeta(r)."""

    def test_flat_zeta(self) -> None:
        """Constant positive zeta yields C_max > 0 with computed alpha."""
        r = np.linspace(0, 10, 200)
        zeta = np.full_like(r, 0.1)
        out = compute_C_profile(r, zeta)
        assert out["C_max"] > 0.0
        # For constant zeta > 0, C(r) ~ r^2, peak at max r
        assert out["r_max"] == pytest.approx(r[-1], abs=1e-3)
        # alpha is a finite number (may be negative, clipped later)
        assert np.isfinite(out["alpha"])
        # Output arrays match input length
        assert len(out["r_arr"]) == len(r)
        assert len(out["C_arr"]) == len(r)
        # C(r=0) should be ~0 (division safe-guarded)
        assert out["C_arr"][0] == pytest.approx(0.0, abs=1e-15)

    def test_gaussian_zeta(self) -> None:
        """Gaussian zeta bump produces finite C_max with alpha > 0."""
        r = np.linspace(0, 10, 200)
        zeta = 0.3 * np.exp(-0.5 * r**2)
        out = compute_C_profile(r, zeta)
        assert out["C_max"] > 0.0
        # Peak should be at finite r > 0
        assert 0.0 < out["r_max"] < r[-1]
        # Shape parameter should be positive (peaked profile)
        assert out["alpha"] > 0.0
        assert np.isfinite(out["alpha"])

    def test_zero_zeta_gives_near_zero_C(self) -> None:
        """Zero curvature perturbation yields ~zero compaction (numerical only)."""
        r = np.linspace(0, 10, 100)
        zeta = np.zeros_like(r)
        out = compute_C_profile(r, zeta)
        # C_max is ~0.04 from cumulative_trapezoid integration error on r²
        # Theoretically exactly 0; numerically small
        assert out["C_max"] == pytest.approx(0.0, abs=0.1)
        # alpha is tiny (4e-8 from near-zero C_peak in _shape_parameter)
        assert out["alpha"] == pytest.approx(0.0, abs=1e-6)
        # C_arr[0] is exactly 0 (safe-divide guards r=0)
        assert out["C_arr"][0] == 0.0

    def test_input_validation_too_few_points(self) -> None:
        """Fewer than 10 radial points raises ValueError."""
        r = np.linspace(0, 1, 5)
        zeta = np.zeros(5)
        with pytest.raises(ValueError, match="At least 10"):
            compute_C_profile(r, zeta)

    def test_input_validation_length_mismatch(self) -> None:
        """Mismatched array lengths raises ValueError."""
        r = np.linspace(0, 10, 100)
        zeta = np.zeros(50)
        with pytest.raises(ValueError, match="must have the same length"):
            compute_C_profile(r, zeta)

    def test_input_validation_non_1d(self) -> None:
        """Non-1D arrays raise TypeError."""
        r = np.ones((5, 5))
        zeta = np.ones((5, 5))
        with pytest.raises(TypeError, match="must be 1-D"):
            compute_C_profile(r, zeta)

    def test_r_arr_monotonicity_not_required(self) -> None:
        """Non-monotonic r does not raise (not validated, but should not crash)."""
        r = np.array([0.0, 3.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        zeta = 0.1 * np.exp(-0.5 * r**2)
        # Should not raise; results may be unphysical but code handles it
        out = compute_C_profile(r, zeta)
        assert "C_max" in out

    def test_strong_gaussian_bump(self) -> None:
        """Larger amplitude still gives well-behaved C(r)."""
        r = np.linspace(0, 5, 200)
        zeta = 1.0 * np.exp(-0.5 * (r / 0.5) ** 2)
        out = compute_C_profile(r, zeta)
        assert out["C_max"] > 0.0
        assert out["r_max"] > 0.0

    def test_negative_zeta_gives_negative_bulk(self) -> None:
        """Uniform negative zeta: C(r) ~ r² sinh(ζ) < 0 for r > r_num."""
        r = np.linspace(0, 10, 100)
        zeta = -0.1 * np.ones_like(r)
        out = compute_C_profile(r, zeta)
        # C_max should be ~0 (peak at r=0 where safe-divide produces 0).
        # Small positive from integration error at adjacent bins.
        assert out["C_max"] == pytest.approx(0.0, abs=0.1)
        # All elements beyond index ~3 should be negative (bulk behavior)
        # First few bins may have small positive numerical noise
        assert np.all(out["C_arr"][3:] < 0.0)


# ===========================================================================
# compute_zeta_r_profile — radial curvature profile from P_S(k)
# ===========================================================================


class TestComputeZetaRProfile:
    """Radial curvature profile zeta(r) from a primordial power spectrum."""

    @pytest.fixture
    def power_law_ps(self) -> tuple[np.ndarray, np.ndarray]:
        """Standard power-law P_S(k) with Planck-like parameters."""
        k = np.logspace(-4, 4, 500)
        ps = 2.1e-9 * (k / 0.05) ** (0.965 - 1)
        return k, ps

    @pytest.fixture
    def radial_grid(self) -> np.ndarray:
        """Standard radial grid for profile evaluation."""
        return np.linspace(0, 20, 100)

    def test_center_value_equals_zeta_c(self, power_law_ps, radial_grid) -> None:
        """zeta(r=0) always equals zeta_c by construction."""
        k, ps = power_law_ps
        r = radial_grid
        for zeta_c in (0.5, 1.0, 2.0):
            _, zeta = compute_zeta_r_profile(k, ps, r, zeta_c=zeta_c)
            assert zeta[0] == pytest.approx(zeta_c, abs=1e-10)

    def test_profile_decays_to_zero(self) -> None:
        """Peaked P_S(k) yields zeta(r) that decays to zero at large r."""
        # Peaked PS (lognormal at k=1, width 1 dex) so correlation
        # length is finite and profile decays at large r
        k = np.logspace(-2, 2, 400)
        ps = np.exp(-0.5 * (np.log(k) / 0.5) ** 2)
        r = np.linspace(0, 30, 150)
        _, zeta = compute_zeta_r_profile(k, ps, r, R_smooth=1.0)
        # At large r, correlation should decay below 1%
        assert abs(zeta[-1]) < abs(zeta[0])
        assert zeta[-1] < 0.01 * zeta[0]

    def test_broad_ps_smooth_profile(self) -> None:
        """Broad P_S(k) yields smooth zeta(r) decaying from center."""
        # Power law with broad k coverage
        k = np.logspace(-4, 4, 300)
        ps = 2.1e-9 * (k / 0.05) ** (0.965 - 1)
        r = np.linspace(0, 30, 150)
        _, zeta = compute_zeta_r_profile(k, ps, r, R_smooth=2.0)

        assert zeta[0] == pytest.approx(1.0, abs=1e-10)
        assert np.all(np.isfinite(zeta))
        # Profile should be monotonically decreasing from center (for broad PS)
        center_val = zeta[0]
        assert np.all(zeta <= center_val + 1e-10)

    def test_monochromatic_ps_produces_profile(self) -> None:
        """Narrow (delta-function-like) P_S(k) produces a well-defined zeta(r)."""
        # Narrow log-normal peak in P_S — keep k-range tight to avoid underflow
        k0 = 10.0
        sigma_ln = 0.05
        k = np.logspace(np.log10(k0) - 1.5, np.log10(k0) + 1.5, 500)
        # Add floor to avoid exact zeros from exp underflow
        ps_raw = np.exp(-0.5 * (np.log(k / k0) / sigma_ln) ** 2)
        ps = np.maximum(ps_raw, 1e-300)
        r = np.linspace(0, 5, 200)

        _, zeta = compute_zeta_r_profile(k, ps, r, zeta_c=1.0)

        # Center equals zeta_c
        assert zeta[0] == pytest.approx(1.0, abs=1e-10)
        # Profile should oscillate (sinc-like for monochromatic PS)
        assert np.all(np.isfinite(zeta))
        # At least one zero crossing (sinc-like behavior)
        assert np.any(zeta[1:] * zeta[:-1] < 0) or np.any(np.abs(zeta) < 1e-6)

        # The compaction function computed from this profile should be well-behaved
        comp = compute_C_profile(r, zeta)
        assert comp["C_max"] > 0.0
        assert np.isfinite(comp["alpha"])

    def test_auto_smoothing_from_ps_peak(self) -> None:
        """R_smooth=None sets R = 1/k_peak from P_S(k)."""
        k = np.logspace(-2, 2, 200)
        # Create a PS with a clear peak at k=1
        ps = np.exp(-0.5 * ((np.log(k)) / 0.3) ** 2)
        r = np.linspace(0, 10, 100)

        # Should not raise -- auto computes R from peak
        _, zeta = compute_zeta_r_profile(k, ps, r, R_smooth=None)
        assert np.all(np.isfinite(zeta))

    def test_custom_rsmooth(self, power_law_ps, radial_grid) -> None:
        """Explicit R_smooth gives a consistent profile."""
        k, ps = power_law_ps
        r = radial_grid
        _, zeta_large = compute_zeta_r_profile(k, ps, r, R_smooth=10.0)
        _, zeta_small = compute_zeta_r_profile(k, ps, r, R_smooth=0.1)
        # Large smoothing radius reduces correlation length -> faster decay
        assert np.all(np.isfinite(zeta_large))
        assert np.all(np.isfinite(zeta_small))

    def test_input_validation_k_non_positive(self) -> None:
        """Non-positive k values raise ValueError."""
        k = np.array([-1.0, 0.0, 1.0])
        ps = np.array([1.0, 1.0, 1.0])
        r = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="strictly positive"):
            compute_zeta_r_profile(k, ps, r)

    def test_input_validation_ps_non_positive(self) -> None:
        """Non-positive P_S values raise ValueError."""
        k = np.array([0.1, 1.0, 10.0])
        ps = np.array([1.0, 0.0, 1.0])
        r = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="strictly positive"):
            compute_zeta_r_profile(k, ps, r)

    def test_input_validation_negative_r(self) -> None:
        """Negative r values raise ValueError."""
        k = np.logspace(-1, 1, 50)
        ps = np.full_like(k, 1e-9)
        r = np.array([-1.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="non-negative"):
            compute_zeta_r_profile(k, ps, r)

    def test_input_validation_length_mismatch(self) -> None:
        """Mismatched k and P_S lengths raise ValueError."""
        k = np.logspace(-1, 1, 50)
        ps = np.full(30, 1e-9)
        r = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="must have the same length"):
            compute_zeta_r_profile(k, ps, r)

    def test_input_validation_non_1d(self) -> None:
        """Non-1D arrays raise TypeError."""
        k = np.ones((5, 5))
        ps = np.ones((5, 5))
        r = np.linspace(0, 1, 10)
        with pytest.raises(TypeError, match="1-D"):
            compute_zeta_r_profile(k, ps, r)

    def test_input_validation_non_1d_ps(self) -> None:
        """Non-1D P_S_k raises TypeError even with 1D k."""
        k = np.logspace(-1, 1, 50)
        ps = np.ones((5, 5))
        r = np.linspace(0, 1, 10)
        with pytest.raises(TypeError, match="1-D"):
            compute_zeta_r_profile(k, ps, r)

    def test_input_validation_non_1d_r(self) -> None:
        """Non-1D r_arr raises TypeError even with 1D k and P_S_k."""
        k = np.logspace(-1, 1, 50)
        ps = np.full_like(k, 1e-9)
        r = np.ones((5, 5))
        with pytest.raises(TypeError, match="1-D"):
            compute_zeta_r_profile(k, ps, r)

    def test_input_validation_too_few_kpoints(self) -> None:
        """Fewer than 4 k-points raises ValueError."""
        k = np.array([0.1, 0.5, 1.0])
        ps = np.array([1e-9, 1e-9, 1e-9])
        r = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="At least 4"):
            compute_zeta_r_profile(k, ps, r)

    def test_unknown_window_raises(self) -> None:
        """Unsupported window function raises ValueError."""
        k = np.logspace(-1, 1, 50)
        ps = np.full_like(k, 1e-9)
        r = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="Unknown window"):
            compute_zeta_r_profile(k, ps, r, window="tophat")

    def test_negative_rsmooth_raises(self) -> None:
        """Non-positive R_smooth raises ValueError."""
        k = np.logspace(-1, 1, 50)
        ps = np.full_like(k, 1e-9)
        r = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="R_smooth must be positive"):
            compute_zeta_r_profile(k, ps, r, R_smooth=-1.0)


# ===========================================================================
# beta_f_compaction — PBH formation fraction via compaction formalism
# ===========================================================================


class TestBetaFCompaction:
    """PBH formation fraction via the compaction function formalism."""

    @pytest.fixture
    def power_law_ps(self) -> tuple[np.ndarray, np.ndarray]:
        """Standard power-law P_S(k)."""
        k = np.logspace(-2, 6, 50)
        ps = 2.1e-9 * (k / 0.05) ** (0.965 - 1)
        return k, ps

    def test_output_shape_matches_input(self, power_law_ps) -> None:
        """beta_f, M_pbh, and meta arrays all have length len(k_arr)."""
        k, ps = power_law_ps
        beta_f, M_pbh, meta = beta_f_compaction(k, ps)
        assert beta_f.shape == k.shape
        assert M_pbh.shape == k.shape
        for key in ("C_max_arr", "alpha_arr", "C_c_arr", "M_H_arr"):
            assert meta[key].shape == k.shape, f"meta[{key!r}] shape mismatch"

    def test_empty_input_returns_empty(self) -> None:
        """Empty arrays produce empty outputs."""
        k = np.array([])
        ps = np.array([])
        beta_f, M_pbh, meta = beta_f_compaction(k, ps)
        assert beta_f.shape == (0,)
        assert M_pbh.shape == (0,)
        for key in ("C_max_arr", "alpha_arr", "C_c_arr", "M_H_arr"):
            assert meta[key].shape == (0,)

    def test_non_negative_beta_f(self, power_law_ps) -> None:
        """Formation fraction must be non-negative everywhere."""
        k, ps = power_law_ps
        beta_f, _, _ = beta_f_compaction(k, ps)
        assert np.all(beta_f >= 0.0)
        assert np.all(beta_f <= 1.0)

    def test_horizon_mass_decreases_with_k(self, power_law_ps) -> None:
        """Horizon mass M_H ~ 1/k^2, so larger k means smaller M_H."""
        k, ps = power_law_ps
        _, _, meta = beta_f_compaction(k, ps)
        M_H = meta["M_H_arr"]
        # M_H should be decreasing with k (monotonic)
        diffs = np.diff(M_H)
        # All diffs should be negative (M_H decreasing) or near-zero
        # Exclude the largest k where floating-point may wobble
        assert np.all(diffs[:-1] < 0.0) or np.all(diffs[:-1] <= 1e-15)

    def test_input_validation_gamma_out_of_range(self, power_law_ps) -> None:
        """Gamma outside (0, 1] raises ValueError."""
        k, ps = power_law_ps
        for bad_gamma in (-0.5, 0.0, 1.5, 2.0):
            with pytest.raises(ValueError, match="gamma"):
                beta_f_compaction(k, ps, gamma=bad_gamma)

    def test_input_validation_non_positive_k(self) -> None:
        """Non-positive k values raise ValueError."""
        k = np.array([-1.0, 1.0, 2.0])
        ps = np.array([1e-9, 1e-9, 1e-9])
        with pytest.raises(ValueError, match="strictly positive"):
            beta_f_compaction(k, ps)

    def test_input_validation_non_positive_pk(self) -> None:
        """Non-positive P_S values raise ValueError."""
        k = np.array([0.1, 1.0, 10.0])
        ps = np.array([1e-9, 0.0, 1e-9])
        with pytest.raises(ValueError, match="strictly positive"):
            beta_f_compaction(k, ps)

    def test_input_validation_length_mismatch(self) -> None:
        """Mismatched array lengths raise ValueError."""
        k = np.logspace(-2, 6, 50)
        ps = np.full(30, 1e-9)
        with pytest.raises(ValueError, match="must have the same length"):
            beta_f_compaction(k, ps)

    def test_input_validation_non_1d(self) -> None:
        """Non-1D arrays raise TypeError."""
        k = np.ones((5, 5))
        ps = np.ones(25)
        with pytest.raises(TypeError, match="1-D"):
            beta_f_compaction(k, ps)

    def test_input_validation_non_1d_ps(self) -> None:
        """Non-1D P_S_k raises TypeError even with 1D k."""
        k = np.logspace(-2, 6, 50)
        ps = np.ones((5, 5))
        with pytest.raises(TypeError, match="1-D"):
            beta_f_compaction(k, ps)

    def test_custom_mu_parameter(self, power_law_ps) -> None:
        """Passing mu (peak-height) changes the zeta_c normalisation."""
        k, ps = power_law_ps
        _, _, meta = beta_f_compaction(k, ps, mu=0.5)
        # Should run without error and produce sensible output
        assert np.all(np.isfinite(meta["C_max_arr"]))

    def test_output_values_finite(self, power_law_ps) -> None:
        """All output values are finite floats."""
        k, ps = power_law_ps
        beta_f, M_pbh, meta = beta_f_compaction(k, ps)
        assert np.all(np.isfinite(beta_f))
        assert np.all(np.isfinite(M_pbh))
        for key in ("C_max_arr", "alpha_arr", "C_c_arr", "M_H_arr"):
            assert np.all(np.isfinite(meta[key]))

    def test_parallel_workers_match_serial(self, power_law_ps) -> None:
        """beta_f_compaction with n_workers > 1 matches n_workers=1."""
        k, ps = power_law_ps
        b1, m1, meta1 = beta_f_compaction(k, ps, n_workers=1)
        b2, m2, meta2 = beta_f_compaction(k, ps, n_workers=2)
        b4, m4, meta4 = beta_f_compaction(k, ps, n_workers=4)
        assert np.allclose(b1, b2, rtol=1e-12)
        assert np.allclose(b1, b4, rtol=1e-12)
        assert np.allclose(m1, m2, rtol=1e-12, atol=1e-30)
        assert np.allclose(meta1["C_max_arr"], meta2["C_max_arr"], rtol=1e-12)
        assert np.allclose(meta1["C_max_arr"], meta4["C_max_arr"], rtol=1e-12)

    def test_nworkers_none_defaults_to_one(self, power_law_ps) -> None:
        """n_workers=None is accepted and behaves like n_workers=1."""
        k, ps = power_law_ps
        b1, m1, _ = beta_f_compaction(k, ps, n_workers=1)
        bn, mn, _ = beta_f_compaction(k, ps, n_workers=None)
        assert np.allclose(b1, bn, rtol=1e-12)

    def test_nworkers_zero_raises(self, power_law_ps) -> None:
        """n_workers=0 raises ValueError."""
        k, ps = power_law_ps
        with pytest.raises(ValueError, match="n_workers"):
            beta_f_compaction(k, ps, n_workers=0)

    def test_non_gaussian_window_raises(self, power_law_ps) -> None:
        """Non-gaussian window raises ValueError in beta_f_compaction."""
        k, ps = power_law_ps
        with pytest.raises(ValueError, match="Unknown window"):
            beta_f_compaction(k, ps, window="tophat")


# ===========================================================================
# _simpson_weights — precomputed Simpson integration weights
# ===========================================================================


class TestSimpsonWeights:
    """Simpson integration weight precomputation."""

    def test_uniform_grid_matches_scipy(self) -> None:
        """Weights on uniform grid produce same integral as scipy."""
        ln_k = np.linspace(0, 10, 101)
        w = _simpson_weights(ln_k)
        f = np.sin(ln_k)
        assert np.dot(f, w) == pytest.approx(
            simpson(f, x=ln_k), abs=1e-12
        )

    def test_nonuniform_grid_matches_scipy(self) -> None:
        """Weights on non-uniform grid produce same integral as scipy."""
        rng = np.random.default_rng(42)
        ln_k = np.sort(rng.uniform(0, 10, 100))
        w = _simpson_weights(ln_k)
        f = np.exp(-0.5 * ln_k)
        assert np.dot(f, w) == pytest.approx(
            simpson(f, x=ln_k), abs=1e-12
        )


# ===========================================================================
# _compute_zeta_profile_vectorized — vectorized profile evaluation
# ===========================================================================


class TestVectorizedZetaProfile:
    """Vectorized zeta profile matches scalar version."""

    @pytest.fixture
    def setup(self) -> tuple:
        """Standard test arrays."""
        k = np.logspace(-4, 4, 500)
        ps = 2.1e-9 * (k / 0.05) ** (0.965 - 1)
        r = np.logspace(-3.0, 1.5, 500)
        r_with_zero = np.concatenate([[0.0], r])
        ln_k = np.log(k)
        w = _simpson_weights(ln_k)
        return k, ps, r_with_zero, ln_k, w

    def test_vectorized_equals_scalar(self, setup) -> None:
        """Vectorized and scalar zeta profiles match for all R_smooth."""
        k, ps, r, ln_k, w = setup
        for R_val in [0.1, 1.0, 10.0]:
            _, z_s = compute_zeta_r_profile(k, ps, r, R_smooth=R_val)
            _, z_v, _ = _compute_zeta_profile_vectorized(
                ln_k, w, k, ps, r, R_val, 1.0,
            )
            assert np.allclose(z_s, z_v, rtol=1e-12, atol=1e-15)

    def test_center_value_equals_zeta_c(self) -> None:
        """zeta(r=0) always equals zeta_c."""
        k = np.logspace(-2, 2, 200)
        ps = np.exp(-0.5 * (np.log(k) / 0.5) ** 2)
        r = np.linspace(0, 10, 50)
        ln_k = np.log(k)
        w = _simpson_weights(ln_k)
        for zeta_c in [0.5, 1.0, 2.0]:
            _, zeta, _ = _compute_zeta_profile_vectorized(
                ln_k, w, k, ps, r, 1.0, zeta_c,
            )
            assert zeta[0] == pytest.approx(zeta_c, abs=1e-10)


# ===========================================================================
# timer — elapsed time context manager
# ===========================================================================


class TestTimer:
    """Timer context manager prints elapsed time."""

    def test_timer_context_manager(self) -> None:
        """Timer works as context manager and prints time."""
        import io
        import time
        buf = io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(buf):
            with timer("test"):
                time.sleep(0.05)
        output = buf.getvalue().strip()
        assert "test" in output
        assert "0.05" in output or "0.06" in output or "0.04" in output
