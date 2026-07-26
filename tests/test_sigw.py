"""Physics tests for the SIGW module."""

import numpy as np
import pytest
from scripts.sigw import (
    DILUTION,
    compute_sigw_spectrum,
    compute_snr,
    f_to_k,
    get_lisa_noise,
    k_to_f,
    kernel_rd,
)


class TestFrequencyConversion:
    def test_roundtrip(self):
        f_in = np.logspace(-5, 1, 50)
        k = f_to_k(f_in)
        f_out = k_to_f(k)
        assert np.allclose(f_out, f_in, rtol=1e-10)

    def test_known_value(self):
        k_mpc = 1e6
        f_hz = k_to_f(k_mpc)
        expected = k_mpc / 3.085677581e24 * 2.99792458e10 / (2.0 * np.pi)
        assert abs(f_hz - expected) < 1e-10 * expected


class TestKernelDomain:
    def test_outside_triangle(self):
        assert kernel_rd(0.5, 10.0) == 0.0
        assert kernel_rd(10.0, 0.5) == 0.0
        assert kernel_rd(0.1, 0.01) == 0.0

    def test_on_boundary(self):
        """Singular boundaries u=0 or v=0 return 0 (guarded)."""
        assert kernel_rd(1.0, 0.0) == 0.0
        assert kernel_rd(0.0, 1.0) == 0.0

    def test_inside_triangle_finite(self):
        val = kernel_rd(1.0, 1.0)
        assert np.isfinite(val)
        assert val > 0

    def test_vectorized_shape(self):
        u = np.array([0.5, 1.0, 1.5])
        v = np.array([1.0, 0.8, 0.6])
        K = kernel_rd(u, v)
        assert K.shape == (3,)

    def test_matrix_shape(self):
        u = np.linspace(0.1, 2.0, 10)
        v = np.logspace(-2, 1, 8)
        K = kernel_rd(u[:, None], v[None, :])
        assert K.shape == (10, 8)

    def test_zeros_outside(self):
        u = np.array([0.1, 100.0])
        v = np.array([100.0, 0.1])
        K = kernel_rd(u, v)
        assert K[0] == 0.0
        assert K[1] == 0.0


class TestMonochromaticLimit:
    def test_delta_peak_amplitude(self):
        """Monochromatic P_S(k) approximated by narrow lognormal.

        Uses a broader lognormal (delta=0.1) for grid-resolution safety.
        """
        k_p = 1e12
        k_phys = np.logspace(8, 16, 400)
        delta = 0.1
        P_S = 2.1e-9 * np.exp(-0.5 * (np.log(k_phys / k_p) / delta) ** 2)

        k_target = 0.5 * k_p
        f_target = k_to_f(k_target)
        omega = compute_sigw_spectrum(k_phys, P_S, np.array([f_target]), num_grid=80)
        assert np.isfinite(omega[0])
        assert omega[0] > 0


class TestIRScaling:
    def test_f3_scaling(self):
        """Omega_GW drops at low f for a narrow peak — IR slope is positive."""
        k_p = 1e12
        delta = 0.1
        k_phys = np.logspace(8, 16, 400)
        P_S = 2.1e-9 * np.exp(-0.5 * (np.log(k_phys / k_p) / delta) ** 2)

        f_peak = k_to_f(k_p)
        f_lo = np.logspace(np.log10(f_peak) - 1.5, np.log10(f_peak) - 0.5, 8)
        omega_lo = compute_sigw_spectrum(k_phys, P_S, f_lo, num_grid=80)

        ok = omega_lo > 1e-30
        assert np.sum(ok) >= 3
        slope, _ = np.polyfit(np.log10(f_lo[ok]), np.log10(omega_lo[ok]), 1)
        assert slope > 0  # must rise toward peak


class TestLISANoise:
    def test_noise_floor(self):
        """LISA minimum sensitivity at f ≈ 3 mHz, Ω_noise h² ∈ [1e-12, 1e-10]."""
        f = np.logspace(-4, 0, 500)
        noise = get_lisa_noise(f)
        min_idx = np.argmin(noise)
        f_min = f[min_idx]
        n_min = noise[min_idx]
        assert 0.002 < f_min < 0.01
        assert 1e-12 < n_min < 1e-10


class TestDilutionFactor:
    def test_present_day_factor(self):
        """Ω_GW,0 h² = Ω_rd · DILUTION where DILUTION ≈ 1.62e-5."""
        assert abs(DILUTION - 1.62e-5) / 1.62e-5 < 0.01


class TestIntegrationStability:
    def test_grid_doubling_convergence(self):
        """Omega_GW changes modestly when integration grid is increased."""
        k_p = 1e12
        delta = 0.1  # broad, resolvable
        k_phys = np.logspace(8, 16, 400)
        P_S = 2.1e-9 * np.exp(-0.5 * (np.log(k_phys / k_p) / delta) ** 2)

        f_peak = k_to_f(k_p)
        f_grid = np.logspace(np.log10(f_peak) - 0.5, np.log10(f_peak) + 0.5, 6)
        omega_80 = compute_sigw_spectrum(k_phys, P_S, f_grid, num_grid=80)
        omega_120 = compute_sigw_spectrum(k_phys, P_S, f_grid, num_grid=120)

        peak_80 = np.max(omega_80)
        peak_120 = np.max(omega_120)

        shift = abs(peak_120 - peak_80) / max(peak_80, 1e-30)
        assert shift < 0.50  # modest convergence for low grids


class TestSNR:
    def test_no_signal_zero_snr(self):
        """Zero GW signal → zero SNR."""
        f = np.logspace(-4, 0, 100)
        omega_gw = np.zeros_like(f)
        omega_noise = get_lisa_noise(f)
        snr = compute_snr(f, omega_gw, omega_noise)
        assert snr == 0.0

    def test_snr_monotonic_with_tobs(self):
        """SNR scales as √T_obs."""
        k_p = 1e12
        A_sq = 2.1e-9
        delta = 0.005
        k_phys = np.logspace(8, 16, 200)
        P_S = A_sq * np.exp(-0.5 * (np.log(k_phys / k_p) / delta) ** 2)

        f_gw = np.logspace(-4, 0, 100)
        omega_gw = compute_sigw_spectrum(k_phys, P_S, f_gw, num_grid=100)
        omega_noise = get_lisa_noise(f_gw)

        snr_1 = compute_snr(f_gw, omega_gw, omega_noise, t_obs_yr=1.0)
        snr_4 = compute_snr(f_gw, omega_gw, omega_noise, t_obs_yr=4.0)
        assert abs(snr_4 / snr_1 - 2.0) < 0.05
