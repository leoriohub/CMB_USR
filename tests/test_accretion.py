"""Unit tests for scripts/accretion.py — PBH accretion models.

Covers PBHAccretion (BHL), PR_Accretion (Park-Ricotti with radiative feedback),
and ChisholmAccretion (legacy constant-factor model).

Physics invariants tested:
  - BHL Ṁ ∝ M² at fixed redshift
  - PR radiative feedback suppresses rate vs BHL
  - Chisholm step function at z=0
  - Monotonic mass evolution from z_form to z=0
  - Cosmology arrays positive-definite with correct shapes
"""

import numpy as np
import pytest
from scripts.accretion import (
    ChisholmAccretion,
    EddingtonAccretion,
    MergerAccretion,
    PBHAccretion,
    PR_Accretion,
    _cosmology_arrays,
    PBHAccretionError,
)


# ── BHL accretion ─────────────────────────────────────────────────────────────


class TestPBHAccretion:
    """Bondi-Hoyle-Lyttleton accretion model tests."""

    def test_BHL_negligible_low_M(self) -> None:
        """BHL λ=0.1, M≤100 → mass growth factor < 1.01 at z=0 from z_form=3400."""
        acc = PBHAccretion(lambda_acc=0.1)
        gf = acc.mass_growth_factor(100.0, z_obs=0.0)
        assert gf > 1.0  # some positive growth
        assert gf < 1.01  # but negligible for low-mass PBHs

    def test_monotonic_evolution(self) -> None:
        """BHL evolve returns M(z) monotonically increasing as z decreases."""
        acc = PBHAccretion(lambda_acc=0.1)
        M_hist, z_hist = acc.evolve(M_form=100.0, z_final=0.0, n_steps=50)

        assert len(M_hist) == 50
        assert len(z_hist) == 50
        # z decreases from z_form (~3400) to 0
        assert np.all(np.diff(z_hist) < 0)
        # M increases as z decreases (mass grows toward present)
        assert np.all(np.diff(M_hist) > 0)

    def test_BHL_accretion_rate_vs_M(self) -> None:
        r"""Ṁ ∝ M²: rate(10, z) / rate(20, z) = (10/20)² = 0.25 exactly."""
        acc = PBHAccretion(lambda_acc=0.1)
        z_test = 1000.0
        r10 = acc.accretion_rate(10.0, z_test)
        r20 = acc.accretion_rate(20.0, z_test)

        # At fixed z all cosmology factors cancel — pure M² scaling
        assert r10 / r20 == pytest.approx(0.25, rel=1e-12)

    def test_accretion_rate_zero_for_non_positive_M(self) -> None:
        """accretion_rate returns 0.0 for M <= 0, not an error."""
        acc = PBHAccretion(lambda_acc=0.1)
        assert acc.accretion_rate(0.0, 100.0) == 0.0
        assert acc.accretion_rate(-5.0, 100.0) == 0.0

    def test_invalid_inputs(self) -> None:
        """PBHAccretionError raised for invalid parameters."""
        # λ outside (0, 1]
        for bad_lambda in (0.0, -0.1, 1.5):
            with pytest.raises(PBHAccretionError, match="lambda_acc"):
                PBHAccretion(lambda_acc=bad_lambda)

        acc = PBHAccretion(lambda_acc=0.1)

        # M_form ≤ 0
        for bad_M in (0.0, -1.0):
            with pytest.raises(PBHAccretionError, match="M_form"):
                acc.M_of_redshift(M_form=bad_M, z_obs=0.0)

        # z_obs > z_form
        with pytest.raises(PBHAccretionError, match="z_obs"):
            acc.M_of_redshift(M_form=1.0, z_obs=2000.0, z_form=1000.0)

        # n_steps ≤ 2
        for bad_n in (2, 1, 0):
            with pytest.raises(PBHAccretionError, match="n_steps"):
                acc.evolve(M_form=1.0, n_steps=bad_n)

        # z_cutoff < 0
        with pytest.raises(PBHAccretionError, match="z_cutoff"):
            PBHAccretion(z_cutoff=-1.0)

    def test_M_of_redshift_returns_form_at_same_z(self) -> None:
        """z_obs == z_form returns M_form unchanged (no integration)."""
        acc = PBHAccretion(lambda_acc=0.1)
        result = acc.M_of_redshift(M_form=42.0, z_obs=3400.0, z_form=3400.0)
        assert result == pytest.approx(42.0)

    def test_evolve_returns_flat_if_z_final_gte_z_form(self) -> None:
        """z_final >= z_form returns flat array (no accretion)."""
        acc = PBHAccretion(lambda_acc=0.1)
        M_hist, z_hist = acc.evolve(M_form=50.0, z_form=100.0, z_final=200.0, n_steps=10)
        assert np.all(M_hist == pytest.approx(50.0))
        assert len(z_hist) == 10

    def test_evolve_rejects_non_positive_M_form(self) -> None:
        """PBHAccretion.evolve raises PBHAccretionError for M_form <= 0."""
        acc = PBHAccretion(lambda_acc=0.1)
        with pytest.raises(PBHAccretionError, match="M_form"):
            acc.evolve(M_form=0.0)
        with pytest.raises(PBHAccretionError, match="M_form"):
            acc.evolve(M_form=-5.0)


# ── Park-Ricotti (radiative feedback) ─────────────────────────────────────────


class TestPR_Accretion:
    """Park-Ricotti accretion model with photo-ionisation feedback tests."""

    def test_PR_vs_BHL_ratio(self) -> None:
        """For M=10 at z=3400, PR Ṁ < BHL Ṁ (larger effective sound speed)."""
        bhl = PBHAccretion(lambda_acc=0.1)
        pr = PR_Accretion(lambda_acc=0.1)
        M_test, z_test = 10.0, 3400.0

        rate_bhl = bhl.accretion_rate(M_test, z_test)
        rate_pr = pr.accretion_rate(M_test, z_test)

        # PR uses c_s_ionized = 23 km/s > cosmological c_s (~10 km/s at z=3400)
        # → larger v_eff → smaller accretion rate
        assert rate_pr < rate_bhl
        assert rate_pr > 0.0

    def test_PR_accretion_rate(self) -> None:
        r"""PR accretion_rate scales with M²: rate(100, z) / rate(10, z) = 100."""
        pr = PR_Accretion(lambda_acc=0.1)
        z_test = 100.0

        r10 = pr.accretion_rate(10.0, z_test)
        r100 = pr.accretion_rate(100.0, z_test)

        assert r10 < r100
        # M² scaling at fixed redshift
        assert r100 / r10 == pytest.approx(100.0, rel=1e-12)

    def test_PR_inherits_validation(self) -> None:
        """PR_Accretion inherits PBHAccretion parameter validation."""
        with pytest.raises(PBHAccretionError, match="lambda_acc"):
            PR_Accretion(lambda_acc=0.0)

        pr = PR_Accretion(lambda_acc=0.1)
        with pytest.raises(PBHAccretionError, match="M_form"):
            pr.M_of_redshift(M_form=-1.0, z_obs=0.0)

    def test_PR_rate_zero_for_non_positive_M(self) -> None:
        """PR accretion_rate returns 0 for M <= 0."""
        pr = PR_Accretion(lambda_acc=0.1)
        assert pr.accretion_rate(0.0, 100.0) == 0.0
        assert pr.accretion_rate(-1.0, 100.0) == 0.0


# ── Chisholm legacy model ─────────────────────────────────────────────────────


class TestChisholmAccretion:
    """Chisholm constant-factor accretion model tests."""

    CHISHOLM_FACTOR = 3.0e7

    def test_Chisholm_step(self) -> None:
        """Chisholm M_of_redshift(1.0, 0) = 3e7, M_of_redshift(1.0, 100) = 1.0."""
        acc = ChisholmAccretion()

        # At z=0, mass multiplied by 3×10⁷
        assert acc.M_of_redshift(1.0, 0.0) == pytest.approx(self.CHISHOLM_FACTOR)
        # At any z > 0, no growth
        assert acc.M_of_redshift(1.0, 100.0) == pytest.approx(1.0)
        assert acc.M_of_redshift(1.0, 3400.0) == pytest.approx(1.0)

    def test_Chisholm_evolve(self) -> None:
        """Chisholm evolve(1e-10, z_form=3000) → M_final = 3e-3 (sub-solar)."""
        acc = ChisholmAccretion()
        M_hist, z_hist = acc.evolve(1e-10, z_form=3000.0, n_steps=100)

        # Final mass = M_form × 3×10⁷ = 1e-10 × 3e7 = 3e-3
        assert M_hist[-1] == pytest.approx(3e-3)
        assert z_hist[-1] == pytest.approx(0.0)

        # Respects n_steps grid
        assert len(M_hist) == 100
        assert M_hist[0] == pytest.approx(1e-10)

    def test_mass_growth_factor(self) -> None:
        """Chisholm mass_growth_factor(1, 0) = 3e7, mass_growth_factor(1, 100) = 1."""
        acc = ChisholmAccretion()

        assert acc.mass_growth_factor(1.0, z_obs=0.0) == pytest.approx(self.CHISHOLM_FACTOR)
        assert acc.mass_growth_factor(1.0, z_obs=100.0) == pytest.approx(1.0)
        assert acc.mass_growth_factor(42.0, z_obs=0.0) == pytest.approx(self.CHISHOLM_FACTOR)

    def test_Chisholm_evolve_nonzero_z_final(self) -> None:
        """Chisholm evolve with z_final > 0 returns flat M (no z=0 step)."""
        acc = ChisholmAccretion()
        M_hist, z_hist = acc.evolve(1e-10, z_form=3000.0, z_final=100.0)

        # No growth because z_final > 0
        assert M_hist[-1] == pytest.approx(1e-10)
        assert z_hist[-1] == pytest.approx(100.0)

    def test_Chisholm_validation(self) -> None:
        """Chisholm raises PBHAccretionError for invalid inputs."""
        acc = ChisholmAccretion()
        with pytest.raises(PBHAccretionError, match="M_form"):
            acc.M_of_redshift(M_form=0.0, z_obs=0.0)
        with pytest.raises(PBHAccretionError, match="M_form"):
            acc.M_of_redshift(M_form=-5.0, z_obs=0.0)
        with pytest.raises(PBHAccretionError, match="M_form"):
            acc.evolve(M_form=0.0)
        with pytest.raises(PBHAccretionError, match="n_steps"):
            acc.evolve(M_form=1.0, n_steps=2)


# ── Eddington-limited accretion ───────────────────────────────────────────────


class TestEddingtonAccretion:
    """Eddington-limited BHL accretion model tests."""

    EDDINGTON_FACTOR_1MSUN = 4.0 * np.pi * 6.67430e-8 * 1.98892e33 * 1.6735575e-24 / (
        0.1 * 6.652458732e-25 * 2.99792458e10
    )

    def test_eddington_rate_analytic(self) -> None:
        """_eddington_rate(1 M_⊙) matches hand-computed value."""
        acc = EddingtonAccretion(f_boost=1e4)
        rate = acc._eddington_rate(1.0)
        # Expected: 4π G M_SUN m_p / (ε σ_T c)
        expected = self.EDDINGTON_FACTOR_1MSUN
        assert rate == pytest.approx(expected, rel=1e-12)

    def test_eddington_rate_scales_linearly(self) -> None:
        """Ṁ_Edd ∝ M — rate(100) / rate(1) == 100."""
        acc = EddingtonAccretion(f_boost=1e4)
        r1 = acc._eddington_rate(1.0)
        r100 = acc._eddington_rate(100.0)
        assert r100 / r1 == pytest.approx(100.0, rel=1e-12)

    def test_accretion_rate_non_positive_M(self) -> None:
        """accretion_rate returns 0.0 for M <= 0."""
        acc = EddingtonAccretion(f_boost=1e4)
        assert acc.accretion_rate(0.0, 100.0) == 0.0
        assert acc.accretion_rate(-5.0, 100.0) == 0.0

    def test_accretion_rate_positive(self) -> None:
        """accretion_rate returns positive for M>0, z>0."""
        acc = EddingtonAccretion(f_boost=1e4)
        rate = acc.accretion_rate(100.0, 500.0)
        assert rate > 0.0

    def test_invalid_f_boost(self) -> None:
        """PBHAccretionError for f_boost <= 0."""
        with pytest.raises(PBHAccretionError, match="f_boost"):
            EddingtonAccretion(f_boost=0.0)
        with pytest.raises(PBHAccretionError, match="f_boost"):
            EddingtonAccretion(f_boost=-1.0)

    def test_invalid_epsilon_rad(self) -> None:
        """PBHAccretionError for epsilon_rad outside (0, 1]."""
        with pytest.raises(PBHAccretionError, match="epsilon_rad"):
            EddingtonAccretion(epsilon_rad=0.0)
        with pytest.raises(PBHAccretionError, match="epsilon_rad"):
            EddingtonAccretion(epsilon_rad=1.5)

    def test_M_of_redshift_monotonic(self) -> None:
        """M_of_redshift(10, 0, 500) > M_of_redshift(1, 0, 500)."""
        acc = EddingtonAccretion(f_boost=1e4)
        m1 = acc.M_of_redshift(1.0, 0.0, 500.0)
        m10 = acc.M_of_redshift(10.0, 0.0, 500.0)
        assert m10 > m1  # larger input → larger output

    def test_evolve_produces_growth(self) -> None:
        """evolve from z=3400 to z=0: final mass > initial mass."""
        acc = EddingtonAccretion(f_boost=1e4)
        M_hist, z_hist = acc.evolve(M_form=1000.0, n_steps=50)
        assert M_hist[-1] > M_hist[0]
        assert np.all(np.diff(z_hist) < 0)  # z decreases
        assert np.all(np.diff(M_hist) > 0)  # M increases

    def test_M_of_redshift_returns_form_at_same_z(self) -> None:
        """z_obs == z_form returns M_form unchanged."""
        acc = EddingtonAccretion(f_boost=1e4)
        result = acc.M_of_redshift(M_form=42.0, z_obs=3400.0, z_form=3400.0)
        assert result == pytest.approx(42.0)

    def test_mass_growth_factor_gt1(self) -> None:
        """mass_growth_factor > 1 for accretion from z=500 to z=0."""
        acc = EddingtonAccretion(f_boost=1e4)
        gf = acc.mass_growth_factor(100.0, z_obs=0.0, z_form=500.0)
        assert gf > 1.0

    def test_inherits_validation(self) -> None:
        """EddingtonAccretion inherits PBHAccretion parameter validation."""
        with pytest.raises(PBHAccretionError, match="lambda_acc"):
            EddingtonAccretion(lambda_acc=0.0, f_boost=1e4)
        acc = EddingtonAccretion(f_boost=1e4)
        with pytest.raises(PBHAccretionError, match="M_form"):
            acc.M_of_redshift(M_form=0.0, z_obs=0.0)


# ── Merger history ─────────────────────────────────────────────────────────────


class TestMergerAccretion:
    """Merger history accretion model tests."""

    def test_merger_growth_factor_analytic(self) -> None:
        """growth_factor(4) = 1 + 0.5*4^0.5 = 2.0 for default params."""
        m = MergerAccretion(A=0.5, gamma=0.5)
        assert m.growth_factor(4.0) == pytest.approx(2.0)

    def test_merger_growth_scaling(self) -> None:
        """growth_factor increases with mass for gamma > 0."""
        m = MergerAccretion(A=0.5, gamma=0.5)
        assert m.growth_factor(100.0) > m.growth_factor(1.0)
        assert m.growth_factor(1.0) > 1.0

    def test_merger_zero_gamma_gives_constant(self) -> None:
        """gamma=0: growth_factor = 1 + A (mass-independent)."""
        m = MergerAccretion(A=0.3, gamma=0.0)
        assert m.growth_factor(1.0) == pytest.approx(1.3)
        assert m.growth_factor(100.0) == pytest.approx(1.3)

    def test_merger_M_of_redshift(self) -> None:
        """M_of_redshift returns M * growth_factor(M)."""
        m = MergerAccretion(A=0.5, gamma=0.5)
        M_form = 100.0
        expected = M_form * m.growth_factor(M_form)
        assert m.M_of_redshift(M_form, 0.0) == pytest.approx(expected)

    def test_merger_validation(self) -> None:
        """PBHAccretionError for invalid parameters."""
        with pytest.raises(PBHAccretionError, match="M_form"):
            MergerAccretion().M_of_redshift(M_form=0.0, z_obs=0.0)
        with pytest.raises(PBHAccretionError, match="M_form"):
            MergerAccretion().M_of_redshift(M_form=-5.0, z_obs=0.0)
        with pytest.raises(PBHAccretionError, match="A must"):
            MergerAccretion(A=-1.0)
        with pytest.raises(PBHAccretionError, match="gamma must"):
            MergerAccretion(gamma=-1.0)
        with pytest.raises(PBHAccretionError, match="n_steps"):
            MergerAccretion().evolve(M_form=1.0, n_steps=2)

    def test_merger_evolve(self) -> None:
        """evolve respects z_cutoff and tracks mass transition correctly."""
        m = MergerAccretion(A=0.5, gamma=0.5)
        M_hist, z_hist = m.evolve(M_form=100.0, n_steps=100)
        expected = 100.0 * m.growth_factor(100.0)
        
        # At early times (z >= 10.0), no mergers should have occurred
        assert M_hist[0] == pytest.approx(100.0)
        # At late times (z = 0.0), full merger growth is applied
        assert M_hist[-1] == pytest.approx(expected)
        assert len(z_hist) == 100

    def test_merger_mass_growth_factor(self) -> None:
        """mass_growth_factor at z=0 equals growth_factor(M_form)."""
        m = MergerAccretion(A=0.5, gamma=0.5)
        gf = m.mass_growth_factor(100.0, z_obs=0.0)
        assert gf == pytest.approx(m.growth_factor(100.0))


# ── Cosmology utilities ───────────────────────────────────────────────────────


class TestCosmologyUtilities:
    """Cosmology array helper function tests."""

    COSMOLOGY = {
        'H0': 67.66,
        'ombh2': 0.02242,
        'omch2': 0.11933,
        'T_cmb': 2.725,
    }

    def test_cosmology_arrays(self) -> None:
        """_cosmology_arrays at z=[0, 500, 3400]: shapes match, all positive."""
        z_arr = np.array([0.0, 500.0, 3400.0])
        result = _cosmology_arrays(z_arr, self.COSMOLOGY)

        # All five arrays present with correct shape
        for key in ('rho_gas', 'c_s', 'v_PBH', 'H', 'rho_crit'):
            arr = result[key]
            assert arr.shape == z_arr.shape, f"{key} shape {arr.shape} != {z_arr.shape}"
            assert np.all(arr > 0), f"{key} has non-positive entries"

    def test_cosmology_arrays_physical_ordering(self) -> None:
        """Densities and expansion rates increase with redshift."""
        z_arr = np.array([0.0, 500.0, 3400.0])
        result = _cosmology_arrays(z_arr, self.COSMOLOGY)

        # rho_gas ∝ (1+z)³ → higher at higher z
        assert result['rho_gas'][2] > result['rho_gas'][1] > result['rho_gas'][0]
        # H(z) increases with z
        assert result['H'][2] > result['H'][1] > result['H'][0]
        # c_s ∝ sqrt(T) ∝ sqrt(1+z) → higher at higher z
        assert result['c_s'][2] > result['c_s'][1] > result['c_s'][0]

    def test_cosmology_arrays_single_element(self) -> None:
        """_cosmology_arrays works with single-element input."""
        result = _cosmology_arrays(np.array([0.0]), self.COSMOLOGY)
        for arr in result.values():
            assert arr.shape == (1,)

    def test_cosmology_arrays_default_cosmology(self) -> None:
        """_cosmology_arrays uses Planck 2018 defaults when cosmology is None."""
        z_arr = np.array([0.0, 1000.0])
        result = _cosmology_arrays(z_arr, self.COSMOLOGY)
        for arr in result.values():
            assert arr.shape == (2,)
            assert np.all(arr > 0)

    def test_cosmology_arrays_v_PBH_formula(self) -> None:
        """v_PBH follows 10 × (1 + z/1000)^{0.5} km/s."""
        z_arr = np.array([0.0, 1000.0, 3000.0])
        result = _cosmology_arrays(z_arr, self.COSMOLOGY)

        # v_PBH at z=0: 10 × (1+0)^{0.5} = 10 km/s
        assert result['v_PBH'][0] == pytest.approx(10.0 * 1e5, rel=1e-12)
        # v_PBH at z=1000: 10 × (1+1)^{0.5} ≈ 14.14 km/s
        expected_v1 = 10.0 * np.sqrt(2.0) * 1e5
        assert result['v_PBH'][1] == pytest.approx(expected_v1, rel=1e-12)
        # v_PBH at z=3000: 10 × (1+3)^{0.5} = 20 km/s
        assert result['v_PBH'][2] == pytest.approx(20.0 * 1e5, rel=1e-12)
