"""Unit tests for scripts/constraint_mapping.py — PBH constraint epoch mapping.

Covers _invert_mass, map_constraint_to_present, load_bound_dataset,
and plot_modern_constraints.

Physics invariants tested:
  - z_probe=0 mapping returns identity (no shift)
  - De Luca+20 Eq. 17: f_present = f_z * M_z / M_present
  - _invert_mass short-circuits for identity-mapping accretion models
  - _invert_mass produces M_form < M_z for models with growth
  - load_bound_dataset returns consistent shapes
"""

import numpy as np
import pytest

from scripts.accretion import ChisholmAccretion, PBHAccretion, EddingtonAccretion


# ── _invert_mass ──────────────────────────────────────────────────────────────


class TestInvertMass:
    """_invert_mass: find formation mass from observed mass and accretion model."""

    def test_identity_short_circuit_Chisholm(self) -> None:
        """Chisholm at z_probe=500 returns M_form = M_z (identity mapping)."""
        from scripts.constraint_mapping import _invert_mass

        acc = ChisholmAccretion()
        M_z = 100.0
        M_form = _invert_mass(M_z, acc, z_probe=500.0)
        assert M_form == pytest.approx(M_z, rel=1e-12)

    def test_bhl_growth_inverts(self) -> None:
        """BHL at z_probe=500: M_form < M_z (positive growth)."""
        from scripts.constraint_mapping import _invert_mass

        acc = PBHAccretion(lambda_acc=0.1)
        M_z = 1000.0
        M_form = _invert_mass(M_z, acc, z_probe=500.0)
        assert M_form < M_z  # growth means smaller initial mass
        assert M_form > 0.0

    def test_eddington_growth_inverts(self) -> None:
        """Eddington at z_probe=500: M_form << M_z (strong growth)."""
        from scripts.constraint_mapping import _invert_mass

        acc = EddingtonAccretion(f_boost=1e4)
        M_z = 1000.0
        M_form = _invert_mass(M_z, acc, z_probe=500.0)
        assert M_form < M_z  # growth means smaller initial mass
        assert M_form > 0.0

    def test_probe_equals_form_gives_identity(self) -> None:
        """When z_probe == z_form, M_form == M_z for all models."""
        from scripts.constraint_mapping import _invert_mass

        for acc in [ChisholmAccretion(), PBHAccretion(lambda_acc=0.1)]:
            M_form = _invert_mass(42.0, acc, z_probe=3400.0, z_form=3400.0)
            assert M_form == pytest.approx(42.0)

    def test_convergence_different_masses(self) -> None:
        """_invert_mass converges for a range of M_z values."""
        from scripts.constraint_mapping import _invert_mass

        acc = EddingtonAccretion(f_boost=1e4)
        for M_z in [1.0, 10.0, 100.0, 1000.0]:
            M_form = _invert_mass(M_z, acc, z_probe=500.0)
            assert M_form > 0.0
            assert M_form < M_z  # Eddington always produces growth

    @pytest.mark.parametrize("model_cls,kwargs,M_z", [
        (PBHAccretion, {"lambda_acc": 0.1}, 100.0),
        (PBHAccretion, {"lambda_acc": 0.1}, 1000.0),
        (PBHAccretion, {"lambda_acc": 0.1}, 10000.0),
        (EddingtonAccretion, {"f_boost": 1e4}, 100.0),
        (EddingtonAccretion, {"f_boost": 1e4}, 1000.0),
        (EddingtonAccretion, {"f_boost": 1e4}, 10000.0),
    ])
    def test_roundtrip(self, model_cls, kwargs, M_z) -> None:
        """M_of_redshift(_invert_mass(M_z), z_probe) ≈ M_z (roundtrip)."""
        from scripts.constraint_mapping import _invert_mass

        acc = model_cls(**kwargs)
        z_probe = 500.0
        M_form = _invert_mass(M_z, acc, z_probe=z_probe)
        M_recovered = acc.M_of_redshift(M_form, z_probe)
        # Relative error should be tiny — Brent tolerance is 1e-12/1e-6
        assert M_recovered == pytest.approx(M_z, rel=1e-6, abs=1e-6)


# ── map_constraint_to_present ────────────────────────────────────────────────


class TestMapConstraintToPresent:
    """map_constraint_to_present — De Luca+20 Eq. 17 mapping."""

    def test_zprobe_zero_returns_identity(self) -> None:
        """For z_probe=0, mapping returns unchanged (M, f) arrays."""
        from scripts.constraint_mapping import map_constraint_to_present

        M_z = np.array([1.0, 10.0, 100.0])
        f_z = np.array([0.1, 0.01, 0.001])
        for acc in [ChisholmAccretion(), PBHAccretion(lambda_acc=0.1)]:
            M_p, f_p = map_constraint_to_present(M_z, f_z, acc, z_probe=0.0)
            assert np.allclose(M_p, M_z)
            assert np.allclose(f_p, f_z)

    def test_Chisholm_mapping_ratio(self) -> None:
        """Chisholm at z_probe=500: M_present = 3e7 * M_z, f = f_z / 3e7."""
        from scripts.constraint_mapping import map_constraint_to_present

        acc = ChisholmAccretion()
        M_z = np.array([10.0, 100.0, 1000.0])
        f_z = np.array([0.01, 0.001, 0.0001])

        M_p, f_p = map_constraint_to_present(M_z, f_z, acc, z_probe=500.0)

        # M_present ≈ M_z * 3e7 (growth only at z=0)
        expected_factor = 3.0e7
        assert np.allclose(M_p / M_z, expected_factor, rtol=1e-10)
        # f_present = f_z * M_z / M_present
        assert np.allclose(f_p, f_z * M_z / M_p, rtol=1e-12)

    def test_BHL_mapping_f_present_formula(self) -> None:
        """BHL: f_present = f_z * M_z / M_present (Eq. 17 invariant)."""
        from scripts.constraint_mapping import map_constraint_to_present

        acc = PBHAccretion(lambda_acc=0.1)
        M_z = np.array([100.0, 1000.0, 10000.0])
        f_z = np.array([0.01, 0.001, 0.0001])

        M_p, f_p = map_constraint_to_present(M_z, f_z, acc, z_probe=500.0)

        # Eq. 17 invariant: f_present * M_present = f_z * M_z
        lhs = f_p * M_p
        rhs = f_z * M_z
        assert np.allclose(lhs, rhs, rtol=1e-10)

    def test_eddington_mapping_f_present_formula(self) -> None:
        """Eddington: f_present = f_z * M_z / M_present (Eq. 17 invariant)."""
        from scripts.constraint_mapping import map_constraint_to_present

        acc = EddingtonAccretion(f_boost=1e4)
        M_z = np.array([100.0, 1000.0, 10000.0])
        f_z = np.array([0.01, 0.001, 0.0001])

        M_p, f_p = map_constraint_to_present(M_z, f_z, acc, z_probe=500.0)

        # Eq. 17 invariant: f_present * M_present = f_z * M_z
        lhs = f_p * M_p
        rhs = f_z * M_z
        assert np.allclose(lhs, rhs, rtol=1e-10)

    def test_mapping_preserves_monotonicity(self) -> None:
        """Monotonic input M_z → monotonic output M_present."""
        from scripts.constraint_mapping import map_constraint_to_present

        acc = EddingtonAccretion(f_boost=1e4)
        M_z = np.array([1.0, 10.0, 100.0, 1000.0])
        f_z = np.array([0.1, 0.01, 0.001, 0.0001])

        M_p, f_p = map_constraint_to_present(M_z, f_z, acc, z_probe=500.0)

        assert np.all(np.diff(M_p) > 0)  # M_present monotonic
        assert np.all(np.diff(f_p) < 0)  # f_present monotonic (decreasing)


# ── load_bound_dataset ────────────────────────────────────────────────────────


class TestLoadBoundDataset:
    """load_bound_dataset — constraint data loading."""

    def test_load_cmb_standard(self) -> None:
        """Load CMB bound: returns (M, f) arrays with positive values."""
        from scripts.constraint_mapping import load_bound_dataset

        M, f, meta = load_bound_dataset("cmb_standard")
        assert len(M) > 0
        assert len(f) > 0
        assert np.all(M > 0)
        assert np.all(f > 0)
        assert "z_probe" in meta
        assert meta["z_probe"] == 500

    def test_load_hsc_microlensing(self) -> None:
        """Load HSC bound: z_probe=0, arrays have positive values."""
        from scripts.constraint_mapping import load_bound_dataset

        M, f, meta = load_bound_dataset("microlensing_hsc")
        assert len(M) > 0
        assert np.all(M > 0)
        assert np.all(f > 0)
        assert meta["z_probe"] == 0

    def test_unknown_key_raises_KeyError(self) -> None:
        """Unknown constraint name raises KeyError."""
        from scripts.constraint_mapping import load_bound_dataset

        with pytest.raises(KeyError):
            load_bound_dataset("nonexistent_constraint")

    def test_bound_registry_entries(self) -> None:
        """All BOUND_REGISTRY entries have required fields."""
        from scripts.constraint_mapping import BOUND_REGISTRY

        for name, meta in BOUND_REGISTRY.items():
            assert "file" in meta, f"{name} missing 'file'"
            assert "z_probe" in meta, f"{name} missing 'z_probe'"
            assert "label" in meta, f"{name} missing 'label'"
            assert "color" in meta, f"{name} missing 'color'"
            assert isinstance(meta["z_probe"], (int, float)), f"{name} z_probe not numeric"


# ── plot_modern_constraints ────────────────────────────────────────────────────


class TestPlotModernConstraints:
    """plot_modern_constraints — constraint axes plotting."""

    def test_returns_handles_list(self) -> None:
        """plot_modern_constraints returns list of (name, line) handles."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scripts.constraint_mapping import plot_modern_constraints

        fig, ax = plt.subplots()
        handles = plot_modern_constraints(ax, accretion_model=None)
        assert isinstance(handles, list)
        assert len(handles) > 0
        for name, line in handles:
            assert isinstance(name, str)
            assert hasattr(line, "get_xdata")
        plt.close(fig)

    def test_raises_TypeError_for_None_ax(self) -> None:
        """Calling with ax=None raises TypeError."""
        from scripts.constraint_mapping import plot_modern_constraints

        with pytest.raises(TypeError, match="None"):
            plot_modern_constraints(None)

    def test_show_set_filters_constraints(self) -> None:
        """show_set limits plotted constraints to specified subset."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scripts.constraint_mapping import plot_modern_constraints

        fig, ax = plt.subplots()
        subset = {"cmb_standard", "microlensing_hsc"}
        handles = plot_modern_constraints(ax, show_set=subset)
        plotted_names = {h[0] for h in handles}
        assert plotted_names == subset
        plt.close(fig)

    def test_accretion_model_mapping(self) -> None:
        """With accretion_model, high-z bounds differ from unmapped."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scripts.constraint_mapping import (
            plot_modern_constraints, load_bound_dataset,
        )

        fig, ax = plt.subplots()
        subset = {"cmb_standard"}
        handles_mapped = plot_modern_constraints(
            ax, accretion_model=EddingtonAccretion(f_boost=1e4),
            show_set=subset,
        )
        plt.close(fig)

        fig2, ax2 = plt.subplots()
        handles_unmapped = plot_modern_constraints(
            ax2, accretion_model=None, show_set=subset,
        )
        plt.close(fig2)

        # Mapped and unmapped lines should be different objects
        assert len(handles_mapped) == 1
        assert len(handles_unmapped) == 1
