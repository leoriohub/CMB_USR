"""Tests for PBH utility functions."""
import pytest
import numpy as np
from scripts.full_pbh_pipeline import mass_from_k
from scripts.constants import k_eq_default, M_eq_default, gamma_default


@pytest.mark.fast
def test_mass_from_k_exact():
    """At k=k_eq, formation mass must be γ * M_eq (by definition of k_eq)."""
    M = mass_from_k(k_eq_default)
    expected = gamma_default * M_eq_default
    assert M == pytest.approx(expected, rel=1e-12)


@pytest.mark.fast
def test_mass_from_k_scaling():
    """mass_from_k scales as k^{-2}: halving k → 4x mass."""
    M_half = mass_from_k(k_eq_default / 2.0)
    M_ref = mass_from_k(k_eq_default)
    assert M_half == pytest.approx(4.0 * M_ref, rel=1e-12)
