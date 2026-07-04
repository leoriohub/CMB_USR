"""Tests for USR utility functions."""
import pytest
import numpy as np
from scripts.usr_utils import compute_epsilon2


@pytest.mark.fast
def test_epsilon2_SR_vanishes():
    """ε₂ = 2ε₁ - 2η_H must be zero in pure de Sitter."""
    eps2 = compute_epsilon2(np.array([0.0]), np.array([0.0]))
    assert eps2[0] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.fast
def test_epsilon2_powerlaw():
    """ε₂ = 2ε₁ - 2η_H for a power-law potential."""
    eps2 = compute_epsilon2(np.array([0.01]), np.array([0.02]))
    # ε₂ = 2*0.01 - 2*0.02 = -0.02
    assert eps2[0] == pytest.approx(-0.02, rel=1e-15)
