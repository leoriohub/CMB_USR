"""Tests for find_end_of_inflation function."""
import pytest
import numpy as np
from pspectrum_pipeline import find_end_of_inflation


@pytest.mark.fast
def test_simple_SR_end():
    """epsH starts <1, crosses 1 once → returns crossing index."""
    n = 100
    epsH = np.ones(n)
    epsH[:60] = np.linspace(0.01, 0.5, 60)
    epsH[60:] = np.linspace(1.5, 3.0, 40)

    idx = find_end_of_inflation(epsH)
    assert idx == 60, f"Expected 60, got {idx}"


@pytest.mark.fast
def test_no_crossing():
    """epsH < 1 everywhere → returns -1."""
    epsH = np.linspace(0.01, 0.9, 100)
    idx = find_end_of_inflation(epsH)
    assert idx == -1, f"Expected -1, got {idx}"


@pytest.mark.fast
def test_transient_spike_ignored():
    """Transient USR spike crosses 1 but drops back; permanent crossing later is chosen."""
    n = 200
    epsH = np.ones(n) * 0.1

    # Transient spike at indices 50-59
    spike_len = 10
    spike_start = 50
    epsH[spike_start:spike_start + spike_len] = np.linspace(1.0, 2.0, spike_len)

    # Permanent crossing at index 150
    crossing_idx = 150
    epsH[crossing_idx:] = np.linspace(1.0, 3.0, n - crossing_idx)

    idx = find_end_of_inflation(epsH)
    assert idx == crossing_idx, (
        f"Expected {crossing_idx} (permanent crossing), "
        f"got {idx} (transient at {spike_start})"
    )


@pytest.mark.fast
def test_single_point_crossing():
    """epsH crosses exactly 1 at one index then stays above."""
    n = 100
    epsH = np.ones(n)
    epsH[:50] = 0.5
    epsH[50:] = 1.5

    idx = find_end_of_inflation(epsH)
    assert idx == 50, f"Expected 50, got {idx}"


@pytest.mark.fast
def test_never_begins():
    """epsH always >= 1 (kinetic-dominated throughout) → returns -1."""
    epsH = np.linspace(1.5, 10.0, 100)
    idx = find_end_of_inflation(epsH)
    assert idx == -1, f"Expected -1, got {idx}"
