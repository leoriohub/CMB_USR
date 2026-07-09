"""
Tests for n_s extraction methods (LSQ and derivative) from scripts.observables.

All tests use synthetic power-law spectra with known n_s — no integration with
inflation solvers or CAMB required.
"""

import numpy as np
import pytest

from scripts.observables import extract_ns


# ── helpers ──────────────────────────────────────────────────────────────────


def _power_law(k, As, ns, k_pivot):
    """Generate a perfect power-law P_S(k) = A_s * (k / k_pivot)^(n_s - 1)."""
    return As * (k / k_pivot) ** (ns - 1.0)


# ── LSQ method tests ─────────────────────────────────────────────────────────


class TestExtractNsLsq:
    """Tests for the LSQ (window-averaged) n_s extraction method."""

    @pytest.mark.fast
    def test_extract_ns_lsq_powerlaw(self):
        """LSQ on a perfect power law recovers n_s to <0.001."""
        k = np.logspace(-4, 0, 500)
        k_pivot = 0.05
        ns_input = 0.965
        As = 2.1e-9
        P_S = _power_law(k, As, ns_input, k_pivot)

        ns, meta = extract_ns(k, P_S, k_pivot, ns_window=4.0, method="lsq")

        assert abs(ns - ns_input) < 0.001, f"n_s={ns}, expected ~{ns_input}"
        assert meta["method"] == "lsq"
        assert meta["k_pivot"] == pytest.approx(0.05)
        assert meta["ns_window"] == pytest.approx(4.0)

    @pytest.mark.fast
    def test_extract_ns_lsq_insufficient_points(self):
        """LSQ returns None when fewer than 3 points fall in the fit window."""
        k = np.array([0.01, 0.1])
        P_S = np.array([1e-9, 2e-9])
        k_pivot = 0.05

        ns, meta = extract_ns(k, P_S, k_pivot, ns_window=4.0, method="lsq")

        assert ns is None
        # The LSQ branch does not set an "error" key; n_modes=0 for <3 points.
        assert "too few" in meta.get("error", "") or meta["n_modes"] < 3


# ── derivative method tests ──────────────────────────────────────────────────


class TestExtractNsDerivative:
    """Tests for the derivative n_s extraction method."""

    @pytest.mark.fast
    def test_extract_ns_derivative_powerlaw(self):
        """Derivative on a smooth power law recovers n_s to <0.001."""
        k = np.logspace(-4, 0, 500)
        k_pivot = 0.05
        ns_input = 0.965
        As = 2.1e-9
        P_S = _power_law(k, As, ns_input, k_pivot)

        ns, meta = extract_ns(k, P_S, k_pivot, method="derivative")

        assert abs(ns - ns_input) < 0.001, f"n_s={ns}, expected ~{ns_input}"
        assert meta["method"] == "derivative"
        assert meta["k_pivot"] == pytest.approx(0.05)
        assert meta["ns_window"] is None, "derivative ignores ns_window"

    @pytest.mark.fast
    def test_extract_ns_derivative_pivot_outside_range(self):
        """Derivative returns None when k_pivot lies outside the k grid."""
        k = np.logspace(-2, 0, 50)  # k in [0.01, 1.0]
        P_S = np.ones_like(k)
        k_pivot = 0.002  # below the k range

        ns, meta = extract_ns(k, P_S, k_pivot, method="derivative")

        assert ns is None
        assert "outside grid" in meta.get("error", ""), (
            f"expected 'outside grid' error, got: {meta.get('error', 'none')}"
        )


# ── default-parameter tests ──────────────────────────────────────────────────


class TestExtractNsDefaults:
    """Tests for default parameter handling."""

    @pytest.mark.fast
    def test_extract_ns_default_is_lsq(self):
        """Default method is 'lsq' and default ns_window is 4.0."""
        defaults = extract_ns.__defaults__

        # Signature: extract_ns(k_phys, P_S, k_pivot, ns_window=4.0, method="lsq")
        # Python __defaults__ aligns from the right:
        #   __defaults__[-2] = ns_window = 4.0
        #   __defaults__[-1] = method    = "lsq"
        assert defaults[-1] == "lsq", (
            f"Expected method default 'lsq', got {defaults[-1]!r}"
        )
        assert defaults[-2] == 4.0, (
            f"Expected ns_window default 4.0, got {defaults[-2]}"
        )

        # Calling without method argument must use LSQ under the hood.
        k = np.logspace(-4, 0, 500)
        k_pivot = 0.05
        ns_input = 0.965
        P_S = _power_law(k, 2.1e-9, ns_input, k_pivot)

        ns, meta = extract_ns(k, P_S, k_pivot)
        assert meta["method"] == "lsq", "No-arg call did not default to 'lsq'"
        assert meta["ns_window"] == pytest.approx(4.0)
        assert abs(ns - ns_input) < 0.001, (
            f"Default (LSQ) on power law: n_s={ns}, expected ~{ns_input}"
        )
