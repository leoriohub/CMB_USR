"""Unit tests for PBH-specific plotting functions.

Covers:
- make_filename() backward compat
- make_filename() with formation/accretion/**extra kwargs
- make_pbh_filename() code mapping
"""

from __future__ import annotations

import pytest

from scripts.plotting import (
    _ACCRETION_CODES,
    _FORMATION_CODES,
    make_filename,
    make_pbh_filename,
)


# ===========================================================================
# make_filename — backward compat (existing calls must NOT change)
# ===========================================================================


class TestMakeFilenameBackwardCompat:
    """Existing call patterns produce identical output."""

    def test_basic_ps(self) -> None:
        """make_filename('ps', 6.60, -0.736, 52.6) unchanged."""
        assert make_filename("ps", 6.60, -0.736, 52.6) \
            == "ps_phi6.60_y0-0.736_nstar52.6.json"

    def test_camb_spectrum(self) -> None:
        """make_filename('camb', 6.60, -0.736, 52.6) unchanged."""
        assert make_filename("camb", 6.60, -0.736, 52.6) \
            == "camb_phi6.60_y0-0.736_nstar52.6.json"

    def test_camb_lcdm_name_only(self) -> None:
        """make_filename('camb_lcdm') unchanged."""
        assert make_filename("camb_lcdm") == "camb_lcdm.json"

    def test_planck_png(self) -> None:
        """make_filename('planck', 6.60, -0.736, 52.6, '.png') unchanged."""
        assert make_filename("planck", 6.60, -0.736, 52.6, ".png") \
            == "planck_phi6.60_y0-0.736_nstar52.6.png"

    def test_name_only_no_params(self) -> None:
        """make_filename('test') with no params unchanged."""
        assert make_filename("test") == "test.json"


# ===========================================================================
# make_filename — new formation/accretion kwargs
# ===========================================================================


class TestMakeFilenameNewKwargs:
    """New formation/accretion and **extra kwargs."""

    def test_formation_accretion_both(self) -> None:
        """Both formation and accretion are appended after nstar."""
        result = make_filename(
            "ps", 8.0, -1e-4, 72.0, formation="cmp", accretion="PR",
        )
        assert result == "ps_phi8.00_y0-0.000_nstar72.0_cmp_PR.json"

    def test_formation_only(self) -> None:
        """Only formation is appended."""
        result = make_filename(
            "ps", 8.0, -1e-4, 72.0, formation="cmp",
        )
        assert result == "ps_phi8.00_y0-0.000_nstar72.0_cmp.json"

    def test_accretion_only(self) -> None:
        """Only accretion is appended."""
        result = make_filename(
            "ps", 8.0, -1e-4, 72.0, accretion="PR",
        )
        assert result == "ps_phi8.00_y0-0.000_nstar72.0_PR.json"

    def test_formation_accretion_extra(self) -> None:
        """Extra kwargs appended as key-value pairs."""
        result = make_filename(
            "ps", 8.0, -1e-4, 72.0,
            formation="cmp", accretion="PR", beta=2e-5, zc=0.077,
        )
        assert result \
            == "ps_phi8.00_y0-0.000_nstar72.0_cmp_PR_beta2e-05_zc0.077.json"

    def test_extra_without_formation(self) -> None:
        """Extra kwargs work without formation/accretion."""
        result = make_filename(
            "ps", 8.0, -1e-4, 72.0, beta=2e-5,
        )
        assert result == "ps_phi8.00_y0-0.000_nstar72.0_beta2e-05.json"

    def test_none_formation_is_ignored(self) -> None:
        """Explicit None for formation is ignored (backward compat)."""
        result = make_filename(
            "ps", 6.60, -0.736, 52.6, formation=None, accretion=None,
        )
        assert result == "ps_phi6.60_y0-0.736_nstar52.6.json"

    def test_empty_formation_is_not_appended(self) -> None:
        """Empty string for formation is treated as absent (no trailing underscore)."""
        result = make_filename(
            "ps", 8.0, -1e-4, 72.0, formation="",
        )
        assert result == "ps_phi8.00_y0-0.000_nstar72.0.json"

    def test_empty_both_formation_accretion(self) -> None:
        """Both empty strings produce standard filename."""
        result = make_filename(
            "ps", 8.0, -1e-4, 72.0, formation="", accretion="",
        )
        assert result == "ps_phi8.00_y0-0.000_nstar72.0.json"


# ===========================================================================
# make_pbh_filename — code mapping
# ===========================================================================


class TestMakePbhFilename:
    """make_pbh_filename short-code mapping."""

    def test_code_mappings(self) -> None:
        """_FORMATION_CODES and _ACCRETION_CODES are defined."""
        assert _FORMATION_CODES == {
            "compaction": "cmp",
            "press_schechter": "psch",
        }
        assert _ACCRETION_CODES == {
            "PR": "PR", "BHL": "BHL", "Eddington": "Edd",
            "Chisholm": "Chs", "Merger": "Mrg",
        }

    def test_compaction_PR(self) -> None:
        """compaction + PR maps to cmp_PR."""
        result = make_pbh_filename(
            "pbh", 8.0, -1e-4, 72.0,
            formation="compaction", accretion="PR", ext=".json",
        )
        assert result == "pbh_phi8.00_y0-0.000_nstar72.0_cmp_PR.json"

    def test_press_schechter_Chisholm(self) -> None:
        """press_schechter + Chisholm maps to psch_Chs."""
        result = make_pbh_filename(
            "pbh", 8.0, -1e-4, 72.0,
            formation="press_schechter", accretion="Chisholm", ext=".json",
        )
        assert result == "pbh_phi8.00_y0-0.000_nstar72.0_psch_Chs.json"

    def test_unknown_name_passes_through(self) -> None:
        """Unknown formation name passes through unchanged."""
        result = make_pbh_filename(
            "pbh", 8.0, -1e-4, 72.0, formation="unknown",
        )
        assert "unknown" in result

    def test_default_ext_is_png(self) -> None:
        """Default extension is .png."""
        result = make_pbh_filename(
            "pbh", 8.0, -1e-4, 72.0, formation="compaction", accretion="PR",
        )
        assert result.endswith(".png")
        assert "_cmp_PR" in result

    def test_extra_kwargs_included(self) -> None:
        """Extra kwargs passed through to make_filename."""
        result = make_pbh_filename(
            "rank02", 8.0, -1e-4, 72.0,
            formation="compaction", accretion="Chs",
            beta=2e-5, zc=0.077,
        )
        assert result == "rank02_phi8.00_y0-0.000_nstar72.0_cmp_Chs_beta2e-05_zc0.077.png"

    def test_no_formation_no_accretion(self) -> None:
        """No formation/accretion produces standard output."""
        result = make_pbh_filename("pbh", 8.0, -1e-4, 72.0)
        assert result == "pbh_phi8.00_y0-0.000_nstar72.0.png"
