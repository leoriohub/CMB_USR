"""Contract tests for config -> model construction.

`model_from_config` is the single entry point every config-driven consumer uses
(pspectrum_pipeline, full_pbh_pipeline, camb_scan). Its dispatch used to be a
hand-rolled if/elif chain duplicated in three places; these tests pin the
observable contract so the dispatch can be restructured without silently
changing which model a config produces, or with which parameters.
"""

import pytest

from models import (
    EzquiagaCHIModel,
    FullHiggsModel,
    HiggsModel,
    PalatiniHiggsModel,
    PunctuatedInflationModel,
    StarobinskyModel,
)
from pspectrum_pipeline import model_from_config
from models.registry import MODEL_REGISTRY, model_names


pytestmark = pytest.mark.fast


def test_every_registered_name_builds():
    for name in model_names():
        model = model_from_config({"model": name})
        assert model is not None
        assert hasattr(model, "f") and hasattr(model, "dfdx")


def test_registry_keys_match_cli_choices():
    # The CLI --model choices come from the registry, so they cannot drift.
    assert "PalatiniHiggsModel" in MODEL_REGISTRY
    assert "HiggsModel" in MODEL_REGISTRY


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        model_from_config({"model": "NoSuchModel"})


def test_default_constructor_params():
    m = model_from_config({"model": "HiggsModel"})
    assert isinstance(m, HiggsModel)
    assert m.lam == pytest.approx(0.13)
    assert m.xi_val == pytest.approx(15000.0)

    m = model_from_config({"model": "FullHiggsModel"})
    assert isinstance(m, FullHiggsModel)
    assert m.lam == pytest.approx(0.1)
    assert m.xi_val == pytest.approx(1000.0)
    assert m.v_vev == pytest.approx(0.0)

    m = model_from_config({"model": "StarobinskyModel"})
    assert isinstance(m, StarobinskyModel)
    assert m.v0 == pytest.approx(4.5e-11)

    m = model_from_config({"model": "PalatiniHiggsModel"})
    assert isinstance(m, PalatiniHiggsModel)
    assert m.lam == pytest.approx(0.13)
    assert m.xi_val == pytest.approx(1e9)

    m = model_from_config({"model": "PunctuatedInflationModel"})
    assert isinstance(m, PunctuatedInflationModel)
    assert m.m == pytest.approx(1.1323e-7)


def test_model_params_override_defaults():
    m = model_from_config({"model": "HiggsModel",
                           "model_params": {"lam": 0.2, "xi": 42.0}})
    assert m.lam == pytest.approx(0.2)
    assert m.xi_val == pytest.approx(42.0)

    m = model_from_config({"model": "PalatiniHiggsModel",
                           "model_params": {"lam": 0.5, "xi": 1e4}})
    assert m.lam == pytest.approx(0.5)
    assert m.xi_val == pytest.approx(1e4)

    m = model_from_config({"model": "StarobinskyModel",
                           "model_params": {"v0": 1.2e-10}})
    assert m.v0 == pytest.approx(1.2e-10)


def test_ics_override_wins_over_model_defaults():
    m = model_from_config({"model": "HiggsModel",
                           "ics": {"x0": 6.4, "y0": -0.475}})
    assert m.x0 == pytest.approx(6.4)
    assert m.y0 == pytest.approx(-0.475)


def test_integration_knobs_override():
    m = model_from_config({"model": "HiggsModel",
                           "model_params": {"T_max": 123.0, "bg_steps": 77}})
    assert m.T_max == pytest.approx(123.0)
    assert m.bg_steps == 77


def test_ezquiaga_inflection_reparametrizes_model():
    base = model_from_config({"model": "EzquiagaCHIModel",
                              "model_params": {"c": 0.77}})
    assert isinstance(base, EzquiagaCHIModel)
    # No inflection block: constructor values survive untouched.
    a_before, b_before = base.a, base.b

    tuned = model_from_config({"model": "EzquiagaCHIModel",
                               "model_params": {"c": 0.77},
                               "inflection": {"x_c": 0.784, "beta": 1e-5}})
    assert tuned.a != a_before or tuned.b != b_before
    # v0 is re-derived from the new (a, b, c); keep it consistent with them.
    assert tuned.v0 == pytest.approx(
        tuned._V0 * tuned.a / (tuned.b * tuned.c) ** 2)
