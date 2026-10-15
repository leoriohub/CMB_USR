"""Single source of truth for config-driven model construction.

Adding a model means writing one factory here and exporting the class from
`models/__init__.py`. `pspectrum_pipeline.model_from_config` and the CLI
`--model` choices both derive from `MODEL_REGISTRY`, so the dispatch cannot
drift between the three places it used to be hand-written.

Each factory takes the config's `model_params` and `inflection` blocks and
returns a configured `InflationModel`. Per-model constructor defaults live
here so that a bare `{"model": "<Name>"}` always builds the same object.
"""

from .higgs import HiggsModel, FullHiggsModel
from .higgs_palatini import PalatiniHiggsModel
from .punctuated import PunctuatedInflationModel
from .ezquiaga_chi import EzquiagaCHIModel, inflection_parameters
from .starobinsky import StarobinskyModel


def _higgs(params, inflection):
    return HiggsModel(lam=params.get("lam", 0.13),
                      xi=params.get("xi", 15000.0))


def _full_higgs(params, inflection):
    return FullHiggsModel(lam=params.get("lam", 0.1),
                          xi=params.get("xi", 1000.0),
                          v_vev=params.get("v_vev", 0.0))


def _palatini_higgs(params, inflection):
    return PalatiniHiggsModel(lam=params.get("lam", 0.13),
                              xi=params.get("xi", 1e9))


def _punctuated(params, inflection):
    return PunctuatedInflationModel(m=params.get("m", 1.1323e-7),
                                    lam=params.get("lam", 3.3299e-15),
                                    phi0=params.get("phi0_inflection", None))


def _starobinsky(params, inflection):
    return StarobinskyModel(v0=params.get("v0", 4.5e-11))


def _ezquiaga(params, inflection):
    model = EzquiagaCHIModel(
        lambda_0=params.get("lambda_0", 2.23e-7),
        b_lambda=params.get("b_lambda", 1.2e-6),
        xi_0=params.get("xi_0", 7.55),
        b_xi=params.get("b_xi", 11.5),
        c=params.get("c", 0.77),
        n_grid=params.get("n_grid", 5000),
    )
    x_c = inflection.get("x_c")
    if x_c is not None:
        beta = inflection.get("beta", 1e-5)
        a_new, b_new = inflection_parameters(x_c, params.get("c", 0.77), beta)
        model.a = a_new
        model.b = b_new
        model.v0 = model._V0 * model.a / (model.b * model.c)**2
        print(f"  Self-consistent inflection params: a={a_new:.4f}, b={b_new:.6f}")
    model.patch_background_solver()
    return model


MODEL_REGISTRY = {
    "HiggsModel": _higgs,
    "FullHiggsModel": _full_higgs,
    "PalatiniHiggsModel": _palatini_higgs,
    "PunctuatedInflationModel": _punctuated,
    "EzquiagaCHIModel": _ezquiaga,
    "StarobinskyModel": _starobinsky,
}


def model_names():
    """Registered model names, sorted -- used for CLI choices and tests."""
    return sorted(MODEL_REGISTRY)
