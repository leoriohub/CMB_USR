"""Cache-provenance regressions for scripts/full_pbh_pipeline.py (issue #34).

Exercises the frozen private contract ``_ps_cache_identity`` /
``_ps_cache_matches`` plus real cache read/write/reuse through
``run_full_pbh_pipeline``. Filenames are never reproduced: the pipeline's own
returned ``ps_path`` is used.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from models.ezquiaga_chi import EzquiagaCHIModel, inflection_parameters
from scripts.full_pbh_pipeline import (
    _ps_cache_identity,
    _ps_cache_matches,
    resolve_kgrid,
    run_full_pbh_pipeline,
)
from scripts.observables import extract_ns, interpolate_As


BASE = dict(
    chi0=8.0,
    y0=-1e-4,
    c=0.77,
    xc=0.784,
    beta=1e-5,
    N_star=65.0,
    N_total=86.7,
    k_pivot=0.002,
)

GRID = resolve_kgrid(1e-4, 1e2, 41)


def _build_model(c, xc, beta, chi0, y0, T_max=None, bg_steps=None):
    model = EzquiagaCHIModel(c=c)
    a, b = inflection_parameters(xc, c, beta=beta)
    model.a = a
    model.b = b
    model.v0 = model._V0 * model.a / (model.b * model.c) ** 2
    model.x0, model.y0 = chi0, y0
    if T_max is not None:
        model.T_max = T_max
    if bg_steps is not None:
        model.bg_steps = bg_steps
    return model


def _identity(**overrides: Any) -> dict[str, Any]:
    """Build a request identity from BASE, applying overrides."""
    params = dict(BASE)
    params.update(overrides)
    k_grid = np.asarray(params.pop("k_grid", GRID), dtype=float)
    fast_sr = params.pop("fast_sr", False)
    t_max = params.pop("T_max", None)
    bg_steps = params.pop("bg_steps", None)
    model = _build_model(
        params["c"], params["xc"], params["beta"], params["chi0"], params["y0"],
        T_max=t_max, bg_steps=bg_steps,
    )
    return _ps_cache_identity(
        model,
        chi0=params["chi0"],
        y0=params["y0"],
        N_star=params["N_star"],
        N_total=params["N_total"],
        k_pivot=params["k_pivot"],
        xc=params["xc"],
        beta=params["beta"],
        k_grid=k_grid,
        fast_sr=fast_sr,
    )


def _saved_record(identity: dict[str, Any], *, writer="full_pbh_pipeline", version=1,
                  identity_key: str | None = "cache_identity", k_grid=None,
                  spectrum=None) -> dict[str, Any]:
    grid = GRID if k_grid is None else np.asarray(k_grid, dtype=float)
    if spectrum is None:
        spectrum = _sentinel(grid)
    record = {
        "_type": "pspectrum",
        "format_version": 1,
        "writer": writer,
        "cache_identity_version": version,
        "metadata": {
            "chi0": BASE["chi0"],
            "y0": BASE["y0"],
            "N_star": BASE["N_star"],
            "k_pivot_Mpc": BASE["k_pivot"],
        },
        "spectrum": {
            "k_phys": [float(x) for x in grid],
            "P_S": [float(x) for x in spectrum],
        },
    }
    if identity_key is not None:
        # Deep-copy so a caller mutating the returned record (e.g. dropping a
        # provenance group to build an incomplete case) can never mutate the
        # expected request identity it was built from.
        record[identity_key] = copy.deepcopy(identity)
    return record


# ── pure identity / matching contract ──────────────────────────────────────


@pytest.mark.fast
def test_nstar_and_pivot_changes_are_misses():
    identity = _identity()
    nstar = _identity(N_star=BASE["N_star"] + 0.05)
    pivot = _identity(k_pivot=0.02)
    for other in (nstar, pivot):
        assert _ps_cache_matches(_saved_record(identity), other) is False
        assert _ps_cache_matches(_saved_record(other), identity) is False


@pytest.mark.fast
def test_model_change_below_old_filename_precision_is_a_miss():
    # The legacy cache name rounded c to 6 significant figures, so these two
    # requests shared a path; exact provenance must still treat them apart.
    identity = _identity(c=0.77)
    shifted = _identity(c=0.7700001)
    assert _ps_cache_matches(_saved_record(identity), shifted) is False


@pytest.mark.fast
def test_inflection_shape_change_is_a_miss():
    identity = _identity()
    shifted = _identity(beta=BASE["beta"] * (1.0 + 1e-7))
    assert _ps_cache_matches(_saved_record(identity), shifted) is False


@pytest.mark.fast
def test_sr_and_ms_requests_are_not_interchangeable():
    ms = _identity()
    sr = _identity(fast_sr=True)
    assert ms["solver"]["source"] == "MS"
    assert sr["solver"]["source"] == "SR"
    assert _ps_cache_matches(_saved_record(ms), sr) is False
    assert _ps_cache_matches(_saved_record(sr), ms) is False


@pytest.mark.fast
def test_background_changes_are_misses():
    identity = _identity()
    for override in ({"T_max": 500.0}, {"bg_steps": 4000},
                     {"N_total": BASE["N_total"] + 0.01}):
        assert _ps_cache_matches(_saved_record(identity), _identity(**override)) is False


@pytest.mark.fast
def test_fixed_recipe_change_is_a_miss():
    identity = _identity()
    for mutate in ({"ms_method": "rk45"}, {"ms_steps": 5001},
                   {"recipe_version": 2}, {"k_start_factor": 200.0},
                   {"normalize_to_As": True}, {"source": "SR"}):
        tampered = copy.deepcopy(identity)
        tampered["solver"].update(mutate)
        assert _ps_cache_matches(_saved_record(tampered), identity) is False


@pytest.mark.fast
def test_full_requested_grid_difference_is_a_miss():
    identity = _identity()
    interior = GRID.copy()
    interior[len(interior) // 2] *= 1.0000001
    assert _ps_cache_matches(_saved_record(identity), _identity(k_grid=interior)) is False

    shorter = resolve_kgrid(1e-4, 1e2, 40)
    assert _ps_cache_matches(_saved_record(identity), _identity(k_grid=shorter)) is False


@pytest.mark.fast
def test_legacy_foreign_and_incomplete_records_are_misses():
    identity = _identity()
    assert _ps_cache_matches(_saved_record(identity, writer="legacy"), identity) is False
    assert _ps_cache_matches(_saved_record(identity, version=2), identity) is False
    assert _ps_cache_matches(
        _saved_record(identity, identity_key=None), identity
    ) is False

    incomplete = _saved_record(identity)
    del incomplete["cache_identity"]["grid"]
    # Editing the stored record must not touch the expected request identity.
    assert "grid" in identity
    assert _ps_cache_matches(incomplete, identity) is False


# ── real cache I/O through the pipeline ────────────────────────────────────


_SENTINEL_K = resolve_kgrid(1e-3, 1e2, 31)


def _sentinel(k):
    k = np.asarray(k, dtype=float)
    return 1e-9 * (k / k[0]) ** (-0.04) * (1.0 + 0.5 * np.log(k / k[0]) ** 2)


def _stub_solvers(monkeypatch):
    """Isolate the expensive background/MS producers at their boundaries."""
    import inf_dyn_background as bg

    n_arr = np.linspace(0.0, 90.0, 101)
    eps_h = np.concatenate([np.linspace(0.05, 0.95, 80), np.linspace(1.2, 3.0, 21)])

    monkeypatch.setattr(bg, "run_background_simulation",
                        lambda model, T: (n_arr.copy(), n_arr.copy(),
                                         n_arr.copy(), n_arr.copy()))
    monkeypatch.setattr(bg, "get_derived_quantities",
                        lambda sol, model: {"N": n_arr, "epsH": eps_h})

    import pspectrum_pipeline as psp

    def fake_pipeline(**kwargs):
        k = np.asarray(kwargs["k_phys_grid"], dtype=float)
        return {"k_phys": k, "P_S": _sentinel(k)}

    monkeypatch.setattr(psp, "run_pspectrum_pipeline", fake_pipeline)


@pytest.mark.integration
def test_cache_io_hit_downstream_force_and_miss(pbh_pipeline_sandbox, monkeypatch):
    _stub_solvers(monkeypatch)
    request: dict[str, Any] = dict(
        chi0=BASE["chi0"], y0=BASE["y0"], N_star=BASE["N_star"],
        beta=BASE["beta"], xc=BASE["xc"], c=BASE["c"],
        workers=1, plot=False, k_min=1e-3, k_max=1e2, num_k=31,
        k_pivot=BASE["k_pivot"], ns_window=3.0,
        formation_model="press_schechter", accretion_model="Chisholm1",
    )

    # 1. Cold write: real artifact inside the sandbox.
    first = run_full_pbh_pipeline(**request)
    assert first["cached"] is False
    path = Path(first["ps_path"]).resolve()
    assert pbh_pipeline_sandbox.root in path.parents
    assert path.exists()

    # Replace the stored spectrum with a distinguishable sentinel.
    record = json.loads(path.read_text())
    stored = np.asarray(record["spectrum"]["P_S"], dtype=float) * 7.0
    record["spectrum"]["P_S"] = stored.tolist()
    path.write_text(json.dumps(record))
    k_saved = np.asarray(record["spectrum"]["k_phys"], dtype=float)

    # 2. Exact hit preserves the stored spectrum and derives observables from it.
    second = run_full_pbh_pipeline(**request)
    assert second["cached"] is True
    np.testing.assert_allclose(second["P_S"], stored)
    expected_ns, _ = extract_ns(k_saved, stored, k_pivot=BASE["k_pivot"],
                                ns_window=3.0)
    assert second["n_s"] == pytest.approx(expected_ns, rel=1e-12)
    assert second["A_s_cmb"] == pytest.approx(
        interpolate_As(k_saved, stored, BASE["k_pivot"]), rel=1e-12
    )

    # 3. Downstream-only knob: still a hit, observables use the current window.
    wider_request: dict[str, Any] = {**request, "ns_window": 5.0}
    wider = run_full_pbh_pipeline(**wider_request)
    assert wider["cached"] is True
    np.testing.assert_allclose(wider["P_S"], stored)
    wide_ns, _ = extract_ns(k_saved, stored, k_pivot=BASE["k_pivot"], ns_window=5.0)
    assert wider["n_s"] == pytest.approx(wide_ns, rel=1e-12)
    assert wider["n_s"] != second["n_s"]

    # 4. force bypasses an existing, matching cache record.
    forced_request: dict[str, Any] = {**request, "force": True}
    forced = run_full_pbh_pipeline(**forced_request)
    assert forced["cached"] is False
    np.testing.assert_allclose(
        forced["P_S"], _sentinel(np.asarray(forced["k_phys"], dtype=float))
    )

    # 5. A changed pivot must not return the stored sentinel.
    changed_request: dict[str, Any] = {**request, "k_pivot": 0.02}
    changed = run_full_pbh_pipeline(**changed_request)
    assert changed["cached"] is False
    np.testing.assert_allclose(
        changed["P_S"], _sentinel(np.asarray(changed["k_phys"], dtype=float))
    )


@pytest.mark.integration
def test_cache_io_sr_reuses_stored_grid_when_requested_differs(
    pbh_pipeline_sandbox, monkeypatch
):
    # SR derives its P_S grid from the background, so a matching record may
    # carry a different ``k_phys`` than the request; cache reuse must still hit
    # and return that stored grid. The SR producer is not the acceptance path
    # here (only cache-hit reuse is), so seed the real artifact through the MS
    # stub, then overwrite it with an analytic SR record whose cache_identity is
    # the exact ``fast_sr`` request identity for the same model/background.
    _stub_solvers(monkeypatch)
    request: dict[str, Any] = dict(
        chi0=BASE["chi0"], y0=BASE["y0"], N_star=BASE["N_star"],
        beta=BASE["beta"], xc=BASE["xc"], c=BASE["c"],
        workers=1, plot=False, k_min=1e-3, k_max=1e2, num_k=31,
        k_pivot=BASE["k_pivot"], ns_window=3.0,
        formation_model="press_schechter", accretion_model="Chisholm1",
    )
    requested = resolve_kgrid(1e-3, 1e2, 31)
    assert np.array_equal(requested, _SENTINEL_K)

    # 1. Cold write through the real pipeline to learn its own artifact path
    #    (never reconstruct the filename) and the runtime N_total.
    seed = run_full_pbh_pipeline(**request)
    assert seed["cached"] is False
    path = Path(seed["ps_path"]).resolve()
    assert pbh_pipeline_sandbox.root in path.parents
    record = json.loads(path.read_text())
    n_total = float(record["metadata"]["N_total"])

    # 2. Replace the stored record with an independently analytic SR
    #    grid/spectrum and the exact matching request identity.
    sr_identity = _identity(fast_sr=True, N_total=n_total, k_grid=requested)
    sr_k = np.logspace(-3, 3, 50)  # SR output grid ≠ requested grid
    sr_P = _sentinel(sr_k) * 3.0
    assert not np.array_equal(sr_k, requested)
    record["cache_identity"] = sr_identity
    record["spectrum"] = {"k_phys": sr_k.tolist(), "P_S": sr_P.tolist()}
    record["metadata"] = {**record["metadata"], "source": "SR",
                          "n_modes": int(sr_k.size)}
    path.write_text(json.dumps(record))

    # 3. The matching SR request hits and reuses the stored grid/spectrum,
    #    deriving real observables from it rather than recomputing SR.
    sr_request: dict[str, Any] = {**request, "fast_sr": True}
    second = run_full_pbh_pipeline(**sr_request)
    assert second["cached"] is True
    np.testing.assert_allclose(second["k_phys"], sr_k)
    np.testing.assert_allclose(second["P_S"], sr_P)
    expected_ns, _ = extract_ns(sr_k, sr_P, k_pivot=BASE["k_pivot"],
                                ns_window=3.0)
    assert second["n_s"] == pytest.approx(expected_ns, rel=1e-12)
    assert second["A_s_cmb"] == pytest.approx(
        interpolate_As(sr_k, sr_P, BASE["k_pivot"]), rel=1e-12
    )
