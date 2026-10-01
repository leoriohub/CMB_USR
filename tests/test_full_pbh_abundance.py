"""Ordering and viability regressions for the PBH abundance consumer (#35).

Runs the production integration/selection block inside
``run_full_pbh_pipeline`` with the formation input controlled at the fixture
boundary. Sorting, filtering, trapz, peak lookup, and viability selection stay
in production code.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.constants import k_eq_default
from scripts.full_pbh_pipeline import run_full_pbh_pipeline


E = math.e
M_BASE = [1.0, E, E ** 2]
F_BASE = [0.1, 0.2, 0.3]


def _stub_background(monkeypatch):
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
        return {"k_phys": k, "P_S": np.full_like(k, 1e-9)}

    monkeypatch.setattr(psp, "run_pspectrum_pipeline", fake_pipeline)


def _run(monkeypatch, scenarios, zeta_c_vals, num_k=3):
    """Drive the real consumer block with controlled (M, f) formation pairs.

    ``scenarios`` maps a ζ_c value to ``(M_formation, f_abundance)``; the stub
    only supplies formation data, never the integration/selection itself. The
    k-grid length must equal the scenario length because the consumer pairs the
    formation abundance with the requesting k array.
    """
    _stub_background(monkeypatch)

    import scripts.compaction as compaction

    def fake_compaction(k_phys, P_S, gamma=None, mu=0.0, n_workers=1):
        masses, fractions = scenarios[round(float(mu), 6)]
        k = np.asarray(k_phys, dtype=float)
        n = min(len(k), len(masses))
        beta_f = np.asarray(fractions[:n], dtype=float) * k_eq_default / k[:n]
        return beta_f, np.asarray(masses[:n], dtype=float), {}

    monkeypatch.setattr(compaction, "beta_f_compaction", fake_compaction)

    return run_full_pbh_pipeline(
        chi0=8.0, y0=-1e-4, N_star=65, beta=1e-5, xc=0.784, c=0.77,
        workers=1, plot=False, k_min=1e-3, k_max=1e2, num_k=num_k,
        formation_model="compaction", accretion_model="Chisholm1",
        zeta_c_vals=zeta_c_vals,
    )


@pytest.mark.fast
def test_descending_pairs_integrate_to_analytic_total(pbh_pipeline_sandbox, monkeypatch):
    res = _run(
        monkeypatch,
        {0.05: (list(reversed(M_BASE)), list(reversed(F_BASE)))},
        [0.05],
    )
    assert res["f_total_best"] == pytest.approx(0.4, abs=1e-12)
    assert res["zeta_c_best"] == 0.05
    assert res["M_peak_best"] == pytest.approx(E ** 2)
    assert list(res["M_best"]) == sorted(res["M_best"])
    assert list(res["M_best"]) == pytest.approx(M_BASE)


@pytest.mark.fast
def test_nonmonotonic_pairs_integrate_to_analytic_total(pbh_pipeline_sandbox, monkeypatch):
    res = _run(
        monkeypatch,
        {0.05: ([E, E ** 2, 1.0], [0.2, 0.3, 0.1])},
        [0.05],
    )
    assert res["f_total_best"] == pytest.approx(0.4, abs=1e-12)
    assert res["M_peak_best"] == pytest.approx(E ** 2)
    assert list(res["M_best"]) == pytest.approx(M_BASE)
    peak_mass = res["M_best"][int(np.argmax(res["f_pbh_best"]))]
    assert peak_mass == pytest.approx(E ** 2)


@pytest.mark.fast
def test_highest_viable_total_wins_and_nonviable_is_excluded(pbh_pipeline_sandbox, monkeypatch):
    scenarios = {
        0.05: (M_BASE, F_BASE),
        0.06: (M_BASE, [0.05, 0.1, 0.15]),
        0.07: (M_BASE, [0.4, 0.6, 0.8]),
    }
    res = _run(monkeypatch, scenarios, [0.05, 0.06, 0.07])
    by_zeta = {r["zeta_c"]: r for r in res["all_results"]}
    assert by_zeta[0.05]["f_total"] == pytest.approx(0.4, abs=1e-12)
    assert by_zeta[0.06]["f_total"] == pytest.approx(0.2, abs=1e-12)
    assert by_zeta[0.07]["f_total"] > 1.0
    assert res["zeta_c_best"] == 0.05
    assert res["f_total_best"] == pytest.approx(0.4, abs=1e-12)


@pytest.mark.fast
def test_nonpositive_mass_is_filtered_before_integration(pbh_pipeline_sandbox, monkeypatch):
    res = _run(monkeypatch, {0.05: ([-1.0, E, E ** 2], [0.1, 0.2, 0.3])}, [0.05])
    # The -1 mass is dropped; the remaining pair integrates to 0.25.
    assert res["f_total_best"] == pytest.approx(0.25, abs=1e-12)
    assert list(res["M_best"]) == pytest.approx([E, E ** 2])
    assert res["M_peak_best"] == pytest.approx(E ** 2)


@pytest.mark.fast
def test_empty_selection_is_not_viable(pbh_pipeline_sandbox, monkeypatch):
    res = _run(monkeypatch, {0.05: (M_BASE, [0.0, 0.0, 0.0])}, [0.05])
    assert res["all_results"][0]["f_total"] == 0.0
    assert res["zeta_c_best"] is None
    assert len(res["M_best"]) == 0
    assert res["f_total_best"] == 0.0


@pytest.mark.fast
def test_single_point_keeps_current_zero_width_behavior(pbh_pipeline_sandbox, monkeypatch):
    res = _run(monkeypatch, {0.05: ([1.0], [0.1])}, [0.05], num_k=1)
    entry = res["all_results"][0]
    assert entry["M_peak"] == pytest.approx(1.0)
    assert entry["f_total"] == 0.0
    assert res["zeta_c_best"] is None
    assert res["M_best"].size == 0
