# AGENTS.md

Cosmological inflation codebase studying the CMB low-ℓ anomaly via Higgs Ultra-Slow Roll (USR) dynamics.

## Project structure

```
models/              — InflationModel base, HiggsModel (high-field approx), FullHiggsModel (exact conformal inversion)
inf_dyn_background.py — Background ODE solver: run_background_simulation(), get_derived_quantities()
inf_dyn_MS_full.py    — Mukhanov-Sasaki perturbation solver: run_ms_simulation(), get_ms_derived_quantities()
numerical_observables_calculation.py — High-level orchestrator: run_inflation_protocol() → (ns, r, P_S)
scripts/
  pspectrum_pipeline.py — Batch P_S(k) across k-modes: run_pspectrum_pipeline(), load_pspectrum()
  camb_wrapper.py       — Sachs-Wolfe C_ell + CAMB integration wrappers
  planck_data.py        — Planck 2018 low-ℓ TT data loader (Commander)
notebooks/           — Jupyter analysis notebooks (see .opencode/agents/notebook.md for editing rules)
outputs/cmb_results/ — Cached simulation results (JSON)
  pspectra/          — P_S(k) cache, named Higgs_Inflation_phi{X}_yi{Y}_run_{uuid}.json
  simulations/       — Full dashboard outputs from notebooks
  c_ell/             — CMB angular power spectra
  plots/             — Generated figures
data/                — Empty placeholder dirs for Planck/ACT likelihood data
```

## No build system, tests, or CI

This is a research physics codebase. There are minimal tests, but no linting, typechecking, Makefiles, pyproject.toml, or requirements.txt. Dependencies: `numpy`, `scipy`, `matplotlib`, `camb` (optional, for full Boltzmann code). To verify changes, run relevant notebook cells or Python scripts directly.

## Path setup (critical for notebooks)

Notebooks must add the repo root to `sys.path` before importing:
```python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
```
The `scripts/` modules do this internally via `ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))`.

## Pipeline execution flow

```
HiggsModel(lam, xi) → model.phi0, model.yi
  → run_background_simulation(model, T_span)    # ~0.008s, returns (x,y,z,n) over time
  → get_derived_quantities(sol, model)           # extracts epsH, etaH, ns, r, P_S
  → run_ms_simulation(model, bg_sol, k_modes)    # ~0.007s per k-mode
  → run_pspectrum_pipeline(...)                   # orchestrates full P_S(k), saves JSON
```

Full P_S(k) with 80 k-modes: ~3s. With dense weighted grid (~181 modes): ~1 min. Two configs (USR+SR) in notebooks: 2+ min first run, instant on subsequent runs via cache.

## Caching pattern

`run_or_load()` in notebooks checks `outputs/cmb_results/pspectra/` for matching (phi0, yi, xi, N_pivot). If found, loads cached JSON. Force recompute with `force_recompute=True`.

## Model classes

- `HiggsModel(xi, lam)` — High-field approximation. `f(x) = (1 - exp(-αx))²`, `α = √(2/3)`. Default ICs: `phi0=5.8, yi=-0.01`.
- `FullHiggsModel(xi, lam)` — Exact conformal inversion via numerical integration on a grid. Slower but more accurate.
- `SmoothUSRTransitionModel` — Analytical smooth USR transition model for comparison.
- All inherit from `InflationModel` (models/base.py). Key method: `get_initial_conditions()` returns `[phi0, yi, zi, Ni]`.

## Physical constants and conventions

- Time unit: `S = 5e-5` (one unit of conformal time)
- Planck normalization: `A_s = 2.1e-9` at pivot `k_* = 0.05 Mpc⁻¹`
- Last scattering distance: `r_ls = 14000 Mpc`
- CMB temperature: `T_cmb = 2.7255 K`
- End of inflation: `ε_H ≥ 1` (Hubble slow-roll parameter)
- USR phase: triggered by negative initial velocity `yi < -0.05` (vs `yi ≈ -0.001` for SR)

## Notebook editing rules

See `.opencode/agents/notebook.md` for full rules. Key points:
- Parse JSON, modify in Python, serialize with `json.dumps(nb, indent=1) + "\n"`
- Preserve `outputs`, `execution_count`, `metadata` on all cells
- Source array lines must end with `\n` except possibly the last
- Validate roundtrip after every edit

## Output file naming

- pspectra cache: `Higgs_Inflation_phi{phi0:.2f}_yi{yi:.3f}_run_{uuid8}.json`
- dashboard results: `powerloss_phi{phi0:.2f}_yi{yi:.3f}_xi{xi:.0f}_Npivot{N_pivot}.json`
- JSON structure: `{metadata, k_phys, P_S}` for pspectra; `{config, primordial, cmb_sw, stats}` for simulations

## Common pitfalls

- `xi=15000` makes the potential very flat → integrator needs more steps → slow. Don't reduce `mxstep` or tolerances.
- `use_weighted=True` in NUM_PARAMS builds ~181 k-modes (not the `num_k=80` default). This is intentional for USR zone resolution.
- The `loss_vs_sr` variable is computed in diagnostics cells but may not be in scope in save cells — compute inline.
- Matplotlib `alpha` must be in [0,1]. When doubling alpha for emphasis, clamp: `min(max(alpha*2, 0.5), 1.0)`.
- Notebook `source` arrays: each line is a separate string. Join with `""`, not `"\n"`. Split with `.splitlines(keepends=True)`.
