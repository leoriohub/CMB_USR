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
  sachs_wolfe.py        — Sachs-Wolfe C_ell approximation for ℓ≤30
  planck_data.py        — Planck 2018 low-ℓ TT data loader (Commander)
  usr_chi2_optimizer.py — (phi0, y0, N_star) parameter search: differential_evolution + chi² + ns + k_dip penalty
notebooks/           — Jupyter analysis notebooks (see .opencode/agents/notebook.md for editing rules)
outputs/
  plots/               — Generated figures
    powerloss/         — USR golden figures (publication quality)
    diagnostics/       — Background evolution and mode checks
    optimizer/         — Optimizer convergence and comparison plots
  simulations/         — Cached run data (JSON)
    pspectra/          — P_S(k) cache, named PS_{Model}_m{...}_phi{...}_y0{...}_Nstar{...}_{uuid}.json
    configs/           — Simulation configuration snapshots
    c_ell/             — CMB angular power spectra (Sachs-Wolfe)
    background/        — Background trajectory data
    logs/              — Optimizer log files (JSONL)
    scans/             — Parameter sweep summaries
data/                — Empty placeholder dirs for Planck/ACT likelihood data
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
HiggsModel(lam, xi) → model.phi0, model.y0
  → run_background_simulation(model, T_span)    # ~0.008s, returns (x,y,z,n) over time
  → get_derived_quantities(sol, model)           # extracts epsH, etaH, ns, r, P_S
  → run_ms_simulation(model, bg_sol, k_modes)    # ~0.007s per k-mode
  → run_pspectrum_pipeline(...)                   # orchestrates full P_S(k), saves JSON
```

Full P_S(k) with 80 k-modes: ~3s. With dense weighted grid (~181 modes): ~1 min. Two configs (USR+SR) in notebooks: 2+ min first run, instant on subsequent runs via cache.

## Caching pattern

`run_or_load()` in notebooks checks `outputs/simulations/pspectra/` for matching (phi0, y0, xi, N_pivot). If found, loads cached JSON. Force recompute with `force_recompute=True`.

## Model classes

- `HiggsModel(xi, lam)` — High-field approximation. `f(x) = (1 - exp(-αx))²`, `α = √(2/3)`. Default ICs: `phi0=5.8, y0=-0.01`.
- `FullHiggsModel(xi, lam)` — Exact conformal inversion via numerical integration on a grid. Slower but more accurate.
- `SmoothUSRTransitionModel` — Analytical smooth USR transition model for comparison.
- All inherit from `InflationModel` (models/base.py). Key method: `get_initial_conditions()` returns `[phi0, y0, zi, Ni]`.

## Physical constants and conventions

- Time unit: `S = 5e-5` (one unit of conformal time)
- Planck normalization: `A_s = 2.1e-9` at pivot `k_* = 0.05 Mpc⁻¹`
- Last scattering distance: `r_ls = 14000 Mpc`
- CMB temperature: `T_cmb = 2.7255 K`
- End of inflation: `ε_H ≥ 1` (Hubble slow-roll parameter)
- USR phase: triggered by negative initial velocity `y0 < -0.05` (vs `y0 ≈ -0.001` for SR)

## Notebook editing rules

See `.opencode/agents/notebook.md` for full rules. Key points:
- Parse JSON, modify in Python, serialize with `json.dumps(nb, indent=1) + "\n"`
- Preserve `outputs`, `execution_count`, `metadata` on all cells
- Source array lines must end with `\n` except possibly the last
- Validate roundtrip after every edit

## Output file naming

- pspectra cache: `Higgs_Inflation_phi{phi0:.2f}_y0{y0:.3f}_run_{uuid8}.json`
- dashboard results: `powerloss_phi{phi0:.2f}_y0{y0:.3f}_xi{xi:.0f}_Npivot{N_pivot}.json`
- JSON structure: `{metadata, k_phys, P_S}` for pspectra; `{config, primordial, cmb_sw, stats}` for simulations

## Common pitfalls

- `xi=15000` makes the potential very flat → integrator needs more steps → slow. Don't reduce `mxstep` or tolerances.
- `use_weighted=True` in NUM_PARAMS builds ~181 k-modes (not the `num_k=80` default). This is intentional for USR zone resolution.
- The `loss_vs_sr` variable is computed in diagnostics cells but may not be in scope in save cells — compute inline.
- Matplotlib `alpha` must be in [0,1]. When doubling alpha for emphasis, clamp: `min(max(alpha*2, 0.5), 1.0)`.
- Notebook `source` arrays: each line is a separate string. Join with `""`, not `"\n"`. Split with `.splitlines(keepends=True)`.

## USR Chi^2 Optimizer

```
scripts/usr_chi2_optimizer.py  — CLI optimizer for (phi0, y0, N_star) search

Usage (quick test):
  python scripts/usr_chi2_optimizer.py --maxiter 2 --popsize 15 --workers 4 --phi0-range 5.20 5.90 --y0-range -0.20 -0.03 --nstar-range 45 62

Usage (full search on lab machine, ~2-3 hours):
  python scripts/usr_chi2_optimizer.py --maxiter 200 --popsize 15 --workers 8 --re-run-best --save-best
```

Objective: `loss = chi² + ns_penalty + k_penalty` where:
- `ns_penalty = ((ns_MS - 0.975) / 0.01)²` (soft ACT constraint)
- `k_penalty = 50 if k_dip ∉ [1e-4, 5e-4]` (dip at ℓ≲5)

Uses `scipy.optimize.differential_evolution` (global, population-based). Logs all evaluations to JSONL. Generates convergence + comparison plots.

## Higgs USR Grid-Scan Optimizer

```
scripts/higgs_usr_optimizer.py  — Grid-scan optimizer for Higgs USR (phi0, y0) search

Usage (quick test, ~5 min):
  python scripts/higgs_usr_optimizer.py --n-phi0 5 --n-y0 5 --num-k 40 --workers 4

Usage (full scan on lab machine, ~2-4 hours):
  python scripts/higgs_usr_optimizer.py --n-phi0 20 --n-y0 20 --num-k 80 --workers 8 --plot-best
```

Deterministic grid scan over (phi0, y0). Automatically computes N_star from the background trajectory to align the USR dip with the target k_dip_range (default 1e-4 to 5e-4 Mpc^-1). Evaluates chi^2 against Planck low-ell TT data via Sachs-Wolfe approximation. Generates:
- JSONL log of all evaluations (outputs/simulations/logs/)
- Results summary JSON (outputs/simulations/scans/)
- Chi^2 and suppression heatmaps (outputs/plots/optimizer/)
- Best-config dashboard with P_S(k) and C_ell comparison (with --plot-best)
