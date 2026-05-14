# context-mode — MANDATORY routing rules

context-mode MCP tools available. Rules protect context window from flooding. One unrouted command dumps 56 KB into context.

## Think in Code — MANDATORY

Analyze/count/filter/compare/search/parse/transform data: **write code** via `context-mode_ctx_execute(language, code)`, `console.log()` only the answer. Do NOT read raw data into context. PROGRAM the analysis, not COMPUTE it. Pure JavaScript — Node.js built-ins only (`fs`, `path`, `child_process`). `try/catch`, handle `null`/`undefined`. One script replaces ten tool calls.

## BLOCKED — do NOT attempt

### curl / wget — BLOCKED
Shell `curl`/`wget` intercepted and blocked. Do NOT retry.
Use: `context-mode_ctx_fetch_and_index(url, source)` or `context-mode_ctx_execute(language: "javascript", code: "const r = await fetch(...)")`

### Inline HTTP — BLOCKED
`fetch('http`, `requests.get(`, `requests.post(`, `http.get(`, `http.request(` — intercepted. Do NOT retry.
Use: `context-mode_ctx_execute(language, code)` — only stdout enters context

### Direct web fetching — BLOCKED
Use: `context-mode_ctx_fetch_and_index(url, source)` then `context-mode_ctx_search(queries)`

## REDIRECTED — use sandbox

### Shell (>20 lines output)
Shell ONLY for: `git`, `mkdir`, `rm`, `mv`, `cd`, `ls`, `npm install`, `pip install`.
Otherwise: `context-mode_ctx_batch_execute(commands, queries)` or `context-mode_ctx_execute(language: "shell", code: "...")`

### File reading (for analysis)
Reading to **edit** → reading correct. Reading to **analyze/explore/summarize** → `context-mode_ctx_execute_file(path, language, code)`.

### grep / search (large results)
Use `context-mode_ctx_execute(language: "shell", code: "grep ...")` in sandbox.

## Tool selection

0. **MEMORY**: `context-mode_ctx_search(sort: "timeline")` — after resume, check prior context before asking user.
1. **GATHER**: `context-mode_ctx_batch_execute(commands, queries)` — runs all commands, auto-indexes, returns search. ONE call replaces 30+. Each command: `{label: "header", command: "..."}`.
2. **FOLLOW-UP**: `context-mode_ctx_search(queries: ["q1", "q2", ...])` — all questions as array, ONE call (default relevance mode).
3. **PROCESSING**: `context-mode_ctx_execute(language, code)` | `context-mode_ctx_execute_file(path, language, code)` — sandbox, only stdout enters context.
4. **WEB**: `context-mode_ctx_fetch_and_index(url, source)` then `context-mode_ctx_search(queries)` — raw HTML never enters context.
5. **INDEX**: `context-mode_ctx_index(content, source)` — store in FTS5 for later search.

## Parallel I/O batches

For multi-URL fetches or multi-API calls, **always** include `concurrency: N` (1-8):

- `context-mode_ctx_batch_execute(commands: [3+ network commands], concurrency: 5)` — gh, curl, dig, docker inspect, multi-region cloud queries
- `context-mode_ctx_fetch_and_index(requests: [{url, source}, ...], concurrency: 5)` — multi-URL batch fetch

**Use concurrency 4-8** for I/O-bound work (network calls, API queries). **Keep concurrency 1** for CPU-bound (npm test, build, lint) or commands sharing state (ports, lock files, same-repo writes).

GitHub API rate-limit: cap at 4 for `gh` calls.

## Output

Terse like caveman. Technical substance exact. Only fluff die.
Drop: articles, filler (just/really/basically), pleasantries, hedging. Fragments OK. Short synonyms. Code unchanged.
Pattern: [thing] [action] [reason]. [next step]. Auto-expand for: security warnings, irreversible actions, user confusion.
Write artifacts to FILES — never inline. Return: file path + 1-line description.
Descriptive source labels for `search(source: "label")`.

## Session Continuity

Skills, roles, and decisions persist for the entire session. Do not abandon them as the conversation grows.

## Memory

Session history is persistent and searchable. On resume, search BEFORE asking the user:

| Need | Command |
|------|---------|
| What did we decide? | `context-mode_ctx_search(queries: ["decision"], source: "decision", sort: "timeline")` |
| What constraints exist? | `context-mode_ctx_search(queries: ["constraint"], source: "constraint")` |

DO NOT ask "what were we working on?" — SEARCH FIRST.
If search returns 0 results, proceed as a fresh session.

## Environment

Conda env: `cmb-anomaly`. Activate before running any project code:

```bash
conda activate cmb-anomaly
```

Setup: `bash setup.sh`

### Package structure

Project is pip-installable (`pip install -e .`):
- `models/` — inflation model classes (Higgs, Punctuated, Quadratic, SmoothUSR)
- `scripts/` — analysis pipeline modules (pspectrum, Sachs-Wolfe, Planck data, optimizers)
- Root-level solvers (`inf_dyn_background.py`, `inf_dyn_MS_full.py`, `numerical_observables_calculation.py`) — importable globally after install
- **No `sys.path` hacks** — all 14 hacks removed; imports work from any directory

## ctx commands

| Command | Action |
|---------|--------|
| `ctx stats` | Call `stats` MCP tool, display full output verbatim |
| `ctx doctor` | Call `doctor` MCP tool, run returned shell command, display as checklist |
| `ctx upgrade` | Call `upgrade` MCP tool, run returned shell command, display as checklist |
| `ctx purge` | Call `purge` MCP tool with confirm: true. Warns before wiping knowledge base. |

After /clear or /compact: knowledge base and session stats preserved. Use `ctx purge` to start fresh.

## Core Memories — Permanent Project Rules

These rules persist across all sessions and AI tools. Do not override unless user explicitly says otherwise.

### 1. Good Runs
Only mark a run/config as "good" when the user explicitly says so (e.g. "this run is good"). Never self-declare a run successful. Never elevate a configuration based on metrics alone — user approval required.

### 2. Publication-Ready Plots
All plots must be ready for two-column publication format:
- Big fonts: axis labels ≥14pt, tick labels ≥12pt, legend ≥11pt, title ≥16pt
- 300 DPI minimum
- Proper aspect ratio: ~3.25-3.5in wide (single-column) or ~7in (full width)
- Colorblind-friendly palette (e.g., Tol, Wong, viridis)
- Minimal whitespace, tight bounding box
- Export PNG only (no PDF)

### 3. Outputs Folder Structure
Strict hierarchy — every output file goes in its correct subdirectory:

| Subdirectory | Contents |
|---|---|
| `outputs/plots/diagnostics/` | Diagnostic/debug plots (epsilon, trajectory checks) |
| `outputs/plots/optimizer/` | Optimizer iteration plots |
| `outputs/plots/powerloss/` | Power-loss mechanism plots (PS, Cℓ, suppression) |
| `outputs/plots/top30_candidates/` | Top candidate comparison plots |
| `outputs/plots/punctuated_potential/` | Punctuated inflation potential plots |
| `outputs/simulations/c_ell/` | Cℓ angular power spectra (JSON) |
| `outputs/simulations/configs/` | Configuration snapshots (JSON) |
| `outputs/simulations/logs/` | Scan logs (CSV, JSONL) |
| `outputs/simulations/pspectra/` | P_S(k) primordial power spectra (JSON) |
| `outputs/simulations/scans/` | Scan result summaries (JSON) |

When creating/running scripts, always write outputs to these directories.

### 4. Notebooks
- Place in `notebooks/` with descriptive names (e.g. `Golden_Config_Comparison.ipynb`).
- A `notebooks/outputs/` dir exists for notebook-scoped temp files.
- Generated plots go to `outputs/plots/` subdirectories, not inside notebooks.

### 5. .md Files Are Public
This repository is public. Do not write into .md files:
- API keys, tokens, credentials
- User-specific internal paths (usernames, home directories)
- Personal or sensitive data
- Embargoed/unpublished results or data
Rule of thumb: if you would not put it on arXiv, do not put it in a .md file.

### 6. Higgs-Only Scope
This project is exclusively about Higgs inflation (ξ=15000, λ=0.13) unless explicitly stated otherwise. Punctuated inflation (m=1.1323e-7, λ=3.3299e-15) is a reference model used **only** for validating solvers and cross-checking pipeline behavior — not as a primary target for analysis, optimization, or plotting. Do not run, tune, or analyze punctuated inflation unprompted.

### 7. Unit Conventions (Planck units, M_P = 1)

The code works in natural Planck units (M_P = 1). The ODE variables are:

| Attribute / Var | Meaning | Formal definition |
|---|---|---|
| `model.x0`, ODE `x` | field value (in Planck units) | `x = φ` (M_P=1, so `φ/M_P = φ`) |
| `model.y0`, ODE `y` | field velocity in code time | `y = dx/dT = φ̇ / (S·M_P²)` |
| ODE `z` | Hubble rate in code units | `z = H / S` |
| ODE `n` | log scale factor | `n = ln(a)` |
| `S` | code time scaling factor | `S = 5e-5` |
| `T` | code time | `T = S·t` (t = physical time) |
| `v0` | potential normalization | `e.g. λ/(4ξ²)` for Higgs |

When setting `model.x0 = 6.60`: initial field φ₀ = 6.60 M_P.
When setting `model.y0 = -0.736`: initial dx/dT = -0.736.

**Backward compat:** `model.phi0` is an alias for `model.x0`.

### 8. Additional Conventions
- Scripts are temporary unless user explicitly says to keep them. Delete analysis scripts after use.
- Heavy compute (scans, optimizations) runs on lab machine via `ssh uni`.
- Long-running jobs use JSONL incremental logging (crash-safe).
- Commit messages: semantic, atomic, imperative mood (e.g. "add: ...", "fix: ...", "refactor: ...").

## Project Context — Higgs USR Inflation

### Goal
Tune initial conditions (φ₀, y₀) and N_star for Higgs inflation (ξ=15000, λ=0.13) to explain the CMB low-ℓ anomaly via P_S(k) suppression.

### Physics Summary
- **Higgs USR**: Starts in kinetic dominance (ε_H=2.15 at N=0), extreme Hubble friction kills it in <0.1 e-fold. Localized dip via ε_H suppression, not a hard cutoff.
- **Punctuated Inflation** (reference model only): Creates a peak via η_H>0 amplification. Aligned at N_star=77.2 → peak at k=10⁻³. Used exclusively for solver validation and cross-checking pipeline behavior.

### Current Best Configs
- Higgs (N_star≥50): φ₀=6.60, y₀=−0.736, N_star=52.6, χ²=16.79
- Deepest dip (N_star≈38): stronger suppression but lower N_star
- Punctuated (reference only): φ₀=12.00, y₀=0.000, N_star=77.2, m=1.1323e-7, λ=3.3299e-15

### Key Constraint
Deep Higgs dips require violent kinetic kicks that shorten total inflation (N_total≈43.6). N_star≥50 configs need higher φ₀ (further on plateau) and milder kicks, which weakens the dip and raises χ².

### Reference Files
- `models/punctuated.py` — Punctuated inflaton (validation only) bg_steps=100k
- `scripts/pspectrum_pipeline.py` — Main CLI for P_S(k) pipelines
- `notebooks/Golden_Config_Comparison.ipynb` — Higgs vs Punctuated comparison

### 9. CAMB C_ell Computation
CAMB is the official Python package (`import camb`), available via pip/conda. `scripts/camb_wrapper.py` is a thin convenience layer — not a custom wrapper.
- `_make_camb_params()`: CAMBparams with Planck 2018 LCDM cosmology (H0=67.66, ombh2=0.02242, omch2=0.11933, tau=0.054, mnu=0.06)
- `compute_cl_full_camb(data)`: Inject custom P_S(k) via `set_initial_power_table()`, returns C_ell^TT/TE/EE (converted from CAMB's ℓ(ℓ+1)/(2π) convention to conventional C_ℓ)
- `compute_cl_camb_powerlaw()`: LCDM baseline via `InitPower.set_params(As=2.1e-9, ns=0.965, r=0)`
- `compute_chi2_camb(data)`: χ² vs Planck 2018 low-ℓ TT with asymmetric Commander errors
- Internally handles k-range extrapolation for CAMB spline
- Validation: `scripts/test_camb_validation.py` (7 tests, subprocess isolation for global state), `scripts/validate_camb_lcdm.py` (Planck LCDM comparison, peak~220)
- Pipeline: Inflation solver → MS solver → P_S(k) → `set_initial_power_table()` → CAMB C_ell → Planck comparison

### 10. Planck Error Bar Convention
Planck low-ℓ data plots use the **cross-correlation estimator** $\hat{C}_\ell$, which can go **negative** at low ℓ ($2\ell+1$ is small). The `plt.errorbar` call must allow this.

**Correct convention:**
```python
ax.errorbar(planck_ells, D_planck,
            yerr=[D_err_upper, D_err_lower],  # UPPER first, LOWER second
            ...)
```
Matplotlib interprets `yerr` as (2, N) where row 0 is subtracted from y (lower bar) and row 1 is added to y (upper bar). Putting the larger error first allows the downward bar to cross zero — correct for the low-ℓ estimator.

All existing plotting functions (`plotting.py`, `validate_camb_lcdm.py`, `plot_top_camb_configs.py`, `check_full_dell.py`) use this convention.

### 11. Core Solver Architecture — DO NOT MODIFY
The root-level solver files (`inf_dyn_background.py`, `inf_dyn_MS_full.py`, `pspectrum_pipeline.py`) are the physics core of the project. Do NOT move, rename, refactor, or modify these files unless explicitly asked by the user. They contain the ODE integration, Mukhanov-Sasaki solver, and pipeline orchestration that every downstream script depends on. Changes to these files can silently break every consumer without visible errors in the modified file itself.
