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
- 300 DPI minimum
- Proper aspect ratio: ~3.25-3.5in wide (single-column) or ~7in (full width)
- Colorblind-friendly palette: TOL colors from `scripts.plotting.TOL`
- Font sizes: use `scripts.plotting.PAPER_RCPARAMS` (9pt labels, 8pt ticks, 7pt legend)
- Minimal whitespace, tight bounding box
- Export PNG only (no PDF)

### 3. Outputs Folder Structure — STRICT FLAT HIERARCHY
Every output file goes in its correct subdirectory. **No per-run subdirectories.** All files are flat within each canonical dir. `scripts/plotting.py` is the single source of truth — import `OUTPUT_DIRS` or `get_path()` instead of hardcoding paths.

| Subdirectory | Contents |
|---|---|
| `outputs/plots/diagnostics/` | Diagnostic/debug plots (epsilon, trajectory checks, background dashboards) |
| `outputs/plots/powerloss/` | Power-loss mechanism plots (Dℓ, suppression per config) |
| `outputs/plots/pspectra/` | P_S(k) power spectrum plots |
| `outputs/plots/optimizer/` | Optimizer iteration plots |
| `outputs/plots/paper/` | Final publication-ready plots |
| `outputs/simulations/c_ell/` | Cℓ angular power spectra (JSON) |
| `outputs/simulations/configs/` | Background trajectory snapshots (JSON) |
| `outputs/simulations/logs/` | Scan/optimizer logs (CSV, JSONL) |
| `outputs/simulations/pspectra/` | P_S(k) primordial power spectra (JSON) |
| `outputs/simulations/scans/` | Scan result summaries (JSON) |
| `outputs/archive/` | Legacy/orphaned content (best_candidates, top30, punctuated_potential, old PDFs) |

**Rules:**
- Use `scripts.plotting.OUTPUT_DIRS` or `scripts.plotting.get_path()` — never hardcode path strings
- Use `scripts.plotting.make_filename()` for all output filenames — never manually construct paths
- Only PNG output (no PDFs)
- No per-config subdirectories — all files flat within each canonical dir
- **Never track or commit files in `outputs/` or `notebooks/`.** These directories are gitignored. Do not use `git add -f` to bypass this.

**Naming Convention — ALL scripts and notebooks MUST use `scripts.plotting.make_filename()`:**

| Type | Prefix | Pattern | Example |
|------|--------|---------|---------|
| P_S(k) JSON | `ps` | `ps_phi{phi0}_y0{y0}_nstar{nstar}.json` | `ps_phi6.60_y0-0.736_nstar52.6.json` |
| C_ell JSON | `camb` | `camb_phi{phi0}_y0{y0}_nstar{nstar}.json` | `camb_phi6.60_y0-0.736_nstar52.6.json` |
| Background config | `config` | `config_phi{phi0}_y0{y0}_nstar{nstar}.json` | `config_phi6.60_y0-0.736_nstar52.6.json` |
| Background plot | `bg` | `bg_phi{phi0}_y0{y0}_nstar{nstar}.png` | `bg_phi6.60_y0-0.736_nstar52.6.png` |
| P_S(k) plot | `ps` | `ps_phi{phi0}_y0{y0}_nstar{nstar}.png` | `ps_phi6.60_y0-0.736_nstar52.6.png` |
| D_ell plot | `dell` | `dell_phi{phi0}_y0{y0}_nstar{nstar}.png` | `dell_phi6.60_y0-0.736_nstar52.6.png` |
| CAMB comparison | `camb` | `camb_phi{phi0}_y0{y0}_nstar{nstar}.png` | `camb_phi6.60_y0-0.736_nstar52.6.png` |
| Planck comparison | `planck` | `planck_phi{phi0}_y0{y0}_nstar{nstar}.png` | `planck_phi6.60_y0-0.736_nstar52.6.png` |

- y0 format: `y0-0.736` (negative), `y0+0.100` (positive) — sign always explicit
- Special files (no config): `camb_lcdm.*`, `pipeline_sanity.*`, `camb_lcdm_validation.*`
- Comparison plots: `{type}_comparison_{label}.{ext}` — e.g. `ps_comparison_top5.png`
- **NEVER** use random hashes, redundant model names, or inconsistent prefixes
- **NEVER** hardcode `outputs/` or filenames: use `get_path()` + `make_filename()`

### 4. Notebooks (Deprecated)
- Notebooks are deprecated and ignored by version control (`notebooks/*.ipynb` is in `.gitignore`).
- All active development has transitioned to a script-only workflow.
- Do not add new notebooks to version control.

### 5. .md Files Are Public
This repository is public. Do not write into .md files:
- API keys, tokens, credentials
- User-specific internal paths (usernames, home directories)
- Personal or sensitive data
- Embargoed/unpublished results or data
Rule of thumb: if you would not put it on arXiv, do not put it in a .md file.

### 6. Higgs and Ezquiaga Scope

The project covers two inflation models:
- **Higgs inflation** (ξ=15000, λ=0.13) — primary target for CMB low-ℓ anomaly analysis.
- **Ezquiaga CHI** (`models/ezquiaga_chi.py`) — critical Higgs inflation with RG-running λ(ξ), PBH-focused, validated against paper reference.

Punctuated inflation (m=1.1323e-7, λ=3.3299e-15) is a reference model used **only** for validating solvers and cross-checking pipeline behavior — not as a primary target for analysis, optimization, or plotting. Do not run, tune, or analyze punctuated inflation unprompted.

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
- **No one-off analysis scripts in `scripts/`.** `scripts/` is for importable modules, pipelines, and reusable tools. One-off experiments belong in:
  1. `notebooks/` as Jupyter notebooks (preferred)
  2. `outputs/archive/` as standalone `.py` files ONLY for reproducibility (clearly labelled)
- **Delete analysis scripts immediately after use.** If a one-off script was written to explore data or test a hypothesis, delete it from git before committing the results. Use Jupyter notebooks in `notebooks/` for transient analysis instead.
- **Before adding a new file to `scripts/`, ask:** Is this importable by other code? If no, it doesn't belong here. Put it in a notebook or `outputs/archive/`.
- **Never commit temp scripts.** If you wrote `scripts/frobnicate_widgets.py` to test an idea, delete it before `git commit`. The idea that survives becomes a proper module or gets documented in AGENTS.md.
- **Never auto-commit or auto-push.** Always ask for explicit approval before any git commit or push.
- **Never touch `paper/images/` or `paper/` unless user explicitly asks.** Plots live in `outputs/plots/`. Only copy to `paper/images/` when user specifically requests it.
- Heavy compute (scans, optimizations) runs on lab machine via `ssh uni`. Lab machine project path: `~/Documentos/CMB_USR/`. Sync only via GitHub push/pull — never rsync the full project.
- **Lab execution pattern (prevents SSH hangs):**
  1. Write script locally, commit+push to GitHub
  2. `ssh uni "cd ~/Documentos/CMB_USR && git pull && source ~/miniconda3/etc/profile.d/conda.sh && conda activate cmb-anomaly && nohup python script.py > ~/jobname.log 2>&1 & echo PID=\$!"`
  3. Track with SHORT timeouts (10-15s): `ssh uni "grep -c 'pattern' ~/jobname.log; tail -3 ~/jobname.log"`
  4. Do NOT use `sleep N && ssh ...` — blocks indefinitely. Instead use polling with short timeouts.
  5. Check completion: `ssh uni "ps aux | grep script.py | grep -v grep | wc -l"`
  6. Results are in JSONL logs under `outputs/simulations/logs/` on the lab. Parse with a script copied via `scp`.
- Long-running jobs use JSONL incremental logging (crash-safe).
- Commit messages: semantic, atomic, imperative mood (e.g. "add: ...", "fix: ...", "refactor: ...").

## Project Context — Higgs USR Inflation

### Goal
Tune initial conditions (φ₀, y₀) and N_star for Higgs inflation (ξ=15000, λ=0.13) to explain the CMB low-ℓ anomaly via P_S(k) suppression.

### Physics Summary
- **Higgs USR**: Starts in kinetic dominance (ε_H=2.15 at N=0), extreme Hubble friction kills it in <0.1 e-fold. Localized dip via ε_H suppression, not a hard cutoff.
- **Punctuated Inflation** (reference model only): Creates a peak via η_H>0 amplification. Aligned at N_star=77.2 → peak at k=10⁻³. Used exclusively for solver validation and cross-checking pipeline behavior.

### Current Best Configs (full-resolution, corrected)
- **Best χ²** (6.40,−0.475,59): χ²_full=2574.2 (+1.2 vs LCDM), D₂=918 μK² (11%↓), supp=31%. Matches LCDM essentially perfectly.
- **Best D₂** (5.75,−0.170,55): χ²_full=2613.8 (+40.7), D₂=677 μK² (34%↓), supp=39%. Best quadrupole suppression.
- **Best balance** (5.70,−0.170,52): χ²_full=2582.6 (+9.6), D₂=847 μK² (18%↓), supp=36%. Good χ² + meaningful D₂ suppression.
- Punctuated (reference only): φ₀=12.00, y₀=0.000, N_star=77.2, m=1.1323e-7, λ=3.3299e-15

### Key Constraint
USR suppression at CMB scales requires fine-tuned initial conditions. The mechanism works (D₂ down 34% at cost of +41 χ²) but no config outperforms LCDM across the full spectrum. Deep dips (D₂<700) come at higher χ²_full cost.

### Reference Files
- `models/punctuated.py` — Punctuated inflaton (validation only) bg_steps=100k
- `scripts/pspectrum_pipeline.py` — Main CLI for P_S(k) pipelines
- `scripts/test_camb_validation.py` — Higgs vs Punctuated validation comparison

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
Planck low-ℓ data (`data/Planck/planck_2018_low_ell_tt.csv`) stores asymmetric
errors as positive magnitudes: `D_err_lower` (amount to subtract) and
`D_err_upper` (amount to add).

**Correct matplotlib convention:**
```python
ax.errorbar(planck_ells, D_planck,
            yerr=[D_err_lower, D_err_upper],  # LOWER first (subtracted), UPPER second (added)
            ...)
```
Matplotlib interprets `yerr` as (2, N) where row 0 is subtracted from y and
row 1 is added to y. Both values are positive magnitudes from the CSV.

**χ² computation:** When computing asymmetric χ², select the error based on
the sign of the residual: use `D_err_upper` if model > data, `D_err_lower`
if model < data. This is already correct in `camb_wrapper.py` and
`check_full_dell.py`.

### 10.5 Pivot Convention — Single k_pivot for As and n_s

The project has **one pivot** — `k_pivot_phys` — that drives BOTH the A_s
normalization (`P_S(k_pivot) = A_s`) AND the n_s extraction (least-squares
fit of ln P_S vs ln k over `[k_pivot/ns_window, k_pivot*ns_window]`). The
fit half-width `ns_window` is the only separate knob.

**Two workflows, two defaults, both user-selectable:**

| Workflow | Default k_pivot | Default ns_window | Fit window |
|----------|---------------|------------------|------------|
| Higgs / power suppression | 0.002 Mpc⁻¹ | 4.0 | [5×10⁻⁴, 8×10⁻³] |
| Ezquiaga / PBH | 0.05 Mpc⁻¹ | 3.0 | [0.017, 0.15] |

**CLI flags (on every script that uses the pivot):**
```
--k-pivot FLOAT                   # drives BOTH As normalization and n_s extraction
--ns-window FLOAT                 # n_s fit half-width [k_pivot/w, k_pivot*w]
--ns-method {lsq,derivative}      # n_s extraction method: lsq=window fit (default), derivative=log-derivative at k_pivot
```
`sweep_pbh_params.py` also keeps `--pivot-k` as a deprecated back-compat
alias for `--k-pivot`.

**Config JSON** (optional, in the `pipeline` block):
```json
"pipeline": { "k_pivot_phys": 0.05, "ns_window": 3.0, ... }
```

**Precedence:** `--k-pivot` CLI > config `pipeline.k_pivot_phys` > script default.

> **ns_method default:** `lsq` (window-averaged fit over `[k_pivot/ns_window, k_pivot*ns_window]`). Use `derivative` for the logarithmic derivative at k_pivot, which is more sensitive to local features.

**Single-pivot invariant:** the SAME `k_pivot` value feeds both the pipeline
call (`k_pivot_phys=...`) and `extract_ns(k_pivot=...)`. Never normalize A_s
at one k and extract n_s at another in the same run.

**n_s extraction lives in `scripts/observables.py`:**
- `extract_ns(k_phys, P_S, k_pivot, ns_window)` — the MS-based n_s (uses P_S)
- `extract_pbh_peak(k_phys, P_S)` — PBH peak (small scales, NOT an n_s)
- SR algebraic n_s (`1 + 2η_H − 4ε_H` at N_pivot, in `background_scan.py` as
  `n_s_sr_formula`) is a DIFFERENT observable — it does not use P_S(k)

**Output JSON records the pivot:** every n_s-bearing output carries
`{k_pivot, ns_window, n_modes, k_range, method}` in metadata. JSONL scan
logs include `k_pivot` and `ns_window` per record.

**Refactor design doc:** `docs/ns_extraction_refactor.md`

### 11. Core Solver Architecture — DO NOT MODIFY
The root-level solver files (`inf_dyn_background.py`, `inf_dyn_MS_full.py`, `pspectrum_pipeline.py`) are the physics core of the project. Do NOT move, rename, refactor, or modify these files unless explicitly asked by the user. They contain the ODE integration, Mukhanov-Sasaki solver, and pipeline orchestration that every downstream script depends on. Changes to these files can silently break every consumer without visible errors in the modified file itself.

### 12. Best Config — χ²-Competitive Suppression

After the `find_end_of_inflation` fix (forward-scan with permanence check), no Higgs USR config outperforms LCDM across the full spectrum. The best configs achieve significant D₂ suppression at modest χ² cost:

| Config | χ²_full (ℓ=2-2508) | D₂ [μK²] | Suppression | Δχ² vs LCDM |
|--------|-------------------|-----------|-------------|-------------|
| 6.40,−0.475,59 | 2574.2 | 918 (−11%) | 31% | +1.2 |
| 5.70,−0.170,52 | 2582.6 | 847 (−18%) | 36% | +9.6 |
| 5.75,−0.170,55 | 2613.8 | 677 (−34%) | 39% | +40.7 |
| 6.55,−0.780,50 | 2637.6 | 835 (−19%) | 47% | +64.6 |
| LCDM | 2573.0 | 1029 | — | — |

**Diagnostic script:** `scripts/run_full_analysis.py` — runs full pipeline, produces broken-axis D_ℓ plot with Planck data.
**Quick scan:** `python scripts/camb_scan.py --phase broad --quick --full-chi2` (~20 min).

**Planck data files:** Downloaded from IRSA (R3.01/R3.02), stored in `data/Planck/`:
- Binned TT/TE/EE spectrum (ℓ≈47-2500)
- Unbinned TT/TE/EE spectrum (ℓ=2-2508)
- Low-ℓ Commander data (ℓ=2-29)

### 13. Background Evolution — Standard Higgs vs Ezquiaga CHI

**Standard Higgs** (`models/higgs.py:HiggsModel`): ODE variable `x = φ` (Jordan frame). Potential `f(x) = (1 - e^{-αx})²`, monotonic decreasing, no features. USR is **kinetic-driven** — comes from initial `y₀` (small `|y₀|` causes a freeze). Kinetic dominance at start (`ε_H≈3`), Hubble friction kills velocity in <0.1 e-fold, field freezes (`ε_H` dips to ~10⁻³), then catches SR attractor. Tunable via `y₀`.

**Ezquiaga CHI** (`models/ezquiaga_chi.py:EzquiagaCHIModel`): Physics described in Jordan frame with `x = φ/μ`. ODE integration uses canonically normalized Einstein frame field χ (numerical spline φ↔χ). Potential (Eq. 6) has a near-inflection from RG running of both λ(x) and ξ(x). Plot potential as V/V₀ vs `x = φ/μ` (paper convention). USR is **potential-driven** — η_H crosses zero from the inflection. Default χ₀=6.0 maps to φ₀≈1.94 M_P. NaN-patches required because field can stall near inflection.

**Why χ exists:** The Jordan frame action has a non-minimal coupling ξ(φ)φ²R. Conformal transforming to the Einstein frame makes gravity canonical but the scalar kinetic term becomes non-canonical. The field redefinition φ→χ (Eq. 5) absorbs this back into a canonical `-½(∂χ)²`. This lets the ODE solver use the standard Klein-Gordon equation `χ'' + 3Hχ' + V'(χ) = 0` without knowing about ξ, the conformal factor Ω, or the transformation chain. The φ↔χ spline is purely computational plumbing — all physics (potential shape, inflection, slow-roll) is in the Jordan frame x = φ/μ.

Parameters (paper's published, ROUNDED):
- `λ₀ = 2.23×10⁻⁷`, `b_λ = 1.2×10⁻⁶` → `a = 5.3812`
- `ξ₀ = 7.55`, `b_ξ = 11.5` → `b = 1.5232`
- `c = 0.77`, `x_c = 0.784`, `κ²μ² = 0.102`

**CRITICAL:** The published values are rounded and do NOT satisfy the inflection conditions (paper's Eqs. a-b). For x_c=0.784, c=0.77, the exact inflection values are:
- `a_exact = 5.335304` (0.86% lower than published)
- `b_exact = 1.519340` (0.25% lower than published)

The paper's rounded parameters give β≈−0.018 (negative — creates a bump, not an inflection), so the field gets trapped at a local minimum. Use `inflection_parameters(x_c, c, β)` from `models/ezquiaga_chi.py` to compute self-consistent values:

```python
from models.ezquiaga_chi import inflection_parameters
a, b = inflection_parameters(0.784, 0.77, beta=1e-5)
# a=5.33530386, b=1.51932465
b_lambda = a * lambda_0   # e.g. 1.190e-6 for lambda_0=2.23e-7
b_xi = b * xi_0           # e.g. 11.471 for xi_0=7.55
```

**Working configurations** (paper's reference: β=10⁻⁵, x_c=0.784, c=0.77):

| χ₀ | φ₀ (M_P) | N_total | k=0.002 at χ | n_s (k=0.002) | n_s (k=0.05) | Match to ref |
|----|----------|---------|-------------|---------------|--------------|-------------|
| 6.5 | 2.31 | 64.1 | 4.82 | 0.85 | 0.79 | Too red, CMB off plateau |
| 7.5 | 3.27 | 78.6 | 6.40 | 0.940 | 0.931 | Close |
| **8.0** | **3.90** | **87.4** | **7.06** | **0.957** | **0.952** | **Matches paper n_s=0.952** |
| 8.5 | 4.69 | 97.0 | 7.69 | 0.967 | 0.964 | Slightly too blue |

With χ₀=8.0, k=0.05 (Planck pivot) at χ≈6.8 gives n_s=0.952 — exact match to paper's reference. The paper's N=65 e-folds and n_s=0.952 require enough plateau before the inflection (χ₀≥8) so CMB scales stay on the flat part.

**Critical: pivot finding by k = a·H, not hardcoded N=55.** The standard N=55 convention assumes a specific expansion history that doesn't hold here. The inflection steals ~33.5 e-folds (ΔN), shifting the k↔N mapping. Always find the pivot by matching k = a·H to the target scale.

**Diagnostic plots**: `scripts.plotting.plot_ezquiaga_diagnostics()` — N-χ, V/V₀ vs x, P_S(N).
**SR vs MS comparison**: `scripts.plotting.plot_ps_sr_ms_comparison()` — P_S(k) overlay with ratio panel.
Pivot found by k=a·H.

**Key contrast**: Standard Higgs USR = initial-condition effect (tune `y₀`). Ezquiaga USR = structural (near-inflection from RG running). PBH-focused — the peak is at small scales (k∼10¹⁴ Mpc⁻¹) for PBH formation (0.01-100 M_⊙).### 15. MS n_s Oscillation vs Smooth SR — Physics, Not Numerical

When sweeping x₀ at fixed y₀, SR n_s varies **monotonically** while MS n_s shows
a small **oscillation** (~0.008 amplitude, ~0.004 x₀ period). This is real
physics, not a solver artifact.

**Cause**: SR evaluates n_s = 1 + 2η_H − 4ε_H at a **single N** (N_pivot).
MS fits the slope of P_S(k) across ~11 k-modes spanning ~1 decade. Each k-mode
freezes at a different N_exit. As x₀ shifts N_pivot, the mapping between
physical k and N_exit shifts — the same k-range samples a slightly different
N-range. The spectral index has running (α_s = dn_s/d ln k), so the average
slope across the window varies. In the transient region (near the breakdown of
SR), α_s ≈ O(0.5), producing the observed oscillation.

SR never sees this: it samples one N, one formula, no running.

**Numerical sanity checks (all negative)**:
- `k_start_factor` variation (10, 100, 1000) → identical n_s (BD error negligible)
- CubicSpline vs quintic z-spline → same oscillation (not from z''/z kinks)
- CubicSpline vs linear interp → oscillation disappears but so does sensitivity
  to real P_S variations (linear is too insensitive)
- Natural vs not-a-knot BC → same oscillation (not from boundary conditions)
- bg_steps 1000 vs 10000 → same oscillation (not from grid resolution)
- Perfectly reproducible: same x₀ gives same n_s to float64 precision

### 16. Figure 3 Reproduction — PBH Abundance (1705.04861)

**Config:** `configs/ezquiaga_fig3.json`
**Plot:** `outputs/plots/pbh/pbh_phi8.00_y0-0.000_nstar87.4.png`
**Run:** `python scripts/pspectrum_pipeline.py --config configs/ezquiaga_fig3.json && python scripts/pbh_abundance.py --ms-json outputs/simulations/pspectra/ps_phi8.00_y0-0.000_nstar65.0.json --zeta-c 0.077`

**All Ezquiaga-related outputs archived in:** `outputs/Ezquiaga/` (with README)

**Parameter comparison vs paper reference:**

| Parameter | Paper Ref | Our Fig3 Config | Δ |
|-----------|-----------|-----------------|---|
| **x_c** (critical point) | 0.784 | 0.784 | — |
| **c** (ξ₀κ²μ²) | 0.77 | 0.77 | — |
| **β** (deviation from inflection) | 10⁻⁵ | 10⁻⁵ | — |
| **χ₀** (initial field) | 8.0 | 8.0 | — |
| **y₀** (initial velocity) | ~0 | -10⁻⁴ | — |
| **λ₀** | 2.23×10⁻⁷ | 2.23×10⁻⁷ | — |
| **ζ_c** (collapse threshold) | 0.052 | **0.077** | +48% |
| **γ** (efficiency) | 0.4 | 0.4 | — |
| **Accretion factor** | 3×10⁷ | 3×10⁷ | — |
| **N_star** | **65** | **65** | ✓ (same) |
| **k_pivot** | **0.05 Mpc⁻¹** | **0.05 Mpc⁻¹** | ✓ (same) |

**Result comparison:**

| Metric | Paper Ref | Our code | Δ |
|--------|-----------|----------|---|
| **Ω_PBH^eq** | **0.42** | **0.27** | 1.6× |
| **μ (peak M_present)** | **~11 M_⊙** | **0.4 M_⊙** | 28× |
| **P_S_peak** | **4.8×10⁻⁵** | **1.04×10⁻⁴** | 2.2× |
| **P_S_peak/As** | **2.3×10⁴** | **5.0×10⁴** | 2.2× |

**Key notes:**
- The paper's **rounded** parameters (a=5.381, b=1.523) give β≈-0.018 (bump, not inflection → field stalls, N_total=182). All runs use `inflection_parameters()` for self-consistent (a, b).
- Our MS solver with the same stated parameters gives the same P_S amplitude (1.04×10⁻⁴ peak) but places the peak at k ≈ 3×10¹⁰ Mpc⁻¹, not k ≈ 6×10⁹. This is a solver implementation difference.
- ζ_c=0.077 is within the paper's stated uncertainty range ζ_c ∈ (0.05, 1) [Sec III].
- Archive at `outputs/Ezquiaga/` contains configs, plots, MS outputs, sweep logs.

### 17. No Inline Python Code

**NEVER** run inline `python -c "..."` or `python <<EOF` for physics analysis. It is non-reproducible, un-tracked, and un-reviewable. Use one of:
- **Config file** + `pspectrum_pipeline.py` for MS computation
- **`scripts/pbh_abundance.py --ms-json`** for PBH abundance
- **`scripts/sweep_pbh_params.py`** for parameter sweeps
- **`scripts/plotting.py`** for plotting

The one exception: short (≤5 line) diagnostics to check file contents or list directories. Any physics computation must use the proper scripts.

### 18. Ezquiaga PBH Mass Shift — LIGO Constraint

The Ezquiaga CHI paper's reference configuration (x_c=0.784, c=0.77, β=10⁻⁵) produces PBHs with present-day mass ~0.4-11 M_⊙ (stellar range), which is **ruled out by LIGO** bounds on PBH dark matter in the 1-100 M_⊙ range.

**Primary project goal for Ezquiaga:** Find parameters that shift the PBH mass distribution to **lower masses** (higher k_peak), targeting the sub-solar gap [10⁻⁶, 10⁻²] M_⊙ or the asteroid gap [10⁻¹⁷, 10⁻¹⁵] M_⊙. These mass ranges are not ruled out by current observations.

**n_s compatibility with Planck is secondary.** The priority is mass range placement, not spectral index fitting.

**Mass ↔ k_peak mapping** (with accretion factor 3×10⁷):

| Target | M_present [M_⊙] | k_peak [Mpc⁻¹] |
|--------|-----------------|----------------|
| LIGO range (ruled out) | 0.1-100 | 2×10⁹–6×10¹⁰ |
| Sub-solar gap | 10⁻⁶–10⁻² | 2×10¹¹–2×10¹³ |
| Asteroid gap | 10⁻¹⁷–10⁻¹⁵ | 6×10¹⁷–6×10¹⁸ |

### 19. Empirical Results from Systematic Sweeps

#### Parameter trends

| Trend | Effect on k_peak | Effect on n_s |
|-------|-----------------|---------------|
| ↑ xc | ↑ higher k (lower M, left) | ↑ bluer |
| ↓ xc | ↓ lower k (higher M, right) | ↓ redder (fails near b<0 bound) |
| ↑ c | ↑ higher k (lower M, left) | ↓ redder (asymptotes ~1.018) |
| ↑ β | ↓ lower k (higher M, right) or kills peak | ↓ redder |
| ↑ N_star | ↑ higher k (lower M, left) | ↑ bluer |
| ↑ χ₀ | saturates for N_total > 165 | tiny effect |

#### Note on x_c effect

At first glance, higher x_c should mean the inflection is reached **sooner** (fewer e-folds from start), so the scales exiting at the inflection should be **larger** (higher mass). However, our sweeps show higher x_c → **higher k (lower mass)**. This is because changing x_c also changes `a` and `b` via `inflection_parameters(x_c, c, beta)`, which redesigns the entire potential — not just shifts the inflection position. The plateau gets qualitatively longer/steeper at higher x_c, which dominates over the simple field-position argument. Two competing effects:

1. *Naive effect:* Higher x_c → inflection reached sooner → larger scales (higher M)
2. *Potential reshaping effect:* Higher x_c → (a, b) change → plateau stretches → more e-folds → smaller scales (lower M)

Effect 2 dominates in our model.

#### USR peak existence criterion

A real USR peak (k_peak > 1e6) appears when **BOTH** conditions hold:
1. N_total > 165 (sufficient e-folds)
2. β < β_critical ≈ 4×10⁻⁴ (at c=1.86, xc=0.79), where β_critical depends on (xc, c)

The physical threshold is the residual slope V'(x_c) at the inflection:
- V'(xc) < ~5×10⁻⁵ → USR peak forms
- V'(xc) > ~7×10⁻⁵ → no USR peak

β controls this slope linearly: V'(xc) ≈ 1.4 × 10⁻⁴ × (β/9×10⁻⁴) at (xc=0.79, c=1.86).

#### High-c, high-β regime

At c=1.86, the plateau is very stretched and the inflection is at a different position in field space. This gave the first **resolved** (non-grid-boundary) peak at k=9.12×10¹⁷ with:
- β=3e-4 → n_s=1.012, asteroid peak at k=9.1e17, M=4.7e-16 M_⊙
- β=5e-4 → n_s=1.000, no USR peak
- β=9e-4 → n_s=0.966, no USR peak

#### What β actually does

β creates a **positive slope** at the inflection point x_c in the potential V(x):
- β=0: V'(xc) ≈ 0 (exact inflection, field stalls → strong USR → asteroid peak)
- β=3e-4: V'(xc) ≈ +4.2×10⁻⁵ (weak USR → weak peak at k=9e17)
- β=9e-4: V'(xc) ≈ +1.25×10⁻⁴ (no USR → no peak, field rolls through)

The potential value V(x_c) changes by only 0.02% across the full β range. The slope at x_c is the key parameter.

**N_star is the dominant knob for PBH mass targeting.** β controls USR strength (peak amplitude), but N* shifts the entire P_S(k) along the k-axis via the k↔N pivot mapping. δ(N*) = +1 shifts k_peak by ×e ≈ ×2.7. The difference between sub-solar (k~10¹¹) and asteroid (k~10¹⁸) masses is δ(N*) ≈ +6 at fixed β. β fine-tunes which specific mass bin within the target regime — N* selects the regime.

### 20. Ezquiaga SM-Allowed Parameter Ranges (from 1705.04861)

From paper lines 302-303 (ΔN ∈ (30,35) for viable PBH production):

| Parameter | Paper Ref | SM-Allowed Range | Derived |
|-----------|-----------|------------------|---------|
| λ₀ | 2.23×10⁻⁷ | (0.01–8)×10⁻⁷ | Higgs quartic at critical scale |
| ξ₀ | 7.55 | **0.5–15** | Non-minimal coupling |
| κ²μ² | 0.102 | **0.05–1.2** | Critical scale squared |
| b_λ | 1.2×10⁻⁶ | (0.008–4)×10⁻⁶ | β_λ running coefficient |
| b_ξ | 11.5 | **1–18** | β_ξ running coefficient |
| **c = ξ₀·κ²μ²** | **0.77** | **[0.025, 18]** | Combined: 0.5×0.05 ≤ c ≤ 15×1.2 |
| β | 10⁻⁵ | **(0.1–9)×10⁻⁴** | From Fig 2 (n_s, r plane) |
| ΔN | 33.5 | **10–45** | From Fig 2 right panel |

**Note:** The paper constrains these to ΔN ∈ (30,35) for "large PBH production." Our solver shows viable USR peaks at lower ΔN as well, so this range is a guide, not a hard limit.

**Sweep coverage of allowed parameter space:**

| Parameter | Allowed | Swept | Fraction |
|-----------|---------|-------|----------|
| c | [0.025, 18] | [0.5, 10] | ~50% |
| β | [10⁻⁶, 9×10⁻⁴] | [10⁻⁶, 9×10⁻⁴] | **100%** |
| x_c | ~[0.75, 0.85] | [0.75, 0.85] | **100%** |
| χ₀ | > x_c | [4.0, 8.0] | partial |
| N_star | [50, 70] | [50, 70] | **100%** |

**Empirical from our sweeps (updated):**
- USR peak appears only when N_total > 165 AND β < β_critical (depends on xc, c)
- n_s asymptotes toward ~1.018 for very high c (5.0+), never crossing below 1
- n_s can cross below 1 only when USR peak is absent (β > β_critical)
- k_peak = 1e18 (asteroid) is stable across c ∈ [0.77, 5.0] at xc≥0.79 (grid boundary)
- First resolved (non-grid-boundary) peak at k=9.1×10¹⁷ at c=1.86, β=3e-4, xc=0.79
- The search plan is documented in `docs/pbh_search_plan.md`

### 21. Best PBH Configs — Sub-solar & Asteroid

Two independently-verified configs producing real (non-boundary) USR peaks with
clean observational-constraint fits, companion JSONs auto-generated on plot output:

| Region | Config | M_peak [M⊙] | f_total | ζ_c | n_s | File |
|--------|--------|-------------|---------|-----|-----|------|
| **Sub-solar** | β=2e-5, N*=66 | 1.97e-05 | 0.183 | 0.0765 | 0.9501 | `configs/subsolar_pbh.json` |
| **Asteroid** | β=1.8e-4, N*=72 | 1.29e-16 | 0.128 | 0.0488 | 0.9663 | `configs/asteroid_pbh.json` |

Both at χ₀=8.0, x_c=0.784, c=0.77. Reproduce with:
```bash
python scripts/full_pbh_pipeline.py --config configs/subsolar_pbh.json --tag rank02
python scripts/full_pbh_pipeline.py --config configs/asteroid_pbh.json --tag rank07
```

### 22. Ezquiaga Parameter Relationships & Config Structure

**Fundamental potential parameters:**
- `a = b_λ / λ₀`, `b = b_ξ / ξ₀` — dimensionless RG ratios. These define the potential shape.
- `λ₀`, `ξ₀` — absolute scale. `λ₀=2.23e-7, ξ₀=7.55` are fixed from SM RG running (paper values).
- `c = ξ₀·κ²μ²` — plateau width. This is the tunable scale parameter.
- `V₀ = λ₀·μ⁴/4` — overall potential energy scale.

**Two ways to specify a config:**

1. **Raw RG coefficients** (`paper.json`): store `b_λ, b_ξ, λ₀, ξ₀, c` directly. Constructor computes `a=b_λ/λ₀`, `b=b_ξ/ξ₀`. No inflection block. This is what the paper literally published — the numbers produce a local minimum (β≈−0.018), not an inflection. Field stalls at N≈182.

2. **Inflection parametrization** (all other configs): store `x_c, c, β`. Pipeline calls `inflection_parameters(x_c, c, β)` which computes the exact `a, b` that satisfy V'(x_c)=V''(x_c)=0 (for β=0) or a controlled deviation (β>0). These override whatever the constructor computed from `b_λ`/`b_ξ`. The `b_λ`/`b_ξ` defaults are irrelevant in this path.

**Key consequence:** The `inflection` parametrization assumes `a = a_exact(x_c,c)`. It ONLY varies `b` via `b = (1-β)·b_exact`. If both `a` and `b` are wrong (as in the paper's published numbers), this parametrization cannot represent them — you need the raw RG path instead.

**Config directory structure:**
- `configs/ezquiaga/` — single-run configs, all in canonical nested format.
  - `paper.json` — raw RG path (NO inflection block). Stalls. Documents the paper's literal published numbers.
  - All others — inflection path (HAS inflection block). Produce USR. Differ only in `c` and `β`.
- `configs/sweeps/pbh/` — grid sweep configs for `sweep_pbh_params.py` (different flat schema).

**What actually varies across working configs:**
| Config | c | β | a_eff | b_eff | b_λ_eff | b_ξ_eff |
|--------|---|---|---|---|---|---|
| beta1e-5 | 0.77 | 1e-5 | 5.335304 | 1.519325 | 1.190e-6 | 11.471 |
| perfect | 0.77 | 0 | 5.335304 | 1.519340 | 1.190e-6 | 11.471 |
| tweaked | 0.771 | 4e-5 | 5.330933 | 1.517840 | 1.189e-6 | 11.460 |
| subsolar | 0.77 | 2e-5 | 5.335304 | 1.519339 | 1.190e-6 | 11.471 |
| asteroid | 0.77 | 1.8e-4 | 5.335304 | 1.519334 | 1.190e-6 | 11.471 |

`λ₀=2.23e-7, ξ₀=7.55` are universal across all. `a_eff` barely varies (only via `c`). The radical physics differences come from `b_eff` at the 6th decimal (controlled by `β`).

### 23. Fortran MS Solver Backend

The Hot Path comoving MS grid integration is ported to native Fortran 90 (`fortran/ms_solver.f90`) with OpenMP multi-threaded parallelization over comoving modes.
- Python bridge: `fortran_ms_solver.py` converts background splines to Fortran memory order (`order='F'`) and calls the library.
- Compilation: Compiled via `f2py` with Meson/Ninja toolchain in `cmb-anomaly` conda env. Make command: `cd fortran && make`.
- Linking: Requires `LDFLAGS="-fopenmp"` to resolve OpenMP runtime linker symbols.
- Execution: Activated via `--backend fortran` flag in `pspectrum_pipeline.py`.
- Validation: `fortran/test_vs_numba.py` checks single-mode trajectory correctness, full comoving grid agreement across three key configurations (within relative difference < 1e-4), and CAMB observable compatibility.
