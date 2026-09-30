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
- **Higgs inflation** (ξ=15000, λ=0.13) — primary target for CMB low-ℓ anomaly analysis. This file.
- **Ezquiaga CHI** (`models/ezquiaga_chi.py`) — critical Higgs inflation with RG-running λ(ξ), PBH-focused. See `ezquiaga/AGENTS.md`.

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
- Heavy compute (scans, optimizations) runs on lab machine via `ssh uni`. **Lab machine project path: `~/Projects/CMB_Anomaly/`** (NOT `~/Documentos/CMB_USR/` — that's the old machine). Sync only via GitHub push/pull — never rsync the full project.
- **Conda is auto-activated on uni — no `source`/`conda activate` prefix needed.** Symlinks in `~/.local/bin` (already first in PATH) point `python`/`pip`/`f2py` at the `cmb-anomaly` env. The project is pip-installed in that env, so modules import from anywhere.
- **No `cd` required.** `ROOT_DIR` is derived from `__file__` (file location), NOT cwd, so `get_path()`/`OUTPUT_DIRS` outputs land in `~/Projects/CMB_Anomaly/outputs/` no matter where the command runs. You can launch from `$HOME` or `/tmp`. Only caveat: pass **absolute paths** for `--config` and `--output-dir` (those are the only cwd-relative args).
- **Lab execution pattern (prevents SSH hangs):**
  1. Write script locally, commit+push to GitHub
  2. `ssh uni "cd ~/Projects/CMB_Anomaly && git pull && nohup python script.py > ~/jobname.log 2>&1 & echo PID=\$!"`
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
- **Best χ²** (6.40,−0.475,59): χ²_full=2574.2 (+1.2 vs LCDM), D₂=997 μK² (3%↓), supp=31%. Matches LCDM essentially perfectly.
- **Best D₂** (5.75,−0.170,55): χ²_full=2613.8 (+40.7), D₂=776 μK² (24%↓), supp=39%. Best quadrupole suppression (true ℓ=2 value; the older quoted 677 μK² was the ℓ=4 multipole — see §12 note).
- **Best balance** (5.70,−0.170,52): χ²_full=2582.6 (+9.6), D₂=851 μK² (17%↓), supp=36%. Good χ² + meaningful D₂ suppression.
- Punctuated (reference only): φ₀=12.00, y₀=0.000, N_star=77.2, m=1.1323e-7, λ=3.3299e-15

### Key Constraint
USR suppression at CMB scales requires fine-tuned initial conditions. The mechanism works (D₂ down 24% at cost of +41 χ²) but no config outperforms LCDM across the full spectrum. Deeper low-ℓ suppression (e.g. D₄ ≈ 677 μK² at ℓ=4, 26% below LCDM) comes at higher χ²_full cost.

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
| Ezquiaga / PBH | 0.002 Mpc⁻¹ | 3.0 | [5×10⁻⁴, 8×10⁻³] |

> **Pivot and N_star are degenerate.** The pivot fixes the code→Mpc⁻¹
> conversion `C = k_pivot_phys / k_pivot_code`, with
> `k_pivot_code = aH(N_total − N_star)` falling as `exp(−N_star)`. Holding
> `C` fixed requires `N_star_new = N_star_old + ln(k_pivot_old/k_pivot_new)`,
> so moving the pivot alone rescales every derived mass by
> `(k_new/k_old)^-2` — 625× for 0.05 → 0.002. Declare the pivot in the
> config, never override it on the CLI for a config that already sets it.

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

**D₂ column correction:** the previously quoted values (918/847/677/835) were the D_ell array elements at index 2, i.e. **ℓ=4**, not the quadrupole (ℓ=2). True quadrupoles (ℓ=2) from the same verification runs:

| Config | χ²_full (ℓ=2-2508) | D₂ (ℓ=2) [μK²] | D₄ (ℓ=4) [μK²] | Suppression (P_S) | Δχ² vs LCDM |
|--------|-------------------|----------------|-----------------|-------------------|-------------|
| 6.40,−0.475,59 | 2574.2 | 997 | 918 | 31% | +1.2 |
| 5.70,−0.170,52 | 2582.6 | 851 | 847 | 36% | +9.6 |
| 5.75,−0.170,55 | 2613.8 | 784 | 677 | 39% | +40.7 |
| 6.55,−0.780,50 | 2637.6 | 811 | 835 | 47% | +64.6 |
| LCDM | 2573.0 | 1029 | 920 | — | — |

Current pipeline (post solve_ivp refactor, T_max=500) shifts the reference config slightly: D₂=776.5 μK² (24%↓), D₄=669.5 μK² (27%↓), Δχ²_lowℓ=−1.60.

**Diagnostic script:** `scripts/run_full_analysis.py` — runs full pipeline, produces broken-axis D_ℓ plot with Planck data.
**Quick scan:** `python scripts/camb_scan.py --phase broad --quick --full-chi2` (~20 min).

**Planck data files:** Downloaded from IRSA (R3.01/R3.02), stored in `data/Planck/`:
- Binned TT/TE/EE spectrum (ℓ≈47-2500)
- Unbinned TT/TE/EE spectrum (ℓ=2-2508)
- Low-ℓ Commander data (ℓ=2-29)

### 13. Background Evolution — Standard Higgs

**Standard Higgs** (`models/higgs.py:HiggsModel`): ODE variable `x = φ` (Jordan
frame). Potential `f(x) = (1 - e^{-αx})²`, monotonic decreasing, no features. USR
is **kinetic-driven** — comes from initial `y₀` (small `|y₀|` causes a freeze).
Kinetic dominance at start (`ε_H≈3`), Hubble friction kills velocity in <0.1
e-fold, field freezes (`ε_H` dips to ~10⁻³), then catches the SR attractor.
Tunable via `y₀`.

**Key contrast:** Standard Higgs USR is an initial-condition effect (tune `y₀`),
whereas Ezquiaga CHI USR is structural (near-inflection from RG running).

> **Ezquiaga CHI** — model parameters, the inflection conditions, config
> structure, parameter trends, allowed ranges, PBH windows and formation/accretion
> metadata — is documented in `ezquiaga/AGENTS.md`. Do not duplicate it here.

### 14. MS n_s Oscillation vs Smooth SR — Physics, Not Numerical

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

### 15. No Inline Python Code

**NEVER** run inline `python -c "..."` or `python <<EOF` for physics analysis. It is non-reproducible, un-tracked, and un-reviewable. Use one of:
- **Config file** + `pspectrum_pipeline.py` for MS computation
- **`scripts/full_pbh_pipeline.py --config <cfg>`** for PBH abundance + mass function
- **`scripts/sweep_pbh_params.py`** for parameter sweeps
- **`scripts/plotting.py`** for plotting

The one exception: short (≤5 line) diagnostics to check file contents or list directories. Any physics computation must use the proper scripts.

### 16. Fortran MS Solver Backend — Primary Backend

The Hot Path comoving MS grid integration is ported to native Fortran 90 (`fortran/ms_solver.f90`) with OpenMP multi-threaded parallelization over comoving modes.

**Status: primary backend.** We are committed to using the Fortran solver as the default MS solver. The Python/Numba backends remain available as fallbacks for cross-checking but are no longer the primary target for new work.

- Python bridge: `fortran_ms_solver.py` converts background splines to Fortran memory order (`order='F'`) and calls the library.
- Compilation: Compiled via `f2py` with Meson/Ninja toolchain in `cmb-anomaly` conda env. Make command: `cd fortran && make`.
- Linking: Requires `LDFLAGS="-fopenmp"` to resolve OpenMP runtime linker symbols.
- Execution: Default backend in `pspectrum_pipeline.py` is `'fortran'`. Use `--backend numba` to fall back for debugging.
- Old solvers: `inf_dyn_MS_full.py` (Python/scipy) and `numba_ms_solver.py` (Numba) are **not deleted**. They serve as reference implementations and debugging fallbacks.
- Validation: `fortran/test_vs_numba.py` checks single-mode trajectory correctness, full comoving grid agreement across three key configurations (within relative difference < 1e-4), and CAMB observable compatibility.

---

**Ezquiaga CHI / PBH work — model, configs, parameter ranges, windows, formation
& accretion metadata — lives in `ezquiaga/AGENTS.md`.**
