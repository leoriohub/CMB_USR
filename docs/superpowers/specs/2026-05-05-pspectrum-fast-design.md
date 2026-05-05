# P_S(k) Pipeline — Fast Variant

## Goal

Create a drop-in‑compatible variant of `scripts/pspectrum_pipeline.py` that runs the **same physics** faster by reducing Python overhead in the per‑mode loop.

## Approach

New file: `scripts/pspectrum_pipeline_fast.py`, identical CLI interface and JSON output
format. Keep the original file untouched as baseline.

## Optimizations

1. **Pre‑bind hot functions** — assign `ms_solver.run_ms_simulation`,
   `ms_solver.get_ms_derived_quantities_with_bg`, `np.linspace`,
   `extract_mode_initial_conditions`, `model.f`, `model.dfdx` to local variables
   outside the loop.

2. **Reuse T_ms array** — allocate a single `T_ms` array and fill it with
   `np.linspace(..., out=T_ms)` instead of allocating a new array every mode.

3. **Tight helper** — inline the extraction + solve logic into a single helper
   `_solve_one_mode_fast()` that avoids function call overhead between extraction
   and solve.

4. **Keep parallel support** — same `--n-cores` / `ProcessPoolExecutor` approach
   as the original.

## Non‑goals

- No changes to ODE integrator tolerances, steps, or solver method.
- No changes to background integration.
- No changes to k‑grid construction or pivot mapping.
- No changes to JSON output schema.

## Validation

Run both variants with identical arguments on the PunctuatedInflationModel and
verify:
- `max(|ΔP_S / P_S|) < 1e-12`
- `max(|ΔP_T / P_T|) < 1e-12`
- Same number of completed / failed modes
