# Fast P_S(k) Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create `scripts/pspectrum_pipeline_fast.py` that runs the same physics faster by reducing Python overhead in the per-mode loop.

**Architecture:** Single-file drop-in replacement of `pspectrum_pipeline.py` with identical CLI, identical JSON output, and identically equivalent numerical results.

**Tech Stack:** Python, numpy, scipy, concurrent.futures, tqdm

---

### Task 1: Create `scripts/pspectrum_pipeline_fast.py`

**Files:**
- Create: `scripts/pspectrum_pipeline_fast.py`

- [ ] **Step 1: Copy the original as starting point**

Copy `scripts/pspectrum_pipeline.py` to `scripts/pspectrum_pipeline_fast.py`.

- [ ] **Step 2: Apply perf optimizations to the new file**

Replace the per-mode computation section (from `bg_interp = ...` through the end of the mode loop) with an optimized version:

```python
    # ── Per-mode hot path optimizations ─────────────────────────────────────
    # Pre-bind hot functions to local variables (avoids global lookups)
    _linspace = np.linspace
    _run_ms = ms_solver.run_ms_simulation
    _get_derived = ms_solver.get_ms_derived_quantities_with_bg
    _extract = extract_mode_initial_conditions
    _f = model.f
    _dfdx = model.dfdx

    # Pre-allocate T_ms array (reused across modes)
    _T_ms = np.empty(ms_steps)

    n_modes = len(k_code_grid)
    checkpoint_interval = max(n_modes // 10, 1)
    t_start_all = time.time()
    failed_modes = []
    errors = []

    bg_interp = ms_solver.build_bg_interpolators(bg_sol, T_span_bg)

    # Build metadata (identical to original)
    run_id = str(uuid.uuid4())[:8]
    metadata = { ... }  # same as original

    def save_checkpoint(iteration):
        ...  # same as original

    def _solve_one_mode_fast(idx, k_code_val):
        """Tight helper: extract + solve in one call, returns (ps, pt, si, err)."""
        try:
            xi, y0v, zi, ni, t_start, t_end, si = _extract(
                bg_sol, T_span_bg, end_idx, k_code_val, k_start_factor
            )
            _linspace(t_start, t_end, ms_steps, out=_T_ms)
            ms_sol = _run_ms(bg_interp, ni, _T_ms, k_code_val, model)
            d = _get_derived(ms_sol, bg_interp, _T_ms, model, k_code_val, ni)
            ps = float(d["P_S"][-1])
            pt = float(d["P_T"][-1])
            if np.isfinite(ps) and ps > 0:
                return ps, pt, si, None
            return None, None, si, f"non-finite P_S={ps}"
        except Exception as e:
            return None, None, -1, str(e)

    # Parallel execution path
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        n_actual = min(n_workers, n_modes, multiprocessing.cpu_count())
        print(f"  Computing {n_modes} k-modes on {n_actual} workers...")
        # (parallel path identical to original — uses _compute_single_mode from
        #  the module level, which we keep identical)
        ...

    # Serial execution path with tight helper
    else:
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(k_code_grid), total=n_modes, desc="  k-modes", unit="mode")
        except ImportError:
            iterator = enumerate(k_code_grid)
            print(f"  Computing {n_modes} k-modes (serial)...")

        for idx, k_code in iterator:
            ps, pt, si, err = _solve_one_mode_fast(idx, k_code)
            if err is not None:
                failed_modes.append(idx)
                errors.append(f"mode {idx} (k={k_code:.4e}): {err}")
            else:
                P_S_raw[idx] = ps
                P_T_raw[idx] = pt
                start_indices[idx] = si
            try:
                save_checkpoint(idx + 1)
            except Exception:
                pass
```

The metadata dict, `save_checkpoint`, `find_end_of_inflation`, `ensure_k_pivot`, `build_weighted_kgrid`, `get_k_pivot_code`, `extract_mode_initial_conditions`, `_compute_single_mode`, `load_pspectrum`, `parse_args`, `build_model`, and `main` functions remain identical to the original.

- [ ] **Step 3: Run syntax check**

```bash
python3 -c "import ast; ast.parse(open('scripts/pspectrum_pipeline_fast.py').read()); print('Syntax OK')"
```

- [ ] **Step 4: Verify both produce identical results**

Run both scripts with a small configuration:

```bash
OMP_NUM_THREADS=1 python3 scripts/pspectrum_pipeline.py \
  --model PunctuatedInflationModel --m 1.1323e-7 --lam 3.3299e-15 \
  --phi0 12 --y0 -0.01424021 \
  --num-k 6 --k-min 1e-4 --k-max 0.5 --N-star 77 \
  --bg-steps 30000 --ms-steps 2000 --T-max 200000 --normalize-to-As --no-save 2>&1
```

Then with `_fast` variant with same args. Compare the returned P_S arrays.

```python
# verification snippet
import json, numpy as np
a = json.load(open("ref.json"))
b = json.load(open("fast.json"))
for key in ["P_S", "P_T", "k_phys", "metadata"]:
    ak = np.array(a.get(key, a["spectrum"].get(key, [])))
    bk = np.array(b.get(key, b["spectrum"].get(key, [])))
    if len(ak) and len(bk):
        print(f"{key}: max|diff| = {np.max(np.abs(ak-bk)):.2e}")
```

- [ ] **Step 5: Commit both spec and implementation**

```bash
git add scripts/pspectrum_pipeline_fast.py docs/superpowers/specs/2026-05-05-pspectrum-fast-design.md docs/superpowers/plans/2026-05-05-pspectrum-fast.md
git commit -m "feat: add fast P_S(k) pipeline variant with reduced Python overhead"
```
