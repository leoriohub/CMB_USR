#!/usr/bin/env python3
"""Quick benchmark: Fortran vs Numba MS solver.

Compares 4 conditions in subprocess isolation:
  Numba(1t)  Numba(Nt)  Fortran(1t)  Fortran(Nt)

Each: 1 cold + N warm solves on a 300-mode grid.
Physics check: P_S max rel diff < 1e-4.

Usage: python fortran/bench_quick.py
Output: outputs/simulations/logs/fortran_bench_quick.json
"""
from scripts.constants import ROOT_DIR
import argparse, json, os, shutil, subprocess, sys, time

OUTPUT_PATH = os.path.join(ROOT_DIR,
    "outputs/simulations/logs/fortran_bench_quick.json")

CONFIG = dict(
    phi0=5.70, y0=-0.170, nstar=52,
    k_min=1e-5, k_max=1.0, n_k=300,
    k_start_factor=100.0, n_warm=2,
)


# ── Worker (subprocess — clean env) ──────────────────────────────────

def _find_end_of_inflation(epsH, window_frac=0.05):
    """pspectrum_pipeline.find_end_of_inflation — inlined for clean subprocess import."""
    window = max(20, int(len(epsH) * window_frac))
    start_idx = -1
    for idx, e in enumerate(epsH):
        if e < 1.0:
            start_idx = idx
            break
    if start_idx == -1:
        return -1
    candidates = []
    for i in range(start_idx + 1, len(epsH)):
        if epsH[i - 1] < 1.0 and epsH[i] >= 1.0:
            candidates.append(i)
    if not candidates:
        return -1
    for idx in candidates:
        end = min(idx + window, len(epsH))
        if sum(epsH[idx:end]) / (end - idx) >= 1.0:
            return idx
    if epsH[-1] >= 1.0:
        for idx in range(len(epsH) - 1, start_idx - 1, -1):
            if epsH[idx] < 1.0:
                return idx + 1
    return -1


def worker_main(backend, nthreads):
    os.environ["NUMBA_NUM_THREADS"] = str(nthreads)
    os.environ["OMP_NUM_THREADS"] = str(nthreads)

    import numpy as np
    from models.higgs import HiggsModel
    import inf_dyn_background as bg_solver
    from numba_ms_solver import build_numba_splines

    model = HiggsModel()
    model.x0 = CONFIG["phi0"]
    model.y0 = CONFIG["y0"]
    T_span_bg = np.linspace(0.0, model.T_max, model.bg_steps)
    bg_sol = bg_solver.run_background_simulation(model, T_span_bg)
    derived_bg = bg_solver.get_derived_quantities(bg_sol, model)
    end_idx = _find_end_of_inflation(derived_bg["epsH"])
    if end_idx == -1:
        end_idx = len(T_span_bg) - 1

    k_phys = np.logspace(np.log10(CONFIG["k_min"]),
                         np.log10(CONFIG["k_max"]), CONFIG["n_k"])
    N_total = derived_bg["N"][end_idx]
    N_pivot = N_total - CONFIG["nstar"]
    pivot_bg_idx = int(np.argmin(np.abs(derived_bg["N"][:end_idx] - N_pivot)))
    k_pivot_code = np.exp(bg_sol[3][pivot_bg_idx]) * bg_sol[2][pivot_bg_idx]
    k_codes = k_pivot_code * (k_phys / 0.002)

    if backend == "numba":
        from numba_ms_solver import numba_run_ms_grid
        bg_coefs = build_numba_splines(bg_sol, T_span_bg)
        def solve():
            return numba_run_ms_grid(
                bg_sol, T_span_bg, end_idx, k_codes, model,
                k_start_factor=CONFIG["k_start_factor"], bg_coefs=bg_coefs)
    else:
        from fortran_ms_solver import fortran_run_ms_grid, HAVE_FORTRAN
        if not HAVE_FORTRAN:
            print(json.dumps({"error": "Fortran module not compiled. Run: cd fortran && make"}))
            sys.exit(1)
        bg_coefs = build_numba_splines(bg_sol, T_span_bg)
        def solve():
            return fortran_run_ms_grid(
                bg_sol, T_span_bg, end_idx, k_codes, model,
                k_start_factor=CONFIG["k_start_factor"], bg_coefs=bg_coefs)

    t0 = time.perf_counter()
    P_S_cold, P_T_cold, _ = solve()
    t_cold = time.perf_counter() - t0

    t_warm = []
    for _ in range(CONFIG["n_warm"]):
        t0 = time.perf_counter()
        P_S, P_T, _ = solve()
        t_warm.append(time.perf_counter() - t0)

    result = dict(
        backend=backend,
        threads=nthreads,
        cold_s=t_cold,
        warm_s=t_warm,
        warm_median_s=float(np.median(t_warm)),
        k_phys=k_phys.tolist(),
        P_S=P_S.tolist(),
        P_T=P_T.tolist(),
    )
    print(json.dumps(result))


# ── Driver ───────────────────────────────────────────────────────────

def _clear_numba_cache():
    count = 0
    for root, _dirs, files in os.walk(ROOT_DIR):
        for f in files:
            if f.endswith((".nbi", ".nbc")):
                os.remove(os.path.join(root, f))
                count += 1
    if count:
        print(f"  Cleared {count} Numba cache file(s)")


def _max_threads():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


def driver_main():
    print("=" * 62)
    print("  Fortran vs Numba — Quick Benchmark")
    print("=" * 62)
    print(f"  Config: φ₀={CONFIG['phi0']}, y₀={CONFIG['y0']}, N*={CONFIG['nstar']}")
    print(f"  Grid:  {CONFIG['n_k']} modes  [{CONFIG['k_min']}–{CONFIG['k_max']} Mpc⁻¹]")
    print(f"  Warm runs: {CONFIG['n_warm']} per condition")
    print()

    _clear_numba_cache()
    max_threads = _max_threads()
    n_warm = CONFIG["n_warm"]

    conditions = [
        ("numba",   1),
        ("numba",   max_threads),
        ("fortran", 1),
        ("fortran", max_threads),
    ]

    results = {}
    worker_script = os.path.abspath(__file__)

    for backend, nthreads in conditions:
        label = f"{backend} {nthreads}t"
        print(f"  [{label}] Running...", end=" ", flush=True)

        env = os.environ.copy()
        env["NUMBA_NUM_THREADS"] = str(nthreads)
        env["OMP_NUM_THREADS"] = str(nthreads)

        proc = subprocess.run(
            [sys.executable, worker_script, "--worker", backend, str(nthreads)],
            capture_output=True, text=True, timeout=600,
            env=env, cwd=ROOT_DIR,
        )

        if proc.returncode != 0:
            print(f"FAILED (exit {proc.returncode})")
            stderr_trunc = proc.stderr.strip()[:300]
            print(f"    stderr: {stderr_trunc}")
            results[label] = {"error": proc.stderr.strip()}
            continue

        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"    stdout: {proc.stdout.strip()[:200]}")
            results[label] = {"error": f"JSON parse: {e}"}
            continue

        if "error" in result:
            print(f"ERROR: {result['error']}")
            results[label] = result
            continue

        results[label] = result
        p_warm = " / ".join(f"{t:.4f}" for t in result["warm_s"])
        print(f"cold {result['cold_s']:.3f}s  warm {result['warm_median_s']:.4f}s  [{p_warm}]")

    # ── Table ──
    print()
    print("─" * 62)
    print(f"  {'Backend':<14} {'Threads':<8} {'Cold (s)':<12} {'Warm (s)':<12} {'Speedup':<10}")
    print(f"  {'─'*14} {'─'*8} {'─'*12} {'─'*12} {'─'*10}")

    ft_max_key = f"fortran {max_threads}t"
    baseline = results.get(ft_max_key, {}).get("warm_median_s")

    for backend, nthreads in conditions:
        key = f"{backend} {nthreads}t"
        r = results.get(key, {})
        if "error" in r:
            print(f"  {backend:<14} {nthreads:<12} {'ERROR'}")
            continue
        cold = r["cold_s"]
        warm = r["warm_median_s"]
        ratio = f"{baseline / warm:.2f}x" if baseline and warm else "—"
        print(f"  {backend:<14} {nthreads:<8} {cold:<12.4f} {warm:<12.4f} {ratio:<10}")

    # ── Physics check ──
    print()
    nb_key = f"numba {max_threads}t"
    ft_key = f"fortran {max_threads}t"
    nb_r = results.get(nb_key, {})
    ft_r = results.get(ft_key, {})

    max_rel = mean_rel = status = None
    if "P_S" in nb_r and "P_S" in ft_r:
        import numpy as np
        ps_nb = np.array(nb_r["P_S"])
        ps_ft = np.array(ft_r["P_S"])
        rel_diff = np.abs(ps_ft - ps_nb) / np.maximum(ps_nb, 1e-300)
        max_rel = float(np.nanmax(rel_diff))
        mean_rel = float(np.nanmean(rel_diff))
        status = "PASS" if max_rel < 1e-4 else "FAIL"
        print(f"  Physics check: {status}  "
              f"max ΔP/P = {max_rel:.4e}  mean ΔP/P = {mean_rel:.4e}")
    else:
        print("  Physics check: SKIPPED — missing data from worker(s)")

    # ── Save ──
    output = dict(
        config=CONFIG,
        max_threads=max_threads,
        conditions=results,
        physics_check=dict(
            max_rel_diff=max_rel,
            mean_rel_diff=mean_rel,
            status=status,
            threshold=1e-4,
        ),
        system=dict(
            python=sys.version,
            hostname=os.uname().nodename,
        ),
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results: {OUTPUT_PATH}")
    print("=" * 62)


# ── Entry ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fortran vs Numba MS solver quick benchmark")
    parser.add_argument("--worker", nargs=2, metavar=("BACKEND", "THREADS"),
                        help="(internal) run worker in subprocess")
    args = parser.parse_args()

    if args.worker:
        worker_main(args.worker[0], int(args.worker[1]))
    else:
        driver_main()
