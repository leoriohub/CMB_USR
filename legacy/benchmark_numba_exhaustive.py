"""
Exhaustive Numba branch benchmark & reproduction test.

Tests correctness + performance across branches and solvers.

Usage:
    # Feature branch: full suite
    python scripts/benchmark_numba_exhaustive.py --branch feature --out outputs/simulations/logs/benchmark_feature.jsonl

    # Main branch: Python-only baseline
    python scripts/benchmark_numba_exhaustive.py --branch main --out outputs/simulations/logs/benchmark_main.jsonl

    # Compare results from both branches
    python scripts/benchmark_numba_exhaustive.py --compare benchmark_main.jsonl benchmark_feature.jsonl

    # Quick (fast configs only, small grid)
    python scripts/benchmark_numba_exhaustive.py --branch feature --quick
"""
import argparse
import json
import os
import sys
import time
import inspect

import numpy as np

from scripts.constants import As, k_pivot_phys
from scripts.plotting import get_path

BEST_CONFIGS = [
    {"label": "best_chi2", "phi0": 6.40, "y0": -0.475, "N_star": 59.0},
    {"label": "best_d2",   "phi0": 5.75, "y0": -0.170, "N_star": 55.0},
    {"label": "balance",   "phi0": 5.70, "y0": -0.170, "N_star": 52.0},
]

SMALL_GRID_KW = dict(n_dense=40, n_outer=20)
FULL_GRID_KW = dict(n_dense=140, n_outer=70)
WORKER_VALS = [1, 4, 8]


def _can_use_numba():
    try:
        from numba_ms_solver import HAVE_NUMBA
        return HAVE_NUMBA
    except (ImportError, ModuleNotFoundError):
        return False


def _has_param(func, name):
    sig = inspect.signature(func)
    return name in sig.parameters


def _ratio_dev(ps_nb, ps_py):
    finite = np.isfinite(ps_py) & (ps_py > 0) & np.isfinite(ps_nb)
    if np.sum(finite) < 5:
        return float("inf")
    return float(np.max(np.abs(ps_nb[finite] / ps_py[finite] - 1.0)))


def run_single_config(model, phi0, y0, N_star, k_grid, n_workers, use_numba):
    from pspectrum_pipeline import run_pspectrum_pipeline
    kw = dict(
        model=model, phi0=phi0, y0=y0, N_star=N_star,
        k_pivot_phys=k_pivot_phys, k_phys_grid=k_grid,
        normalize_to_As=True, As=As,
        n_workers=n_workers,
        save_outputs=False,
    )
    try:
        if _has_param(run_pspectrum_pipeline, "use_numba"):
            kw["use_numba"] = use_numba
    except Exception:
        pass

    t0 = time.time()
    r = run_pspectrum_pipeline(**kw)
    elapsed = time.time() - t0

    if r["status"] != "success":
        return None, dict(status="error", message=r.get("message", "?"), elapsed=elapsed)

    n_ok = int(r["metadata"].get("n_completed", 0))
    n_total = int(r["metadata"].get("num_k", n_ok))
    rate = n_ok / max(elapsed, 0.001)
    return r, dict(
        status="ok",
        elapsed=round(elapsed, 2),
        modes_ok=n_ok,
        modes_total=n_total,
        rate_mode_s=round(rate, 2),
    )


def compute_dell_chi2(ps_data):
    return _compute_dell_chi2_branch_agnostic(ps_data)


def _compute_chi2_branch_agnostic(D, ells):
    """Compute chi2 for D_ell vs Planck, works on both branches."""
    from scripts.planck_data import get_planck_data_asymmetric
    p_ells, p_D, p_lo, p_hi = get_planck_data_asymmetric()
    try:
        from scripts.chi2_analysis import chi2_model_lcdm as chi2_fn
        chi2_m, _ = chi2_fn(D, ells, ell_max=29)
    except (ImportError, TypeError):
        from scripts.chi2_analysis import _chi2_model_lcdm
        from scripts.camb_wrapper import compute_cl_camb_powerlaw
        from scripts.planck_data import C_ell_to_d_ell
        ells_l, C_l, _, _ = compute_cl_camb_powerlaw(ell_max=2500)
        D_l = C_ell_to_d_ell(ells_l, C_l)
        chi2_m, _, _ = _chi2_model_lcdm(D, ells, D_l, ells_l, p_ells, p_D, p_lo, p_hi)
    return float(chi2_m)


def _compute_dell_chi2_branch_agnostic(ps_data):
    """Compute D_ell + chi2, works on both branches."""
    from scripts.camb_wrapper import compute_cl_full_camb, compute_cl_camb_powerlaw
    from scripts.planck_data import C_ell_to_d_ell
    try:
        ells, C_tt, _, _ = compute_cl_full_camb(ps_data, ell_max=2500)
        D = C_ell_to_d_ell(ells, C_tt)
        ells_pl, C_pl, _, _ = compute_cl_camb_powerlaw(ell_max=2500)
        D_pl = C_ell_to_d_ell(ells_pl, C_pl)
        chi2_m = _compute_chi2_branch_agnostic(D, ells)
        chi2_l = _compute_chi2_branch_agnostic(D_pl, ells_pl)
        return dict(
            ells=ells.tolist(),
            D_ell=D.tolist(),
            chi2_model=round(float(chi2_m), 2),
            chi2_lcdm=round(float(chi2_l), 2),
            delta_chi2=round(float(chi2_m - chi2_l), 2),
            D2=round(float(D[0]), 1),
        )
    except Exception as e:
        return dict(error=str(e))


def run_lcdm_baseline():
    from scripts.camb_wrapper import compute_cl_camb_powerlaw
    from scripts.planck_data import C_ell_to_d_ell
    t0 = time.time()
    ells, C_tt, _, _ = compute_cl_camb_powerlaw(ell_max=2500)
    D = C_ell_to_d_ell(ells, C_tt)
    chi2_m = _compute_chi2_branch_agnostic(D, ells)
    elapsed = time.time() - t0
    return dict(
        label="lcdm",
        solver="powerlaw",
        branch="any",
        n_workers=0,
        elapsed=round(elapsed, 2),
        D2=round(float(D[0]), 1),
        chi2_model=round(float(chi2_m), 2),
        chi2_lcdm=round(float(chi2_m), 2),
        delta_chi2=0.0,
    )


def run_tests(branch, output_path, use_numba, workers, quick):
    from pspectrum_pipeline import build_weighted_kgrid
    from models.higgs import HiggsModel

    grid_kw = SMALL_GRID_KW if quick else FULL_GRID_KW
    k_grid = build_weighted_kgrid(1e-5, 1.0, k_pivot_phys, **grid_kw)

    model = HiggsModel(lam=0.13, xi=15000.0)
    results = []
    use_nb = use_numba and _can_use_numba()

    solver_label = "numba" if use_nb else "python"

    for cfg in BEST_CONFIGS:
        for nw in workers:
            print(f"  {cfg['label']}: {solver_label} {nw}w  "
                  f"φ₀={cfg['phi0']} y₀={cfg['y0']:.3f} N*={cfg['N_star']:.0f}  "
                  f"grid={len(k_grid)} modes",
                  flush=True)

            r, meta = run_single_config(
                model, cfg["phi0"], cfg["y0"], cfg["N_star"],
                k_grid, n_workers=nw, use_numba=use_nb,
            )

            entry = dict(
                label=cfg["label"], phi0=cfg["phi0"], y0=cfg["y0"],
                N_star=cfg["N_star"], branch=branch, solver=solver_label,
                n_workers=nw, grid_size=len(k_grid),
            )
            entry.update(meta)

            if r is not None:
                ps_dev = _ratio_dev(r["P_S"], r["P_S"])  # 0 for single run
                entry["max_self_dev"] = 0.0

                dell = compute_dell_chi2({k: r[k] for k in ("k_phys", "P_S")})
                entry["dell"] = dell

                print(f"    → {meta['elapsed']:.1f}s  "
                      f"{meta['rate_mode_s']:.1f} mode/s  "
                      f"D₂={dell.get('D2', '?'):.0f} "
                      f"Δχ²={dell.get('delta_chi2', '?'):.1f}",
                      flush=True)
            else:
                print(f"    → FAIL: {meta.get('message', '?')}", flush=True)

            results.append(entry)

    return results


def run_numba_comparison(branch, output_path, quick):
    """On feature branch: run Python + Numba for same configs, compare P_S(k)."""
    from pspectrum_pipeline import build_weighted_kgrid
    from models.higgs import HiggsModel

    grid_kw = SMALL_GRID_KW if quick else FULL_GRID_KW
    k_grid = build_weighted_kgrid(1e-5, 1.0, k_pivot_phys, **grid_kw)
    model = HiggsModel(lam=0.13, xi=15000.0)
    results = []

    print(f"\n  Numba vs Python comparison ({len(k_grid)} modes, serial)", flush=True)

    for cfg in BEST_CONFIGS:
        r_py, m_py = run_single_config(model, cfg["phi0"], cfg["y0"],
                                        cfg["N_star"], k_grid, 1, False)
        r_nb, m_nb = run_single_config(model, cfg["phi0"], cfg["y0"],
                                        cfg["N_star"], k_grid, 1, True)

        entry = dict(
            label=cfg["label"], phi0=cfg["phi0"], y0=cfg["y0"],
            N_star=cfg["N_star"], branch=branch, grid_size=len(k_grid),
        )

        if r_py and r_nb:
            ps_dev = _ratio_dev(r_nb["P_S"], r_py["P_S"])
            entry["ps_ratio_dev"] = float(round(ps_dev, 6))

            d_py = compute_dell_chi2({k: r_py[k] for k in ("k_phys", "P_S")})
            d_nb = compute_dell_chi2({k: r_nb[k] for k in ("k_phys", "P_S")})

            d_dev = float(np.max(np.abs(
                np.array(d_nb.get("D_ell", [0])) - np.array(d_py.get("D_ell", [0]))
            ))) if d_py.get("D_ell") and d_nb.get("D_ell") else None

            entry["python"] = {**m_py, "dell": d_py}
            entry["numba"] = {**m_nb, "dell": d_nb}
            entry["ps_ratio_dev"] = ps_dev
            entry["d_ell_max_abs_diff"] = d_dev
            entry["chi2_diff"] = (
                abs(d_nb.get("delta_chi2", 0) - d_py.get("delta_chi2", 0))
                if "delta_chi2" in (d_nb or {}) and "delta_chi2" in (d_py or {})
                else None
            )

            status = "PASS" if ps_dev < 1e-4 else "WARN" if ps_dev < 1e-3 else "FAIL"
            print(f"  [{status}] {cfg['label']}: "
                  f"P_S max|ratio-1|={ps_dev:.2e}  "
                  f"Δχ² diff={entry.get('chi2_diff', '?'):.2e}  "
                  f"Python={m_py['elapsed']:.1f}s  "
                  f"Numba={m_nb['elapsed']:.1f}s  "
                  f"speedup={m_py['elapsed']/max(m_nb['elapsed'],0.01):.1f}x",
                  flush=True)
        else:
            entry["python"] = m_py
            entry["numba"] = m_nb
            print(f"  [SKIP] {cfg['label']}: "
                  f"py={r_py is not None} nb={r_nb is not None}", flush=True)

        results.append(entry)

    return results


def run_parallel_benchmark(branch, output_path, quick):
    """Benchmark parallel scaling for 1 worker on both solvers."""
    from pspectrum_pipeline import build_weighted_kgrid
    from models.higgs import HiggsModel

    grid_kw = SMALL_GRID_KW if quick else FULL_GRID_KW
    k_grid = build_weighted_kgrid(1e-5, 1.0, k_pivot_phys, **grid_kw)
    model = HiggsModel(lam=0.13, xi=15000.0)
    cfg = BEST_CONFIGS[2]
    results = []

    print(f"\n  Parallel scaling: φ₀={cfg['phi0']} y₀={cfg['y0']:.3f} "
          f"N*={cfg['N_star']:.0f}  grid={len(k_grid)} modes", flush=True)

    for solver in ["python", "numba"]:
        use_nb = (solver == "numba")
        if use_nb and not _can_use_numba():
            continue
        for nw in WORKER_VALS:
            r, meta = run_single_config(
                model, cfg["phi0"], cfg["y0"], cfg["N_star"],
                k_grid, n_workers=nw, use_numba=use_nb,
            )
            entry = dict(
                label=cfg["label"], phi0=cfg["phi0"], y0=cfg["y0"],
                N_star=cfg["N_star"], branch=branch, solver=solver,
                n_workers=nw, grid_size=len(k_grid),
            )
            entry.update(meta)
            results.append(entry)

            print(f"    {solver:>8} {nw:>2d}w: "
                  f"{meta['elapsed']:>7.1f}s  "
                  f"{meta['rate_mode_s']:>7.1f} mode/s",
                  flush=True)

    return results


def collate_branch(output_path, branch, quick):
    """Run full test suite for one branch."""
    print(f"\n{'='*70}", flush=True)
    print(f"  BENCHMARK: branch={branch}  quick={quick}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Numba available: {_can_use_numba()}", flush=True)
    print(f"  Output: {output_path}", flush=True)
    print(f"{'='*70}", flush=True)

    all_results = []
    has_numba = _can_use_numba()

    # LCDM baseline
    print("\n  LCDM baseline...", flush=True)
    lcdm = run_lcdm_baseline()
    all_results.append(lcdm)
    print(f"    D₂={lcdm['D2']:.0f}  χ²={lcdm['chi2_model']:.1f}  "
          f"Δχ²={lcdm['delta_chi2']:.1f}", flush=True)

    # Plasma solver: Python (always)
    print("\n  Python solver tests", flush=True)
    py_results = run_tests(branch, output_path, use_numba=False,
                           workers=WORKER_VALS, quick=quick)
    all_results.extend(py_results)

    # Numba comparison (feature branch only)
    if has_numba:
        print("\n  Numba vs Python direct comparison", flush=True)
        comp = run_numba_comparison(branch, output_path, quick)
        all_results.append({"_type": "comparison", "runs": comp})

        print("\n  Numba solver tests", flush=True)
        nb_results = run_tests(branch, output_path, use_numba=True,
                               workers=WORKER_VALS, quick=quick)
        all_results.extend(nb_results)

        print("\n  Parallel scaling benchmark", flush=True)
        scaling = run_parallel_benchmark(branch, output_path, quick)
        all_results.append({"_type": "scaling", "runs": scaling})

    # Write
    with open(output_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\n  Results written: {output_path}", flush=True)

    # Summary table
    print_summary(all_results)
    return all_results


def print_summary(results):
    """Print human-readable summary table."""
    print(f"\n{'='*70}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    header = f"  {'Config':<14} {'Branch':<10} {'Solver':<8} {'W':<4} "
    header += f"{'Time(s)':<8} {'Mode/s':<8} {'D₂':<8} {'Δχ²':<8}"
    print(header)
    print(f"  {'-'*14} {'-'*10} {'-'*8} {'-'*4} {'-'*8} {'-'*8} "
          f"{'-'*8} {'-'*8}", flush=True)

    for r in results:
        if r.get("_type") in ("comparison", "scaling"):
            continue
        label = r.get("label", "?")
        branch = r.get("branch", "?")
        solver = r.get("solver", "?")
        nw = r.get("n_workers", "?")
        elapsed = r.get("elapsed", "?")
        rate = r.get("rate_mode_s", "?")
        dell = r.get("dell", {}) or {}
        d2 = dell.get("D2", "?")
        dchi2 = dell.get("delta_chi2", "?")
        print(f"  {label:<14} {str(branch):<10} {solver:<8} {str(nw):<4} "
              f"{str(elapsed):<8} {str(rate):<8} {str(d2):<8} "
              f"{str(dchi2):<8}", flush=True)
    print(flush=True)


def compare_results(path1, path2):
    """Compare two benchmark output files (main vs feature)."""
    def load(path):
        items = []
        with open(path) as f:
            for line in f:
                items.append(json.loads(line.strip()))
        return items

    r1 = load(path1)
    r2 = load(path2)

    print(f"\n{'='*70}", flush=True)
    print(f"  CROSS-BRANCH COMPARISON", flush=True)
    print(f"  A: {path1} ({len(r1)} records)", flush=True)
    print(f"  B: {path2} ({len(r2)} records)", flush=True)
    print(f"{'='*70}", flush=True)

    # Match Python solver results across branches
    def key(r):
        return (r.get("label"), r.get("solver"), r.get("n_workers"))

    py1 = {key(r): r for r in r1 if r.get("solver") == "python" and not r.get("_type")}
    py2 = {key(r): r for r in r2 if r.get("solver") == "python" and not r.get("_type")}

    common = set(py1.keys()) & set(py2.keys())
    if common:
        print(f"\n  Python solver match ({len(common)} configs):", flush=True)
        print(f"  {'Config':<14} {'W':<4} {'D₂ (A)':<10} {'D₂ (B)':<10} "
              f"{'ΔD₂':<10} {'Time(A)':<10} {'Time(B)':<10} {'ΔTime%':<10}", flush=True)
        print(f"  {'-'*14} {'-'*4} {'-'*10} {'-'*10} {'-'*10} "
              f"{'-'*10} {'-'*10} {'-'*10}", flush=True)
        for k in sorted(common):
            a, b = py1[k], py2[k]
            d2a = (a.get("dell") or {}).get("D2", "?")
            d2b = (b.get("dell") or {}).get("D2", "?")
            ta = a.get("elapsed", 0)
            tb = b.get("elapsed", 0)
            dt_pct = f"{100*(tb-ta)/max(ta,0.01):+.0f}%" if ta and tb else "?"
            dd2 = f"{d2b - d2a:+.1f}" if isinstance(d2a, float) and isinstance(d2b, float) else "?"
            print(f"  {k[0]:<14} {str(k[2]):<4} {str(d2a):<10} "
                  f"{str(d2b):<10} {dd2:<10} {str(ta):<10} {str(tb):<10} "
                  f"{dt_pct}", flush=True)

    # Numba comparison section
    comp_records = [r for r in r2 if r.get("_type") == "comparison"]
    for rec in comp_records:
        for run in rec.get("runs", []):
            py = run.get("python", {})
            nb = run.get("numba", {})
            dev = run.get("ps_ratio_dev")
            print(f"\n  [{run['label']}] P_S dev={dev:.2e}, "
                  f"Δχ² diff={run.get('chi2_diff', '?')}: "
                  f"Python={py.get('elapsed','?'):.1f}s, "
                  f"Numba={nb.get('elapsed','?'):.1f}s, "
                  f"speedup={py.get('elapsed',1)/max(nb.get('elapsed',0.01),0.01):.1f}x",
                  flush=True)

    # Scaling section
    scaling_records = [r for r in r2 if r.get("_type") == "scaling"]
    for rec in scaling_records:
        print(f"\n  Parallel scaling:", flush=True)
        print(f"  {'Solver':<10} {'W':<4} {'Time(s)':<10} {'Mode/s':<10}", flush=True)
        print(f"  {'-'*10} {'-'*4} {'-'*10} {'-'*10}", flush=True)
        for run in rec.get("runs", []):
            print(f"  {run.get('solver','?'):<10} {run.get('n_workers','?'):<4} "
                  f"{run.get('elapsed','?'):<10} {run.get('rate_mode_s','?'):<10}",
                  flush=True)

    print(flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Exhaustive Numba branch benchmark & reproduction test"
    )
    parser.add_argument("--branch", default="feature",
                        help="Branch name for record-keeping")
    parser.add_argument("--out", default=None,
                        help="Output JSONL path")
    parser.add_argument("--quick", action="store_true",
                        help="Smaller grid for faster runs")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="Compare two benchmark output files")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    out_path = args.out or get_path("logs", f"benchmark_{args.branch}_{int(time.time())}.jsonl")

    if args.branch == "any":
        args.branch = "current"

    collate_branch(out_path, args.branch, args.quick)

    print(f"\n  To compare with another branch:", flush=True)
    print(f"    python scripts/benchmark_numba_exhaustive.py "
          f"--compare path/to/benchmark_main.jsonl {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
