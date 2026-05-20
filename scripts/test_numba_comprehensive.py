"""
Comprehensive Numba MS solver test suite.

Tests P_S(k) shape, D_ell via CAMB, random sweep, large grid, Punctuated model,
JIT timing, and parallel scaling. Designed for tmux/mosh-uni workflow.

Usage:
    # Local (fast tests 1-5)
    python -m scripts.test_numba_comprehensive

    # Lab machine (full suite including scaling)
    mosh-uni
    cd ~/Documentos/CMB_USR
    git pull
    conda activate cmb-anomaly
    python -m scripts.test_numba_comprehensive --lab > ~/numba_test.log 2>&1
    Ctrl+b d
    # Check progress:
    ssh uni "tail -10 ~/numba_test.log"
"""
import argparse
import sys
import time
import os

import numpy as np

from models import HiggsModel, PunctuatedInflationModel
from pspectrum_pipeline import (
    run_pspectrum_pipeline, build_weighted_kgrid,
)
from scripts.constants import As, k_pivot_phys


PASS = 0
FAIL = 0
WARN = 0

_TOL_CRITICAL = dict(
    ps_shape=1e-4, dell=1e-4, chi2=0.5,
    sweep=1e-3, large=1e-4, punctuated=1e-4,
)
_TOL_WARN = dict(
    ps_shape=1e-3, dell=1e-3, chi2=1.0,
    sweep=5e-3, large=1e-3, punctuated=1e-3,
)


def _run_both(model, phi0, y0, N_star, k_grid, ms_steps=5000):
    """Run both Python and Numba solvers, return (P_S_py, P_S_nb, meta)."""
    r_py = run_pspectrum_pipeline(
        model=model, phi0=phi0, y0=y0, N_star=N_star,
        k_pivot_phys=k_pivot_phys, k_phys_grid=k_grid,
        normalize_to_As=True, As=As,
        n_workers=1, use_numba=False,
        ms_steps=ms_steps, save_outputs=False,
    )
    r_nb = run_pspectrum_pipeline(
        model=model, phi0=phi0, y0=y0, N_star=N_star,
        k_pivot_phys=k_pivot_phys, k_phys_grid=k_grid,
        normalize_to_As=True, As=As,
        n_workers=1, use_numba=True,
        ms_steps=ms_steps, save_outputs=False,
    )
    if r_py["status"] != "success":
        return None, None, f"Python failed: {r_py.get('message', '?')}"
    if r_nb["status"] != "success":
        return None, None, f"Numba failed: {r_nb.get('message', '?')}"
    return r_py["P_S"], r_nb["P_S"], r_py["metadata"]


def _ratio_dev(ps_nb, ps_py):
    """max |ratio - 1| over finite P_S points."""
    finite = np.isfinite(ps_py) & (ps_py > 0) & np.isfinite(ps_nb)
    if np.sum(finite) < 5:
        return float("inf")
    return float(np.max(np.abs(ps_nb[finite] / ps_py[finite] - 1.0)))


def _test(label, dev, critical, warn_lev):
    global PASS, FAIL, WARN
    status = "PASS"
    if dev > critical:
        status = "FAIL"
        FAIL += 1
    elif dev > warn_lev:
        status = "WARN"
        WARN += 1
    else:
        PASS += 1
    print(f"  [{status}] {label}: max |ratio-1| = {dev:.2e}")
    sys.stdout.flush()


def test_ps_shape():
    """Test 1: P_S(k) full shape for one Higgs config."""
    print("\n[1/7] P_S(k) full shape — Higgs φ₀=5.70 y₀=-0.170 N*=52")
    m = HiggsModel(lam=0.13, xi=15000.0)
    k = build_weighted_kgrid(1e-5, 1.0, k_pivot_phys, n_dense=60, n_outer=30)
    ps_py, ps_nb, meta = _run_both(m, 5.70, -0.170, 52.0, k)
    if ps_py is None:
        print(f"  [FAIL] {meta}"); return
    dev = _ratio_dev(ps_nb, ps_py)
    _test("P_S shape", dev, _TOL_CRITICAL["ps_shape"], _TOL_WARN["ps_shape"])


def test_camb_dell():
    """Test 2: D_ell and chi2 via CAMB."""
    print("\n[2/7] D_ell + χ² via CAMB")
    from scripts.camb_wrapper import compute_cl_full_camb, compute_chi2_camb
    from scripts.planck_data import C_ell_to_d_ell

    m = HiggsModel(lam=0.13, xi=15000.0)
    k = build_weighted_kgrid(1e-5, 1.0, k_pivot_phys, n_dense=60, n_outer=30)
    ps_py, ps_nb, _ = _run_both(m, 5.70, -0.170, 52.0, k)
    if ps_py is None:
        print(f"  [FAIL] pipeline failed"); return

    data_py = {"k_phys": k, "P_S": ps_py}
    data_nb = {"k_phys": k, "P_S": ps_nb}

    try:
        ells, C_tt_py, _, _ = compute_cl_full_camb(data_py, ell_max=2500)
        _, C_tt_nb, _, _ = compute_cl_full_camb(data_nb, ell_max=2500)
    except Exception as e:
        print(f"  [FAIL] CAMB: {e}"); return

    D_py = C_ell_to_d_ell(ells, C_tt_py)
    D_nb = C_ell_to_d_ell(ells, C_tt_nb)
    dev = float(np.max(np.abs(D_nb / D_py - 1.0)))
    _test("D_ell", dev, _TOL_CRITICAL["dell"], _TOL_WARN["dell"])

    try:
        chi2_m, chi2_l, dchi2 = compute_chi2_camb(data_nb, ell_max=29)
        chi2_m_py, chi2_l_py, dchi2_py = compute_chi2_camb(data_py, ell_max=29)
    except Exception as e:
        print(f"  [WARN] χ² compute failed: {e}"); return

    chi2_dev = abs(dchi2 - dchi2_py)
    chi2_status = "PASS" if chi2_dev < _TOL_CRITICAL["chi2"] else "FAIL"
    print(f"  [{chi2_status}] Δχ² diff = {chi2_dev:.2e} "
          f"(numba={dchi2:.2f}, python={dchi2_py:.2f})")
    if chi2_dev >= _TOL_CRITICAL["chi2"]:
        global FAIL; FAIL += 1
    else:
        global PASS; PASS += 1


def test_random_sweep(n_configs=20):
    """Test 3: Random config sweep."""
    print(f"\n[3/7] Random sweep ({n_configs} configs)")
    rng = np.random.RandomState(42)
    m = HiggsModel(lam=0.13, xi=15000.0)
    k = build_weighted_kgrid(1e-5, 1.0, k_pivot_phys, n_dense=20, n_outer=10)

    max_dev = 0.0
    for i in range(n_configs):
        phi0 = rng.uniform(5.5, 7.5)
        y0 = -rng.uniform(0.05, 0.8)
        N_star = rng.uniform(50, 60)
        ps_py, ps_nb, _ = _run_both(m, phi0, y0, N_star, k)
        if ps_py is None:
            print(f"  [{i+1}/{n_configs}] SKIP φ₀={phi0:.2f} y₀={y0:+.3f} N*={N_star:.1f}")
            continue
        dev = _ratio_dev(ps_nb, ps_py)
        if dev > max_dev:
            max_dev = dev
        if (i+1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{n_configs}] max dev so far = {max_dev:.2e}  "
                  f"(last: φ₀={phi0:.2f})", flush=True)

    _test(f"Sweep (n={n_configs})", max_dev, _TOL_CRITICAL["sweep"], _TOL_WARN["sweep"])


def test_large_grid():
    """Test 4: Large weighted grid."""
    print("\n[4/7] Large weighted grid (n_dense=200, n_outer=100)")
    m = HiggsModel(lam=0.13, xi=15000.0)
    k = build_weighted_kgrid(1e-5, 1.0, k_pivot_phys, n_dense=200, n_outer=100)
    ps_py, ps_nb, _ = _run_both(m, 5.70, -0.170, 52.0, k)
    if ps_py is None:
        print(f"  [FAIL] pipeline failed"); return
    dev = _ratio_dev(ps_nb, ps_py)
    _test(f"Large grid ({len(k)} modes)", dev, _TOL_CRITICAL["large"], _TOL_WARN["large"])


def test_punctuated():
    """Test 5: Punctuated model."""
    print("\n[5/7] Punctuated model (reference only)")
    m = PunctuatedInflationModel(m=1.1323e-7, lam=3.3299e-15)
    k = build_weighted_kgrid(1e-5, 1.0, k_pivot_phys, n_dense=40, n_outer=20)
    ps_py, ps_nb, _ = _run_both(m, 12.0, 0.0, 77.2, k, ms_steps=10000)
    if ps_py is None:
        print(f"  [SKIP] Punctuated pipeline failed (expected low N_total)")
        return
    dev = _ratio_dev(ps_nb, ps_py)
    _test("Punctuated", dev, _TOL_CRITICAL["punctuated"], _TOL_WARN["punctuated"])


def test_jit_timing():
    """Test 6: JIT timing breakdown (local only, 2 runs)."""
    print("\n[6/7] JIT timing breakdown")
    m = HiggsModel(lam=0.13, xi=15000.0)
    k = build_weighted_kgrid(1e-5, 1.0, k_pivot_phys, n_dense=40, n_outer=20)
    for label in ["First (includes JIT)", "Second (cached)"]:
        t0 = time.time()
        r = run_pspectrum_pipeline(
            m, phi0=5.70, y0=-0.170, N_star=52.0,
            k_phys_grid=k, use_numba=True,
            n_workers=1, save_outputs=False,
        )
        t = time.time() - t0
        ok = r["metadata"]["n_completed"]
        print(f"  {label}: {t:.1f}s, {ok}/{len(k)} modes")
        sys.stdout.flush()


def test_parallel_scaling():
    """Test 7: Parallel scaling (lab machine, 300 modes)."""
    print("\n[7/7] Parallel scaling (300 modes)")
    k = build_weighted_kgrid(1e-5, 1.0, k_pivot_phys, n_dense=200, n_outer=100)
    print(f"  Grid: {len(k)} modes")
    m = HiggsModel(lam=0.13, xi=15000.0)

    for solver in ["Python", "Numba"]:
        for nw in [1, 4, 8, 16]:
            use_nb = (solver == "Numba")
            t0 = time.time()
            r = run_pspectrum_pipeline(
                m, phi0=5.70, y0=-0.170, N_star=52.0,
                k_phys_grid=k, use_numba=use_nb,
                n_workers=nw, save_outputs=False,
            )
            t = time.time() - t0
            ok = r["metadata"]["n_completed"]
            rate = ok / max(t, 0.01)
            print(f"  {solver:>8} {nw:>2d}w: {t:>6.1f}s  {rate:>6.1f} modes/s  "
                  f"({ok}/{len(k)} ok)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Numba MS solver test")
    parser.add_argument("--lab", action="store_true", help="Run full suite including scaling")
    parser.add_argument("--sweep-configs", type=int, default=20, help="Configs in random sweep")
    parser.add_argument("--skip-camb", action="store_true", help="Skip CAMB test (slow)")
    args = parser.parse_args()

    from numba_ms_solver import HAVE_NUMBA
    if not HAVE_NUMBA:
        print("ERROR: Numba not available. Install: conda install numba")
        sys.exit(1)

    print(f"Numba MS Solver — Comprehensive Test Suite")
    print(f"{'='*60}")
    print(f"  Lab mode: {args.lab}")
    print(f"  Sweep configs: {args.sweep_configs}")
    print(f"  Tolerances (critical): {_TOL_CRITICAL}")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Tests 1-5: correctness (always run)
    test_ps_shape()
    if not args.skip_camb:
        test_camb_dell()
    test_random_sweep(n_configs=args.sweep_configs)
    test_large_grid()
    test_punctuated()

    # Test 6: timing (local + lab)
    test_jit_timing()

    # Test 7: scaling (lab only)
    if args.lab:
        test_parallel_scaling()
    else:
        print("\n[7/7] SKIP (use --lab for parallel scaling)")

    # Summary
    total = PASS + FAIL + WARN
    print(f"\n{'='*60}")
    print(f"  Results: {PASS} pass, {FAIL} fail, {WARN} warn ({total} total)")
    print(f"{'='*60}")
    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()
