import os
import sys
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.higgs import HiggsModel
import inf_dyn_background as bg_solver
from numba_ms_solver import numba_run_ms, build_numba_splines
from pspectrum_pipeline import (
    run_pspectrum_pipeline,
    find_end_of_inflation,
    extract_mode_initial_conditions
)
from fortran_ms_solver import fortran_run_ms_grid, HAVE_FORTRAN
from scripts.camb_wrapper import compute_cl_full_camb, compute_chi2_camb
from scripts.planck_data import C_ell_to_d_ell

if not HAVE_FORTRAN:
    print("CRITICAL: Fortran module is not compiled. Please run `make` in fortran/ first.")
    sys.exit(1)

# Import compiled module for Level 1 wrapper call
import ms_solver_fort as _fort

# Publication-ready plotting setup
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Inter', 'Outfit', 'Roboto']
})

def run_level1_single_mode():
    print("\n=== LEVEL 1: Single-Mode Verification ===")
    model = HiggsModel()
    model.x0 = 5.70
    model.y0 = -0.170
    T_span_bg = np.linspace(0.0, model.T_max, model.bg_steps)
    bg_sol = bg_solver.run_background_simulation(model, T_span_bg)
    derived_bg = bg_solver.get_derived_quantities(bg_sol, model)
    end_idx = find_end_of_inflation(derived_bg["epsH"])
    if end_idx == -1:
        end_idx = len(T_span_bg) - 1

    # Pivot scale in physical and code units
    k_phys = 0.002
    # We need to find k_pivot_code first to scale it
    N_total = derived_bg["N"][end_idx]
    N_pivot = N_total - 52.0  # N_star = 52
    pivot_bg_idx = int(np.argmin(np.abs(derived_bg["N"][:end_idx] - N_pivot)))
    k_pivot_code = np.exp(bg_sol[3][pivot_bg_idx]) * bg_sol[2][pivot_bg_idx]
    k_code = k_pivot_code * (k_phys / 0.002)

    # Initial conditions
    k_start_factor = 100.0
    xi_mode, y0_mode, zi_mode, ni_mode, t_start, t_end, start_idx = extract_mode_initial_conditions(
        bg_sol, T_span_bg, end_idx, k_code, k_start_factor
    )
    
    ms_steps = 5000
    T_ms = np.linspace(t_start, t_end, ms_steps)
    k_rel = k_code * np.exp(-ni_mode)

    # Solve Numba
    bg_coefs = build_numba_splines(bg_sol, T_span_bg)
    out_numba = numba_run_ms(bg_sol, T_span_bg, T_ms, ni_mode, k_code, model, bg_coefs=bg_coefs)

    # Solve Fortran via wrapper
    from fortran_ms_solver import _pack_spline_coefs
    bc_arr = _pack_spline_coefs(bg_coefs)
    
    # y0 state initialization
    yv = zi_mode / k_rel
    vi = 1.0 / np.sqrt(2.0 * k_rel)
    y0_state = np.array([vi, k_rel/np.sqrt(2.0*k_rel)*yv, yv*vi, -k_rel/np.sqrt(2.0*k_rel)*(1-yv*yv),
                         vi, k_rel/np.sqrt(2.0*k_rel)*yv, yv*vi, -k_rel/np.sqrt(2.0*k_rel)*(1-yv*yv)], dtype=np.float64)

    out_fort = _fort.integrate_dp5_wrapper(
        y0_state, t_start, t_end, T_ms, bc_arr, k_rel, ni_mode, model.S, model.v0, model.alpha, 0
    )

    # Relative errors
    abs_diff = np.abs(out_fort - out_numba)
    rel_err = abs_diff / np.maximum(np.abs(out_numba), 1e-30)
    max_rel_err = np.max(rel_err)

    # Compute final P_S, P_T
    y_end_numba = out_numba[:, -1]
    y_end_fort = out_fort[:, -1]
    
    y_bg_end = bg_sol[1][end_idx]
    z_bg_end = bg_sol[2][end_idx]
    n_end_rel = bg_sol[3][end_idx] - ni_mode
    epsH = max(y_bg_end**2 / (2.0 * z_bg_end**2), 1e-30)
    inv_A2 = np.exp(-2.0 * n_end_rel)
    
    # Numba values
    zeta2_nb = (y_end_numba[0]**2 + y_end_numba[2]**2) * inv_A2 * (model.S**2) / (2.0 * epsH)
    PS_nb = (k_rel**3 * zeta2_nb) / (2.0 * np.pi**2)
    h2_nb = (y_end_numba[4]**2 + y_end_numba[6]**2) * inv_A2 * (model.S**2)
    PT_nb = 4.0 * (k_rel**3 * h2_nb) / (np.pi**2)

    # Fortran values
    zeta2_ft = (y_end_fort[0]**2 + y_end_fort[2]**2) * inv_A2 * (model.S**2) / (2.0 * epsH)
    PS_ft = (k_rel**3 * zeta2_ft) / (2.0 * np.pi**2)
    h2_ft = (y_end_fort[4]**2 + y_end_fort[6]**2) * inv_A2 * (model.S**2)
    PT_ft = 4.0 * (k_rel**3 * h2_ft) / (np.pi**2)

    err_ps = np.abs(PS_ft - PS_nb) / PS_nb
    err_pt = np.abs(PT_ft - PT_nb) / PT_nb

    print(f"Max relative ODE trajectory error: {max_rel_err:.4e}")
    print(f"Final P_S relative error: {err_ps:.4e}")
    print(f"Final P_T relative error: {err_pt:.4e}")

    assert max_rel_err < 0.1, f"Level 1 failed: trajectory error {max_rel_err:.4e} >= 0.1"
    assert err_ps < 1e-5, f"Level 1 failed: P_S error {err_ps:.4e} >= 1e-5"
    assert err_pt < 1e-5, f"Level 1 failed: P_T error {err_pt:.4e} >= 1e-5"
    print("Level 1 validation: PASS")
    return max_rel_err, err_ps, err_pt


def run_level2_grid_and_level3_camb():
    print("\n=== LEVEL 2 & 3: Grid & Pipeline Validation ===")
    configs = [
        {"label": "Best chi2", "phi0": 6.40, "y0": -0.475, "nstar": 59},
        {"label": "Best D2", "phi0": 5.75, "y0": -0.170, "nstar": 55},
        {"label": "Best balance", "phi0": 5.70, "y0": -0.170, "nstar": 52}
    ]

    for cfg in configs:
        label = cfg["label"]
        print(f"\nConfig: {label} (phi0={cfg['phi0']}, y0={cfg['y0']}, N_star={cfg['nstar']})")
        model = HiggsModel()
        model.x0 = cfg["phi0"]
        model.y0 = cfg["y0"]
        
        # 1. Run baseline Numba pipeline
        print("  Running Numba baseline...")
        res_nb = run_pspectrum_pipeline(
            model=model, phi0=cfg["phi0"], y0=cfg["y0"], N_star=cfg["nstar"],
            use_numba=True, backend='numba', save_outputs=False
        )
        
        # 2. Run Fortran pipeline
        print("  Running Fortran backend...")
        res_ft = run_pspectrum_pipeline(
            model=model, phi0=cfg["phi0"], y0=cfg["y0"], N_star=cfg["nstar"],
            use_numba=False, backend='fortran', save_outputs=False
        )

        # 3. Compare grid results (Level 2)
        ps_nb = res_nb["P_S"]
        ps_ft = res_ft["P_S"]
        pt_nb = res_nb["P_T"]
        pt_ft = res_ft["P_T"]
        
        start_idx_nb = res_nb["metadata"]["pivot_bg_idx"] # Wait, need start index array
        # Let's compare the arrays directly
        # Find raw P_S, P_T (before normalization) to check solver output directly
        # Wait, the pipeline res contains normalized P_S and P_T. Since normalization is As/P_S_pivot,
        # relative errors should be identical.
        rel_diff_ps = np.abs(ps_ft - ps_nb) / ps_nb
        max_rel_ps = np.nanmax(rel_diff_ps)
        mean_rel_ps = np.nanmean(rel_diff_ps)
        
        rel_diff_pt = np.abs(pt_ft - pt_nb) / pt_nb
        max_rel_pt = np.nanmax(rel_diff_pt)
        
        print(f"  P_S max rel diff: {max_rel_ps:.4e}")
        print(f"  P_S mean rel diff: {mean_rel_ps:.4e}")
        print(f"  P_T max rel diff: {max_rel_pt:.4e}")
        
        assert max_rel_ps < 1e-4, f"Level 2 failed for P_S: max diff {max_rel_ps:.4e} >= 1e-4"
        assert mean_rel_ps < 5e-5, f"Level 2 failed for P_S: mean diff {mean_rel_ps:.4e} >= 5e-5"
        assert max_rel_pt < 1e-4, f"Level 2 failed for P_T: max diff {max_rel_pt:.4e} >= 1e-4"
        
        # 4. Compare CAMB (Level 3)
        print("  Running CAMB evaluations...")
        # Prepare spectrum data dict
        ps_data_nb = {"k_phys": res_nb["k_phys"], "P_S": ps_nb}
        ps_data_ft = {"k_phys": res_ft["k_phys"], "P_S": ps_ft}
        
        ells_nb, C_TT_nb, _, _ = compute_cl_full_camb(ps_data_nb, ell_max=2500)
        ells_ft, C_TT_ft, _, _ = compute_cl_full_camb(ps_data_ft, ell_max=2500)
        
        D_TT_nb = C_ell_to_d_ell(ells_nb, C_TT_nb)
        D_TT_ft = C_ell_to_d_ell(ells_ft, C_TT_ft)
        
        D2_nb = D_TT_nb[ells_nb == 2][0]
        D2_ft = D_TT_ft[ells_ft == 2][0]
        
        chi2_nb_m, _, _ = compute_chi2_camb(ps_data_nb, ell_max=29)
        chi2_ft_m, _, _ = compute_chi2_camb(ps_data_ft, ell_max=29)
        
        print(f"  Numba D2: {D2_nb:.4f} uK^2 | Fortran D2: {D2_ft:.4f} uK^2 (diff: {abs(D2_ft-D2_nb):.4f})")
        print(f"  Numba chi2 (l<30): {chi2_nb_m:.4f} | Fortran chi2: {chi2_ft_m:.4f} (diff: {abs(chi2_ft_m-chi2_nb_m):.4f})")
        
        assert abs(D2_ft - D2_nb) < 0.1, f"Level 3 failed for D2: difference {abs(D2_ft-D2_nb):.4e} >= 0.1 uK^2"
        assert abs(chi2_ft_m - chi2_nb_m) < 0.01, f"Level 3 failed for chi2: difference {abs(chi2_ft_m-chi2_nb_m):.4e} >= 0.01"

        # 5. Plot comparisons for the Best Balance config
        if label == "Best balance":
            # Plot 1: P_S comparison
            os.makedirs("outputs/plots/diagnostics", exist_ok=True)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6), gridspec_kw={'height_ratios': [2, 1]})
            
            ax1.loglog(res_nb["k_phys"], ps_nb, color='#1f77b4', linestyle='-', linewidth=2.0, label='Numba JIT')
            ax1.loglog(res_ft["k_phys"], ps_ft, color='#ff7f0e', linestyle='--', linewidth=2.0, label='Fortran (OpenMP)')
            ax1.set_ylabel(r'$P_\mathcal{R}(k)$')
            ax1.set_title('Primordial Power Spectrum Comparison')
            ax1.legend(loc='lower left')
            ax1.grid(True, which='both', linestyle=':', alpha=0.5)
            
            ax2.semilogx(res_nb["k_phys"], rel_diff_ps, color='#2ca02c', linewidth=1.5)
            ax2.axhline(1e-5, color='r', linestyle=':', label='Threshold (1e-5)')
            ax2.set_xlabel(r'$k\ [Mpc^{-1}]$')
            ax2.set_ylabel(r'$|\Delta P_\mathcal{R} / P_\mathcal{R}|$')
            ax2.legend(loc='upper right')
            ax2.grid(True, which='both', linestyle=':', alpha=0.5)
            
            plt.tight_layout()
            ps_plot_path = "outputs/plots/diagnostics/fortran_vs_numba_ps_comparison.png"
            plt.savefig(ps_plot_path, dpi=300)
            plt.close()
            print(f"  Saved P_S comparison plot: {ps_plot_path}")

            # Plot 2: D_ell comparison
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6), gridspec_kw={'height_ratios': [2, 1]})
            
            # Linear for l < 30, log for l > 30 is good, but for comparison simple semilogx is clean
            ax1.semilogx(ells_nb, D_TT_nb, color='#1f77b4', linestyle='-', linewidth=2.0, label='Numba JIT')
            ax1.semilogx(ells_ft, D_TT_ft, color='#ff7f0e', linestyle='--', linewidth=2.0, label='Fortran (OpenMP)')
            ax1.set_ylabel(r'$D_\ell^{TT}\ [\mu K^2]$')
            ax1.set_title('Angular Power Spectrum Comparison')
            ax1.legend(loc='lower right')
            ax1.grid(True, which='both', linestyle=':', alpha=0.5)
            
            ax2.semilogx(ells_nb, np.abs(D_TT_ft - D_TT_nb), color='#d62728', linewidth=1.5)
            ax2.axhline(0.1, color='gray', linestyle=':', label='Threshold (0.1 uK^2)')
            ax2.set_xlabel(r'$\ell$')
            ax2.set_ylabel(r'$|\Delta D_\ell^{TT}|\ [\mu K^2]$')
            ax2.legend(loc='upper right')
            ax2.grid(True, which='both', linestyle=':', alpha=0.5)
            
            plt.tight_layout()
            dell_plot_path = "outputs/plots/diagnostics/fortran_vs_numba_dell_comparison.png"
            plt.savefig(dell_plot_path, dpi=300)
            plt.close()
            print(f"  Saved D_ell comparison plot: {dell_plot_path}")

    print("Level 2 & 3 validation: PASS")


def run_level4_benchmark():
    print("\n=== LEVEL 4: Performance Benchmark ===")
    model = HiggsModel()
    model.x0 = 5.70
    model.y0 = -0.170
    
    T_span_bg = np.linspace(0.0, model.T_max, model.bg_steps)
    bg_sol = bg_solver.run_background_simulation(model, T_span_bg)
    derived_bg = bg_solver.get_derived_quantities(bg_sol, model)
    end_idx = find_end_of_inflation(derived_bg["epsH"])
    if end_idx == -1:
        end_idx = len(T_span_bg) - 1

    # Large k-grid to stress-test parallelism (200 modes)
    k_grid = np.logspace(np.log10(1e-5), np.log10(1.0), 200)
    N_total = derived_bg["N"][end_idx]
    N_pivot = N_total - 52.0
    pivot_bg_idx = int(np.argmin(np.abs(derived_bg["N"][:end_idx] - N_pivot)))
    k_pivot_code = np.exp(bg_sol[3][pivot_bg_idx]) * bg_sol[2][pivot_bg_idx]
    k_codes = k_pivot_code * (k_grid / 0.002)

    # 1. Benchmark solver hot path (Numba vs Fortran)
    numba_times = []
    fortran_times = []
    
    # We do 5 runs, discard first, report median of remaining 4
    for r in range(5):
        # Numba
        bg_coefs = build_numba_splines(bg_sol, T_span_bg)
        # Numba has no grid solver directly, it uses serial or multiprocessing.
        # To measure the core serial overhead of Numba solver, let's time the serial grid loop in python
        # or use run_pspectrum_pipeline with n_workers=1.
        # But wait! To make a fair comparison:
        # Fortran solve_ms_grid is parallelized.
        # Numba can use n_workers=12 (parallel multiprocessing).
        # Let's benchmark the pipeline call itself to compare real end-to-end performance!
        
        # Pipeline Numba
        t0 = time.perf_counter()
        run_pspectrum_pipeline(
            model=model, phi0=5.70, y0=-0.170, N_star=52,
            use_numba=True, backend='numba', save_outputs=False, n_workers=12,
            k_phys_grid=k_grid
        )
        t_nb = time.perf_counter() - t0
        numba_times.append(t_nb)
        
        # Pipeline Fortran
        t0 = time.perf_counter()
        run_pspectrum_pipeline(
            model=model, phi0=5.70, y0=-0.170, N_star=52,
            use_numba=False, backend='fortran', save_outputs=False,
            k_phys_grid=k_grid
        )
        t_ft = time.perf_counter() - t0
        fortran_times.append(t_ft)
        
        print(f"  Run {r+1}/5: Numba = {t_nb:.3f}s | Fortran = {t_ft:.3f}s")

    # Discard first and get median
    nb_median = np.median(numba_times[1:])
    ft_median = np.median(fortran_times[1:])
    speedup = nb_median / ft_median
    
    print(f"\nMedian pipeline runtime (200 modes, 12 cores):")
    print(f"  Numba JIT:  {nb_median:.3f} s")
    print(f"  Fortran OM: {ft_median:.3f} s")
    print(f"  Speedup:    {speedup:.2f}x")

    # Save benchmark log
    os.makedirs("outputs/simulations/logs", exist_ok=True)
    bench_data = {
        "config": {"phi0": 5.70, "y0": -0.170, "nstar": 52},
        "num_k": 200,
        "omp_threads": 12,
        "numba_pipeline_median_s": nb_median,
        "fortran_pipeline_median_s": ft_median,
        "speedup_pipeline": speedup,
        "validation": "PASS"
    }
    with open("outputs/simulations/logs/fortran_benchmark.json", "w") as f:
        json.dump(bench_data, f, indent=2)
    print("Saved benchmark log to outputs/simulations/logs/fortran_benchmark.json")

    return bench_data

if __name__ == "__main__":
    print("Starting Fortran vs Numba Validation Suite...")
    run_level1_single_mode()
    run_level2_grid_and_level3_camb()
    run_level4_benchmark()
    print("\nALL VALIDATION LEVELS PASSED SUCCESSFULY!")
