#########################################################################################################
#
# Numerical Observables Calculation Module
#
# This module provides a high-level function to calculate ns and r by solving
# exact background and perturbation equations. 
#
#########################################################################################################

import numpy as np
import os
import json
import datetime
import uuid
from typing import Dict, Any

# Core solver imports
import inf_dyn_background as bg_solver
import inf_dyn_MS_full as ms_solver

def run_inflation_protocol(model, phi0: float, yi: float, delta: float = 1e-4, N_star: float = 60.0, output_dir: str = "outputs/results", T_span_bg: np.ndarray = None, save_to_file: bool = True) -> Dict[str, Any]:
    """
    Coordinates the calculation of cosmological observables by mapping them from the end of inflation.
    
    By evolving the background to find the exact moment when inflation ends (epsH = 1), 
    we trace back N_star e-folds to anchor our pivot scale (k_pivot). We then integrate the Mukhanov-Sasaki 
    perturbation equations across a narrow band around k_pivot, allowing us to capture any 
    transient non-slow-roll dynamics when computing the spectral index (n_s) and tensor-to-scalar ratio (r).
    """
    # Anchor the specified initial conditions to the model instance.
    model.phi0 = phi0
    model.yi = yi
    
    # Evolve the background fields to capture the full dynamic trajectory.
    if T_span_bg is None:
        T_span_bg = np.linspace(0, 5000, 10000)
        
    bg_sol = bg_solver.run_background_simulation(model, T_span_bg)
    derived_bg = bg_solver.get_derived_quantities(bg_sol, model)


    # Identify the exact end of inflation (where the slow-roll parameter epsH crosses 1).
    epsH = derived_bg['epsH']
    N_efolds = derived_bg['N']
    
    # To ignore early transients where epsH >= 1, we wait for inflation to start (epsH < 1),
    # and then find the first time it ends (epsH >= 1) afterwards.
    in_inflation = False
    end_idx = -1
    for i in range(len(epsH)):
        if not in_inflation:
            if epsH[i] < 1.0:
                in_inflation = True
        else:
            if epsH[i] >= 1.0:
                end_idx = i
                break

    if end_idx == -1:
        return {"status": "error", "message": f"Inflation did not end in window for phi0={phi0}, yi={yi}"}

    
    N_total = N_efolds[end_idx]

    # Map the pivot scale by rewinding exactly N_star e-folds from the end of inflation.
    # Check if we have enough inflation e-folds for N_star
    if N_total < N_star:
        return {"status": "error", "message": f"Total inflation ({N_total:.2f}) is less than N_star ({N_star})"}

    N_pivot = N_total - N_star
    pivot_idx = np.argmin(np.abs(N_efolds[:end_idx] - N_pivot))
    
    z_pivot = bg_sol[2][pivot_idx]
    a_pivot = np.exp(bg_sol[3][pivot_idx])
    k_pivot_code = a_pivot * z_pivot

    # Extract Slow Roll approximations at pivot scale
    ns_SR = derived_bg['ns'][pivot_idx]
    r_SR = derived_bg['r'][pivot_idx]
    
    # Evaluate the exact Mukhanov-Sasaki perturbations. 
    # We compute the power spectrum for k_pivot and its immediate neighbors to compute the precise spectral slope.
    # Using the corrected naming and delta logic
    ks_code_list = [k_pivot_code * (1 - delta), k_pivot_code, k_pivot_code * (1 + delta)]
    results = []
    
    # Build background interpolation once — avoids re-integrating background per k-mode
    bg_interp = ms_solver.build_bg_interpolators(bg_sol, T_span_bg)
    
    # Precompute log(a*z) for start-index lookup
    n_bg = bg_sol[3]
    z_bg = bg_sol[2]
    log_az = n_bg + np.log(z_bg)
    
    for k_code in ks_code_list:
        # Initial condition point: k = 100 a*z
        target_start = np.log(k_code) - np.log(100) 
        
        start_idx = np.argmin(np.abs(log_az[:end_idx] - target_start))
        start_idx = max(start_idx, 0)
        
        ni = bg_sol[3][start_idx]
        
        # Time span for this mode
        t_start = T_span_bg[start_idx]
        t_end = T_span_bg[end_idx]
        T_ms = np.linspace(t_start, t_end, 5000)
        
        # Solve MS perturbations only (background from interpolation)
        ms_sol = ms_solver.run_ms_simulation(bg_interp, ni, T_ms, k_code, model)
        derived = ms_solver.get_ms_derived_quantities_with_bg(ms_sol, bg_interp, T_ms, model, k_code, ni)
        results.append((derived['P_S'][-1], derived['P_T'][-1]))
    
    # Compute the final observables using a finite difference method across our narrow k-band.
    log_k = np.log(ks_code_list)
    log_Ps = np.log([res[0] for res in results])
    slope = (log_Ps[2] - log_Ps[0]) / (log_k[2] - log_k[0])
    ns = 1 + slope
    r_val = results[1][1] / results[1][0]
    
    # Persist the configuration and generated observables for downstream analysis.
    if save_to_file:
        output_path = save_results_to_json(model, ns, r_val, ns_SR, r_SR, delta, k_pivot_code, N_total, N_efolds[pivot_idx], results, ks_code_list, output_dir)
    else:
        output_path = None
    
    return {
        "status": "success",
        "ns": float(ns),
        "r": float(r_val),
        "ns_SR": float(ns_SR),
        "r_SR": float(r_SR),
        "N_total": float(N_total),
        "P_S": float(results[1][0]),
        "output_file": output_path
    }

def save_results_to_json(model, ns, r_val, ns_SR, r_SR, delta, k_pivot_code, N_total, N_pivot, results_list, ks_code_list, output_dir="outputs/results") -> str:
    """Utility function to save the standardized JSON output."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    run_id = str(uuid.uuid4())[:8]
    data = {
        "metadata": {
            "run_id": run_id, 
            "timestamp": datetime.datetime.now().isoformat(),
            "description": "Batch automated calculation of ns and r from exact MS equations."
        },
        "model_parameters": {
            "name": model.name, 
            "xi": getattr(model, 'xi_val', None), 
            "lambda": getattr(model, 'lam', None),
            "phi0": float(model.phi0), 
            "yi": float(model.yi), 
            "S": float(model.S)
        },
        "numerical_settings": {
            "delta_finite_difference": float(delta),
            "k_start_factor": 100.0
        },
        "background_info": {
            "total_efolds": float(N_total),
            "exit_N_simulation": float(N_pivot)
        },
        "observables": {
            "n_s": float(ns), 
            "r": float(r_val), 
            "n_s_SR": float(ns_SR),
            "r_SR": float(r_SR),
            "P_S_pivot": float(results_list[1][0]),
            "P_T_pivot": float(results_list[1][1])
        },
        "spectrum_scan": [
            {
                "label": label, 
                "k_code": float(k), 
                "P_S": float(Ps),
                "P_T": float(Pt)
            } 
            for (label, k, (Ps, Pt)) in zip(["k_minus", "k_pivot", "k_plus"], ks_code_list, results_list)
        ]
    }
    
    safe_name = model.name.replace(' ', '_').replace('(', '').replace(')', '')
    filename = f"{safe_name}_phi{model.phi0:.2f}_yi{model.yi:.3f}_run_{run_id}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
        
    return filepath
