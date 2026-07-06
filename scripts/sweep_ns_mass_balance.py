#!/usr/bin/env python3
"""Targeted sweep: can we break the n_s-mass degeneracy with x_c in (0.785, 0.79)?

Tests: x_c=0.786,0.787,0.788 at multiple β and N* to see if we can get
n_s > 0.96 with sub-solar PBH (not grid-boundary).
"""

import os
import json
import time
import argparse
import numpy as np
from scipy.special import erfc
from scripts.constants import ROOT_DIR
from scripts.observables import extract_ns, interpolate_As, extract_pbh_peak

ACCRETION = 3e7; K_EQ = 0.0104; M_EQ = 3.0e17; GAMMA = 0.4
C = 0.77

ZETA_C_VALS = [0.05, 0.055, 0.06, 0.065, 0.068, 0.07, 0.072,
               0.075, 0.077, 0.078, 0.08, 0.083, 0.085, 0.1]

def _pbh_weighted_kgrid():
    k_cmb = np.logspace(np.log10(0.005), np.log10(0.5), 15)
    k_pbh = np.logspace(6, 22, 350)
    return np.unique(np.concatenate([k_cmb, k_pbh]))

def run_one(xc, beta, nstar, chi0=8.0, workers=8, k_pivot=0.05,
         ns_window=3.0, ns_method='lsq'):
    from models.ezquiaga_chi import EzquiagaCHIModel, inflection_parameters
    from inf_dyn_background import run_background_simulation, get_derived_quantities
    from pspectrum_pipeline import run_pspectrum_pipeline

    model = EzquiagaCHIModel(c=C)
    a, b = inflection_parameters(xc, C, beta=beta)
    model.a, model.b = a, b
    model.v0 = model._V0 * model.a / (model.b * model.c) ** 2
    model.x0 = chi0; model.y0 = -1e-4
    model.patch_background_solver()

    T = np.linspace(0, model.T_max, model.bg_steps)
    bg_sol = run_background_simulation(model, T)
    derived = get_derived_quantities(bg_sol, model)
    epsH = derived['epsH']
    end_candidates = np.where(epsH >= 1.0)[0]
    end_idx = int(end_candidates[0]) if len(end_candidates) > 0 else len(epsH) - 1
    N_total = float(derived['N'][end_idx])

    k_grid = _pbh_weighted_kgrid()
    result = run_pspectrum_pipeline(
        model=model, phi0=None, y0=None,
        k_phys_grid=k_grid, k_pivot_phys=k_pivot,
        N_star=nstar, k_start_factor=100.0, ms_steps=5000,
        normalize_to_As=False, save_outputs=False, n_workers=workers,
    )

    k_phys = np.array(result['k_phys'])
    P_S = np.array(result['P_S'])

    # n_s via chosen method (window fit or log-derivative at k_pivot).
    n_s, _ns_meta = extract_ns(k_phys, P_S, k_pivot=k_pivot,
                               ns_window=ns_window if ns_method == "lsq" else None,
                               method=ns_method)
    A_s = interpolate_As(k_phys, P_S, k_pivot)

    # P_S peak
    pbh_mask = k_phys > 1.0
    if np.any(pbh_mask):
        ps_pbh = P_S[pbh_mask]; k_pbh = k_phys[pbh_mask]
        peak_i = int(np.argmax(ps_pbh))
        k_peak = float(k_pbh[peak_i]); ps_peak = float(ps_pbh[peak_i])
        on_b = bool(k_peak <= 1e6*1.01 or k_peak >= 1e22*0.99)
    else:
        k_peak, ps_peak, on_b = np.nan, np.nan, False

    # PBH abundance sweep
    results = []
    for zc in ZETA_C_VALS:
        bf = erfc(zc / np.sqrt(2 * np.clip(P_S, 1e-300, None)))
        beq = np.clip(bf * k_phys / K_EQ, 0.0, 1.0)
        M = GAMMA * M_EQ * (K_EQ / k_phys) ** 2 * ACCRETION
        ok = np.isfinite(M) & np.isfinite(beq) & (M > 0)
        M_ok, beq_ok = M[ok], beq[ok]
        order = np.argsort(M_ok)
        M_ok, beq_ok = M_ok[order], beq_ok[order]
        f_total = float(np.trapezoid(beq_ok, np.log(np.maximum(M_ok, 1e-300))))
        peak_i = int(np.argmax(beq_ok))
        M_peak = float(M_ok[peak_i]) if len(M_ok) > 0 else np.nan
        results.append({'zeta_c': zc, 'f_total': f_total, 'M_peak': M_peak})

    viable = [r for r in results if r['f_total'] <= 1.0 and r['f_total'] > 1e-6]
    best = max(viable, key=lambda r: r['f_total']) if viable else None

    return {
        'xc': xc, 'c': C, 'beta': beta, 'chi0': chi0, 'N_star': nstar,
        'N_total': N_total, 'n_s': n_s, 'A_s': A_s,
        'k_pivot': k_pivot, 'ns_window': None,
        'k_peak': k_peak, 'P_S_peak': ps_peak, 'on_grid_boundary': on_b,
        'best_zeta_c': best['zeta_c'] if best else None,
        'f_total_best': best['f_total'] if best else 0,
        'M_peak_best': best['M_peak'] if best else np.nan,
        'max_f_total': max(r['f_total'] for r in results),
        'all_zeta': results,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=8)
    p.add_argument(
        '--k-pivot', type=float, default=0.05,
        help='Pivot k_pivot_phys (Mpc^-1) — drives BOTH As normalization and '
             'n_s extraction. Planck default 0.05; Higgs low-ell 0.002.',
    )
    p.add_argument(
        '--ns-window', type=float, default=3.0,
        help='Fit half-width for lsq method [k_pivot/w, k_pivot*w] (ignored for derivative)',
    )
    p.add_argument(
        '--ns-method', choices=['lsq', 'derivative'], default='lsq',
        help="n_s extraction method: lsq=window fit (default), derivative=log-derivative at k_pivot",
    )
    args = p.parse_args()

    xc_vals = [0.786, 0.787, 0.788]
    beta_vals = [5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4]
    nstar_vals = [55, 60, 65]

    total = len(xc_vals) * len(beta_vals) * len(nstar_vals)
    print(f'Targeted n_s–mass sweep: {total} configs')
    print(f'  x_c = {xc_vals}')
    print(f'  β   = {beta_vals}')
    print(f'  N*  = {nstar_vals}')

    out_dir = os.path.join(ROOT_DIR, 'outputs/simulations/logs')
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'sweep_ns_mass_balance.jsonl')

    all_results = []
    t0 = time.time()
    completed = 0

    for xc in xc_vals:
        for beta in beta_vals:
            for nstar in nstar_vals:
                completed += 1
                print(f'\n[{completed}/{total}] x_c={xc} β={beta:.0e} N*={nstar}')
                try:
                    r = run_one(xc, beta, nstar, workers=args.workers,
                                k_pivot=args.k_pivot,
                                ns_window=args.ns_window,
                                ns_method=args.ns_method)
                    all_results.append(r)

                    # Write log entry
                    entry = {k: v for k, v in r.items() if k != 'all_zeta'}
                    entry['zeta_c'] = r['best_zeta_c']
                    entry['f_total'] = r['f_total_best']
                    entry['M_present'] = r['M_peak_best']
                    with open(log_path, 'a') as f:
                        json.dump(entry, f); f.write('\n')

                    M_str = f"{r['M_peak_best']:.2e}" if r['M_peak_best'] else 'N/A'
                    bf_str = f"{r['f_total_best']:.4e}" if r['f_total_best'] else '0'
                    flag = ' B' if r['on_grid_boundary'] else '  '
                    print(f"  n_s={r['n_s']:.4f} N_total={r['N_total']:.1f} M={M_str} f={bf_str} k_peak={r['k_peak']:.2e}{flag}")

                except Exception as e:
                    print(f'  ERROR: {e}')
                    import traceback; traceback.print_exc()

    elapsed = time.time() - t0
    print(f'\n{"="*80}')
    print(f'Completed {completed}/{total} in {elapsed:.0f}s ({elapsed/max(completed,1):.1f}s/config)')

    # Summary table
    print(f'\n{"xc":<6} {"β":>10} {"N*":<5} {"Ntot":<7} {"n_s":<7} {"M_peak":<12} {"f_best":<10} {"k_peak":<12} {"flag"}')
    print('-' * 75)
    for r in all_results:
        flag = 'B' if r['on_grid_boundary'] else '.'
        M = f"{r['M_peak_best']:.2e}" if r['M_peak_best'] and not np.isnan(r['M_peak_best']) else '—'
        f = f"{r['f_total_best']:.3e}" if r['f_total_best'] else '0'
        print(f'{r["xc"]:<6.3f} {r["beta"]:>10.0e} {r["N_star"]:<5.0f} {r["N_total"]:<7.1f} {r["n_s"]:<7.4f} {M:<12} {f:<10} {r["k_peak"]:<12.2e} {flag}')

if __name__ == '__main__':
    main()
