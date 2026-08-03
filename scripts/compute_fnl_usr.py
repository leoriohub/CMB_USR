"""
compute_fnl_usr.py — f_NL^(local) for scales exiting during USR via phase-space δN.

The single-trajectory φ-only δN (scripts/compute_fnl.py) breaks down for modes
exiting during the non-attractor (USR) phase: there N is a function of BOTH
(φ, ẏ), and the flat-slice perturbation of a mode carries a velocity component
set by the linear dynamics, not by the attractor relation.

Method (tree-level local bispectrum — hybrid linear + classical):
  1. Background: one reference trajectory (x₀, y₀).
  2. For each scale N_exit: linear flat-gauge perturbation (δx, δy) at the
     slice N_slice = N_exit + offset, read from the Mukhanov-Sasaki mode
     function (v = a·δφ/S in flat gauge, so
        δx ∝ v·e^{-(n-ni)},   δy ∝ (v' − z·v)·e^{-(n-ni)} ).
  3. Phase-space response N(x, y): re-integrate 9 separate-universe patches
     from (x_s ± hx, y_s ± hy) to ε_H = 1; centered finite differences →
     N_x, N_y, N_xx, N_xy, N_yy.
  4. f_NL^(local)(k) = (5/6) (N_xx + 2 N_xy r + N_yy r²) / (2 (N_x + N_y r)²),
     with r = δy/δx the mode direction at the slice.
  5. Sanity anchors: r = r_adia ≡ ẏ/ẋ (time-shift direction) must give
     f_NL ≈ 0 (Maldacena consistency); deep-SR scales must give
     f_NL ≈ (5/12)(1 − n_s).

Usage:
    python scripts/compute_fnl_usr.py --x0 5.75 --y0 -0.170 --nstar 55 \
        --nexits 0.3,0.6,1.0,1.5,2.0,3.0,5.0,40.0 --json /tmp/fnl_usr.json
"""

import argparse
import json
import sys
import time

import numpy as np

from models.higgs import HiggsModel
from inf_dyn_background import run_background_simulation, get_derived_quantities
from pspectrum_pipeline import find_end_of_inflation, extract_mode_initial_conditions
from numba_ms_solver import numba_run_ms, build_numba_splines
from scripts.compute_fnl import compute_fnl


def frac_N_end(derived_bg, end_idx):
    """N_total via fractional interpolation at ε_H = 1 (matches compute_fnl.py)."""
    N_arr = derived_bg['N']
    epsH = derived_bg['epsH']
    N_total = float(N_arr[end_idx])
    if 0 < end_idx < len(epsH) and epsH[end_idx] > 1.0 >= epsH[end_idx - 1]:
        f = (1.0 - epsH[end_idx - 1]) / (epsH[end_idx] - epsH[end_idx - 1])
        if 0.0 <= f <= 1.0:
            N_total = float(N_arr[end_idx - 1] + f * (N_arr[end_idx] - N_arr[end_idx - 1]))
    return N_total


def run_patch_N_total(lam, xi, x_p, y_p, T_span):
    """N_total (e-folds from the patch start to the end of inflation).

    A separate-universe patch: homogeneous universe with initial data
    (x_p, y_p); z comes from the Friedmann constraint via
    InflationModel.get_initial_conditions().
    """
    model = HiggsModel(lam=lam, xi=xi)
    model.x0 = x_p
    model.y0 = y_p
    bg = run_background_simulation(model, T_span)
    der = get_derived_quantities(bg, model)
    end_idx = find_end_of_inflation(der['epsH'])
    if end_idx == -1:
        raise RuntimeError(f"patch (x={x_p:.6g}, y={y_p:.6g}) never ends inflation")
    return frac_N_end(der, end_idx)


def phase_space_response(model, bg_sol, derived_bg, end_idx, T_span_bg, N_slice, hx, hy):
    """9-patch response of N(x, y) at the slice N_slice -> (N_x, N_y, N_xx, N_xy, N_yy)."""
    N_arr = derived_bg['N']
    x_s = float(np.interp(N_slice, N_arr, bg_sol[0]))
    y_s = float(np.interp(N_slice, N_arr, bg_sol[1]))

    lam = model.lam
    xi = model.xi_val

    pts = {
        'c':  (x_s,       y_s),
        'xp': (x_s + hx,  y_s),
        'xm': (x_s - hx,  y_s),
        'yp': (x_s,       y_s + hy),
        'ym': (x_s,       y_s - hy),
        'pp': (x_s + hx,  y_s + hy),
        'pm': (x_s + hx,  y_s - hy),
        'mp': (x_s - hx,  y_s + hy),
        'mm': (x_s - hx,  y_s - hy),
    }
    N = {}
    for key, (x_p, y_p) in pts.items():
        N[key] = run_patch_N_total(lam, xi, x_p, y_p, T_span_bg)

    Nc = N['c']
    N_x = (N['xp'] - N['xm']) / (2.0 * hx)
    N_y = (N['yp'] - N['ym']) / (2.0 * hy)
    N_xx = (N['xp'] - 2.0 * Nc + N['xm']) / hx ** 2
    N_yy = (N['yp'] - 2.0 * Nc + N['ym']) / hy ** 2
    N_xy = (N['pp'] - N['pm'] - N['mp'] + N['mm']) / (4.0 * hx * hy)

    return {
        'N_slice': float(N_slice),
        'x_s': x_s, 'y_s': y_s,
        'N_center': Nc,
        'N_x': N_x, 'N_y': N_y,
        'N_xx': N_xx, 'N_xy': N_xy, 'N_yy': N_yy,
        'hx': hx, 'hy': hy,
        'patches': N,
    }


def mode_direction(model, bg_sol, derived_bg, T_span_bg, end_idx, k_code, N_slice,
                   bg_coefs=None, k_start_factor=100.0, ms_steps=4000):
    """Flat-gauge (δx, δy) direction r = δy/δx of the mode at N_slice.

    From the MS mode function: v = a·δφ/S (flat gauge) so
        δx ∝ v·e^{-(n-ni)},   δy ∝ (v' − z·v)·e^{-(n-ni)}
    Only the RATIO matters for f_NL (normalization-independent).
    """
    N_arr = derived_bg['N']
    z_s = float(np.interp(N_slice, N_arr, bg_sol[2]))
    t_slice = float(np.interp(N_slice, N_arr, T_span_bg))

    xi, y0v, zi, ni, t_start, t_end, start_idx = extract_mode_initial_conditions(
        bg_sol, T_span_bg, end_idx, k_code, k_start_factor
    )
    if t_slice <= t_start:
        raise ValueError(
            f"N_slice={N_slice:.3f} is before the MS start "
            f"(t_start={t_start:.4f}, t_slice={t_slice:.4f})"
        )
    T_ms = np.linspace(t_start, t_slice, ms_steps)
    ms = numba_run_ms(bg_sol, T_span_bg, T_ms, ni, k_code, model,
                      bg_coefs=bg_coefs, method='dp5')
    v_s = ms[0, -1]
    vT_s = ms[1, -1]
    u_s = ms[2, -1]
    uT_s = ms[3, -1]
    denom = v_s * v_s + u_s * u_s
    if denom < 1e-300:
        raise RuntimeError(f"mode ≈0 at slice (N_slice={N_slice:.3f})")
    # Covariance ratio ⟨δx δy⟩/⟨δx²⟩ at the slice (phase-robust):
    # δx ∝ v·e^{-(n-ni)}, δy ∝ (v' − z·v)·e^{-(n-ni)}  →
    # r = Re(v* v')/|v|² − z
    r = (v_s * vT_s + u_s * uT_s) / denom - z_s

    # Adiabatic (time-shift) direction from the background at the slice:
    # r_adia = ẏ/ẋ = (dydT)/y, with dydT = -3zy - v0·dfdx/S²
    x_s = float(np.interp(N_slice, N_arr, bg_sol[0]))
    y_s = float(np.interp(N_slice, N_arr, bg_sol[1]))
    dydT = -3.0 * z_s * y_s - model.v0 * model.dfdx(x_s) / model.S ** 2
    r_adia = dydT / y_s if abs(y_s) > 1e-300 else np.inf

    return {
        'r': float(r),
        'r_adia': float(r_adia),
        '|u/v|': float(abs(u_s / v_s)) if abs(v_s) > 1e-300 else float('inf'),
        'k_code': float(k_code),
        't_start': float(t_start),
        'N_ms_start': float(N_arr[start_idx]),
    }


def fnl_from_response(N_x, N_y, N_xx, N_xy, N_yy, r):
    """f_NL^(local) = (5/6)·ζ₂/ζ₁² with ζ₁ = N_x δx + N_y δy,
    ζ₂ = ½(N_xx δx² + 2 N_xy δx δy + N_yy δy²), r = δy/δx."""
    lin = N_x + N_y * r
    quad = N_xx + 2.0 * N_xy * r + N_yy * r ** 2
    if abs(lin) < 1e-300:
        return float('inf')
    return (5.0 / 6.0) * quad / (2.0 * lin ** 2)


def main():
    parser = argparse.ArgumentParser(
        description='f_NL^(local) for USR-exiting scales via phase-space δN'
    )
    parser.add_argument('--x0', type=float, default=5.75)
    parser.add_argument('--y0', type=float, default=-0.170)
    parser.add_argument('--nstar', type=float, default=55.0)
    parser.add_argument('--nexits', type=str,
                        default='0.15,0.3,0.6,1.0,1.5,2.0,3.0,5.0,10.0,40.0',
                        help='Comma-separated N_exit values (e-folds from start)')
    parser.add_argument('--slice-offset', type=float, default=1.0,
                        help='E-folds after exit where the δN slice sits '
                             '(must be DURING/just after USR for USR scales)')
    parser.add_argument('--hx', type=float, default=5e-3,
                        help='Finite-difference step in x (φ/M_P)')
    parser.add_argument('--hy', type=float, default=None,
                        help='Finite-difference step in y (default: 0.1·|y_s|)')
    parser.add_argument('--lam', type=float, default=0.13)
    parser.add_argument('--xi', type=float, default=15000.0)
    parser.add_argument('--bg-steps', type=int, default=50000)
    parser.add_argument('--ms-steps', type=int, default=4000)
    parser.add_argument('--k-start-factor', type=float, default=100.0)
    parser.add_argument('--json', type=str, default=None)
    args = parser.parse_args()

    nexits = [float(v) for v in args.nexits.split(',')]

    model = HiggsModel(lam=args.lam, xi=args.xi)
    model.x0 = args.x0
    model.y0 = args.y0
    T_span = np.linspace(0.0, model.T_max, args.bg_steps)

    t0 = time.time()
    print(f"Integrating reference background: x₀={args.x0}, y₀={args.y0} ...")
    bg_sol = run_background_simulation(model, T_span)
    derived_bg = get_derived_quantities(bg_sol, model)
    end_idx = find_end_of_inflation(derived_bg['epsH'])
    if end_idx == -1:
        print("ERROR: Inflation never ends or never starts.", file=sys.stderr)
        sys.exit(1)
    N_total = frac_N_end(derived_bg, end_idx)
    N_pivot = N_total - args.nstar
    print(f"  N_total={N_total:.6f}, N_pivot={N_pivot:.6f}, end_idx={end_idx}")

    bg_coefs = build_numba_splines(bg_sol, T_span, model=model)

    # k_phys calibration: physical pivot k is a fiducial convention; the
    # absolute normalization of a (Ai=1e-5) is arbitrary, so calibrate:
    # k_phys = k_code · (k_pivot_phys / k_pivot_code)
    from scripts.constants import k_pivot_phys
    n_p = float(np.interp(N_pivot, derived_bg['N'], bg_sol[3]))
    z_p = float(np.interp(N_pivot, derived_bg['N'], bg_sol[2]))
    k_pivot_code = float(np.exp(n_p) * z_p)
    k_cal = k_pivot_phys / k_pivot_code

    # Single-trajectory (interpolated) f_NL at the pivot, for comparison
    traj = compute_fnl(model, bg_sol, derived_bg, end_idx, args.nstar)
    print(f"  [old method] f_NL(trajectory, pivot) = {traj['f_NL']:.4f} "
          f"(SR x-check f_NL^SR={traj['f_NL_sr']:.4f})")

    # Center patch: one run shared by all scales
    x_c = float(np.interp(N_pivot, derived_bg['N'], bg_sol[0]))
    y_c = float(np.interp(N_pivot, derived_bg['N'], bg_sol[1]))
    N_center_shared = run_patch_N_total(args.lam, args.xi, x_c, y_c, T_span)
    print(f"  [consistency] center patch N={N_center_shared:.6f} vs "
          f"N_total−N_pivot={N_total - N_pivot:.6f}")

    print()
    header = (f"{'N_exit':>6} {'k_phys':>10} {'r_mode':>10} {'r_adia':>10} "
              f"{'f_NL(mode)':>10} {'f_NL(adia)':>10} {'N_x':>8} {'N_y':>8} "
              f"{'N_xx':>9} {'N_yy':>9}")
    print(header)
    print('-' * len(header))

    results = []
    for N_exit in nexits:
        if N_exit >= N_total - 2.0:
            print(f"  skip N_exit={N_exit} (too close to end)")
            continue
        # k exiting at N_exit: k_code = a·z = exp(n)·z
        n_e = float(np.interp(N_exit, derived_bg['N'], bg_sol[3]))
        z_e = float(np.interp(N_exit, derived_bg['N'], bg_sol[2]))
        k_code = float(np.exp(n_e) * z_e)

        N_slice = min(N_exit + args.slice_offset, N_total - 1.5)
        hy = args.hy if args.hy is not None else max(1e-6, 0.1 * abs(float(np.interp(N_slice, derived_bg['N'], bg_sol[1]))))
        try:
            md = mode_direction(model, bg_sol, derived_bg, T_span, end_idx,
                                k_code, N_slice, bg_coefs=bg_coefs,
                                k_start_factor=args.k_start_factor,
                                ms_steps=args.ms_steps)
            resp = phase_space_response(model, bg_sol, derived_bg, end_idx,
                                        T_span, N_slice, args.hx, hy)
            fnl_mode = fnl_from_response(resp['N_x'], resp['N_y'], resp['N_xx'],
                                         resp['N_xy'], resp['N_yy'], md['r'])
            fnl_adia = fnl_from_response(resp['N_x'], resp['N_y'], resp['N_xx'],
                                         resp['N_xy'], resp['N_yy'], md['r_adia'])
            k_phys = k_code * k_cal
            print(f"{N_exit:6.2f} {k_phys:10.3e} {md['r']:10.4g} {md['r_adia']:10.4g} "
                  f"{fnl_mode:10.4g} {fnl_adia:10.4g} {resp['N_x']:8.3g} {resp['N_y']:8.3g} "
                  f"{resp['N_xx']:9.3g} {resp['N_yy']:9.3g}")
            results.append({
                'N_exit': N_exit, 'k_phys': k_phys, 'N_slice': N_slice,
                'r_mode': md['r'], 'r_adia': md['r_adia'],
                'f_NL_mode': fnl_mode, 'f_NL_adia': fnl_adia,
                '|u/v|': md['|u/v|'],
                **{k2: resp[k2] for k2 in
                   ('N_x', 'N_y', 'N_xx', 'N_xy', 'N_yy', 'N_center', 'hx', 'hy')},
            })
        except Exception as e:
            print(f"{N_exit:6.2f} ERROR: {e}")
            results.append({'N_exit': N_exit, 'error': str(e)})

    print(f"\nTotal wall time: {time.time() - t0:.1f} s")
    if args.json:
        with open(args.json, 'w') as f:
            json.dump({
                'config': {'x0': args.x0, 'y0': args.y0, 'nstar': args.nstar,
                           'N_total': N_total, 'N_pivot': N_pivot},
                'trajectory_fnl': traj['f_NL'],
                'results': results,
            }, f, indent=2, default=str)
        print(f"Results saved to {args.json}")


if __name__ == '__main__':
    main()
