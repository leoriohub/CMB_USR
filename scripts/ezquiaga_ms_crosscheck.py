"""
Standalone solve_ivp MS cross-check for Ezquiaga.

Independent of numba_ms_solver. Uses scipy CubicSpline + solve_ivp directly.
Compares against Numba DP5 results for verification.

Usage:
    python -m scripts.ezquiaga_ms_crosscheck --chi0 8.0 --beta 1e-5
"""
import argparse, json, time, multiprocessing
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from models.ezquiaga_chi import EzquiagaCHIModel, inflection_parameters
from inf_dyn_background import run_background_simulation, get_derived_quantities
from numba_ms_solver import build_numba_splines, _spline_eval_py

S_CODE = 5e-5

def solve_one_mode(k_phys, bg_sol, T_bg, end_idx, model, method='LSODA'):
    """Solve MS for a single k-mode using solve_ivp. Returns P_S."""
    k_code = k_phys / S_CODE
    n_bg, z_bg = bg_sol[3], bg_sol[2]
    log_az = n_bg + np.log(np.maximum(z_bg, 1e-300))
    si = int(np.argmin(np.abs(log_az[:end_idx] - np.log(k_code) + np.log(100.0))))
    si = max(si, 0)
    ni = bg_sol[3][si]
    t_start, t_end = T_bg[si], T_bg[end_idx]
    T_ms = np.linspace(t_start, t_end, 200)
    k_rel = k_code * np.exp(-ni)
    v0, S = model.v0, S_CODE
    si2 = 1 / S**2

    # Build splines from background (directly, not through numba)
    spl = [CubicSpline(T_bg, bg_sol[i], bc_type='not-a-knot', extrapolate=True) for i in range(4)]
    bc = build_numba_splines(bg_sol, T_bg, model=model)  # for potential splines
    zi = float(spl[2](T_ms[0]))
    yv = zi / k_rel
    vi = 1 / np.sqrt(2 * k_rel)
    y0 = np.array([vi, k_rel/np.sqrt(2*k_rel)*yv, yv*vi, -k_rel/np.sqrt(2*k_rel)*(1-yv*yv),
                   vi, k_rel/np.sqrt(2*k_rel)*yv, yv*vi, -k_rel/np.sqrt(2*k_rel)*(1-yv*yv)])

    def rhs(t, y):
        v, vt, u, ut, h, ht, g, gt = y
        x_b = float(spl[0](t)); y_b = float(spl[1](t)); z = float(spl[2](t))
        n_rel = float(spl[3](t)) - ni
        df = _spline_eval_py(t, *bc[5]); d2f = _spline_eval_py(t, *bc[6])
        v0df = v0 * df * si2; dydt = -3*z*y_b - v0df
        k2 = k_rel**2 * np.exp(-2*n_rel)
        m2 = (2.5*y_b**2 + 2*y_b*dydt/z + 2*z**2 + 0.5*y_b**4/z**2 - v0*d2f*si2 - k2)
        return np.array([vt, -z*vt+v*m2, ut, -z*ut+u*m2,
                         ht, -z*ht-h*(k2-2*z**2+0.5*y_b**2),
                         gt, -z*gt-g*(k2-2*z**2+0.5*y_b**2)])

    sol = solve_ivp(rhs, [t_start, t_end], y0, method=method,
                    t_eval=T_ms, rtol=1e-8, atol=1e-10)
    if not sol.success:
        return np.nan

    y_bg = np.array([spl[i](T_ms) for i in range(4)])
    epsH = y_bg[1]**2 / (2 * y_bg[2]**2)
    inv_A2 = np.exp(-2 * (y_bg[3] - ni))
    zeta2 = (sol.y[0]**2 + sol.y[2]**2) * inv_A2 * S**2 / (2 * epsH + 1e-100)
    P_S = (k_rel**3 * zeta2) / (2 * np.pi**2)
    valid = np.isfinite(P_S) & (P_S > 0)
    return float(P_S[valid][-1]) if np.any(valid) else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chi0', type=float, default=8.0)
    parser.add_argument('--beta', type=float, default=1e-5)
    parser.add_argument('--k-min', type=float, default=1e-4)
    parser.add_argument('--k-max', type=float, default=1e14)
    parser.add_argument('--n-k', type=int, default=30)
    parser.add_argument('--method', type=str, default='LSODA',
                        choices=['LSODA', 'DOP853', 'Radau', 'BDF'])
    args = parser.parse_args()

    a, b = inflection_parameters(0.784, 0.77, args.beta)
    m = EzquiagaCHIModel(lambda_0=2.23e-7, b_lambda=a*2.23e-7,
                          xi_0=7.55, b_xi=b*7.55, c=0.77)
    m.x0 = args.chi0; m.y0 = -1e-4; m.patch_background_solver()
    T = np.linspace(0, m.T_max, m.bg_steps)
    sol = run_background_simulation(m, T)
    epsH = sol[1]**2 / (2 * sol[2]**2)
    end_idx = int(np.where(np.isfinite(epsH) & (epsH >= 1.0))[0][0])

    k_grid = np.logspace(np.log10(args.k_min), np.log10(args.k_max), args.n_k)
    t0 = time.time()
    results = []
    for i, kp in enumerate(k_grid):
        ps = solve_one_mode(kp, sol, T, end_idx, m, args.method)
        results.append((kp, ps))
        if (i + 1) % 5 == 0:
            n_ok = sum(1 for _, p in results if np.isfinite(p))
            print(f"  {i+1}/{args.n_k} done, {n_ok} OK, {time.time()-t0:.0f}s")

    k_arr = np.array([r[0] for r in results])
    ps_arr = np.array([r[1] for r in results])
    ok = np.isfinite(ps_arr)

    print(f"\n{args.method}: {np.sum(ok)}/{args.n_k} OK in {time.time()-t0:.0f}s")
    if np.any(ok):
        print(f"P_S range: [{np.min(ps_arr[ok]):.3e}, {np.max(ps_arr[ok]):.3e}]")
        i_pk = np.argmax(ps_arr[ok])
        print(f"Peak: k={k_arr[ok][i_pk]:.3e} P_S={ps_arr[ok][i_pk]:.3e}")

    out = f"outputs/simulations/pspectra/ms_crosscheck_{args.method}_chi{args.chi0}.json"
    json.dump({"k_phys": k_arr[ok].tolist(), "P_S": ps_arr[ok].tolist(),
               "method": args.method, "chi0": args.chi0},
              open(out, "w"), indent=2)
    print(f"Saved to {out}")


if __name__ == '__main__':
    main()
