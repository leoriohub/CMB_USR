"""
Ezquiaga CHI: compare P_S(k) from SR approximation vs full MS solver.

Usage:
    python -m scripts.ezquiaga_ps_check --chi0 8.0 --beta 1e-5 --n-k 80
"""
import sys, os
import argparse
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from models.ezquiaga_chi import EzquiagaCHIModel, inflection_parameters
from inf_dyn_background import run_background_simulation, get_derived_quantities
from scripts.plotting import save_fig

X_C, C_VAL = 0.784, 0.77
S_CODE = 5e-5

def compute_ps_sr(bg_sol, end_idx):
    x, y, z, n = bg_sol
    epsH = y**2 / (2 * z**2)
    k_phys = np.exp(n[:end_idx+1]) * z[:end_idx+1] * S_CODE
    e = epsH[:end_idx+1]
    P_S = np.where(np.isfinite(e) & (e > 0),
                   (S_CODE * z[:end_idx+1])**2 / (8 * np.pi**2 * e),
                   np.nan)
    return k_phys, P_S, e

def _solve_batch(args):
    k_phys_batch, bg_sol, T_span_bg, end_idx, model_params = args
    from scipy.interpolate import CubicSpline
    import inf_dyn_MS_full as ms_solver
    from numba_ms_solver import numba_run_ms, build_numba_splines

    m = EzquiagaCHIModel()
    m.a, m.b, m.v0, m.x0, m.y0 = model_params
    m.patch_background_solver()

    interp = tuple(
        CubicSpline(T_span_bg, bg_sol[i], bc_type='not-a-knot', extrapolate=True)
        for i in range(4)
    )
    bg_coefs = build_numba_splines(bg_sol, T_span_bg, model=m)
    n_bg, z_bg = bg_sol[3], bg_sol[2]
    log_az = n_bg + np.log(np.maximum(z_bg, 1e-300))
    t_end = T_span_bg[end_idx]

    results = []
    for kp in k_phys_batch:
        k_code = kp / S_CODE
        si = int(np.argmin(np.abs(log_az[:end_idx] - np.log(k_code) + np.log(100.0))))
        si = max(si, 0)
        ni = bg_sol[3][si]
        T_ms = np.linspace(T_span_bg[si], t_end, 5000)
        try:
            sol = numba_run_ms(bg_sol, T_span_bg, T_ms, ni, k_code, m, bg_coefs=bg_coefs)
            d = ms_solver.get_ms_derived_quantities_with_bg(sol, interp, T_ms, m, k_code, ni)
            ps = float(d["P_S"][-1])
            if np.isfinite(ps) and ps > 0:
                results.append((kp, ps))
            else:
                results.append((kp, None))
        except Exception as e:
            results.append((kp, None))
    return results

def main():
    parser = argparse.ArgumentParser(description="SR vs MS P_S(k) for Ezquiaga")
    parser.add_argument("--chi0", type=float, default=8.0)
    parser.add_argument("--beta", type=float, default=1e-5)
    parser.add_argument("--k-min", type=float, default=3e-7)
    parser.add_argument("--k-max", type=float, default=30)
    parser.add_argument("--n-k", type=int, default=80)
    parser.add_argument("--n-workers", type=int, default=min(8, multiprocessing.cpu_count()))
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b = inflection_parameters(X_C, C_VAL, args.beta)
    model = EzquiagaCHIModel()
    model.a = a; model.b = b
    model.v0 = model._V0 * model.a / (model.b * model.c)**2
    model.x0 = args.chi0
    model.y0 = -1e-4
    model.patch_background_solver()

    print(f"Ezquiaga PS check: chi0={args.chi0}, beta={args.beta}, n_workers={args.n_workers}")
    print(f"  a={a:.5f}, b={b:.5f}, v0={model.v0:.6e}")

    T = np.linspace(0, model.T_max, model.bg_steps)
    bg_sol = run_background_simulation(model, T)
    derived = get_derived_quantities(bg_sol, model)

    epsH = derived["epsH"]
    eps1_arr = np.where(np.isfinite(epsH) & (epsH >= 1.0))[0]
    end_idx = int(eps1_arr[0]) if len(eps1_arr) > 0 else len(epsH) - 1
    N_total = float(derived["N"][end_idx])
    print(f"  N_total = {N_total:.1f}, end_idx = {end_idx}")

    k_sr, Ps_sr, _ = compute_ps_sr(bg_sol, end_idx)
    valid = np.where(np.isfinite(Ps_sr) & (Ps_sr > 0))[0]
    k_sr, Ps_sr = k_sr[valid], Ps_sr[valid]

    k_grid = np.unique(np.logspace(np.log10(args.k_min), np.log10(args.k_max), args.n_k))
    k_grid = k_grid[k_grid < k_sr[-1]]
    n_k = len(k_grid)
    print(f"  Running MS for {n_k} modes on {args.n_workers} workers...")

    chunks = np.array_split(k_grid, args.n_workers)
    model_params = (model.a, model.b, model.v0, model.x0, model.y0)
    batch_args = [(ch, bg_sol, T, end_idx, model_params) for ch in chunks]

    Ps_ms = np.full(n_k, np.nan)
    k_to_result = {kp: i for i, kp in enumerate(k_grid)}

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = [pool.submit(_solve_batch, ba) for ba in batch_args]
        done = 0
        for future in as_completed(futures):
            for kp, ps in future.result():
                if kp in k_to_result and ps is not None:
                    Ps_ms[k_to_result[kp]] = ps
            done += 1
            print(f"    batch {done}/{args.n_workers}", flush=True)

    elapsed = time.time() - t0
    n_ok = int(np.sum(np.isfinite(Ps_ms)))
    print(f"  MS done: {n_ok}/{n_k} OK in {elapsed:.0f}s")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.loglog(k_sr, Ps_sr, "-", color="tab:blue", lw=1.5, label="SR approx")
    ok_ms = np.isfinite(Ps_ms)
    if np.any(ok_ms):
        ax1.loglog(k_grid[ok_ms], Ps_ms[ok_ms], "o", color="tab:red",
                   ms=4, label=f"MS solver ({n_ok}/{n_k})")
    ax1.axvline(0.002, color="grey", ls=":", lw=0.8, alpha=0.5)
    ax1.annotate("k=0.002", xy=(0.002, ax1.get_ylim()[0]), fontsize=8,
                 color="grey", ha="center")
    ax1.legend(fontsize=11)
    ax1.set_ylabel(r"$P_{\mathcal{R}}(k)$", fontsize=14)
    ax1.tick_params(labelsize=12)
    ax1.grid(True, which="both", alpha=0.2)

    interp_ms = np.interp(k_sr, k_grid[ok_ms], Ps_ms[ok_ms],
                          left=np.nan, right=np.nan)
    ratio = interp_ms / Ps_sr
    ax2.semilogx(k_sr, ratio, "-", color="tab:blue", lw=1.2)
    ax2.axhline(1.0, color="grey", ls="--", lw=0.8)
    ax2.set_xlabel(r"$k$ [Mpc$^{-1}$]", fontsize=14)
    ax2.set_ylabel("MS / SR", fontsize=14)
    ax2.tick_params(labelsize=12)
    rmin, rmax = np.nanmin(ratio), np.nanmax(ratio)
    if not np.isfinite(rmin): rmin = 0.5
    if not np.isfinite(rmax): rmax = 1.5
    ax2.set_ylim(max(0, rmin - 0.2), min(5, rmax + 0.2))
    ax2.grid(True, which="both", alpha=0.2)
    ax2.axhline(1.0, color="grey", ls="--", lw=0.8)

    fig.tight_layout()
    save_fig(fig, f"ezquiaga_ps_check_chi{args.chi0}", "diagnostics", dpi=200)
    print(f"  Saved plot to outputs/plots/diagnostics/")

    for label, k_piv in [("k=0.002", 0.002), ("k=0.05", 0.05)]:
        pi = int(np.argmin(np.abs(k_sr - k_piv)))
        if pi < 5 or pi > len(k_sr) - 6:
            continue
        d5 = 5
        lk = np.log(k_sr[pi-d5:pi+d5+1])
        lp_sr = np.log(Ps_sr[pi-d5:pi+d5+1])
        ns_sr = 1 + (lp_sr[-1] - lp_sr[0]) / (lk[-1] - lk[0])
        ns_ms = None
        if np.isfinite(interp_ms[pi]):
            lp_ms = np.log(np.maximum(interp_ms[pi-d5:pi+d5+1], 1e-300))
            ns_ms = 1 + (lp_ms[-1] - lp_ms[0]) / (lk[-1] - lk[0])
        msg = f"  {label}: n_s(SR)={ns_sr:.4f}"
        if ns_ms is not None:
            msg += f", n_s(MS)={ns_ms:.4f}"
        print(msg)

if __name__ == "__main__":
    main()
