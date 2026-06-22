"""Background-only fast scan for PBH parameter search.

Runs ONLY the background ODE solver (no MS) across (x_c, c, β, χ₀, N_star)
to map N_total, SR n_s, and USR status across parameter space.

~100x faster than full MS sweeps. Identifies "interesting regions"
for targeted MS validation using physically motivated N_total thresholds.
"""

import os
import sys
import json
import time
import argparse
import numpy as np


def model_from_params(x_c, c, beta):
    from models.ezquiaga_chi import EzquiagaCHIModel, inflection_parameters

    m = EzquiagaCHIModel(c=c)
    a, b = inflection_parameters(x_c, c, beta=beta)
    m.a = a
    m.b = b
    m.v0 = m._V0 * m.a / (m.b * m.c) ** 2
    return m


def find_end_of_inflation_simple(epsH):
    for i in range(len(epsH) - 5):
        if all(epsH[i + j] < 0.1 for j in range(5)):
            sr_start = i
            break
    else:
        sr_start = int(0.1 * len(epsH))
    for i in range(sr_start + 1, len(epsH)):
        if epsH[i - 1] < 1.0 and epsH[i] >= 1.0:
            return i
    below = np.where(epsH < 1.0)[0]
    if len(below) > 0:
        return int(below[-1])
    return len(epsH) - 1


def infer_usr_type(N_total):
    if N_total > 165:
        return "strong"
    elif N_total > 130:
        return "transitional"
    else:
        return "weak"


def analyze_background(model, chi0, y0, N_star, pivot_k=0.05):
    from inf_dyn_background import run_background_simulation, get_derived_quantities

    T_span = np.linspace(0, model.T_max, model.bg_steps)
    bg_sol = run_background_simulation(model, T_span)
    derived = get_derived_quantities(bg_sol, model)

    epsH = derived["epsH"]
    end_idx = find_end_of_inflation_simple(epsH)
    if end_idx >= len(epsH) - 1 or end_idx < 10:
        end_idx = len(epsH) - 1

    N_arr = derived["N"][: end_idx + 1]
    N_total = float(N_arr[-1])
    if N_total < N_star:
        return None

    N_pivot = N_total - N_star
    eps_pivot = float(np.interp(N_pivot, N_arr, epsH[: end_idx + 1]))
    eta_pivot = float(np.interp(N_pivot, N_arr, derived["etaH"][: end_idx + 1]))
    n_s_sr = 1 + 2 * eta_pivot - 4 * eps_pivot
    usr_type = infer_usr_type(N_total)

    return {
        "N_total": N_total,
        "N_pivot": N_pivot,
        "n_s_sr": n_s_sr,
        "eps_pivot": eps_pivot,
        "eta_pivot": eta_pivot,
        "usr_type": usr_type,
    }


def run_scan(
    xc_vals,
    c_vals,
    beta_vals,
    chi0_vals,
    N_star_vals,
    y0=-1e-4,
    pivot_k=0.05,
    log_path=None,
    progress_fn=None,
):
    results = []
    total = (
        len(xc_vals) * len(c_vals) * len(beta_vals) * len(chi0_vals) * len(N_star_vals)
    )
    fh = None
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = open(log_path, "w")
    idx = 0
    last_flush = 0
    for xc in xc_vals:
        for c in c_vals:
            for beta in beta_vals:
                for chi0 in chi0_vals:
                    for N_star in N_star_vals:
                        idx += 1
                        if progress_fn:
                            progress_fn(idx, total, xc, c, beta, chi0, N_star)
                        try:
                            m = model_from_params(xc, c, beta)
                            m.x0 = chi0
                            m.y0 = y0
                            m.patch_background_solver()
                            analysis = analyze_background(m, chi0, y0, N_star, pivot_k)
                            if analysis is None:
                                continue
                            result = {
                                "x_c": xc,
                                "c": c,
                                "beta": beta,
                                "chi0": chi0,
                                "N_star": N_star,
                            }
                            result.update(analysis)
                            results.append(result)
                            if fh:
                                json.dump(result, fh)
                                fh.write("\n")
                                if idx - last_flush >= 100:
                                    fh.flush()
                                    last_flush = idx
                        except Exception as e:
                            err = {
                                "x_c": xc,
                                "c": c,
                                "beta": beta,
                                "chi0": chi0,
                                "N_star": N_star,
                                "error": str(e),
                            }
                            results.append(err)
                            if fh:
                                json.dump(err, fh)
                                fh.write("\n")
                        # Always flush at x_c changes
                        if fh and idx % 1000 == 0:
                            fh.flush()
    if fh:
        fh.close()
    return results


def write_log(results, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        for r in results:
            json.dump(r, f)
            f.write("\n")


def print_summary(results, n_show=20):
    scored = []
    for r in results:
        if "error" in r:
            continue
        ns = r.get("n_s_sr", 1) or 1
        ut = r.get("usr_type", "weak")
        score = 0.0
        if ut == "strong":
            score += 20
        elif ut == "transitional":
            score += 10
        score -= abs(ns - 0.965) * 200
        scored.append((score, r))
    scored.sort(key=lambda x: -x[0])

    print(
        f"\n{'Rank':<5} {'xc':<7} {'c':<7} {'beta':<9} {'chi0':<5} "
        f"{'N*':<4} {'Ntot':<7} {'ns_sr':<8} {'usr_type':<14} {'Npiv':<8}"
    )
    print("-" * 85)
    for i, (s, r) in enumerate(scored[:n_show]):
        ns = r.get("n_s_sr", "N/A")
        print(
            f"{i + 1:<5} {r['x_c']:<7.4f} {r['c']:<7.4f} {r['beta']:<9.1e} "
            f"{r['chi0']:<5.1f} {r['N_star']:<4.0f} {r['N_total']:<7.1f} "
            f"{ns if isinstance(ns, str) else f'{ns:.4f}':<8} "
            f"{r.get('usr_type', '?'):<14} "
            f"{r.get('N_pivot', 0):<8.1f}"
        )


def main():
    p = argparse.ArgumentParser(
        description="Background-only fast scan for PBH parameter search"
    )
    p.add_argument("--x_c-lo", type=float, default=0.776)
    p.add_argument("--x_c-hi", type=float, default=0.792)
    p.add_argument("--n-xc", type=int, default=9)
    p.add_argument("--c-lo", type=float, default=0.77)
    p.add_argument("--c-hi", type=float, default=0.77)
    p.add_argument("--n-c", type=int, default=1)
    p.add_argument("--beta-vals", type=float, nargs="+", default=[1e-5, 2e-5, 3e-5])
    p.add_argument("--chi0-vals", type=float, nargs="+", default=[7.0])
    p.add_argument("--N-star-vals", type=float, nargs="+", default=[50, 60, 70])
    p.add_argument("--y0", type=float, default=-1e-4)
    p.add_argument("--pivot-k", type=float, default=0.05)
    p.add_argument("--log", default="outputs/simulations/logs/bg_scan.jsonl")
    args = p.parse_args()

    xc_vals = np.linspace(args.x_c_lo, args.x_c_hi, args.n_xc)
    c_vals = np.linspace(args.c_lo, args.c_hi, args.n_c)
    beta_vals = args.beta_vals
    chi0_vals = args.chi0_vals
    N_star_vals = args.N_star_vals

    total = (
        len(xc_vals) * len(c_vals) * len(beta_vals) * len(chi0_vals) * len(N_star_vals)
    )
    print(
        f"Scan: {len(xc_vals)} x_c × {len(c_vals)} c × "
        f"{len(beta_vals)} β × {len(chi0_vals)} χ₀ × "
        f"{len(N_star_vals)} N* = {total} configs  pivot_k={args.pivot_k}"
    )

    t0 = time.time()

    def progress(i, n, xc, c, beta, chi0, N_star):
        elapsed = time.time() - t0
        rate = i / max(elapsed, 0.01)
        eta = (n - i) / rate
        sys.stdout.write(
            f"\r  [{i}/{n}] x_c={xc:.4f} c={c:.3f} β={beta:.1e} "
            f"χ₀={chi0:.1f} N*={N_star:.0f}  "
            f"[{elapsed:.0f}s<{eta:.0f}s, {rate:.1f}/s]  "
        )
        sys.stdout.flush()

    results = run_scan(
        xc_vals,
        c_vals,
        beta_vals,
        chi0_vals,
        N_star_vals,
        y0=args.y0,
        pivot_k=args.pivot_k,
        log_path=args.log,
        progress_fn=progress,
    )
    print(f"\n  Done in {time.time() - t0:.1f}s  ({len(results)} of {total})")

    print(f"  Log: {args.log}")
    print_summary(results)

    strong = sum(1 for r in results if r.get("usr_type") == "strong")
    trans = sum(1 for r in results if r.get("usr_type") == "transitional")
    weak = sum(1 for r in results if r.get("usr_type") == "weak")
    print(f"\nUSR: {strong} strong, {trans} transitional, {weak} weak")


if __name__ == "__main__":
    main()
