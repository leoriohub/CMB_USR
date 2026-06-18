"""Background-only fast scan for PBH parameter search.

Runs ONLY the background ODE solver (no MS) across (x_c, c, β, χ₀, N_star)
to map USR duration (ΔN) and SR n_s across parameter space.

~100x faster than full MS sweeps. Identifies "interesting regions"
where ΔN > 3 AND n_s ≈ 0.965 for targeted MS validation.
"""

import os
import sys
import json
import time
import argparse
import numpy as np

# Fixed cosmology (matches sweep_pbh_params.py)
ACCRETION = 3e7
K_EQ = 0.0104
M_EQ = 3.0e17
GAMMA = 0.4

EPS_USR = 0.01       # ε_H below this threshold = in USR phase (unused, kept as ref)
EPS_SR = 0.01        # ε_H above this after initial kinetic phase = SR
SKIP_FRAC = 0.05     # fraction of trajectory to skip for initial transient


def model_from_params(x_c, c, beta):
    """Build Ezquiaga CHI model from (x_c, c, beta)."""
    from models.ezquiaga_chi import EzquiagaCHIModel, inflection_parameters
    m = EzquiagaCHIModel(c=c)
    a, b = inflection_parameters(x_c, c, beta=beta)
    m.a = a
    m.b = b
    m.v0 = m._V0 * m.a / (m.b * m.c) ** 2
    return m


def find_usr_phase(N_arr, epsH_arr, etaH_arr):
    """Find the USR phase from ε_H(N) trajectory.

    USR is identified as the deep dip in ε_H (much lower than SR plateau).
    Uses an adaptive threshold based on the minimum ε_H value.

    Returns dict with:
      - N_start (float): e-fold where USR begins
      - N_end (float): e-fold where USR ends
      - Delta_N (float): duration of USR phase (0 if no USR)
      - eps_min (float): minimum ε_H during USR
      - eps_min_idx (int): index of minimum ε_H
    """
    # Skip initial transient (kinetic phase or SR startup)
    skip = min(len(epsH_arr) - 1, max(10, int(SKIP_FRAC * len(epsH_arr))))

    eps = epsH_arr[skip:]
    N = N_arr[skip:]

    # Find the global minimum of ε_H (deepest part of USR)
    eps_min_idx_local = int(np.argmin(eps))
    eps_min = eps[eps_min_idx_local]

    # No USR if minimum is not very small (near-SR values)
    if eps_min > 1e-4:
        return {"N_start": None, "N_end": None, "Delta_N": 0.0,
                "eps_min": None, "eps_min_idx": None}

    # Expand outward from the minimum
    # USR region: ε_H < REL_THRESH * eps_min (factor above the dip floor)
    REL_THRESH = 1000  # ε_H rises 10^3 above min = end of USR phase

    lo = eps_min_idx_local
    while lo > 0 and eps[lo] < eps_min * REL_THRESH:
        lo -= 1
    hi = eps_min_idx_local
    while hi < len(eps) - 1 and eps[hi] < eps_min * REL_THRESH:
        hi += 1

    start_idx = int(lo) + skip
    end_idx = int(hi) + skip
    
    if end_idx >= len(N_arr):
        end_idx = len(N_arr) - 1
    if start_idx >= end_idx:
        return {"N_start": None, "N_end": None, "Delta_N": 0.0,
                "eps_min": None, "eps_min_idx": None}

    Delta_N = float(N_arr[end_idx] - N_arr[start_idx])

    return {
        "N_start": float(N_arr[start_idx]),
        "N_end": float(N_arr[end_idx]),
        "Delta_N": Delta_N,
        "eps_min": float(eps_min),
        "eps_min_idx": int(eps_min_idx_local + skip),
    }


def find_end_of_inflation_simple(epsH):
    """Find where ε_H permanently crosses 1 (end of inflation).

    Faster version of pspectrum_pipeline.find_end_of_inflation:
    finds the first crossing after the trajectory enters SR.
    """
    # Find SR entry: first time ε_H drops below EPS_SR after initial kinetic peak
    sr_start = None
    for i in range(1, len(epsH)):
        if epsH[i] < EPS_SR and epsH[i] > epsH[i - 1]:
            # ε_H is small and rising — this is SR
            sr_start = None  # keep looking for first entry
        if epsH[i] < EPS_SR and sr_start is None and epsH[i - 1] >= EPS_SR:
            sr_start = i
            break

    if sr_start is None:
        sr_start = int(0.1 * len(epsH))  # fallback

    # After SR entry, find first ε_H crossing 1
    for i in range(sr_start + 1, len(epsH)):
        if epsH[i - 1] < 1.0 and epsH[i] >= 1.0:
            return i

    # Fallback: last index where ε_H < 1
    below_one = np.where(epsH < 1.0)[0]
    if len(below_one) > 0:
        return int(below_one[-1])
    return len(epsH) - 1


def estimate_peak_k(N_arr, eps_arr, bg_sol, end_idx, N_pivot, eps_min_idx, pivot_k=0.05):
    """Estimate k_peak from a*H at the time of minimum ε_H.

    The peak in P_S(k) during USR occurs at the scale that exits
    when ε_H is at its minimum:
      k_peak / k_pivot = (a*H)_eps_min / (a*H)_pivot
    """
    z_arr = bg_sol[2][:end_idx + 1]
    n_arr = bg_sol[3][:end_idx + 1]
    log_aH = n_arr + np.log(np.maximum(z_arr, 1e-300))

    log_aH_pivot = float(np.interp(N_pivot, N_arr, log_aH))
    if eps_min_idx is not None and eps_min_idx < len(log_aH):
        log_aH_peak = float(log_aH[eps_min_idx])
    else:
        return None

    return pivot_k * np.exp(log_aH_peak - log_aH_pivot)


def analyze_background(model, chi0, y0, N_star, pivot_k=0.05):
    """Run background only, extract USR diagnostics and SR n_s.

    Parameters
    ----------
    model : EzquiagaCHIModel
        Configured model instance.
    chi0, y0 : float
        Initial conditions.
    N_star : float
        E-folds before end for CMB pivot exit.
    pivot_k : float
        CMB pivot scale [Mpc^-1].

    Returns
    -------
    dict with all diagnostics, or None if N_total < N_star.
    """
    from inf_dyn_background import run_background_simulation, get_derived_quantities

    # 1. Run background
    T_span = np.linspace(0, model.T_max, model.bg_steps)
    bg_sol = run_background_simulation(model, T_span)
    derived = get_derived_quantities(bg_sol, model)

    # 2. Find end of inflation
    epsH = derived["epsH"]
    end_idx = find_end_of_inflation_simple(epsH)
    if end_idx >= len(epsH) - 1 or end_idx < 10:
        end_idx = len(epsH) - 1

    N_arr = derived["N"][:end_idx + 1]
    N_total = float(N_arr[-1])
    if N_total < N_star:
        return None  # not enough e-folds

    eps_arr = epsH[:end_idx + 1]
    eta_arr = derived["etaH"][:end_idx + 1]

    # 3. Find USR phase
    usr = find_usr_phase(N_arr, eps_arr, eta_arr)
    Delta_N = usr["Delta_N"]
    N_inflection_start = usr["N_start"]
    N_inflection_end = usr["N_end"]

    # 4. Pivot position and SR n_s
    N_pivot = N_total - N_star

    eps_pivot = float(np.interp(N_pivot, N_arr, eps_arr))
    eta_pivot = float(np.interp(N_pivot, N_arr, eta_arr))
    n_s_sr = 1 + 2 * eta_pivot - 4 * eps_pivot

    pivot_before_inflection = (
        N_inflection_start is not None and N_pivot < N_inflection_start
    )

    # 5. Estimate k_peak and mass (using ε_H minimum position)
    M_peak_estimate = None
    k_peak_estimate = None
    if Delta_N > 0 and usr["eps_min_idx"] is not None:
        k_peak_estimate = float(estimate_peak_k(
            N_arr, eps_arr, bg_sol, end_idx, N_pivot,
            usr["eps_min_idx"], pivot_k))
        if k_peak_estimate and k_peak_estimate > 0:
            M_form = GAMMA * M_EQ * (K_EQ / k_peak_estimate) ** 2
            M_peak_estimate = float(M_form * ACCRETION)

    # 6. Inflection flatness V''(x_c) in the model's internal units
    V_pp = None
    try:
        V_pp = float(model._d2Vdx2(float(model._x_of_chi(float(N_inflection_start)))) if N_inflection_start is not None
                     else model._d2Vdx2(float(model._x_of_chi(0))))
    except:
        pass

    return {
        "N_total": N_total,
        "N_inflection_start": N_inflection_start,
        "N_inflection_end": N_inflection_end,
        "Delta_N": Delta_N,
        "eps_min": usr["eps_min"],
        "N_pivot": N_pivot,
        "n_s_sr": n_s_sr,
        "eps_pivot": eps_pivot,
        "eta_pivot": eta_pivot,
        "pivot_before_inflection": pivot_before_inflection,
        "k_peak_estimate": k_peak_estimate,
        "M_peak_estimate": M_peak_estimate,
        "V_double_prime": V_pp,
        # Full arrays (for plotting / debugging)
        "_N_arr": N_arr.tolist(),
        "_epsH_arr": eps_arr.tolist(),
        "_etaH_arr": eta_arr.tolist(),
    }


def run_scan(xc_vals, c_vals, beta_vals, chi0_vals, N_star_vals,
             y0=-1e-4, pivot_k=0.05, progress_fn=None):
    """Run background-only scan over parameter grid."""
    results = []
    total = (len(xc_vals) * len(c_vals) * len(beta_vals)
             * len(chi0_vals) * len(N_star_vals))
    idx = 0
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

                            analysis = analyze_background(
                                m, chi0, y0, N_star, pivot_k)
                            if analysis is None:
                                continue

                            # Build clean result (no array data)
                            result = {
                                "x_c": xc,
                                "c": c,
                                "beta": beta,
                                "chi0": chi0,
                                "N_star": N_star,
                            }
                            for k, v in analysis.items():
                                if not k.startswith("_"):
                                    result[k] = v
                            results.append(result)
                        except Exception as e:
                            results.append({
                                "x_c": xc,
                                "c": c,
                                "beta": beta,
                                "chi0": chi0,
                                "N_star": N_star,
                                "error": str(e),
                            })
    return results


def write_log(results, log_path):
    """Write results to JSONL log."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        for r in results:
            json.dump(r, f)
            f.write("\n")


def print_summary(results, n_show=20):
    """Print formatted summary table, sorted by interestingness."""
    # Prioritize: has USR + n_s close to 0.965 + pivot before inflection
    scored = []
    for r in results:
        if "error" in r:
            continue
        score = 0.0
        dn = r.get("Delta_N", 0) or 0
        ns = r.get("n_s_sr", 1) or 1
        pbi = r.get("pivot_before_inflection", False)

        if dn > 3:
            score += 10  # has USR
        score -= abs(ns - 0.965) * 100  # closer to 0.965 = better
        if pbi:
            score += 5  # pivot on plateau
        if dn > 0:
            mk = r.get("M_peak_estimate", 0) or 0
            if 1e-17 <= mk <= 1e-15:
                score += 20  # asteroid gap
            elif 1e-6 <= mk <= 1e-2:
                score += 15  # sub-solar gap
        scored.append((score, r))

    scored.sort(key=lambda x: -x[0])

    print(f"\n{'Rank':<5} {'xc':<7} {'c':<7} {'beta':<9} {'chi0':<5} "
          f"{'N*':<4} {'Ntot':<7} {'dN':<6} {'ns_sr':<8} {'pivot_before':<13} "
          f"{'k_peak':<10} {'M_peak':<10} {'bin':<18}")
    print("-" * 130)
    for i, (s, r) in enumerate(scored[:n_show]):
        dn = r.get("Delta_N", 0)
        dn_str = f"{dn:.1f}" if dn else "NONE"
        ns_str = f"{r.get('n_s_sr', 'N/A'):.4f}" if r.get('n_s_sr') is not None else "N/A"
        pbi = r.get("pivot_before_inflection", False)
        kp = r.get("k_peak_estimate", 0)
        kp_str = f"{kp:.3e}" if kp else "N/A"
        mp = r.get("M_peak_estimate", 0)
        mp_str = f"{mp:.4e}" if mp else "N/A"

        # Classify mass bin from estimate
        bin_str = "no_USR"
        if dn and dn > 0 and mp and mp > 0:
            if mp < 1e-17:
                bin_str = "too_light"
            elif mp < 1e-15:
                bin_str = "asteroid_gap"
            elif mp < 1e-6:
                bin_str = "intermediate"
            elif mp < 1e-2:
                bin_str = "sub_solar_gap"
            elif mp < 1:
                bin_str = "sub_stellar"
            elif mp < 100:
                bin_str = "stellar_ligo"
            else:
                bin_str = "massive"
        elif dn and dn > 0:
            bin_str = "USR_noM"

        print(f"{i + 1:<5} {r['x_c']:<7.4f} {r['c']:<7.4f} {r['beta']:<9.1e} "
              f"{r['chi0']:<5.1f} {r['N_star']:<4.0f} {r['N_total']:<7.1f} "
              f"{dn_str:<6} {ns_str:<8} {str(pbi):<13} {kp_str:<10} "
              f"{mp_str:<10} {bin_str:<18}")


def main():
    p = argparse.ArgumentParser(
        description="Background-only fast scan for PBH parameter search")
    p.add_argument("--x_c-lo", type=float, default=0.776)
    p.add_argument("--x_c-hi", type=float, default=0.792)
    p.add_argument("--n-xc", type=int, default=9)
    p.add_argument("--c-lo", type=float, default=0.77)
    p.add_argument("--c-hi", type=float, default=0.77)
    p.add_argument("--n-c", type=int, default=1)
    p.add_argument("--beta-vals", type=float, nargs="+",
                   default=[1e-5, 2e-5, 3e-5])
    p.add_argument("--chi0-vals", type=float, nargs="+", default=[7.0])
    p.add_argument("--N-star-vals", type=float, nargs="+", default=[50, 60, 70])
    p.add_argument("--y0", type=float, default=-1e-4)
    p.add_argument("--pivot-k", type=float, default=0.05,
                   help="CMB pivot scale [Mpc^-1]")
    p.add_argument("--log", default="outputs/simulations/logs/bg_scan.jsonl")
    args = p.parse_args()

    xc_vals = np.linspace(args.x_c_lo, args.x_c_hi, args.n_xc)
    c_vals = np.linspace(args.c_lo, args.c_hi, args.n_c)
    beta_vals = args.beta_vals
    chi0_vals = args.chi0_vals
    N_star_vals = args.N_star_vals

    total = (len(xc_vals) * len(c_vals) * len(beta_vals)
             * len(chi0_vals) * len(N_star_vals))
    print(f"Background scan: {len(xc_vals)} x_c × {len(c_vals)} c × "
          f"{len(beta_vals)} β × {len(chi0_vals)} χ₀ × "
          f"{len(N_star_vals)} N* = {total} configs  "
          f"pivot_k={args.pivot_k}")

    t0 = time.time()

    def progress(i, n, xc, c, beta, chi0, N_star):
        elapsed = time.time() - t0
        rate = i / max(elapsed, 0.01)
        eta = (n - i) / rate
        sys.stdout.write(
            f"\r  [{i}/{n}] x_c={xc:.4f} c={c:.3f} β={beta:.1e} "
            f"χ₀={chi0:.1f} N*={N_star:.0f}  "
            f"[{elapsed:.0f}s<{eta:.0f}s, {rate:.1f}cfg/s]  ")
        sys.stdout.flush()

    results = run_scan(xc_vals, c_vals, beta_vals, chi0_vals, N_star_vals,
                       y0=args.y0, pivot_k=args.pivot_k,
                       progress_fn=progress)
    print(f"\n  Done in {time.time() - t0:.1f}s  "
          f"({len(results)} successful of {total})")

    write_log(results, args.log)
    print(f"  Log: {args.log}")

    print_summary(results)

    # Quick stats
    has_usr = sum(1 for r in results
                  if not r.get("error") and (r.get("Delta_N") or 0) > 3)
    ns_ok = sum(1 for r in results
                if not r.get("error") and r.get("n_s_sr") is not None
                and 0.95 <= r.get("n_s_sr", 2) <= 0.98)
    pivot_before = sum(1 for r in results
                       if not r.get("error")
                       and r.get("pivot_before_inflection", False))
    print(f"\nSummary: {has_usr} configs with ΔN>3, "
          f"{ns_ok} with n_s∈[0.95,0.98], "
          f"{pivot_before} with pivot before inflection")


if __name__ == "__main__":
    main()
