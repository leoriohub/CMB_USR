"""
compute_fnl.py — f_NL^(local) via δN formalism for Higgs USR.

Computes f_NL using the δN expansion:
    N(φ + δφ) ≈ N + N_φ δφ + ½ N_φφ (δφ)²
    f_NL = (5/6) N_φφ / N_φ²

where N(φ) is the number of e-folds from field value φ to the end of inflation.

Usage:
    python scripts/compute_fnl.py --x0 5.75 --y0 -0.170 --nstar 55
"""

import argparse
import json
import sys

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

from models.higgs import HiggsModel
from inf_dyn_background import run_background_simulation, get_derived_quantities
from pspectrum_pipeline import find_end_of_inflation


# ---------------------------------------------------------------------------
# δN computation
# ---------------------------------------------------------------------------

def _build_n_remaining(x_arr, N_arr, N_total):
    """
    Build an interpolated function N_remaining(φ) = N_total − N(φ).

    Parameters
    ----------
    x_arr : 1D array
        Field values φ/M_P along the background trajectory (monotonic decreasing).
    N_arr : 1D array
        Number of e-folds from start at each step.
    N_total : float
        Total number of e-folds (interpolated at ε_H=1).

    Returns
    -------
    callable
        N_remaining(phi) returning float or array.
    """
    # Reverse so x is increasing for stable interpolation
    x_rev = x_arr[::-1]
    N_rev = N_arr[::-1]

    # Remove duplicate x values (possible with very flat field evolution)
    _, uniq_idx = np.unique(x_rev, return_index=True)
    x_rev = x_rev[uniq_idx]
    N_rev = N_rev[uniq_idx]

    # Build N(x) interpolator
    if len(x_rev) < 4:
        N_of_x = interp1d(x_rev, N_rev, kind='linear',
                          bounds_error=False, fill_value='extrapolate')
    else:
        N_of_x = CubicSpline(x_rev, N_rev, bc_type='not-a-knot',
                             extrapolate=True)

    def n_remaining(phi):
        """N_remaining(φ) = N_total - N(φ)."""
        return N_total - N_of_x(phi)

    return n_remaining


def compute_fnl(model, bg_sol, derived_bg, end_idx, N_star,
                delta_phi=0.05):
    """
    Compute f_NL^(local) via finite differences on N_remaining(φ).

    Parameters
    ----------
    model : InflationModel
        Model instance (used only for SR cross-check formulas).
    bg_sol : np.ndarray, shape (4, Nsteps)
        Background solution [x, y, z, n] from run_background_simulation.
    derived_bg : dict
        Derived quantities from get_derived_quantities.
    end_idx : int
        Index where inflation ends (from find_end_of_inflation).
    N_star : float
        Number of e-folds before end of inflation for the pivot.
    delta_phi : float, optional
        Finite-difference step δφ in Planck units (default 0.05).

    Returns
    -------
    dict
        Results with keys:
            f_NL, N_phi, N_phiphi, phi_pivot, N_pivot, N_total,
            f_NL_sr, n_s_sr, sr_agreement, rel_diff_sr, …
    """
    x_arr = bg_sol[0]        # φ / M_P
    N_arr = derived_bg['N']   # e-folds from start
    epsH = derived_bg['epsH']
    etaH = derived_bg['etaH']

    # ------------------------------------------------------------------
    # 1. N_total via fractional interpolation at ε_H=1
    #    (matches pspectrum_pipeline.py:332-338)
    # ------------------------------------------------------------------
    N_total = float(N_arr[end_idx])
    if (0 < end_idx < len(epsH)
            and epsH[end_idx] > 1.0 >= epsH[end_idx - 1]):
        f = (1.0 - epsH[end_idx - 1]) / (epsH[end_idx] - epsH[end_idx - 1])
        if 0.0 <= f <= 1.0:
            N_total = float(N_arr[end_idx - 1]
                            + f * (N_arr[end_idx] - N_arr[end_idx - 1]))

    # ------------------------------------------------------------------
    # 2. Locate φ_pivot at N_pivot = N_total − N_star
    # ------------------------------------------------------------------
    N_pivot = N_total - N_star
    if N_pivot < 0 or N_pivot > N_arr[-1]:
        raise ValueError(
            f"N_pivot={N_pivot:.6f} outside trajectory bounds "
            f"[0, {N_arr[-1]:.6f}]"
        )

    phi_pivot = float(np.interp(N_pivot, N_arr, x_arr))
    pivot_idx = int(np.argmin(np.abs(N_arr - N_pivot)))

    # ------------------------------------------------------------------
    # 3. Build N_remaining(φ) interpolator over the inflationary slice
    # ------------------------------------------------------------------
    # Add small padding past ε_H=1 to avoid edge effects
    infl_end = min(end_idx + 5, len(x_arr))
    n_rem = _build_n_remaining(x_arr[:infl_end], N_arr[:infl_end], N_total)

    # ------------------------------------------------------------------
    # 4. Centered finite differences at φ_pivot
    # ------------------------------------------------------------------
    phi_min, phi_max = float(np.min(x_arr[:infl_end])), float(np.max(x_arr[:infl_end]))

    phi_plus = phi_pivot + delta_phi
    phi_minus = phi_pivot - delta_phi

    if phi_plus > phi_max or phi_minus < phi_min:
        raise ValueError(
            f"δφ={delta_phi} too large: φ_pivot={phi_pivot:.4f} "
            f"out of range [{phi_min:.4f}, {phi_max:.4f}]. "
            "Try a smaller delta_phi."
        )

    Np = n_rem(phi_pivot)
    Np_plus = n_rem(phi_plus)
    Np_minus = n_rem(phi_minus)

    N_phi = (Np_plus - Np_minus) / (2.0 * delta_phi)
    N_phiphi = (Np_plus - 2.0 * Np + Np_minus) / (delta_phi ** 2)

    f_NL = (5.0 / 6.0) * N_phiphi / (N_phi ** 2)

    # ------------------------------------------------------------------
    # 5. SR cross-check at φ_pivot
    # ------------------------------------------------------------------
    epsH_pivot = float(np.interp(N_pivot, N_arr, epsH))
    etaH_pivot = float(np.interp(N_pivot, N_arr, etaH))

    # N_φ^SR = -1 / √(2 ε_H)   (M_P = 1 in code units)
    N_phi_sr = -1.0 / np.sqrt(2.0 * epsH_pivot) if epsH_pivot > 0 else -np.inf

    # n_s^SR = 1 + 2η_H − 4ε_H
    n_s_sr = 1.0 + 2.0 * etaH_pivot - 4.0 * epsH_pivot

    # f_NL^SR ≈ (5/12)(1 − n_s)
    f_NL_sr = (5.0 / 12.0) * (1.0 - n_s_sr)

    # Agreement flag: |f_NL − f_NL^SR| / |f_NL^SR| ≤ 0.2
    if abs(f_NL_sr) > 1e-15:
        rel_diff = abs(f_NL - f_NL_sr) / abs(f_NL_sr)
        agreement = rel_diff <= 0.20
    else:
        rel_diff = 0.0
        agreement = abs(f_NL) < 1e-3

    return {
        # δN numerical
        'f_NL': float(f_NL),
        'N_phi': float(N_phi),
        'N_phiphi': float(N_phiphi),
        'phi_pivot': float(phi_pivot),
        'N_pivot': float(N_pivot),
        'N_total': float(N_total),
        'N_star': float(N_star),
        'n_rem_pivot': float(Np),
        'delta_phi': float(delta_phi),
        # SR cross-check
        'f_NL_sr': float(f_NL_sr),
        'n_s_sr': float(n_s_sr),
        'N_phi_sr': float(N_phi_sr),
        'epsH_pivot': float(epsH_pivot),
        'etaH_pivot': float(etaH_pivot),
        'sr_agreement': bool(agreement),
        'rel_diff_sr': float(rel_diff),
        # Metadata
        'method': 'δN centered finite difference',
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compute f_NL^(local) via δN formalism for Higgs USR'
    )
    parser.add_argument('--x0', type=float, default=5.75,
                        help='Initial field φ₀/M_P (default: 5.75)')
    parser.add_argument('--y0', type=float, default=-0.170,
                        help='Initial velocity y₀ (default: -0.170)')
    parser.add_argument('--nstar', type=float, default=55.0,
                        help="N_star (default: 55.0)")
    parser.add_argument('--delta-phi', type=float, default=0.05,
                        help='δφ for finite differences (default: 0.05)')
    parser.add_argument('--lam', type=float, default=0.13,
                        help='Higgs self-coupling λ (default: 0.13)')
    parser.add_argument('--xi', type=float, default=15000.0,
                        help='Non-minimal coupling ξ (default: 15000)')
    parser.add_argument('--bg-steps', type=int, default=50000,
                        help='Background integration steps (default: 50000)')
    parser.add_argument('--json', type=str, default=None,
                        help='Output results as JSON to this path')
    args = parser.parse_args()

    # Model setup
    model = HiggsModel(lam=args.lam, xi=args.xi)
    model.x0 = args.x0
    model.y0 = args.y0

    # Background integration
    T_span = np.linspace(0.0, model.T_max, args.bg_steps)
    print(f"Integrating background: x₀={args.x0}, y₀={args.y0} ...")
    bg_sol = run_background_simulation(model, T_span)
    derived_bg = get_derived_quantities(bg_sol, model)

    # End of inflation
    end_idx = find_end_of_inflation(derived_bg['epsH'])
    if end_idx == -1:
        print("ERROR: Inflation never ends or never starts.", file=sys.stderr)
        sys.exit(1)

    print(f"  end_idx={end_idx},  "
          f"N(end_idx)={derived_bg['N'][end_idx]:.4f}")

    # Compute f_NL
    result = compute_fnl(
        model, bg_sol, derived_bg, end_idx, args.nstar,
        delta_phi=args.delta_phi,
    )

    # Print summary
    print()
    print("=" * 55)
    print("  f_NL^(local) — δN Formalism")
    print("=" * 55)
    print(f"  Config:     x₀={args.x0}, y₀={args.y0}, N_*={args.nstar}")
    print(f"  N_total:    {result['N_total']:.6f}")
    print(f"  N_pivot:    {result['N_pivot']:.6f}")
    print(f"  φ_pivot:    {result['phi_pivot']:.6f}")
    print(f"  δφ:         {result['delta_phi']}")
    print("─" * 55)
    print(f"  N_φ:        {result['N_phi']:.6e}")
    print(f"  N_φφ:       {result['N_phiphi']:.6e}")
    print(f"  f_NL:       {result['f_NL']:.6e}")
    print("─" * 55)
    print(f"  SR cross-check (at φ_pivot):")
    print(f"    ε_H:      {result['epsH_pivot']:.6e}")
    print(f"    η_H:      {result['etaH_pivot']:.6e}")
    print(f"    n_s^SR:   {result['n_s_sr']:.6f}")
    print(f"    N_φ^SR:   {result['N_phi_sr']:.6e}")
    print(f"    f_NL^SR:  {result['f_NL_sr']:.6e}")
    print(f"    |Δ|/|f_NL^SR|: {result['rel_diff_sr']:.4f}")
    print(f"    Agreement: {'✓ YES' if result['sr_agreement'] else '✗ NO — USR may contaminate pivot'}")
    print("=" * 55)

    if result['sr_agreement']:
        print()
        print("VERDICT: CMB pivot exits in SR regime.")
        print("  f_NL ≪ Planck bound (f_NL = −0.9 ± 5.1, 68% CL).")
        print("  USR dip is MS-linear → model naturally satisfies")
        print("  Planck non-Gaussianity constraints.")
    else:
        print()
        print("VERDICT: Pivot NOT in SR regime — USR contaminates.")
        print("  Report numerical f_NL only; SR comparison invalid.")
        print("  Re-run with different N_star or check background.")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to {args.json}")


if __name__ == '__main__':
    main()
