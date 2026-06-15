"""PBH abundance for Ezquiaga CHI (1705.04861).

Press-Schechter formalism:
  β_f(M)  = erfc(ζ_c / √(2 P_R(k)))
  M(k)    = γ · M_eq · (k_eq/k)²           (RD reentry)
  β_eq(M) = β_f(M) · k / k_eq              (linear growth during RD)

Masses are shifted to present-day via accretion factor from the paper:
  M_present = M_form × accretion_factor
  accretion_factor = 3×10⁷  (Chisholm 2006)

Two modes:
  --ms-json : Load MS P_S(k) from a saved pspectrum JSON (k in Mpc⁻¹)
  SR mode   : Run background + SR P_S(k) (fast exploration)

Output follows json-logging-conventions.
"""

import os
import json
import time
import argparse
import numpy as np
from scipy.special import erfc

from scripts.constants import (
    zeta_c_default, gamma_default, k_eq_default, M_eq_default,
    k_pivot_phys, S, ROOT_DIR,
)
from scripts.plotting import (
    make_filename, save_fig, plot_pbh_abundance,
)

S_CODE = S
ACCRETION_FACTOR = 3e7  # mass growth from equality to today (Chisholm 2006)


def mass_from_k(k_phys, gamma=gamma_default, k_eq=k_eq_default, M_eq=M_eq_default):
    k = np.asarray(k_phys, dtype=float)
    return gamma * M_eq * (k_eq / k) ** 2


def beta_f_press_schechter(P_S, zeta_c=zeta_c_default):
    ps = np.asarray(P_S, dtype=float)
    return erfc(zeta_c / np.sqrt(2 * np.clip(ps, 1e-300, None)))


def beta_equality(beta_f, k_phys, k_eq=k_eq_default):
    return np.asarray(beta_f) * np.asarray(k_phys) / k_eq


def compute_pbh_abundance(k_phys, P_S, gamma=gamma_default,
                          zeta_c=zeta_c_default, k_eq=k_eq_default,
                          M_eq=M_eq_default):
    k = np.asarray(k_phys, dtype=float)
    M = mass_from_k(k, gamma, k_eq, M_eq)
    bf = beta_f_press_schechter(P_S, zeta_c)
    beq = beta_equality(bf, k, k_eq)

    ok = np.isfinite(M) & np.isfinite(beq) & (M > 0) & (beq >= 0)
    M_ok = M[ok]
    beq_ok = np.maximum(beq[ok], 0.0)
    bf_ok = bf[ok]

    order = np.argsort(M_ok)
    return {
        "M": M_ok[order],
        "beta_f": bf_ok[order],
        "f_pbh": beq_ok[order],
    }


def load_ms_json(path):
    """Load MS P_S(k) from a saved pspectrum JSON.

    Expects structure: {
      "spectrum": {"k_phys": [...], "P_S": [...]},
      "metadata": {...}
    }
    Returns (k_phys, P_S, metadata).
    """
    with open(path) as f:
        d = json.load(f)

    spec = d.get("spectrum", d)
    k = np.array([v if v is not None else np.nan for v in spec["k_phys"]],
                 dtype=float)
    ps = np.array([v if v is not None else np.nan for v in spec["P_S"]],
                  dtype=float)
    meta = d.get("metadata", {})

    ok = np.isfinite(k) & np.isfinite(ps) & (k > 0) & (ps > 0)
    order = np.argsort(k[ok])
    return k[ok][order], ps[ok][order], meta


def run_from_ms(ms_path, gamma, zeta_c, k_eq, M_eq,
                save_json=True, plot=True, category="pbh",
                accretion=ACCRETION_FACTOR, use_old_bounds=False,
                smooth_bounds=False):
    k_phys, P_S, meta = load_ms_json(ms_path)
    chi0 = meta.get("x0", meta.get("chi0", 0.0))
    y0 = meta.get("y0", -1e-4)
    N_total = meta.get("N_total", 0)

    pbh = compute_pbh_abundance(k_phys, P_S, gamma, zeta_c, k_eq, M_eq)
    M_form, f_pbh = pbh["M"], pbh["f_pbh"]
    M = M_form * accretion  # present-day mass after accretion/merging

    mass_lo, mass_hi = 1e-12, 1e5
    m_range = (M >= mass_lo) & (M <= mass_hi)
    f_total = np.trapezoid(f_pbh[m_range], np.log(M[m_range])) if np.sum(m_range) > 1 else 0.0
    peak_i = int(np.argmax(f_pbh)) if len(f_pbh) > 0 else 0
    M_peak = M[peak_i] if len(M) > 0 else np.nan

    metadata = {
        "source": "MS",
        "ms_path": os.path.basename(ms_path),
        "chi0": chi0, "y0": y0, "N_total": N_total,
        "gamma": gamma, "zeta_c": zeta_c,
        "k_eq_Mpc": k_eq, "M_eq_Msun": M_eq,
        "f_pbh_total": float(f_total),
        "M_peak_Msun": float(M_peak),
        "n_modes": len(k_phys),
    }

    n_tr = round(N_total, 1)
    output_path = None
    if save_json:
        rec = {
            "_type": "result",
            "format_version": 1,
            "metadata": metadata,
            "spectrum": {"k_phys": k_phys.tolist(), "P_S": P_S.tolist()},
            "pbh": {"M_Msun": M.tolist(), "f_pbh": f_pbh.tolist()},
        }
        fname = make_filename("pbh", chi0, y0, n_tr, ".json")
        pbh_dir = os.path.join(ROOT_DIR, "outputs/simulations/logs")
        os.makedirs(pbh_dir, exist_ok=True)
        output_path = os.path.join(pbh_dir, fname)
        with open(output_path, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"  Saved: {output_path}")

    if plot and len(M) > 0:
        suffix = ""
        if use_old_bounds:
            suffix += "_oldbounds"
        if smooth_bounds:
            suffix += "_smooth"
        fname = make_filename("pbh", chi0, y0, n_tr, suffix)
        plot_pbh_abundance(
            M, f_pbh, zeta_c=zeta_c, gamma=gamma,
            model_label=f"Ezquiaga CHI χ₀={chi0}",
            filename=fname, category=category,
            use_old_bounds=use_old_bounds,
            smooth_bounds=smooth_bounds,
        )

    print(f"  N_total={N_total:.1f}, modes={len(k_phys)}, "
          f"f_PBH_total={f_total:.4e}, M_peak={M_peak:.2e} M⊙")

    return {"M": M, "f_pbh": f_pbh, "metadata": metadata, "output_path": output_path}


def run_from_sr(chi0=8.0, y0=-1e-4, bg_steps=5000, T_max=1000.0,
                gamma=gamma_default, zeta_c=zeta_c_default,
                k_eq=k_eq_default, M_eq=M_eq_default,
                save_json=True, plot=True, category="pbh",
                accretion=ACCRETION_FACTOR, use_old_bounds=False,
                smooth_bounds=False):
    from models.ezquiaga_chi import EzquiagaCHIModel, inflection_parameters
    from inf_dyn_background import run_background_simulation, get_derived_quantities
    from scripts.plotting import compute_ps_sr

    model = EzquiagaCHIModel()
    a_ref, b_ref = inflection_parameters(0.784, 0.77, beta=1e-5)
    model.a = a_ref; model.b = b_ref
    model.v0 = model._V0 * model.a / (model.b * model.c) ** 2
    model.patch_background_solver()
    model.x0 = chi0; model.y0 = y0
    model.bg_steps = bg_steps; model.T_max = T_max

    T = np.linspace(0, T_max, bg_steps)
    bg_sol = run_background_simulation(model, T)
    derived = get_derived_quantities(bg_sol, model)
    epsH = derived["epsH"]
    end_candidates = np.where(epsH >= 1.0)[0]
    end_idx = int(end_candidates[0]) if len(end_candidates) > 0 else len(epsH) - 1
    N_total = float(derived["N"][end_idx])

    k_cu, P_S_sr, _ = compute_ps_sr(bg_sol, end_idx)
    valid = np.isfinite(P_S_sr) & (P_S_sr > 0) & np.isfinite(k_cu) & (k_cu > 0)
    k_cu = k_cu[valid]; P_S_sr = P_S_sr[valid]
    N_v = derived["N"][:end_idx + 1][valid]

    N_pivot = N_total - 60.0
    z_bg = bg_sol[2]; n_bg = bg_sol[3]; N_arr = derived["N"]
    zp = float(np.interp(N_pivot, N_arr, z_bg))
    ap = float(np.exp(np.interp(N_pivot, N_arr, n_bg)))
    k_piv = ap * zp * S_CODE
    k_phys = k_cu * (k_pivot_phys / k_piv)

    n_be = N_total - N_v
    mask = n_be < 40
    k_phys = k_phys[mask]; P_S_sr = P_S_sr[mask]

    pbh = compute_pbh_abundance(k_phys, P_S_sr, gamma, zeta_c, k_eq, M_eq)
    M_form, f_pbh = pbh["M"], pbh["f_pbh"]
    M = M_form * accretion

    mass_lo, mass_hi = 1e-12, 1e5
    m_range = (M >= mass_lo) & (M <= mass_hi)
    f_total = np.trapezoid(f_pbh[m_range], np.log(M[m_range])) if np.sum(m_range) > 1 else 0.0
    peak_i = int(np.argmax(f_pbh)) if len(f_pbh) > 0 else 0
    M_peak = M[peak_i] if len(M) > 0 else np.nan

    metadata = {
        "source": "SR", "chi0": chi0, "y0": y0, "N_total": N_total,
        "gamma": gamma, "zeta_c": zeta_c,
        "k_eq_Mpc": k_eq, "M_eq_Msun": M_eq,
        "f_pbh_total": float(f_total), "M_peak_Msun": float(M_peak),
    }

    n_tr = round(N_total, 1)
    output_path = None
    if save_json:
        rec = {
            "_type": "result", "format_version": 1,
            "metadata": metadata,
            "spectrum": {"k_phys": k_phys.tolist(), "P_S": P_S_sr.tolist()},
            "pbh": {"M_Msun": M.tolist(), "f_pbh": f_pbh.tolist()},
        }
        fname = make_filename("pbh", chi0, y0, n_tr, ".json")
        od = os.path.join(ROOT_DIR, "outputs/simulations/logs")
        os.makedirs(od, exist_ok=True)
        output_path = os.path.join(od, fname)
        with open(output_path, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"  Saved: {output_path}")

    if plot and len(M) > 0:
        suffix = ""
        if use_old_bounds:
            suffix += "_oldbounds"
        if smooth_bounds:
            suffix += "_smooth"
        fname = make_filename("pbh", chi0, y0, n_tr, suffix)
        plot_pbh_abundance(
            M, f_pbh, zeta_c=zeta_c, gamma=gamma,
            model_label=f"Ezquiaga CHI χ₀={chi0}",
            filename=fname, category=category,
            use_old_bounds=use_old_bounds,
            smooth_bounds=smooth_bounds,
        )

    print(f"  N_total={N_total:.1f}, modes={len(k_phys)}, "
          f"f_PBH_total={f_total:.4e}, M_peak={M_peak:.2e} M⊙")

    return {"M": M, "f_pbh": f_pbh, "metadata": metadata, "output_path": output_path}


def main():
    p = argparse.ArgumentParser(description="PBH abundance for Ezquiaga CHI")
    p.add_argument("--ms-json", type=str, default=None,
                   help="Load MS P_S from pspectrum JSON")
    p.add_argument("--chi0", type=float, default=8.0)
    p.add_argument("--y0", type=float, default=-1e-4)
    p.add_argument("--gamma", type=float, default=gamma_default)
    p.add_argument("--zeta-c", type=float, default=zeta_c_default)
    p.add_argument("--accretion", type=float, default=ACCRETION_FACTOR,
                   help="Mass growth from equality to today (default 3e7)")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--use-old-bounds", action="store_true",
                   help="Use old 2017 paper bounds instead of modern ones")
    p.add_argument("--smooth-bounds", action="store_true",
                   help="Smooth sparse bounds data using Pchip shape-preserving interpolation")
    args = p.parse_args()

    if args.ms_json:
        run_from_ms(
            args.ms_json, args.gamma, args.zeta_c,
            k_eq_default, M_eq_default,
            save_json=not args.no_save, plot=not args.no_plot,
            accretion=args.accretion,
            use_old_bounds=args.use_old_bounds,
            smooth_bounds=args.smooth_bounds,
        )
    else:
        run_from_sr(
            chi0=args.chi0, y0=args.y0,
            gamma=args.gamma, zeta_c=args.zeta_c,
            save_json=not args.no_save, plot=not args.no_plot,
            accretion=args.accretion,
            use_old_bounds=args.use_old_bounds,
            smooth_bounds=args.smooth_bounds,
        )


if __name__ == "__main__":
    main()
