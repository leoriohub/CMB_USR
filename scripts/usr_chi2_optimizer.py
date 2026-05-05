"""
USR Chi^2 Optimizer: Automated (phi0, y0, N_star) search.

Searches the Higgs USR parameter space for configurations that:
  1. Minimize low-ell chi^2 against Planck 2018 Commander data
  2. Maintain ns_MS close to a target (default 0.975, ACT prior)
  3. Place the P_S(k) dip in a target k-range (default 1e-4 to 5e-4 Mpc^-1)

Uses scipy.optimize.differential_evolution for global optimization.
Logs all evaluations to a JSONL file for analysis and convergence plotting.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.pspectrum_pipeline import run_pspectrum_pipeline, build_weighted_kgrid
from scripts.sachs_wolfe import compute_cl_sw, compute_cl_sw_powerlaw
from scripts.planck_data import get_planck_data, C_ell_to_d_ell
from scripts.constants import As, k_pivot_phys, r_ls, T_cmb
from models import HiggsModel


# ── Helper functions ─────────────────────────────────────────────────────────


def find_k_dip(k_phys, P_S):
    """Find the k value at the USR dip (local minimum for k < 0.01).
    
    Returns -1.0 if no clear dip is found.
    """
    mask = k_phys < 0.01
    if not np.any(mask):
        return -1.0
    k_sub = k_phys[mask]
    ps_sub = P_S[mask]
    i_dip = int(np.argmin(ps_sub))
    if i_dip == 0 or i_dip == len(ps_sub) - 1:
        return -1.0
    return float(k_sub[i_dip])


def extract_ns_ms(k_phys, P_S, pivot_k):
    """Local spectral index at pivot via finite differences.
    
    ns(k_pivot) = 1 + d ln(P_S) / d ln(k) evaluated around pivot.
    Falls back to 0.975 if pivot is at the edge of the k-grid.
    """
    idx = int(np.argmin(np.abs(k_phys - pivot_k)))
    if idx == 0 or idx == len(k_phys) - 1:
        return 0.975
    dlnP = np.log(P_S[idx + 1]) - np.log(P_S[idx - 1])
    dlnk = np.log(k_phys[idx + 1]) - np.log(k_phys[idx - 1])
    return float(1.0 + dlnP / dlnk)


def compute_chi2(pipeline_result, ell_max=29):
    """Diagonal chi^2 vs Planck 2018 low-ell TT (Sachs-Wolfe approximation)."""
    data = {
        "k_phys": pipeline_result["k_phys"],
        "P_S": pipeline_result["P_S"],
    }
    ells_model, C_ell = compute_cl_sw(data, ell_max=ell_max)
    D_model = C_ell_to_d_ell(ells_model, C_ell)

    ells_pl, D_pl, D_err = get_planck_data()
    mask = ells_pl <= ell_max
    D_pl_m = D_pl[mask]
    D_err_m = D_err[mask]
    D_interp = np.interp(ells_pl[mask], ells_model, D_model)
    return float(np.sum(((D_interp - D_pl_m) / D_err_m) ** 2))


# ── Objective function ────────────────────────────────────────────────────────


def build_objective(model_kwargs, num_kwargs, ns_target, ns_tol,
                    k_dip_range, log_path):
    """Return a closure: objective(params) -> loss for scipy.optimize.
    
    Loss = chi^2 + ns_penalty + k_penalty
      - ns_penalty = ((ns_MS - ns_target) / ns_tol)^2
      - k_penalty = 0 if dip in target zone, else 50
      - Pipeline failure returns 1000
    """
    log_file = open(log_path, "a") if log_path else None
    _eval_counter = [0]

    def objective(params):
        phi0 = float(params[0])
        y0 = float(params[1])
        N_star = float(params[2])
        _eval_counter[0] += 1
        eval_num = _eval_counter[0]

        model = HiggsModel(lam=model_kwargs["lam"], xi=model_kwargs["xi"])

        try:
            result = run_pspectrum_pipeline(
                model=model, phi0=phi0, y0=y0,
                k_min=num_kwargs["k_min"],
                k_max=num_kwargs["k_max"],
                k_pivot_phys=num_kwargs["k_pivot_phys"],
                N_star=N_star,
                normalize_to_As=True,
                As=num_kwargs["As"],
                n_workers=num_kwargs.get("workers", 1),
                num_k=num_kwargs.get("num_k", 80),
                save_outputs=False,
            )
        except Exception as exc:
            _log_eval(log_file, eval_num, phi0, y0, N_star,
                      loss=1000.0, status="exception", error=str(exc))
            return 1000.0

        if result["status"] != "success":
            _log_eval(log_file, eval_num, phi0, y0, N_star,
                      loss=1000.0, status="failed",
                      error=result.get("message", "unknown"))
            return 1000.0

        try:
            chi2 = compute_chi2(result, ell_max=num_kwargs.get("ell_max", 29))
            ns_MS = extract_ns_ms(
                result["k_phys"], result["P_S"],
                result["metadata"]["k_pivot_phys"],
            )
            k_dip = find_k_dip(result["k_phys"], result["P_S"])
        except Exception as exc:
            _log_eval(log_file, eval_num, phi0, y0, N_star,
                      loss=1000.0, status="analysis_error", error=str(exc))
            return 1000.0

        ns_penalty = ((ns_MS - ns_target) / ns_tol) ** 2
        if 0 < k_dip_range[0] <= k_dip <= k_dip_range[1]:
            k_penalty = 0.0
        else:
            k_penalty = 50.0
        loss = chi2 + ns_penalty + k_penalty

        _log_eval(log_file, eval_num, phi0, y0, N_star,
                  chi2=chi2, ns_MS=ns_MS, k_dip=k_dip,
                  loss=loss, status="ok")

        if eval_num % 10 == 0:
            s = (f"\r  [{eval_num:5d}] loss={loss:.2f}  "
                 f"chi2={chi2:.2f}  ns={ns_MS:.5f}  "
                 f"k_dip={k_dip:.3e}  "
                 f"phi0={phi0:.3f}  y0={y0:.4f}  N*={N_star:.1f}")
            print(s, end="", flush=True)

        return loss

    return objective


def _log_eval(log_file, eval_num, phi0, y0, N_star, **kw):
    if log_file is None:
        return
    entry = {
        "eval": eval_num,
        "phi0": round(phi0, 6),
        "y0": round(y0, 6),
        "N_star": round(N_star, 4),
        "timestamp": datetime.now().isoformat(),
    }
    entry.update(kw)
    log_file.write(json.dumps(entry) + "\n")
    log_file.flush()


# ── Optimisation ──────────────────────────────────────────────────────────────


def run_optimizer(args):
    """Run scipy.optimize.differential_evolution."""
    from scipy.optimize import differential_evolution

    model_kwargs = {"xi": args.xi, "lam": args.lam}
    num_kwargs = {
        "k_min": args.k_min,
        "k_max": args.k_max,
        "k_pivot_phys": k_pivot_phys,
        "As": As,
        "ell_max": args.ell_max,
        "workers": args.workers,
        "num_k": args.num_k,
    }

    log_path = os.path.abspath(args.log) if args.log else None
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    objective = build_objective(
        model_kwargs, num_kwargs,
        ns_target=args.ns_target,
        ns_tol=args.ns_tol,
        k_dip_range=args.k_dip_range,
        log_path=log_path,
    )

    bounds = [
        (args.phi0_range[0], args.phi0_range[1]),
        (args.y0_range[0], args.y0_range[1]),
        (args.nstar_range[0], args.nstar_range[1]),
    ]

    print(f"\n  Starting DE: popsize={args.popsize}, maxiter={args.maxiter}, "
          f"bounds={bounds}", flush=True)
    print(f"  Logging to: {log_path}", flush=True)
    t0 = time.time()

    result = differential_evolution(
        objective,
        bounds,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        updating="deferred",
        workers=1,
        polish=False,
        tol=0.001,
    )

    elapsed = time.time() - t0
    print(flush=True)
    print(f"\n  Optimization complete in {elapsed:.0f}s", flush=True)
    print(f"  Best: phi0={result.x[0]:.4f}, y0={result.x[1]:.4f}, "
          f"N_star={result.x[2]:.2f}, loss={result.fun:.3f}", flush=True)

    return result


# ── Post-processing ──────────────────────────────────────────────────────────


def re_run_best(best_params, args):
    """Full pipeline at best config with diagnostics and plots."""
    phi0, y0, N_star = best_params
    print(f"\n{'='*60}", flush=True)
    print(f"  Re-running at best: phi0={phi0:.4f}, y0={y0:.4f}, "
          f"N_star={N_star:.2f}", flush=True)
    print(f"{'='*60}", flush=True)

    model = HiggsModel(lam=args.lam, xi=args.xi)

    output_dir = os.path.join(ROOT_DIR, "outputs/simulations/pspectra")
    os.makedirs(output_dir, exist_ok=True)

    k_grid = build_weighted_kgrid(args.k_min, args.k_max, k_pivot_phys)

    result = run_pspectrum_pipeline(
        model=model, phi0=phi0, y0=y0,
        k_min=args.k_min, k_max=args.k_max,
        k_pivot_phys=k_pivot_phys,
        N_star=N_star,
        normalize_to_As=True, As=As,
        output_dir=output_dir, save_outputs=True,
        k_phys_grid=k_grid,
        n_workers=args.workers,
    )

    if result["status"] != "success":
        print(f"  Pipeline failed: {result.get('message', 'unknown')}", flush=True)
        return

    print(f"  P_S(k) saved -> {result.get('output_file', 'N/A')}", flush=True)

    meta = result["metadata"]
    chi2 = compute_chi2(result, ell_max=args.ell_max)
    ns_MS = extract_ns_ms(result["k_phys"], result["P_S"],
                          meta["k_pivot_phys"])
    k_dip = find_k_dip(result["k_phys"], result["P_S"])
    N_total = meta["N_total"]

    ells_lcdm, C_lcdm, _ = compute_cl_sw_powerlaw(
        k_min=args.k_min, k_max=args.k_max,
        As=As, ns=0.965, ell_max=args.ell_max,
    )
    D_lcdm = C_ell_to_d_ell(ells_lcdm, C_lcdm)

    ells_pl, D_pl, D_err = get_planck_data()
    pl_mask = ells_pl <= args.ell_max
    D_lcdm_ip = np.interp(ells_pl[pl_mask], ells_lcdm, D_lcdm)
    chi2_lcdm = float(np.sum(
        ((D_lcdm_ip - D_pl[pl_mask]) / D_err[pl_mask]) ** 2
    ))
    dof = int(np.sum(pl_mask))

    dip_ps = np.interp(k_dip, result["k_phys"], result["P_S"]) if k_dip > 0 else As
    dip_sup = (dip_ps / As - 1) * 100 if k_dip > 0 else 0.0

    print(f"\n  chi2={chi2:.2f} (dof={dof})  LCDM chi2={chi2_lcdm:.2f}", flush=True)
    print(f"  Δchi2 = {chi2 - chi2_lcdm:+.2f}", flush=True)
    print(f"  ns_MS={ns_MS:.5f}  k_dip={k_dip:.3e}  "
          f"dip_sup={dip_sup:.1f}%  N_total={N_total:.1f}", flush=True)

    # Save summary
    sim_dir = os.path.join(ROOT_DIR, "outputs/simulations/configs")
    os.makedirs(sim_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "params": {"phi0": phi0, "y0": y0, "N_star": N_star},
        "chi2": chi2, "chi2_lcdm": chi2_lcdm,
        "dchi2_vs_lcdm": chi2 - chi2_lcdm, "dof": dof,
        "ns_MS": ns_MS, "k_dip": k_dip,
        "dip_suppression_pct": dip_sup,
        "N_total": N_total,
        "output_file": result.get("output_file"),
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = os.path.join(sim_dir, f"best_config_{date_str}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved -> {summary_path}", flush=True)

    # Plot
    try:
        ells_usr, C_usr = compute_cl_sw(result, ell_max=args.ell_max)
        D_usr = C_ell_to_d_ell(ells_usr, C_usr)
        plot_dashboard(ells_usr, D_usr, ells_lcdm, D_lcdm,
                       ells_pl[pl_mask], D_pl[pl_mask], D_err[pl_mask],
                       phi0, y0, N_star, k_dip, chi2, chi2_lcdm, date_str)
        print(f"  Plot saved", flush=True)
    except Exception as exc:
        print(f"  Warning: plot failed: {exc}", flush=True)

    return summary


def plot_dashboard(ells_usr, D_usr, ells_lcdm, D_lcdm,
                   ells_pl, D_pl, D_err, phi0, y0, N_star,
                   k_dip, chi2, chi2_lcdm, date_str):
    """D_ell comparison and suppression ratio plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.errorbar(ells_pl, D_pl, yerr=D_err, fmt="ko", ms=5,
                 capsize=3, label="Planck 2018")
    ax1.semilogx(ells_usr, D_usr, "r-o", ms=4, lw=2, label="USR Best")
    ax1.semilogx(ells_lcdm, D_lcdm, "gray", ls="--", lw=1.5,
                 label="LCDM (ns=0.965)")
    ax1.axvspan(2, 5, color="blue", alpha=0.06,
                label="k_dip target (ell=2-7)")
    ax1.set_xlabel("Multipole $\\ell$")
    ax1.set_ylabel("$D_\\ell$ [$\\mu$K$^2$]")
    ax1.set_title("CMB Low-$\\ell$ Power Spectrum")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_xlim(1.5, 30)
    ax1.grid(True, alpha=0.3)

    ratio = D_usr / np.interp(ells_usr, ells_lcdm, D_lcdm)
    ax2.semilogx(ells_usr, ratio, "r-", lw=1.5, label="USR / LCDM")
    ax2.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5)
    ax2.axvspan(2, 5, color="blue", alpha=0.06)
    ax2.fill_between(ells_usr, ratio, 1.0, alpha=0.2, color="red",
                     where=(ratio < 1.0))
    ax2.set_xlabel("Multipole $\\ell$")
    ax2.set_ylabel("$D_\\ell / D_\\ell^{\\rm LCDM}$")
    ax2.set_title("Suppression Relative to LCDM")
    ax2.legend(fontsize=8)
    ax2.set_xlim(1.5, 30)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Best USR: $\\phi_0$={phi0:.2f}, $y_0$={y0:.3f}, "
        f"$N_{{*}}$={N_star:.0f}, $k_{{\\rm dip}}$={k_dip:.2e}\n"
        f"$\\chi^2$={chi2:.1f}  LCDM $\\chi^2$={chi2_lcdm:.1f}  "
        f"$\\Delta\\chi^2$={chi2 - chi2_lcdm:+.1f}",
        fontsize=11, fontweight="bold",
    )

    plot_dir = os.path.join(ROOT_DIR, "outputs/plots/optimizer")
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, f"best_config_{date_str}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_convergence(log_path):
    """Convergence metrics from JSONL log."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    ok_recs = [r for r in records if r.get("status") == "ok"]
    if not ok_recs:
        print("  No successful evals for convergence plot.", flush=True)
        return

    evals = np.arange(len(ok_recs))
    chi2 = np.array([r["chi2"] for r in ok_recs])
    ns = np.array([r["ns_MS"] for r in ok_recs])
    kd = np.array([r.get("k_dip", -1.0) for r in ok_recs])
    loss = np.array([r["loss"] for r in ok_recs])
    best = np.minimum.accumulate(loss)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(evals, loss, "b.", alpha=0.25, ms=2, label="all")
    axes[0].plot(evals, best, "r-", lw=2, label="running best")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Optimizer Convergence")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0.975, color="k", ls="--", alpha=0.5, label="target")
    axes[1].axhline(0.985, color="k", ls=":", alpha=0.3)
    axes[1].axhline(0.965, color="k", ls=":", alpha=0.3)
    axes[1].plot(evals, ns, "g.", alpha=0.25, ms=2)
    axes[1].set_ylabel("ns_MS")
    axes[1].set_ylim(0.94, 1.02)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].axhline(1e-4, color="k", ls="--", alpha=0.3)
    axes[2].axhline(5e-4, color="k", ls="--", alpha=0.3)
    ok_k = kd > 0
    if np.any(ok_k):
        axes[2].semilogy(evals[ok_k], kd[ok_k], "m.", alpha=0.25, ms=2)
    axes[2].set_ylabel("k_dip [Mpc$^{-1}$]")
    axes[2].set_xlabel("Evaluation number")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    base = os.path.splitext(log_path)[0]
    out = base + "_convergence.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Convergence plot -> {out}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="USR Chi^2 Optimizer: search (phi0, y0, N_star) for "
                    "best Planck low-ell fit"
    )
    p.add_argument("--method", default="differential_evolution",
                   choices=["differential_evolution"])
    p.add_argument("--phi0-range", nargs=2, type=float, default=[5.20, 5.90],
                   metavar=("MIN", "MAX"))
    p.add_argument("--y0-range", nargs=2, type=float, default=[-0.20, -0.03],
                   metavar=("MIN", "MAX"))
    p.add_argument("--nstar-range", nargs=2, type=float, default=[45.0, 62.0],
                   metavar=("MIN", "MAX"))
    p.add_argument("--ns-target", type=float, default=0.975)
    p.add_argument("--ns-tol", type=float, default=0.01)
    p.add_argument("--k-dip-range", nargs=2, type=float,
                   default=[1e-4, 5e-4], metavar=("MIN", "MAX"))
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--popsize", type=int, default=15)
    p.add_argument("--workers", type=int, default=4,
                   help="k-mode parallel workers inside pipeline")
    p.add_argument("--xi", type=float, default=15000.0)
    p.add_argument("--lam", type=float, default=0.13)
    p.add_argument("--log", type=str, default=None,
                   help="JSONL log (auto-generated if omitted)")
    p.add_argument("--save-best", action="store_true")
    p.add_argument("--re-run-best", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-min", type=float, default=1e-5)
    p.add_argument("--k-max", type=float, default=1.0)
    p.add_argument("--ell-max", type=int, default=29)
    p.add_argument("--num-k", type=int, default=80,
                   help="k-modes per eval during search (use weighted grid "
                        "for re-run)")
    return p.parse_args(argv)


def main():
    args = parse_args()

    if args.log is None:
        ds = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log = os.path.join(ROOT_DIR,
                                f"outputs/simulations/logs/optimizer_log_{ds}.jsonl")

    print(f"{'='*60}", flush=True)
    print(f"  USR Chi^2 Optimizer", flush=True)
    print(f"  phi0=[{args.phi0_range[0]:.2f}, {args.phi0_range[1]:.2f}]  "
          f"y0=[{args.y0_range[0]:.3f}, {args.y0_range[1]:.3f}]  "
          f"N*=[{args.nstar_range[0]:.1f}, {args.nstar_range[1]:.1f}]",
          flush=True)
    print(f"  ns_target={args.ns_target}  ns_tol={args.ns_tol}  "
          f"k_dip_target=[{args.k_dip_range[0]:.1e},{args.k_dip_range[1]:.1e}]",
          flush=True)
    print(f"  maxiter={args.maxiter}  popsize={args.popsize}  "
          f"workers={args.workers}", flush=True)
    print(f"{'='*60}", flush=True)

    opt_result = run_optimizer(args)

    if args.save_best or args.re_run_best:
        re_run_best(opt_result.x, args)

    if args.log and os.path.exists(args.log):
        plot_convergence(args.log)

    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
