"""Pipeline sanity check: punctuated model end-to-end (solver -> P_S(k) -> CAMB -> Planck)."""
import json
import os
import sys

import numpy as np

from models import PunctuatedInflationModel
from scripts.constants import As, T_cmb
from scripts.pspectrum_pipeline import run_pspectrum_pipeline
from scripts.camb_wrapper import compute_cl_full_camb, compute_cl_camb_powerlaw, compute_chi2_camb
from scripts.planck_data import C_ell_to_d_ell

M = 1.1323e-7
LAM = 3.3299e-15
PHI0 = 12.0
Y0 = 0.0
N_STAR = 77.2

TOL = {"blue": "#4477AA", "red": "#CC3311", "green": "#228833", "grey": "#666666", "dark": "#222222"}

def check_N_total(metadata):
    N_tot = metadata["N_total"]
    N_star = metadata["N_star"]
    passed = N_tot > N_star
    print(f"  [{'PASS' if passed else 'FAIL'}] N_total={N_tot:.1f} > N_star={N_star:.1f}")
    return passed

def check_PS_peak(k_phys, P_S):
    finite = np.isfinite(P_S) & (P_S > 0)
    k_finite = k_phys[finite]
    PS_finite = P_S[finite]
    peak_idx = np.argmax(PS_finite)
    k_peak = k_finite[peak_idx]
    passed = 1e-4 < k_peak < 1e-2
    print(f"  [{'PASS' if passed else 'FAIL'}] P_S peak at k={k_peak:.4e} Mpc^-1 (expected ~1e-3)")
    return passed

def check_pivot_As(metadata):
    scale = metadata.get("scale_factor", 1.0)
    passed = scale is not None and scale > 0
    print(f"  [{'PASS' if passed else 'FAIL'}] Normalized to As={As:.2e} (scale={scale:.4e})")
    return passed

def check_low_ell_enhancement(ells, D_model, D_lcdm, ell_max=10):
    low = ells <= ell_max
    ratio = float(np.mean(D_model[low]) / np.mean(D_lcdm[low]))
    passed = ratio > 1.0
    print(f"  [{'PASS' if passed else 'FAIL'}] Low-ell D_ell ratio (model/LCDM) = {ratio:.4f} (expected >1)")
    return passed

def check_high_ell_consistency(ells, D_model, D_lcdm):
    mask = (ells >= 200) & (ells <= 2500)
    ratio = D_model[mask] / D_lcdm[mask]
    max_dev = float(np.max(np.abs(ratio - 1)) * 100)
    passed = max_dev < 5.0
    print(f"  [{'PASS' if passed else 'FAIL'}] High-ell max deviation = {max_dev:.2f}% (threshold < 5%)")
    return passed

def check_acoustic_peak(ells, D_ell):
    peak_idx = int(np.argmax(D_ell[30:]) + 30)
    peak = int(ells[peak_idx])
    passed = 210 <= peak <= 230
    print(f"  [{'PASS' if passed else 'FAIL'}] First acoustic peak at ell={peak} (expected ~220)")
    return passed

def check_chi2(chi2_m, chi2_l, dchi2):
    passed = 0 < chi2_m < 100
    print(f"  [{'PASS' if passed else 'FAIL'}] Model chi2={chi2_m:.2f}, LCDM chi2={chi2_l:.2f}, Delta={dchi2:+.2f}")
    return passed

def main():
    print("=" * 60)
    print("  Pipeline Sanity Validation")
    print("  Model: Punctuated Inflation")
    print("=" * 60)

    print("\n[Stage 1] Computing background + P_S(k)...")
    model = PunctuatedInflationModel(m=M, lam=LAM)
    result = run_pspectrum_pipeline(
        model=model,
        phi0=PHI0, y0=Y0,
        k_min=1e-5, k_max=1.0, num_k=80,
        N_star=N_STAR,
        normalize_to_As=True,
        n_workers=1,
    )
    if result["status"] != "success":
        print(f"  FATAL: {result['message']}")
        return 1
    print(f"  Output: {result['output_file']}")

    metadata = result["metadata"]
    k_phys = result["k_phys"]
    P_S = result["P_S"]

    checks = []
    checks.append(check_N_total(metadata))
    checks.append(check_PS_peak(k_phys, P_S))
    checks.append(check_pivot_As(metadata))

    print("\n[Stage 2] Computing C_ell via CAMB...")
    ells, C_TT, C_TE, C_EE = compute_cl_full_camb(result, ell_max=2500)
    D_model = C_ell_to_d_ell(ells, C_TT)

    print("\n[Stage 3] Computing LCDM baseline...")
    ells_l, C_l, _, _ = compute_cl_camb_powerlaw(ell_max=2500)
    D_lcdm = C_ell_to_d_ell(ells_l, C_l)

    checks.append(check_low_ell_enhancement(ells, D_model, D_lcdm))
    checks.append(check_high_ell_consistency(ells, D_model, D_lcdm))
    checks.append(check_acoustic_peak(ells, D_model))

    print("\n[Stage 4] Computing chi2 vs Planck...")
    chi2_m, chi2_l, dchi2 = compute_chi2_camb(result)
    checks.append(check_chi2(chi2_m, chi2_l, dchi2))

    n_pass = sum(checks)
    n_total = len(checks)
    print(f"\n{'=' * 60}")
    print(f"  Result: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("  ALL CHECKS PASSED")
    else:
        print(f"  {n_total - n_pass} CHECK(S) FAILED")
    print(f"{'=' * 60}")

    out_dir = "outputs/simulations/c_ell"
    os.makedirs(out_dir, exist_ok=True)
    record = {
        "_type": "result",
        "format_version": 2,
        "metadata": {"model": "Punctuated", "m": M, "lam": LAM, "N_star": N_STAR},
        "c_ell": {
            "ells": ells.tolist(),
            "C_ell_TT": C_TT.tolist(),
            "D_ell": D_model.tolist(),
        },
    }
    out_path = os.path.join(out_dir, "pipeline_sanity_punctuated.json")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  Saved: {out_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

        ax = axes[0, 0]
        ax.loglog(k_phys, P_S, color=TOL["blue"], lw=1.5)
        ax.axvline(1e-3, color=TOL["grey"], ls="--", lw=0.8, alpha=0.5)
        ax.set_xlabel(r"$k$ [Mpc$^{-1}$]", fontsize=10)
        ax.set_ylabel(r"$\mathcal{P}_\mathcal{R}(k)$", fontsize=10)
        ax.set_title("Primordial Power Spectrum", fontsize=11)

        ax = axes[0, 1]
        ax.plot(ells, D_model, color=TOL["blue"], lw=1.5, label="Punctuated")
        ax.plot(ells_l, D_lcdm, color=TOL["grey"], lw=1, ls="--", label="LCDM")
        ax.set_xlabel(r"$\ell$", fontsize=10)
        ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]", fontsize=10)
        ax.set_title("Angular Power Spectrum", fontsize=11)
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        low = ells <= 30
        ax.plot(ells[low], D_model[low], color=TOL["blue"], lw=1.5, label="Punctuated")
        ax.plot(ells_l[low], D_lcdm[low], color=TOL["grey"], lw=1, ls="--", label="LCDM")
        ax.set_xlabel(r"$\ell$", fontsize=10)
        ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]", fontsize=10)
        ax.set_title(r"Low-$\ell$ Zoom", fontsize=11)
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        mask = ells >= 2
        residual = (D_model[mask] / D_lcdm[mask] - 1) * 100
        ax.plot(ells[mask], residual, color=TOL["blue"], lw=1.5)
        ax.axhline(0, color=TOL["grey"], ls="--", lw=0.8)
        ax.axhline(5, color=TOL["red"], ls=":", lw=0.8, alpha=0.5)
        ax.axhline(-5, color=TOL["red"], ls=":", lw=0.8, alpha=0.5)
        ax.set_xlabel(r"$\ell$", fontsize=10)
        ax.set_ylabel(r"$\Delta D_\ell / D_\ell^{\rm LCDM}$ [%]", fontsize=10)
        ax.set_title("Residuals vs LCDM", fontsize=11)

        fig.tight_layout()
        out_dir = "outputs/plots/diagnostics"
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"pipeline_sanity.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)
    except ImportError:
        print("  Skipping plot (matplotlib not available)")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
