"""
Validate CAMB setup against Planck 2018 LCDM best-fit C_ell.

Runs CAMB with a power-law primordial spectrum (As=2.1e-9, ns=0.965)
and compares the resulting C_ell^TT against the Planck 2018 best-fit
LCDM model. This validates that the CAMB configuration, cosmological
parameters, and normalization are all correct.

Usage:
    python scripts/validate_camb_lcdm.py
"""

import json
import os
import sys

import numpy as np
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.camb_wrapper import compute_cl_camb_powerlaw
from scripts.constants import As, k_pivot_phys, T_cmb
from scripts.planck_data import get_planck_data_asymmetric


TOL = {
    "blue": "#4477AA",
    "red": "#CC3311",
    "green": "#228833",
    "grey": "#666666",
    "dark": "#222222",
}


def validate_LCDM():
    """Run CAMB LCDM and validate against Planck best-fit."""
    ell_max = 2500
    print(f"Running CAMB LCDM validation (ell_max={ell_max})...")

    ells, C_TT, C_TE, C_EE = compute_cl_camb_powerlaw(ell_max=ell_max)

    ell_factor = ells * (ells + 1) / (2 * np.pi) * (T_cmb * 1e6) ** 2
    D_TT = C_TT * ell_factor

    planck_ells, D_planck, D_err_lower, D_err_upper = get_planck_data_asymmetric()

    # ── Validation checks ──
    print()
    print("  CAMB LCDM validation results:")
    print(f"  {'=' * 50}")

    # Check 1: Low-ell (2-29) residuals vs Planck
    ell_low = ells[(ells >= 2) & (ells <= 29)]
    D_low = D_TT[(ells >= 2) & (ells <= 29)]
    planck_interp = interp1d(planck_ells, D_planck, kind="linear",
                             bounds_error=False, fill_value="extrapolate")
    D_planck_at_ells = planck_interp(ell_low)
    residuals = (D_low - D_planck_at_ells) / D_planck_at_ells * 100
    max_res = float(np.max(np.abs(residuals)))
    print(f"  Low-ell (2-29) D_TT vs Planck: max residual = {max_res:.2f}%")

    # Check 2: First acoustic peak position
    peak_idx = int(np.argmax(D_TT[30:]) + 30)
    peak_ell = int(ells[peak_idx])
    print(f"  First acoustic peak: ell = {peak_ell} (expected ~220)")
    print(f"  D_TT at peak: {D_TT[peak_idx]:.1f} muK^2")

    # Check 3: Acoustic trough position
    trough_search = D_TT[peak_idx - 50:peak_idx]
    trough_idx_local = int(np.argmin(trough_search))
    trough_ell = int(ells[peak_idx - 50 + trough_idx_local])
    print(f"  Pre-peak trough: ell = {trough_ell} (expected ~180)")

    print(f"  {'=' * 50}")
    print()

    # ── Plot 1: Full-sky D_ell ──
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.errorbar(
        planck_ells, D_planck,
        yerr=[D_err_upper, D_err_lower],
        fmt="o", color=TOL["dark"], capsize=2, capthick=0.8,
        markersize=3, elinewidth=0.8, alpha=0.7,
        label="Planck 2018 TT",
    )
    ax.semilogy(ells, D_TT, "-", color=TOL["blue"], lw=1.5,
                label="CAMB LCDM (power-law)")

    ax.set_xlabel(r"$\ell$", fontsize=14)
    ax.set_ylabel(r"$D_\ell^{TT}\ [\mu{\rm K}^2]$", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(1.5, ell_max)

    fig.tight_layout()
    out_dir = "outputs/plots/diagnostics"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "camb_lcdm_validation.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    # ── Plot 2: Low-ell zoom ──
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.errorbar(
        planck_ells, D_planck,
        yerr=[D_err_upper, D_err_lower],
        fmt="o", color=TOL["dark"], capsize=3, capthick=1,
        markersize=4, elinewidth=1,
        label="Planck 2018 low-ell TT",
    )
    ax.semilogy(ells[ells <= 30], D_TT[ells <= 30], "-",
                color=TOL["blue"], lw=1.5, label="CAMB LCDM")

    ax.set_xlabel(r"$\ell$", fontsize=14)
    ax.set_ylabel(r"$D_\ell^{TT}\ [\mu{\rm K}^2]$", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(1.5, 31)

    fig.tight_layout()
    path = os.path.join(out_dir, "camb_lcdm_low_ell.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

    # ── Save results ──
    record = {
        "metadata": {"model": "LCDM_powerlaw", "As": As, "ns": 0.965},
        "ells": ells.tolist(),
        "C_TT": C_TT.tolist(),
        "C_TE": C_TE.tolist(),
        "C_EE": C_EE.tolist(),
        "D_TT": D_TT.tolist(),
        "validation": {
            "peak_ell": peak_ell,
            "peak_D_TT": float(D_TT[peak_idx]),
            "trough_ell": trough_ell,
            "max_low_ell_residual_pct": max_res,
        },
    }
    out_path = os.path.join("outputs/simulations/c_ell", "camb_lcdm_validation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  Saved: {out_path}")
    print()

    return record


if __name__ == "__main__":
    record = validate_LCDM()
    print("  Validation complete.")
    if record["validation"]["peak_ell"] not in range(200, 250):
        print(f"  WARNING: Peak at ell={record['validation']['peak_ell']} "
              f"(expected ~220). Check cosmology parameters.")
    if record["validation"]["max_low_ell_residual_pct"] > 10:
        print(f"  WARNING: Large low-ell residuals "
              f"({record['validation']['max_low_ell_residual_pct']:.1f}%). "
              f"Check normalization.")
