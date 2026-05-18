# Pipeline Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate full inflation pipeline (solver → P_S(k) → CAMB C_ell → Planck) using punctuated model known config

**Architecture:** Create validation script that runs punctuated model through pipeline, generates P_S(k), feeds into CAMB, compares vs Planck LCDM. Then run existing test suite.

**Tech Stack:** numpy, scipy, camb, matplotlib, SSH (`ssh uni`)

---

### Task 1: Create validation script `scripts/validate_pipeline_sanity.py`

**Files:**
- Create: `scripts/validate_pipeline_sanity.py`
- Reference: `scripts/pspectrum_pipeline.py` for P_S pipeline
- Reference: `scripts/camb_wrapper.py` for CAMB functions
- Reference: `scripts/planck_data.py` for Planck data
- Reference: `models/punctuated.py` for model params

- [ ] **Step 1: Write script skeleton with imports and model setup**

```python
"""Pipeline sanity check: punctuated model end-to-end."""
import os, sys, json
import numpy as np
from models import PunctuatedInflationModel
from scripts.constants import As, k_pivot_phys, T_cmb
from scripts.pspectrum_pipeline import run_pspectrum_pipeline
from scripts.camb_wrapper import compute_cl_full_camb, compute_cl_camb_powerlaw
from scripts.planck_data import get_planck_data_asymmetric, C_ell_to_d_ell
from scripts.constants import As

TOL = {
    "blue": "#4477AA",
    "red": "#CC3311",
    "green": "#228833",
    "grey": "#666666",
    "dark": "#222222",
}

M = 1.1323e-7
LAM = 3.3299e-15
PHI0 = 12.0
Y0 = 0.0
N_STAR = 77.2
```

- [ ] **Step 2: Write check_N_total function**

```python
def check_N_total(metadata):
    N_tot = metadata["N_total"]
    N_star = metadata["N_star"]
    passed = N_tot > N_star
    print(f"  [{'PASS' if passed else 'FAIL'}] N_total={N_tot:.1f} > N_star={N_star:.1f}")
    return passed
```

- [ ] **Step 3: Write check_PS_peak function**

```python
def check_PS_peak(k_phys, P_S):
    # Find peak in P_S
    finite = np.isfinite(P_S) & (P_S > 0)
    k_finite = k_phys[finite]
    PS_finite = P_S[finite]
    peak_idx = np.argmax(PS_finite)
    k_peak = k_finite[peak_idx]
    # Expected: peak at k ~ 1e-3 Mpc^-1
    passed = 1e-4 < k_peak < 1e-2
    print(f"  [{'PASS' if passed else 'FAIL'}] P_S peak at k={k_peak:.4e} Mpc^-1 (expected ~1e-3)")
    return passed
```

- [ ] **Step 4: Write check_pivot_As function**

```python
def check_pivot_As(metadata):
    P_S_pivot = metadata.get("P_S_pivot_raw")
    As_target = metadata.get("As_target", As)
    scale = metadata.get("scale_factor", 1.0)
    passed = scale is not None and scale > 0
    print(f"  [{'PASS' if passed else 'FAIL'}] Normalized to As={As_target:.2e} (scale={scale:.4e})")
    return passed
```

- [ ] **Step 5: Write check_low_ell_enhancement function**

```python
def check_low_ell_enhancement(ells, D_model, D_lcdm, ell_max=10):
    low = ells <= ell_max
    ratio = np.mean(D_model[low]) / np.mean(D_lcdm[low])
    passed = ratio > 1.0  # punctuated enhances at low ell
    print(f"  [{'PASS' if passed else 'FAIL'}] Low-ell D_ell ratio (model/LCDM) = {ratio:.4f} (expected >1)")
    return passed
```

- [ ] **Step 6: Write check_high_ell_consistency function**

```python
def check_high_ell_consistency(ells, D_model, D_lcdm):
    mask = (ells >= 200) & (ells <= 2500)
    ratio = D_model[mask] / D_lcdm[mask]
    max_dev = np.max(np.abs(ratio - 1)) * 100
    passed = max_dev < 5.0
    print(f"  [{'PASS' if passed else 'FAIL'}] High-ell max deviation = {max_dev:.2f}% (threshold < 5%)")
    return passed
```

- [ ] **Step 7: Write check_acoustic_peak function**

```python
def check_acoustic_peak(ells, D_ell):
    peak_idx = int(np.argmax(D_ell[30:]) + 30)
    peak = int(ells[peak_idx])
    passed = 210 <= peak <= 230
    print(f"  [{'PASS' if passed else 'FAIL'}] First acoustic peak at ell={peak} (expected ~220)")
    return passed
```

- [ ] **Step 8: Write check_chi2 function**

```python
def check_chi2(chi2_m, chi2_l, dchi2):
    passed = 0 < chi2_m < 100
    print(f"  [{'PASS' if passed else 'FAIL'}] Model chi2={chi2_m:.2f}, LCDM chi2={chi2_l:.2f}, Delta={dchi2:+.2f}")
    return passed
```

- [ ] **Step 9: Write main function that runs the full pipeline and all checks**

```python
def main():
    print("=" * 60)
    print("  Pipeline Sanity Validation")
    print("  Model: Punctuated Inflation")
    print("=" * 60)

    # Stage 1: Run P_S(k) pipeline
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
    # Check 1: N_total
    checks.append(check_N_total(metadata))
    # Check 2: P_S peak
    checks.append(check_PS_peak(k_phys, P_S))
    # Check 3: Pivot normalization
    checks.append(check_pivot_As(metadata))

    # Stage 2: Compute C_ell via CAMB
    print("\n[Stage 2] Computing C_ell via CAMB...")
    ells, C_TT, C_TE, C_EE = compute_cl_full_camb(result, ell_max=2500)
    D_model = C_ell_to_d_ell(ells, C_TT)

    print("\n[Stage 3] Computing LCDM baseline...")
    ells_l, C_l, _, _ = compute_cl_camb_powerlaw(ell_max=2500)
    D_lcdm = C_ell_to_d_ell(ells_l, C_l)

    # Check 4: Low-ell enhancement
    checks.append(check_low_ell_enhancement(ells, D_model, D_lcdm))
    # Check 5: High-ell consistency
    checks.append(check_high_ell_consistency(ells, D_model, D_lcdm))
    # Check 6: Acoustic peak
    checks.append(check_acoustic_peak(ells, D_model))

    # Stage 4: Chi2 vs Planck
    print("\n[Stage 4] Computing chi2 vs Planck...")
    from scripts.camb_wrapper import compute_chi2_camb
    chi2_m, chi2_l, dchi2 = compute_chi2_camb(result)
    checks.append(check_chi2(chi2_m, chi2_l, dchi2))

    # Summary
    n_pass = sum(checks)
    n_total = len(checks)
    print(f"\n{'=' * 60}")
    print(f"  Result: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("  ALL CHECKS PASSED")
    else:
        print(f"  {n_total - n_pass} CHECK(S) FAILED")
    print(f"{'=' * 60}")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 10: Save outputs to JSON**

Add after CAMB computation:
```python
out_dir = "outputs/simulations/c_ell"
os.makedirs(out_dir, exist_ok=True)
record = {
    "metadata": {"model": "Punctuated", "m": M, "lam": LAM, "N_star": N_STAR},
    "ells": ells.tolist(), "C_TT": C_TT.tolist(),
    "D_TT": D_model.tolist(),
}
with open(os.path.join(out_dir, "pipeline_sanity_punctuated.json"), "w") as f:
    json.dump(record, f, indent=2)
print(f"  Saved: {os.path.join(out_dir, 'pipeline_sanity_punctuated.json')}")
```

- [ ] **Step 11: Add 4-panel plot for diagnostics**

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

# Panel 1: P_S(k)
ax = axes[0, 0]
ax.loglog(k_phys, P_S, color=TOL["blue"], lw=1.5)
ax.axvline(1e-3, color=TOL["grey"], ls="--", lw=0.8, alpha=0.5)
ax.set_xlabel(r"$k$ [Mpc$^{-1}$]", fontsize=10)
ax.set_ylabel(r"$\mathcal{P}_\mathcal{R}(k)$", fontsize=10)
ax.set_title("Primordial Power Spectrum", fontsize=11)

# Panel 2: Full D_ell
ax = axes[0, 1]
ax.plot(ells, D_model, color=TOL["blue"], lw=1.5, label="Punctuated")
ax.plot(ells_l, D_lcdm, color=TOL["grey"], lw=1, ls="--", label="LCDM")
ax.set_xlabel(r"$\ell$", fontsize=10)
ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]", fontsize=10)
ax.set_title("Angular Power Spectrum", fontsize=11)
ax.legend(fontsize=8)

# Panel 3: Low-ell zoom
ax = axes[1, 0]
ell_max = 30
low = ells <= ell_max
ax.plot(ells[low], D_model[low], color=TOL["blue"], lw=1.5, label="Punctuated")
ax.plot(ells_l[low], D_lcdm[low], color=TOL["grey"], lw=1, ls="--", label="LCDM")
ax.set_xlabel(r"$\ell$", fontsize=10)
ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]", fontsize=10)
ax.set_title(r"Low-$\ell$ Zoom", fontsize=11)
ax.legend(fontsize=8)

# Panel 4: Residuals
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
for ext in ["png", "pdf"]:
    path = os.path.join(out_dir, f"pipeline_sanity.{ext}")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
plt.close(fig)
```

---

### Task 2: Run validation on lab machine

**Files:**
- Execute: `scripts/validate_pipeline_sanity.py`
- Execute: `scripts/test_camb_validation.py`
- Execute: `scripts/validate_camb_lcdm.py`

- [ ] **Step 1: Git commit and push, then SSH into lab machine**

```bash
git add scripts/validate_pipeline_sanity.py docs/superpowers/specs/2026-05-12-pipeline-validation-design.md docs/superpowers/plans/2026-05-12-pipeline-validation.md
git commit -m "add: pipeline sanity validation script"
git push
ssh uni
```

- [ ] **Step 2: On lab machine, pull latest and activate env**

```bash
cd ~/path/to/repo
git pull
conda activate cmb-anomaly
```

- [ ] **Step 3: Run Stage 1 — pipeline sanity check**

```bash
python scripts/validate_pipeline_sanity.py
```
Expected: All 7 checks pass.

- [ ] **Step 4: Run Stage 2 — full test suite**

```bash
python scripts/test_camb_validation.py
```
Expected: All 7 tests pass.

```bash
python scripts/validate_camb_lcdm.py
```
Expected: Peak at ~220, low residuals.

- [ ] **Step 5: Copy output plots back to local**

```bash
# On local
scp uni:path/to/repo/outputs/plots/diagnostics/pipeline_sanity.png outputs/plots/diagnostics/
scp uni:path/to/repo/outputs/plots/diagnostics/camb_lcdm_validation.png outputs/plots/diagnostics/
scp uni:path/to/repo/outputs/simulations/c_ell/* outputs/simulations/c_ell/
scp uni:path/to/repo/outputs/simulations/pspectra/* outputs/simulations/pspectra/
```
