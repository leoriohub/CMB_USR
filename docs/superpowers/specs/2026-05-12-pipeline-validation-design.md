# Pipeline Validation Design

## Goal
Reproduce known CMB results using the punctuated inflation model to verify the full pipeline (solver → P_S(k) → CAMB C_ell → Planck) is working correctly.

## Approach: Two-stage validation

### Stage 1: Focused Sanity Check
New script `scripts/validate_pipeline_sanity.py` that:
1. Runs punctuated model (m=1.1323e-7, λ=3.3299e-15, φ₀=12.0, y₀=0.0, N_star=77.2) through P_S(k) pipeline
2. Feeds P_S(k) into CAMB via `compute_cl_full_camb()`
3. Computes LCDM baseline via `compute_cl_camb_powerlaw()`
4. Compares against Planck data
5. Prints pass/fail per check

### Stage 2: Full Test Suite
Run existing `scripts/test_camb_validation.py` + `scripts/validate_camb_lcdm.py` using generated P_S(k) files.

## Validation Checks

| # | Check | Method | Tolerance |
|---|-------|--------|-----------|
| 1 | N_total ≥ N_star | Background solver | N_total > 77.2 |
| 2 | P_S(k) peak at k≈10⁻³ | MS solver | peak within 1 decade of 10⁻³ |
| 3 | P_S(k) at pivot = A_s | Normalization | exact (2.1e-9) |
| 4 | Low-ℓ D_ell (ℓ=2-10) > LCDM | CAMB | direction only (enhancement) |
| 5 | High-ℓ D_ell (ℓ=200-2500) ≈ LCDM | CAMB | < 5% deviation |
| 6 | First acoustic peak at ℓ≈220 | CAMB | ±10 |
| 7 | CAMB LCDM χ² (low-ℓ) in [15, 30] | chi2 | sanity range |

## Outputs
- `outputs/simulations/pspectra/PS_Punctuated_*.json` — P_S(k)
- `outputs/simulations/c_ell/C_ell_camb_Punctuated_*.json` — C_ell
- `outputs/simulations/c_ell/camb_lcdm_validation.json` — LCDM C_ell
- Console pass/fail report

## Execution
On lab machine via `ssh uni`:
```bash
conda activate cmb-anomaly
python scripts/validate_pipeline_sanity.py      # Stage 1
python scripts/test_camb_validation.py           # Stage 2
python scripts/validate_camb_lcdm.py             # Stage 2
```
