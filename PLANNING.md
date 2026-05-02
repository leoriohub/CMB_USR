# Planning: CMB Low-ℓ Anomaly from Non-SR Initial Conditions in Higgs Inflation

## Overview

**Hypothesis:** Non-slow-roll (non-SR) initial conditions in Higgs inflation can produce a transient Ultra-Slow-Roll (USR) phase at early times (large scales), suppressing the primordial power spectrum $P_S(k)$ at low $k$ and thereby explaining the observed lack of CMB power at low multipoles ($\ell \lesssim 30$).

**The gap:** The codebase can compute exact $P_S(k)$ from Mukhanov-Sasaki integration, but no pipeline exists to convert this into a CMB angular power spectrum $C_\ell$ or to perform a quantitative comparison with Planck/ACT data. This document outlines a roadmap to bridge that gap.

## How It All Fits Together

```
Model parameters (φ₀, yᵢ, ξ, λ)
    │
    ▼
Background evolution (run_background_simulation)
    │  ε_H(N), H(N)
    ▼
Mukhanov-Sasaki solver (run_ms_simulation, ~80 k-modes)
    │
    ▼
Primordial power spectrum P_S(k)    ← You are here
    │
    ├──► CAMB (Boltzmann code): P_S(k) → C_ℓ^TT/EE/TE
    │     C_ℓ = ∫(dk/k) P_S(k) · |Θ_ℓ(k)|²   ← radiation transfer functions
    │
    └──► Sachs-Wolfe approx (ℓ<30): C_ℓ ≈ (2π²/9) ∫(dk/k) P_S(k) · j_ℓ²(k r_ls)
    │
    ▼
    C_ℓ^TT   vs.   Planck/ACT data
    │
    ▼
    χ²(θ) = Σ_ℓ (C_ℓ^model(θ) - C_ℓ^data)² / σ_ℓ²
    │
    ├──► Grid scan: Δχ²(θ) across (φ₀, yᵢ) parameter space
    └──► Bayesian: p(θ|data) ∝ exp(-½χ²) · π(θ)  →  posterior over ICs

---

## Phase 1: Characterise the P_S(k) Suppression from Non-SR ICs

**Goal:** Systematically map the parameter space $(\phi_0, y_i, \xi)$ to find where USR-driven suppression occurs at the right scale and amplitude.

### Step 1.1 — Background survey

- **Tool:** `run_background_simulation` + `get_derived_quantities`
- **Model:** `HiggsModel` (approximated) and `FullHiggsModel` (exact)
- **Parameter space:** $\phi_0 \in [5.0, 7.0]$, $y_i \in [-1.0, 0.0]$, $\xi \in [1000, 20000]$
- **Deliverable:** Heatmap of USR duration $(\Delta N_{\text{USR}})$ vs. $(\phi_0, y_i)$ for each $\xi$, identifying "golden trajectories" where the USR dip ($\epsilon_2 \approx -6$) occurs at $N_{\text{back}} \approx 60\text{–}70$ e-folds before end of inflation (i.e., at the largest observable scales).

### Step 1.2 — Full MS power spectrum scan

- **Tool:** `run_ms_simulation` (exact finite-time Bunch-Davies ICs; use `inf_dyn_MS_full.py`, not the old version)
- **Procedure:** For each golden trajectory from Step 1.1, compute $P_S(k)$ for $k \in [10^{-5}, 1] \, \text{Mpc}^{-1}$ ($\sim 80\text{–}100$ log-spaced modes).
- **Normalization:** Anchor to Planck $A_s = 2.1 \times 10^{-9}$ at pivot $k_* = 0.05 \, \text{Mpc}^{-1}$.
- **Deliverable:** $P_S(k)$ curves for each golden trajectory, with the suppression dip quantified by:
  - Dip amplitude: $A_{\text{dip}} = P_S(k_{\text{dip}}) / P_S^{\text{(SR)}}(k_{\text{dip}})$
  - Dip location: $k_{\text{dip}}$ (and corresponding $\ell_{\text{dip}} \approx k_{\text{dip}} \times 14000$)
  - Dip width: $\Delta \log k$ at half-suppression

### Step 1.3 — Cross-check: `HiggsModel` vs `FullHiggsModel`

- **Tool:** `FullHiggsModel`
- **Purpose:** Validate that the suppression features from the approximated Higgs model are genuine physics (not artifacts of the high-field approximation), by comparing with the exact conformal inversion model.

---

## Phase 2: Compute CMB Angular Power Spectrum $C_\ell$

**Goal:** Convert the primordial $P_S(k)$ into the CMB temperature $C_\ell^{TT}$ that can be compared directly with Planck data.

### Option A: CAMB (Preferred — accurate)

- **Tool:** [CAMB](https://camb.readthedocs.io/) (Code for Anisotropies in the Microwave Background)
- **Workflow:**
  1. Install CAMB via `pip install camb`
  2. Set base $\Lambda$CDM cosmological parameters from Planck 2018 best-fit (from `data/PLANCK_contours.csv` or Planck papers):
     - $\Omega_b h^2 = 0.0224$, $\Omega_c h^2 = 0.120$, $H_0 = 67.3$, $\tau = 0.054$, $n_s = 0.965$, $A_s = 2.1 \times 10^{-9}$
  3. Define a custom primordial power spectrum by interpolating the computed $P_S(k)$ from Phase 1 onto a fine $k$ grid CAMB expects (via `camb.set_params(pk=...)`).
  4. CAMB will convolve $P_S(k)$ with the radiation transfer functions to produce $C_\ell^{TT}$, $C_\ell^{EE}$, $C_\ell^{TE}$.

### Option B: Sachs-Wolfe integral (ℓ < 30, simplest)

For $\ell \lesssim 30$, the dominant contribution is the Sachs-Wolfe effect:

$$C_\ell^{TT} \approx \frac{2\pi^2}{9} \int \frac{dk}{k} \, \mathcal{P}_S(k) \, j_\ell^2(k r_{ls})$$

where $r_{ls} \approx 14000 \, \text{Mpc}$ and $j_\ell$ is a spherical Bessel function. This is ~10 lines with `scipy.special.spherical_jn` and is essentially free computationally.

- **Good enough for:** scoping whether the suppression shape maps to the low-$\ell$ deficit
- **Not good for:** fitting high-$\ell$ (acoustic peaks, Silk damping), polarization, or the reionization bump
- **Recommendation:** Start here to validate the idea, then upgrade to CAMB only if the SW result is promising.

### Option C: CLASS (Alternative to CAMB)

- [CLASS](https://github.com/lesgourg/class_public) is a viable open-source alternative if CAMB proves difficult to interface.

---

## Phase 3: Compare with CMB Data

**Goal:** Quantitatively assess whether the USR-suppressed spectrum fits Planck/ACT better than the power-law $\Lambda$CDM null hypothesis.

### Step 3.1 — Obtain low-ℓ CMB data

- **Planck 2018:** Download low-$\ell$ likelihood data from the [Planck Legacy Archive](https://pla.esac.esa.int/). The low-$\ell$ TT likelihood uses $\ell \in [2, 29]$.
- **ACT DR4/DR6:** Download ACT power spectrum data for cross-check.
- **Clipping:** The anomaly is most significant for $\ell \lesssim 30$, so focus analysis there.
- **Data files:** Already in `data/Planck/` (currently empty) and `data/ACT/` (empty); these directories should be populated with the relevant likelihood data.

### Step 3.2 — $\chi^2$ analysis

- Compute $\chi^2 = \sum_{\ell} \frac{(C_\ell^{\text{model}} - C_\ell^{\text{data}})^2}{\sigma_\ell^2}$ separately for:
  - $\ell \in [2, 29]$ (low-$\ell$, where anomaly lives)
  - $\ell \in [30, 2500]$ (high-$\ell$, where $\Lambda$CDM fits well)
  - Full range $[2, 2500]$
- Compare with the baseline $\Lambda$CDM $\chi^2$ (using the power-law $P_S(k) = A_s (k/k_*)^{n_s-1}$).
- **Key metric:** $\Delta \chi^2 = \chi^2_{\text{USR}} - \chi^2_{\Lambda\text{CDM}}$. A negative $\Delta \chi^2$ for the low-$\ell$ bin (with minimal penalty at high-$\ell$) supports the hypothesis.

### Step 3.3 — Bayesian evidence (optional, advanced)

- Compute the Bayesian evidence ratio (Bayes factor) between the USR model and the power-law $\Lambda$CDM model using nested sampling (e.g., `dynesty` or `multinest`) with the $(\phi_0, y_i, \xi)$ parameters of the Higgs model as free priors.

---

## Phase 4: Sensitivity and Robustness Tests

**Goal:** Ensure the result is robust to modeling choices.

### Step 4.1 — Dependence on non-minimal coupling $\xi$

- Repeat the full pipeline for $\xi \in \{1000, 5000, 10000, 15000, 17000, 20000\}$.
- Check: Is the suppression universally present across $\xi$, or only for certain values?
- **Expected:** Larger $\xi$ flattens the plateau more, potentially giving longer USR durations and stronger suppression.

### Step 4.2 — Dependence on the equation of state after inflation

- The matching of $k$ to $\ell$ depends on the post-inflationary thermal history (through $r_{ls}$, the distance to last scattering).
- Vary $N_{\text{pivot}}$ (currently 60) from 50 to 70 to test sensitivity of the suppression position to the assumed thermal history.

### Step 4.3 — Higgs self-coupling $\lambda$

- Test $\lambda \in \{0.01, 0.05, 0.1, 0.13, 0.5\}$ at fixed $\xi$.
- The plateau height scales as $V_0 \propto \lambda / \xi^2$, so changing $\lambda$ primarily rescales the overall normalization (absorbed by $A_s$) but could affect the shape of the transition.

### Step 4.4 — Comparison with exact vs. approximated Higgs model

- Compare `HiggsModel` (high-field approximation) vs. `FullHiggsModel` (exact numerical conformal inversion).
- If both give the same suppression features, the approximated model is sufficient; otherwise, the exact model should be used.

### Step 4.5 — Tensor consistency check

- Compute $r$ (tensor-to-scalar ratio) for each trajectory using the same MS pipeline.
- Verify that $r$ remains within Planck/BICEP/Keck bounds ($r < 0.036$ at 95% CL). Elevated $r$ could rule out a trajectory regardless of the low-$\ell$ fit.

---

## Phase 5: Interpretation and Publication

**Goal:** Produce a compelling physics story.

### Deliverables checklist

- [ ] **Notebook 1:** `Background_Survey.ipynb` — Phase 1 heatmaps and golden trajectory identification
- [ ] **Notebook 2:** `Full_Power_Spectrum.ipynb` — Full $P_S(k)$ for golden trajectories
- [ ] **Notebook 3:** `CMB_Power_Spectrum.ipynb` — CAMB/CLASS integration, $C_\ell$ computation
- [ ] **Notebook 4:** `Data_Comparison.ipynb` — $\chi^2$ analysis, plots of $C_\ell$ vs. Planck data
- [ ] **Notebook 5:** `Robustness.ipynb` — Sensitivity tests across $\xi$, $\lambda$, $N_{\text{pivot}}$
- [ ] **Paper figure mockups:**
  - Fig 1: Higgs potential $V(\phi)$ showing plateau where USR occurs
  - Fig 2: USR duration heatmap + golden trajectory markers
  - Fig 3: $P_S(k)$ showing suppression at low $k$ for the best-fit ICs
  - Fig 4: $C_\ell^{TT}$ from USR model vs. Planck data, highlighting $\Delta \chi^2$ improvement
  - Fig 5: Parameter space constraints in $(\phi_0, y_i)$ with Planck $n_s$-$r$ contours overlaid

### Computational cost (actual benchmarks)

All benchmarks done on a standard laptop (WSL, single core, no parallelization):

| Step | Time |
|------|------|
| Background solve | **0.008s** |
| Single MS mode (median) | **0.007s** |
| Single MS mode (mean) | **0.036s** |
| Full $P_S(k)$ (80 modes) | **~3s** |
| CAMB ($\ell_{\max}=2500$) | **0.37s** |
| CAMB ($\ell_{\max}=30$, low-ℓ only) | **0.12s** |
| SW integral (scipy, ℓ<30) | **<0.01s** |

**Projected total costs:**

| Scenario | MS solver | CAMB | Total wall time |
|----------|-----------|------|----------------|
| Grid scan (50 φ₀ × 10 yᵢ = 500 traj.) | ~25 min | ~1 min | **~26 min** |
| SW-only (500 traj., ℓ<30 scoping) | ~25 min | 0 | **~25 min** |
| Scan across 5 ξ values (2500 traj.) | ~2 hrs | ~5 min | **~2 hrs** |
| MCMC (10k evals, full Cℓ) | ~8 hrs | ~1 hr | **~9 hrs** |

**Conclusion:** This is eminently feasible. A full grid scan with CAMB finishes in under 30 minutes. Even a full MCMC analysis can run overnight. The MS solver dominates the cost, not the Boltzmann code.

### Potential pitfalls

1. ~~**Computational cost:**~~ Resolved — see benchmarks above. Each trajectory costs ~3s.
2. **CAMB interface:** CAMB expects either a parameterized primordial spectrum or a tabulated one on a specific $k$ grid. The custom primordial spectrum interface in CAMB (`camb.set_params(Pk=...)`) can be non-trivial. **Mitigation:** Test with a few mock spectra first; use the `lensing` module if needed. For ℓ<30, the SW analytic integral avoids this entirely.
3. **Low-ℓ likelihood:** The Planck low-ℓ likelihood is a highly non-Gaussian, simulation-based likelihood (the "Commander" or "SimAll" likelihoods), not a simple $\chi^2$ on the binned $C_\ell$'s. A Gaussian approximation on binned $C_\ell$ may suffice for initial scoping but not for a definitive claim. **Mitigation:** For rigorous results, use the official Planck likelihood code (e.g., `clik`).
4. **Anomaly statistical significance:** The low-$\ell$ anomaly is only $\sim 2\text{–}3\sigma$ in Planck. Even a perfect fit from USR may not be statistically preferred over $\Lambda$CDM with a Bayesian penalty for extra parameters. Be upfront about this.

---

## Quickstart: What to Do First

```bash
# 1. Install CAMB
pip install camb

# 2. Test CAMB integration with a power-law spectrum
python -c "import camb; print(camb.__version__)"

# 3. Reproduce the existing CMB_Anomaly notebook results
# (open notebooks/CMB_Anomaly.ipynb, run all cells)

# 4. Modify the CMB_Anomaly notebook to save P_S(k) as a numpy array,
# then feed it into CAMB to get C_ell's. This is the simplest test.

# 5. If CAMB works, generalize to the grid scan pipeline.
```

---

## Code Architecture for the Pipeline

```
notebooks/
├── Phase1_Background_Survey.ipynb     # heatmaps, golden trajectories
├── Phase2_Full_PSpectrum.ipynb         # full P_S(k) for trajectories
├── Phase3_CMB_Power.ipynb             # CAMB integration, C_ell
├── Phase4_Data_Comparison.ipynb        # χ² vs Planck data
├── Phase5_Robustness.ipynb             # sensitivity tests
└── CMB_Anomaly.ipynb                  # existing — reference only

scripts/
├── pspectrum_pipeline.py              # batch MS solves across k-modes
├── camb_wrapper.py                    # wraps CAMB for custom P(k)
└── chi2_analysis.py                   # computes Δχ² / AIC / BIC

outputs/
└── cmb_results/
    ├── golden_trajectories.json
    ├── pspectra/
    ├── c_ell/
    └── chi2_summary.csv
```
