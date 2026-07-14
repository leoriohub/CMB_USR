"""Validate Starobinsky model background and P_S(k)."""
import numpy as np
import sys
sys.path.insert(0, ".")

from models.starobinsky import StarobinskyModel
from inf_dyn_background import run_background_simulation, get_derived_quantities
from pspectrum_pipeline import run_pspectrum_pipeline
from scripts.observables import extract_ns

model = StarobinskyModel(v0=4.5e-11)
model.x0 = 5.5
model.y0 = -1e-4
model.T_max = 3000.0  # Starobinsky plateau is long — need more code time

T = np.linspace(0, model.T_max, model.bg_steps)
sol = run_background_simulation(model, T)
derived = get_derived_quantities(sol, model)

eps1 = derived["epsH"]
N_arr = derived["N"]

final_eps = float(eps1[-1])
final_N = float(N_arr[-1])
print(f"N_total = {final_N:.1f}")
print(f"phi0={model.x0}  y0={model.y0}")
print(f"  T_end = {T[-1]:.1f}")
print(f"  Final epsH = {final_eps:.4f}")
print(f"  Max epsH = {float(np.nanmax(eps1)):.4f}")
print(f"  Min epsH = {float(np.nanmin(eps1[eps1 > 0])):.4g}")

# Find where epsH crosses 1, if ever
cross = np.where(np.isfinite(eps1) & (eps1 >= 1.0))[0]
if len(cross) > 0:
    end = int(cross[0])
    N_total = float(N_arr[end])
    print(f"  epsH>=1 at N = {N_total:.1f}, T = {T[end]:.1f}")
else:
    N_total = float(N_arr[-1])
    print("  epsH never reaches 1.0 within T_max")

k_pivot = 0.002
k_min, k_max = 1e-5, 1.0
n_k = 80
k_grid = np.logspace(np.log10(k_min), np.log10(k_max), n_k)

result = run_pspectrum_pipeline(
    model,
    T_span_bg=T,
    k_phys_grid=k_grid,
    k_pivot_phys=k_pivot,
    N_star=55,
    normalize_to_As=True,
    backend="fortran",
    ms_method="dp5",
    save_outputs=False,
)

if result["status"] != "success":
    print(f"  Pipeline FAILED: {result.get('message', '')}")
    sys.exit(1)

k_phys = result["k_phys"]
P_S = result["P_S"]

n_s, ns_meta = extract_ns(k_phys, P_S, k_pivot=k_pivot, ns_window=4.0, method="lsq")
print(f"n_s(k={k_pivot}) = {n_s:.4f}")

N_pivot_val = N_total - 55
pivot_idx = int(np.argmin(np.abs(N_arr - N_pivot_val)))
r_sr = 16.0 * eps1[pivot_idx]
print(f"r (SR at N_pivot) = {r_sr:.4f}")
print(f"n_s = {n_s:.4f}, r = {r_sr:.4f}")

from scripts.camb_wrapper import compute_cl_camb_powerlaw, compute_cl_full_camb, compute_chi2_camb, C_ell_to_d_ell

print("\n── CAMB C_ell ──")
print("Computing LCDM baseline...")
ells_pl, C_ell_pl, _, _ = compute_cl_camb_powerlaw()
D_ell_pl = C_ell_to_d_ell(ells_pl, C_ell_pl)
print(f"  LCDM D₂ = {D_ell_pl[0]:.0f} μK²")

print("Computing Starobinsky C_ell via CAMB...")
ps_data = {
    "k_phys": result["k_phys"],
    "P_S": result["P_S"],
    "k_pivot": k_pivot,
}
ells_star, C_ell_star, _, _ = compute_cl_full_camb(ps_data)
D_ell_star = C_ell_to_d_ell(ells_star, C_ell_star)
print(f"  Starobinsky D₂ = {D_ell_star[0]:.0f} μK²")

chi2_m, chi2_l, chi2_diff = compute_chi2_camb(ps_data)
print(f"\nχ² (low-ℓ, ℓ=2-29):")
print(f"  Starobinsky = {chi2_m:.1f}")
print(f"  LCDM        = {chi2_l:.1f}")
print(f"  Δχ²         = {chi2_diff:.1f}")
