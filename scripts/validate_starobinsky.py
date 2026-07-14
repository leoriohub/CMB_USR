"""Validate Starobinsky model background and P_S(k)."""
import numpy as np
import sys
sys.path.insert(0, ".")

from models.starobinsky import StarobinskyModel
from inf_dyn_background import run_background_simulation, get_derived_quantities

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
    print("  epsH never reaches 1.0 within T_max")
