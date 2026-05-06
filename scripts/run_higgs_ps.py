import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import HiggsModel
from scripts.pspectrum_pipeline import run_pspectrum_pipeline

model = HiggsModel(lam=0.13, xi=15000.0)
model.phi0 = 5.70
model.y0 = -0.12

k_min = 1e-5
k_max = 0.5
num_k = 80

# Get N_total and compute N_star from model defaults
import inf_dyn_background as bg_solver
from scripts.pspectrum_pipeline import find_end_of_inflation, get_k_pivot_code
T_span = np.linspace(0, model.T_max, model.bg_steps)
sol = bg_solver.run_background_simulation(model, T_span)
d = bg_solver.get_derived_quantities(sol, model)
end_idx = find_end_of_inflation(d['epsH'])
eps_inf = d['epsH'][d['N'] < d['N'][end_idx]]
dip_N = d['N'][np.argmin(eps_inf[10:]) + 10]
N_total = d['N'][end_idx]
N_after_dip = N_total - dip_N
N_star_target = N_after_dip - 4.8
print(f"N_total={N_total:.1f}, dip_N={dip_N:.1f}, N_after_dip={N_after_dip:.1f}, N_star={N_star_target:.1f}")
print("Running P_S(k) pipeline (auto T_span from model)...")
ps_data = run_pspectrum_pipeline(
    model, 
    k_min=k_min, 
    k_max=k_max, 
    num_k=num_k,
    N_star=N_star_target
)

if ps_data['status'] == 'error':
    print("Error:", ps_data['message'])
    sys.exit(1)

k_phys = ps_data['k_phys']
P_S = ps_data['P_S']

plt.figure(figsize=(8, 6))
plt.semilogx(k_phys, P_S * 1e9, 'b-', linewidth=2, label=f'Higgs (N_*={N_star_target:.1f}, y0={model.y0})')
plt.xlim(1e-5, 0.5)
plt.ylim(0, 5)
plt.xlabel('k [Mpc$^{-1}$]')
plt.ylabel('P(k) x $10^{-9}$')
plt.title('Primordial Power Spectrum - Higgs Inflation')
plt.xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1], ['1e-5','1e-4','1e-3','1e-2','0.1'])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
os.makedirs('outputs/plots', exist_ok=True)
plt.savefig('outputs/plots/higgs_PS_auto.png', dpi=150)
print("Saved outputs/plots/higgs_PS_auto.png")
