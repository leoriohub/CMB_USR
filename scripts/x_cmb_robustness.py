"""
Reproduce Table I: robustness of x_cmb against xi variations.

For each xi, lambda is rescaled to maintain tree-level CMB normalization:
    lambda = 0.13 * (xi / 15000)^2

Tests both approximate (HiggsModel) and full (FullHiggsModel) potentials,
at N_star = 55 and 60.

Usage:
    conda activate cmb-anomaly
    python scripts/x_cmb_robustness.py
"""
import numpy as np

from models import HiggsModel, FullHiggsModel
import inf_dyn_background as bg_solver
from pspectrum_pipeline import find_end_of_inflation


xi_values = [10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 15000.0]
lam_bench = 0.13
xi_bench = 15000.0
N_star_values = [55, 60]

T_MAX = 2000.0
BG_STEPS = 100000
T_span = np.linspace(0, T_MAX, BG_STEPS)


def compute_xcmb(ModelClass, lam, xi, N_star, x0, y0):
    try:
        model = ModelClass(lam=lam, xi=xi)
    except Exception:
        return None

    model.x0 = x0
    model.y0 = y0
    model.T_max = T_MAX
    model.bg_steps = BG_STEPS

    try:
        bg = bg_solver.run_background_simulation(model, T_span)
    except Exception:
        return None

    derived = bg_solver.get_derived_quantities(bg, model)

    end_idx = find_end_of_inflation(derived["epsH"])
    if end_idx < 1:
        return None

    N_total = float(derived["N"][end_idx])
    if N_total < N_star:
        return None

    N_pivot = N_total - N_star
    pivot_idx = int(np.argmin(np.abs(derived["N"][:end_idx] - N_pivot)))
    return float(bg[0][pivot_idx])


def format_row(xi, lam_val, vals):
    parts = [f"{xi:>10.1f}", f"{lam_val:.3e}"]
    for v in vals:
        if v is None:
            parts.append(f"{'N/A':>10}")
        else:
            parts.append(f"{v:>10.5f}")
    return "  ".join(parts)


def main():
    y0_sr = -1e-6
    x0 = 5.7

    results = {"table_55_60": []}

    # -- Tabla principal: ambos modelos, N*=55 y N*=60 --
    print("=" * 70)
    print(f"  {'xi':>8}  {'lambda':>12}  {'appx(N*=55)':>12}  {'full(N*=55)':>12}  {'appx(N*=60)':>12}  {'full(N*=60)':>12}")
    print("-" * 70)

    for xi in xi_values:
        lam_val = lam_bench * (xi / xi_bench) ** 2

        row = {"xi": xi, "lambda": lam_val, "N_star_55": {}, "N_star_60": {}}
        vals = []

        for ns in N_star_values:
            for model_name, ModelClass in [("approx", HiggsModel), ("full", FullHiggsModel)]:
                xc = compute_xcmb(ModelClass, lam_val, xi, ns, x0, y0_sr)
                vals.append(xc)
                if ns == 55:
                    row["N_star_55"][model_name] = xc
                else:
                    row["N_star_60"][model_name] = xc

        if xi == xi_values[0]:
            print("-" * 70)

        print(f"  {xi:>8.1f}  {lam_val:>12.3e}  {vals[0]:>12.5f}  {vals[1]:>12.5f}  {vals[2]:>12.5f}  {vals[3]:>12.5f}")
        results["table_55_60"].append(row)

    print("-" * 70)

    # -- Escaneo N_star para el modelo approx, xi=15000 --
    print()
    print("  N_star scan (approx, xi=15000):")
    print(f"  {'N*':>4}  {'x_cmb':>10}  {'N*':>4}  {'x_cmb':>10}")
    print("  " + "-" * 25)
    for ns1, ns2 in zip(range(40, 58), range(58, 76)):
        xc1 = compute_xcmb(HiggsModel, lam_bench, xi_bench, ns1, x0, y0_sr)
        xc2 = compute_xcmb(HiggsModel, lam_bench, xi_bench, ns2, x0, y0_sr)
        print(f"  {ns1:>4d}  {xc1:>10.5f}  {ns2:>4d}  {xc2:>10.5f}" if xc1 else f"  {ns1:>4d}  {'N/A':>10}")

    # -- Papel: x_cmb = 5.42354 para xi=15000 -> encontrar N_star --
    print()
    print("  Searching N_star for paper's x_cmb=5.42354...")
    for ns in range(60, 70):
        xc = compute_xcmb(HiggsModel, lam_bench, xi_bench, ns, x0, y0_sr)
        if xc:
            print(f"    N*={ns:>2d}:  x_cmb={xc:.5f}  (paper: 5.42354, diff={xc-5.42354:+.5f})")

    # -- N_total por modelo --
    print()
    print("  N_total summary (approx, xi=15000):")
    m = HiggsModel(lam=lam_bench, xi=xi_bench)
    m.x0 = x0; m.y0 = y0_sr; m.T_max = T_MAX; m.bg_steps = BG_STEPS
    bg = bg_solver.run_background_simulation(m, T_span)
    d = bg_solver.get_derived_quantities(bg, m)
    ei = find_end_of_inflation(d["epsH"])
    if ei > 0:
        print(f"    N_total = {d['N'][ei]:.2f}")
        for ns in [55, 60, 63]:
            npi = d['N'][ei] - ns
            pi = int(np.argmin(np.abs(d['N'][:ei] - npi)))
            print(f"    N*={ns:>2d}:  N_pivot={npi:.2f}  x_cmb={bg[0][pi]:.5f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
