"""
Ezquiaga CHI diagnostics: background scan + 3 diagnostic plots.
CLI: python -m scripts.ezquiaga_diagnostics

Pivot found by k = a*H matching the target scale, not hardcoded N.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.ezquiaga_chi import EzquiagaCHIModel, inflection_parameters
from inf_dyn_background import run_background_simulation, get_derived_quantities
from scripts.plotting import save_fig, TOL

X_C = 0.784
C_VAL = 0.77
K_PIVOT = 0.002  # Mpc^-1, CMB pivot


def scan_beta(betas, chi0=6.0, x_c=X_C, c_val=C_VAL, verbose=True):
    results = []
    for beta in betas:
        a, b = inflection_parameters(x_c, c_val, beta)
        model = EzquiagaCHIModel()
        model.a = a
        model.b = b
        model.v0 = model._V0 * model.a / (model.b * model.c)**2
        model.x0 = chi0
        model.patch_background_solver()

        T = np.linspace(0, model.T_max, model.bg_steps)
        bg = run_background_simulation(model, T)
        derived = get_derived_quantities(bg, model)

        epsH = derived["epsH"]
        N = derived["N"]
        eps_end = np.where(np.isfinite(epsH) & (epsH >= 1.0))[0]
        N_end = N[eps_end[0]] if len(eps_end) > 0 else N[-1]
        N_total = float(N_end)
        found = len(eps_end) > 0
        results.append((beta, a, b, N_total, float(N_end), found))
        if verbose:
            flag = " ***" if (55 <= N_total <= 80 and found) else ""
            print(f"  beta={beta:.6f}: N_total={N_total:.1f}, end={found}{flag}")
    return results


def make_model(beta, x_c=X_C, c_val=C_VAL):
    a, b = inflection_parameters(x_c, c_val, beta)
    model = EzquiagaCHIModel()
    model.a = a
    model.b = b
    model.v0 = model._V0 * model.a / (model.b * model.c)**2
    return model


def plot_N_chi(chi, N_std, end_idx, pivot_idx, chi0):
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    ax.plot(chi[:end_idx + 1], N_std[:end_idx + 1], "-", color=TOL["blue"], lw=1.3)
    ax.plot(chi[pivot_idx], N_std[pivot_idx], "o", color=TOL["green"], ms=4)
    ax.annotate("pivot", xy=(chi[pivot_idx], N_std[pivot_idx]),
                fontsize=8, color=TOL["green"])
    ax.set_xlabel("chi", fontsize=14)
    ax.set_ylabel("N", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_xlim(0, chi0 + 0.5)
    ax.set_ylim(-2, N_std[0] + 2)
    fig.tight_layout()
    save_fig(fig, "ezquiaga_Nchi", "diagnostics")


def plot_V_shape(model, end_idx, chi, pivot_idx):
    """Potential V/V0 vs x = phi/mu (Jordan frame, paper convention)."""
    # Map pivot and end chi to Jordan frame x
    x_end = model._x_of_chi(float(chi[end_idx]))
    x_pivot = model._x_of_chi(float(chi[pivot_idx]))
    x_max = model._x_of_chi(float(chi[0]))
    x_grid = np.linspace(0.05, 15, 3000)
    f_vals = np.array([model._V(float(x)) for x in x_grid])

    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    ax.plot(x_grid, f_vals, "-", color=TOL["blue"], lw=1.3)
    ax.axvline(x_end, color=TOL["grey"], ls="--", lw=0.8, alpha=0.5)
    ax.annotate("end", xy=(x_end, ax.get_ylim()[1]),
                xytext=(5, 5), textcoords="offset points", fontsize=8, color=TOL["grey"])
    ax.axvline(x_pivot, color=TOL["green"], ls="-.", lw=0.8, alpha=0.5)
    ax.annotate("pivot", xy=(x_pivot, ax.get_ylim()[1]),
                xytext=(5, 5), textcoords="offset points", fontsize=8, color=TOL["green"])
    ax.axvspan(0, x_max, alpha=0.08, color=TOL["blue"])
    ax.set_xlim(0, 12)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_xlabel(r"$x = \phi/\mu$", fontsize=14)
    ax.set_ylabel(r"$V/V_0$", fontsize=14)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    save_fig(fig, "ezquiaga_Vshape", "diagnostics")


def plot_PS_N(N_std, P_S, end_idx, pivot_idx, start_N):
    mask = N_std[:end_idx + 1] <= start_N
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    ax.semilogy(N_std[:end_idx + 1][mask], P_S[:end_idx + 1][mask], "-", color=TOL["blue"], lw=1.3)
    ax.axvline(N_std[pivot_idx], color=TOL["green"], ls="-.", lw=0.8, alpha=0.4)
    ax.annotate("pivot", xy=(N_std[pivot_idx], ax.get_ylim()[0]),
                xytext=(5, 10), textcoords="offset points", fontsize=8, color=TOL["green"])
    ax.set_xlabel("N", fontsize=14)
    ax.set_ylabel("P_S", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_xlim(0, max(N_std[0] + 2, 70))
    fig.tight_layout()
    save_fig(fig, "ezquiaga_PS_N", "diagnostics")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chi0", type=float, default=6.0, help="Initial field value")
    parser.add_argument("--k-pivot", type=float, default=K_PIVOT,
                        help="CMB pivot scale in Mpc^-1 (default 0.002)")
    args = parser.parse_args()
    chi0 = args.chi0
    k_pivot = args.k_pivot

    print("=== Ezquiaga CHI Diagnostics ===")
    print(f"  x_c = {X_C}, c = {C_VAL}, chi0 = {chi0}, k_pivot = {k_pivot} Mpc^-1")
    print()

    print("Beta scan:")
    betas = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2]
    results = scan_beta(betas, chi0=chi0)
    print()

    candidates = [r for r in results if r[3] >= 55 and r[3] <= 80 and r[5]]
    if not candidates:
        candidates = [r for r in results if r[5]]
    if not candidates:
        print("  ERROR: no viable background")
        sys.exit(1)
    candidates.sort(key=lambda r: abs(r[3] - 65))
    beta_best, a_best, b_best, N_total_best, _, _ = candidates[0]
    print(f"Best: beta={beta_best:.6f}, a={a_best:.3f}, b={b_best:.5f}, N_total={N_total_best:.1f}")
    print()

    model = make_model(beta_best)
    model.x0 = chi0
    model.y0 = -1e-4
    print(f"  a={model.a:.5f}, b={model.b:.5f}, chi0={chi0}")

    T = np.linspace(0, model.T_max, model.bg_steps)
    bg = run_background_simulation(model, T)
    derived = get_derived_quantities(bg, model)
    epsH = derived["epsH"]
    N = derived["N"]
    chi = bg[0]
    z_bg = bg[2]

    end_idx = np.where(np.isfinite(epsH) & (epsH >= 1.0))[0][0]
    N_end_val = float(N[end_idx])
    N_std = N_end_val - N[:end_idx + 1]

    # Find pivot from k = a*H = k_pivot
    S = 5e-5
    k_phys = np.exp(bg[3][:end_idx + 1]) * z_bg[:end_idx + 1] * S
    pivot_idx = int(np.argmin(np.abs(np.log10(k_phys) - np.log10(k_pivot))))
    pivot_chi = float(chi[pivot_idx])

    print(f"  chi0={chi[0]:.2f} -> chi_end={chi[end_idx]:.2f} (N_total={N_end_val:.1f})")
    print(f"  pivot at k={k_phys[pivot_idx]:.4e} Mpc^-1, N_std={N_std[pivot_idx]:.1f}, chi={pivot_chi:.2f}")
    print(f"  epsH(end)={float(epsH[end_idx]):.3f}")
    print()

    epsH_clip = np.clip(epsH[:end_idx + 1], 1e-30, None)
    P_S = (S * z_bg[:end_idx + 1])**2 / (8 * np.pi**2 * epsH_clip)

    settled = np.where(epsH[:end_idx + 1] > 1e-6)[0]
    start_idx = settled[0] if len(settled) > 0 else 0
    start_N = N_std[start_idx]

    # Compute n_s(k) from P_S(k) log-log slope
    ns_arr = np.full(end_idx + 1, np.nan)
    for idx in range(start_idx + 5, end_idx - 5):
        d = 5
        lk = np.log(k_phys[idx-d:idx+d+1])
        lp = np.log(P_S[idx-d:idx+d+1])
        ns_arr[idx] = 1 + (lp[-1] - lp[0]) / (lk[-1] - lk[0])

    ns_pivot_slope = float(ns_arr[pivot_idx]) if pivot_idx >= start_idx + 5 and pivot_idx < end_idx - 5 else float('nan')
    n_s_sr = 1 + 2*float(derived['etaH'][pivot_idx]) - 4*float(derived['epsH'][pivot_idx])

    print(f"  n_s at pivot (P_S log-log slope) = {ns_pivot_slope:.4f}")
    print(f"  n_s at pivot (SR formula)        = {n_s_sr:.4f}")
    print()

    print("Plotting...")
    plot_N_chi(chi, N_std, end_idx, pivot_idx, chi0)
    plot_V_shape(model, end_idx, chi, pivot_idx)
    plot_PS_N(N_std, P_S, end_idx, pivot_idx, start_N)
    print("Done.")


if __name__ == "__main__":
    main()
