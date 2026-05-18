"""
Reproduce Fig. 3: USR duration heatmap over initial-condition phase space.

Grid: x₀ ∈ [5.18, 5.93] (100 pts), |y₀| ∈ [0.01, 0.20] (100 pts).
Color = e-folds where ε₂ < -5.5.
Reference line at x = 5.42 (CMB pivot for N* = 60, TOL blue dashed).
Uses multiprocessing (12 workers) for the grid scan.
"""
import multiprocessing as mp
import time
import numpy as np
import matplotlib.pyplot as plt
from scripts.plotting import get_path, TOL

N_WORKERS = 12


def _compute_worker(args):
    """Worker function for multiprocessing (top-level for pickling)."""
    idx, phi_val, y_val, xi, lam, bg_steps = args
    from scripts.usr_utils import run_bg_for_usr, measure_usr_duration
    N_arr, epsH_arr, eps2_arr = run_bg_for_usr(
        phi_val, y_val, xi=xi, lam=lam, bg_steps=bg_steps
    )
    return idx, measure_usr_duration(N_arr, epsH_arr, eps2_arr, threshold=-5.5)


def _set_fonts():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.dpi": 300, "font.family": "serif",
    })


def main():
    xi = 15000.0
    lam = 0.13
    bg_steps = 30000  # matches notebook

    # Grid parameters (paper Fig. 3 range)
    phi_min, phi_max, N_phi = 5.18, 5.93, 200
    y_min_abs, y_max_abs, N_y = 0.01, 0.20, 200

    # CMB pivot x_cmb (N* = 60 SR trajectory, ~5.42)
    x_star = 5.42

    phi_grid = np.linspace(phi_min, phi_max, N_phi)
    y_grid = np.linspace(-y_min_abs, -y_max_abs, N_y)  # matches notebook: [-0.01, ..., -0.20]

    _set_fonts()

    n_total = N_phi * N_y
    print(f"  Grid: {N_phi}×{N_y} = {n_total} points")
    print(f"  Workers: {N_WORKERS}")
    print(f"  x₀ ∈ [{phi_min:.2f}, {phi_max:.2f}]")
    print(f"  |y₀| ∈ [{y_min_abs:.2f}, {y_max_abs:.2f}]")
    t0 = time.time()

    # Build task list: (idx, phi, y, xi, lam, bg_steps)
    tasks = [(i, phi_val, y_val, xi, lam, bg_steps)
             for i, (phi_val, y_val) in enumerate(
                 (p, y) for y in y_grid for p in phi_grid
             )]

    grid_dur = np.zeros((N_y, N_phi))
    done = 0

    with mp.Pool(N_WORKERS) as pool:
        for idx, val in pool.imap_unordered(_compute_worker, tasks):
            j = idx % N_phi  # phi column (inner loop)
            i = idx // N_phi # y row (outer loop)
            grid_dur[i, j] = val
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{n_total} ({100*done/n_total:.0f}%)")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({elapsed / n_total:.2f}s/point)")

    # -- Plot --
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    # Cell edges: N+1 vertices for N rectilinear cells (prevents pcolormesh artifacts)
    phi_edges = np.linspace(phi_min, phi_max, N_phi + 1)
    y_edges = np.linspace(y_min_abs, y_max_abs, N_y + 1)
    X_edges, Y_edges = np.meshgrid(phi_edges, y_edges)

    mesh = ax.pcolormesh(X_edges, Y_edges, grid_dur, shading="flat", cmap="magma")
    cbar = fig.colorbar(mesh, ax=ax, label="USR Duration [N]")
    cbar.ax.tick_params(labelsize=7)

    ax.axvline(x=x_star, color=TOL["blue"], linestyle="--", linewidth=1.0,
               label=rf"$x_* = {x_star}$")
    ax.legend(loc="upper right", framealpha=0.8)

    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$|y_0|$")

    out = get_path("paper", "paper_fig3_usr_heatmap.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
