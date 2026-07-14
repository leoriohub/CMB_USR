"""2D USR terrain heatmaps from trained PSEmulator.

CLI for generating pcolormesh heatmaps of emulator parameter space.

Usage:
    python scripts/plot_terrain.py --x-param beta --y-param c --color k_peak
    python scripts/plot_terrain.py --x-param xc --y-param n_star --color n_s --density 50
"""
import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, BoundaryNorm

from scripts.emulator.model import PSEmulator
from scripts.emulator.postprocess import extract_usr_info
from scripts.plotting import PAPER_RCPARAMS, TOL, save_fig

PARAMS = {
    "x_c": {"bounds": (0.75, 0.85), "default": 0.784, "label": r"$x_c$"},
    "c": {"bounds": (0.5, 5.0), "default": 0.77, "label": r"$c$"},
    "beta": {"bounds": (1e-6, 9e-4), "default": 1e-5, "label": r"$\beta$"},
    "chi0": {"bounds": (5.0, 9.0), "default": 8.0, "label": r"$\chi_0$"},
    "n_star": {"bounds": (50, 75), "default": 65, "label": r"$N_*$"},
}

COLOR_INFO = {
    "k_peak": {"mode": "continuous", "label": r"$k_{\mathrm{peak}}$ [Mpc$^{-1}$]"},
    "n_s": {"mode": "continuous", "label": r"$n_s$"},
    "f_total": {"mode": "continuous", "label": r"$f_{\mathrm{total}}$"},
    "mass_bin": {"mode": "categorical", "label": "Mass bin"},
    "usr_type": {"mode": "categorical", "label": "USR type"},
}

MASS_BIN_MAP = {
    "unknown": -1, "too_light": 0, "asteroid_gap": 1, "intermediate": 2,
    "sub_solar_gap": 3, "sub_stellar": 4, "stellar_ligo_ruled_out": 5, "massive": 6,
}

USR_TYPE_MAP = {"weak": 0, "transitional": 1, "strong": 2}


def _to_int(v, mapping):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v)
    return mapping.get(s, np.nan)


def make_terrain_map(emulator, x_param, y_param, color="k_peak", density=100,
                     fix_params=None):
    """Generate 2D terrain data for a given color metric.

    Returns (X, Y, Z) where X, Y are 2D meshgrid, Z is the metric array.
    """
    if fix_params is None:
        fix_params = {}

    x_bounds = PARAMS[x_param]["bounds"]
    y_bounds = PARAMS[y_param]["bounds"]
    x_vals = np.linspace(x_bounds[0], x_bounds[1], density)
    y_vals = np.linspace(y_bounds[0], y_bounds[1], density)
    X, Y = np.meshgrid(x_vals, y_vals)

    param_list = []
    for j in range(density):
        for i in range(density):
            params = {}
            for p in PARAMS:
                if p == x_param:
                    params[p] = x_vals[i]
                elif p == y_param:
                    params[p] = y_vals[j]
                else:
                    params[p] = fix_params.get(p, PARAMS[p]["default"])
            param_list.append(params)

    k_grid, ps_array = emulator.predict_batch(param_list)

    is_cat = COLOR_INFO[color]["mode"] == "categorical"
    Z = np.full((density, density), np.nan, dtype=float)

    for j in range(density):
        for i in range(density):
            idx = j * density + i
            try:
                ps = ps_array[idx]
                info = extract_usr_info(k_grid, ps)
                val = info.get(color)
            except Exception:
                val = None

            if val is None or (isinstance(val, float) and np.isnan(val)):
                Z[j, i] = np.nan
            elif is_cat:
                if color == "mass_bin":
                    Z[j, i] = _to_int(val, MASS_BIN_MAP)
                elif color == "usr_type":
                    Z[j, i] = _to_int(val, USR_TYPE_MAP)
                else:
                    Z[j, i] = np.nan
            else:
                Z[j, i] = float(val)

    return X, Y, Z


def _configure_colormap(color, Z):
    valid = Z[np.isfinite(Z)]

    if color == "k_peak":
        vmin = np.nanmin(Z)
        vmax = np.nanmax(Z)
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            vmin, vmax = 1, 1e18
        if vmin <= 0:
            vmin = max(np.nanmin(Z[Z > 0]), 1.0) if np.any(Z > 0) else 1.0
        if vmin >= vmax:
            vmin, vmax = 1.0, 1e18
        return LogNorm(vmin=vmin, vmax=vmax), "viridis", COLOR_INFO[color]["label"]

    elif color == "n_s":
        return Normalize(vmin=0.85, vmax=1.05), "RdBu_r", COLOR_INFO[color]["label"]

    elif color == "f_total":
        pos = Z[Z > 0]
        vmin = max(np.nanmin(pos), 1e-10) if len(pos) else 1e-10
        vmax = np.nanmax(Z) if len(valid) else 1.0
        if vmin >= vmax:
            vmin, vmax = 1e-10, 1.0
        return LogNorm(vmin=vmin, vmax=vmax), "plasma", COLOR_INFO[color]["label"]

    elif color == "mass_bin":
        bins = np.arange(-0.5, 7.5, 1.0)
        norm = BoundaryNorm(bins, ncolors=min(8, len(bins) - 1), clip=True)
        return norm, "tab10", COLOR_INFO[color]["label"]

    elif color == "usr_type":
        bins = np.arange(-0.5, 3.5, 1.0)
        norm = BoundaryNorm(bins, ncolors=3, clip=True)
        return norm, "coolwarm", COLOR_INFO[color]["label"]

    return Normalize(), "viridis", color


def plot_terrain(X, Y, Z, x_param, y_param, color="k_peak", output=None):
    """Create a publication-ready terrain heatmap. Returns (fig, ax)."""
    with plt.rc_context(PAPER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(7, 5))

        valid = np.isfinite(Z)
        if not valid.any():
            ax.text(0.5, 0.5, "No valid data points", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_xlabel(PARAMS[x_param]["label"])
            ax.set_ylabel(PARAMS[y_param]["label"])
            save_fig(fig, output or "terrain_empty", category="diagnostics")
            return fig, ax

        norm, cmap, label = _configure_colormap(color, Z)

        masked = np.ma.masked_invalid(Z)
        pcm = ax.pcolormesh(X, Y, masked, norm=norm, cmap=cmap, shading="auto")

        cb = fig.colorbar(pcm, ax=ax, label=label)

        if color == "mass_bin":
            cb.set_ticks(range(7))
            cb.set_ticklabels(["Too light", "Asteroid", "Intermed.", "Sub-solar",
                               "Sub-stell.", "Stellar", "Massive"])
        elif color == "usr_type":
            cb.set_ticks(range(3))
            cb.set_ticklabels(["Weak", "Trans.", "Strong"])

        ax.set_xlabel(PARAMS[x_param]["label"])
        ax.set_ylabel(PARAMS[y_param]["label"])

        save_fig(fig, output or f"terrain_{x_param}_{y_param}_{color}",
                 category="diagnostics")
        return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Generate 2D USR terrain heatmaps from trained emulator."
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained PSEmulator joblib file")
    parser.add_argument("--x-param", type=str, default="beta",
                        choices=list(PARAMS.keys()),
                        help="X-axis parameter")
    parser.add_argument("--y-param", type=str, default="c",
                        choices=list(PARAMS.keys()),
                        help="Y-axis parameter")
    parser.add_argument("--density", type=int, default=100,
                        help="Grid points per axis (default: 100)")
    parser.add_argument("--color", type=str, default="k_peak",
                        choices=list(COLOR_INFO.keys()),
                        help="Color metric to plot")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output filename (without extension)")
    for p in PARAMS:
        parser.add_argument(f"--{p}", type=float, default=PARAMS[p]["default"],
                            help=f"Fixed {p} (default: {PARAMS[p]['default']})")

    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        candidates = [
            "outputs/simulations/emulator/model.joblib",
            "outputs/simulations/emulator/emulator_model.joblib",
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break
        if model_path is None:
            print("Error: No model found. Use --model to specify path.",
                  file=sys.stderr)
            sys.exit(1)

    print(f"Loading model from: {model_path}")
    emu = PSEmulator.load(model_path)
    print(f"Model: {emu}")

    fix_params = {p: getattr(args, p) for p in PARAMS}

    print(f"Generating {args.density}x{args.density} terrain: "
          f"x={args.x_param}, y={args.y_param}, color={args.color}")
    X, Y, Z = make_terrain_map(
        emu, args.x_param, args.y_param, color=args.color,
        density=args.density, fix_params=fix_params,
    )

    if args.output is None:
        args.output = f"terrain_{args.x_param}_{args.y_param}_{args.color}"
    print("Plotting...")
    fig, ax = plot_terrain(X, Y, Z, args.x_param, args.y_param,
                           color=args.color, output=args.output)
    valid_count = int(np.sum(np.isfinite(Z)))
    total = Z.size
    print(f"Done. Valid points: {valid_count}/{total} "
          f"({100.0 * valid_count / total:.1f}%)")


if __name__ == "__main__":
    main()
