#!/usr/bin/env python3
"""Interactive Explorer CLI for Ezquiaga CHI NN emulator.

Usage:
  Single-shot:  python scripts/explorer.py --xc 0.784 --c 0.77 --beta 1e-5 --chi0 8.0 --n-star 65
  REPL:         python scripts/explorer.py
"""

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from scripts.emulator import PSEmulator, extract_usr_info
from scripts.plotting import save_fig, TOL, PAPER_RCPARAMS

DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "outputs", "simulations", "emulator", "model.joblib"
)
PLOT_FILENAME = "explorer.png"
K_PIVOT = 0.05


def load_model(model_path=None):
    """Load trained PSEmulator. Returns None if not found."""
    path = model_path or DEFAULT_MODEL_PATH
    if not os.path.isfile(path):
        print(f"Model not found: {path}", file=sys.stderr)
        print("Train the emulator first or specify --model-path.", file=sys.stderr)
        return None
    try:
        emu = PSEmulator.load(path)
        if not emu.is_trained:
            print(f"Model at {path} is untrained.", file=sys.stderr)
            return None
        print(f"Loaded emulator from {path}")
        return emu
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return None


def format_metric(val, fmt=".4f"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:{fmt}}"


def predict_and_plot(emu, xc, c, beta, chi0, n_star, verbose=True):
    """Run prediction, print metrics, save plot. Returns metrics dict."""
    k_grid, P_S = emu.predict(xc, c, beta, chi0, n_star)
    info = extract_usr_info(k_grid, P_S)

    n_s = info.get("n_s")
    A_s = info.get("A_s")
    k_peak = info.get("k_peak")
    M_peak = info.get("M_peak")
    mass_bin = info.get("mass_bin")
    f_total = info.get("f_total")
    usr_type = info.get("usr_type")

    if verbose:
        print(f"  n_s       = {format_metric(n_s)}")
        print(f"  A_s       = {format_metric(A_s, '.4e')}")
        print(f"  k_peak    = {format_metric(k_peak, '.4e')}")
        print(f"  M_peak    = {format_metric(M_peak, '.4e')}")
        print(f"  mass_bin  = {mass_bin or 'N/A'}")
        print(f"  f_total   = {format_metric(f_total, '.4f')}")
        print(f"  usr_type  = {usr_type or 'N/A'}")

    with plt.rc_context(PAPER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.loglog(k_grid, P_S, "-", color=TOL["red"], lw=1.5, label="Emulator")

        ax.axvline(K_PIVOT, color=TOL["green"], ls="--", lw=1.0, alpha=0.7)
        ax.annotate(
            r"$k_{\rm pivot}=0.05$",
            xy=(K_PIVOT, ax.get_ylim()[0]),
            fontsize=7,
            color=TOL["green"],
            ha="center",
            va="bottom",
        )

        if k_peak is not None and np.isfinite(k_peak) and k_peak > 0:
            ps_at_peak = np.interp(k_peak, k_grid, P_S)
            ax.plot(
                k_peak, ps_at_peak,
                marker="v", color=TOL["green"], ms=6, lw=0,
                label=f"Peak: k={k_peak:.2e}",
            )

        metrics_lines = [
            f"$n_s$      = {format_metric(n_s)}",
            f"$A_s$      = {format_metric(A_s, '.4e')}",
            f"$k_{{\\rm peak}}$ = {format_metric(k_peak, '.4e')}",
            f"$M_{{\\rm peak}}$ = {format_metric(M_peak, '.4e')}",
            f"mass_bin = {mass_bin or 'N/A'}",
            f"$f_{{\\rm total}}$   = {format_metric(f_total, '.4f')}",
            f"usr_type = {usr_type or 'N/A'}",
        ]
        params_lines = [
            f"$x_c$={xc:.4f}  $c$={c:.4f}",
            f"$\\beta$={beta:.1e}  $\\chi_0$={chi0:.1f}",
            f"$N_*$={n_star:.1f}",
        ]
        text_str = "\n".join(params_lines + [""] + metrics_lines)
        ax.text(
            0.98, 0.98, text_str,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            horizontalalignment="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor=TOL["grey"]),
        )

        ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
        ax.set_ylabel(r"$\mathcal{P}_{\mathcal{R}}(k)$")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.2, which="both")

        fig.tight_layout()
        save_fig(fig, PLOT_FILENAME, category="diagnostics")

    return info


def interactive_loop(emu):
    """Run interactive REPL."""
    print("\nEzquiaga Explorer (type 'q' to quit)")
    print("Enter: xc c beta chi0 n_star")
    print("  e.g. 0.784 0.77 1e-5 8.0 65")
    print()
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            break

        parts = line.split()
        if len(parts) != 5:
            print("  Expected 5 values: xc c beta chi0 n_star")
            continue
        try:
            xc, c, beta, chi0, n_star = map(float, parts)
        except ValueError:
            print("  Invalid numbers. Enter: xc c beta chi0 n_star")
            continue

        try:
            predict_and_plot(emu, xc, c, beta, chi0, n_star, verbose=True)
            print(f"  Plot: outputs/plots/diagnostics/{PLOT_FILENAME}")
        except Exception as e:
            print(f"  Error: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Explorer CLI for Ezquiaga CHI NN emulator."
    )
    parser.add_argument("--model-path", type=str, default=None,
                        help=f"Path to trained model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--xc", type=float, default=None,
                        help="Critical field value x_c")
    parser.add_argument("--c", type=float, default=None,
                        help="Plateau width parameter c")
    parser.add_argument("--beta", type=float, default=None,
                        help="Deviation from inflection")
    parser.add_argument("--chi0", type=float, default=None,
                        help="Initial field value chi0")
    parser.add_argument("--n-star", type=float, default=None,
                        help="Number of e-folds to pivot")
    return parser.parse_args()


def main():
    args = parse_args()
    emu = load_model(args.model_path)
    if emu is None:
        sys.exit(1)

    has_cli_params = all(
        getattr(args, p) is not None
        for p in ("xc", "c", "beta", "chi0", "n_star")
    )

    if has_cli_params:
        predict_and_plot(emu, args.xc, args.c, args.beta, args.chi0, args.n_star)
        print(f"  Plot: outputs/plots/diagnostics/{PLOT_FILENAME}")
    else:
        interactive_loop(emu)


if __name__ == "__main__":
    main()
