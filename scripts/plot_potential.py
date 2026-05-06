"""
Plot inflationary potential V(phi)/V0 for any InflationModel subclass.

Usage:
  python scripts/plot_potential.py Punctuated --m 1.1323e-7 --lam 3.3299e-15 --x-range 0 3 --mark-inflection
  python scripts/plot_potential.py Higgs --lam 0.13 --xi 15000 --x-range 0 8
  python scripts/plot_potential.py FullHiggs --xi 1000 --lam 0.1 --x-range 0 8
  python scripts/plot_potential.py Quadratic --x-range 0 20
  python scripts/plot_potential.py --list

Output: interactive display or save to PNG via --save DIR
"""

import argparse
import os
import sys
import textwrap

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import numpy as np
import matplotlib.pyplot as plt

from models import (InflationModel, QuadraticModel, HiggsModel,
                    FullHiggsModel, SmoothUSRTransitionModel,
                    PunctuatedInflationModel)


MODEL_REGISTRY = {
    "Quadratic": (QuadraticModel, {}),
    "Higgs": (HiggsModel, {"lam": 0.13, "xi": 15000.0}),
    "FullHiggs": (FullHiggsModel, {"lam": 0.1, "xi": 1000.0, "v_vev": 0.0}),
    "SmoothUSR": (SmoothUSRTransitionModel,
                  {"alpha": 22.63, "mu": 2.0294, "eps_sr1": 1e-6, "H0": 1.0}),
    "Punctuated": (PunctuatedInflationModel,
                   {"m": 1.1323e-7, "lam": 3.3299e-15, "phi0": None}),
}

DEFAULT_XRANGES = {
    "Punctuated": (0.0, 3.0),
    "Higgs": (0.0, 8.0),
    "FullHiggs": (0.0, 8.0),
    "SmoothUSR": None,
    "Quadratic": (0.0, 20.0),
}


def list_models():
    print("Available models:")
    for name, (cls, defaults) in MODEL_REGISTRY.items():
        sig = ", ".join(f"{k}={v}" for k, v in defaults.items())
        print(f"  {name}  ({sig})")


def parse_model_kwargs(unknown_args):
    kwargs = {}
    i = 0
    while i < len(unknown_args):
        key = unknown_args[i].lstrip("-")
        if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("-"):
            val_str = unknown_args[i + 1]
            i += 2
        else:
            i += 1
            continue
        try:
            if "." in val_str or "e" in val_str.lower():
                kwargs[key] = float(val_str)
            else:
                kwargs[key] = int(val_str)
        except ValueError:
            kwargs[key] = val_str
    return kwargs


def main():
    parser = argparse.ArgumentParser(
        description="Plot V(phi)/V0 for any inflationary model")
    parser.add_argument("model", nargs="?", help="Model name (use --list to see options)")
    parser.add_argument("--list", action="store_true", help="Show available models")
    parser.add_argument("--x-range", nargs=2, type=float, default=None,
                        metavar=("MIN", "MAX"),
                        help="Phi range for the plot")
    parser.add_argument("--n-points", type=int, default=500,
                        help="Number of points in phi grid")
    parser.add_argument("--mark-inflection", action="store_true",
                        help="Mark inflection point (Punctuated model only)")
    parser.add_argument("--title", type=str, default=None,
                        help="Custom plot title")
    parser.add_argument("--save", type=str, default=None, metavar="DIR",
                        help="Save plot to directory instead of showing interactively")

    known, unknown = parser.parse_known_args()

    if known.list:
        list_models()
        return

    if known.model is None:
        parser.print_help()
        return

    model_name = known.model.capitalize()
    model_name_matches = [n for n in MODEL_REGISTRY if n.lower().startswith(model_name.lower())]

    if not model_name_matches:
        print(f"Unknown model: {known.model}")
        list_models()
        return

    model_name = model_name_matches[0]
    model_cls, model_defaults = MODEL_REGISTRY[model_name]

    user_kwargs = parse_model_kwargs(unknown)
    for k, v in model_defaults.items():
        user_kwargs.setdefault(k, v)

    valid_params = set(model_defaults.keys())
    filtered_kwargs = {k: v for k, v in user_kwargs.items() if k in valid_params}

    model = model_cls(**filtered_kwargs)

    if known.x_range is not None:
        x_min, x_max = known.x_range
    else:
        if model_name == "SmoothUSR":
            x_min = float(model.phi_grid[0])
            x_max = float(model.phi_grid[-1])
        else:
            x_min, x_max = DEFAULT_XRANGES.get(model_name, (0.0, 10.0))

    phi_vals = np.linspace(x_min, x_max, known.n_points)
    V_vals = np.array([model.f(x) for x in phi_vals])

    plt.figure(figsize=(7, 4.5))
    plt.plot(phi_vals, V_vals, "b-", linewidth=1.5)
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$V(\phi)\,/\,V_0$")
    plt.grid(True, alpha=0.3)

    if known.mark_inflection:
        if hasattr(model, "phi0_inflection"):
            phi_infl = model.phi0_inflection
            plt.axvline(phi_infl, color="red", ls="--", lw=1.2, alpha=0.7,
                        label=rf"Inflection $\phi_0 = {phi_infl:.4f}$")
            plt.legend(fontsize=9)
        else:
            print(f"Warning: --mark-inflection not supported for {model_name}")

    if known.title:
        plt.title(known.title, fontsize=11)
    else:
        param_str = ", ".join(f"{k}={v}" for k, v in filtered_kwargs.items())
        wrapped = textwrap.fill(param_str, width=70)
        plt.title(f"{model.name}\n{wrapped}", fontsize=10)

    plt.tight_layout()

    if known.save:
        os.makedirs(known.save, exist_ok=True)
        out_name = f"potential_{model_name}_{model_cls.__name__}.png"
        out_path = os.path.join(known.save, out_name)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
