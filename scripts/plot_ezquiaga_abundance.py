"""Overlay the PBH mass functions of several Ezquiaga CHI configs.

The sub-solar, asteroid and stellar windows share the same axes and the
same observational bounds, so a single frame compares them better than
three separate panels.  This script runs each config through the pipeline
(reusing the cached power spectrum where the provenance matches),
converts to the present-day fraction f(M) = Omega_PBH/Omega_DM with the
standard Carr & Kuhnel conversion, and draws them together.

Usage
-----
    python scripts/plot_ezquiaga_abundance.py \
        --config configs/ezquiaga/fm_curv_stellar.json:0.044 \
        --config configs/ezquiaga/fm_curv_subsolar.json:0.052 \
        --config configs/ezquiaga/fm_curv_asteroid.json:0.042 \
        --output fig_pbh_windows
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

import pspectrum_pipeline as psp
from scripts.full_pbh_pipeline import compute_pbh_abundance
from scripts.plotting import plot_pbh_abundance_multi
from scripts.constants import gamma_default, k_eq_default, M_eq_default

# Saturated colours, deliberately distinct from the pastel constraint
# bands so the model curves read on top of them.
CURVE_COLORS = ["#CC3311", "#332288", "#117733", "#882255"]

NICE_NAMES = {"stellar": "Stellar", "subsolar": "Sub-solar", "asteroid": "Asteroid"}


def load_config(path):
    with open(path) as f:
        return json.load(f)


def mass_function(cfg, zeta_c):
    """Return (M, f) for one config at the given collapse threshold."""
    model = psp.model_from_config(cfg)
    pipeline = cfg.get("pipeline", {})
    out = psp.run_pspectrum_pipeline(
        model=model,
        phi0=None,
        y0=None,
        N_star=pipeline.get("N_star", 65.0),
        k_pivot_phys=pipeline.get("k_pivot_phys", 0.05),
        num_k=pipeline.get("num_k", 1000),
        k_min=pipeline.get("k_min", 1e-10),
        k_max=pipeline.get("k_max", 1e20),
        backend="fortran",
        ms_steps=5000,
        normalize_to_As=False,
        n_workers=pipeline.get("n_cores", 8),
        save_outputs=False,
    )
    k = np.asarray(out["k_phys"], dtype=float)
    ps = np.asarray(out["P_S"], dtype=float)
    pbh = compute_pbh_abundance(
        k, ps, gamma_default, zeta_c, k_eq_default, M_eq_default
    )
    return pbh["M"], pbh["f_pbh"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        action="append",
        required=True,
        metavar="PATH:ZETA_C",
        help="Config JSON and collapse threshold, colon separated. Repeatable.",
    )
    ap.add_argument("--output", default="pbh_windows", help="Output filename stem")
    args = ap.parse_args()

    curves = []
    for i, spec in enumerate(args.config):
        path, zeta_c = spec.rsplit(":", 1)
        zeta_c = float(zeta_c)
        cfg = load_config(path)
        M, f = mass_function(cfg, zeta_c)
        peak = int(np.argmax(f))
        stem = os.path.splitext(os.path.basename(path))[0].replace("fm_curv_", "")
        name = NICE_NAMES.get(stem, stem.replace("_", " ").title())
        # Quote only the peak mass.  The crest height is not the f_PBH of
        # Table II (that is the integral over the whole curve), so putting
        # it in the legend invites the reader to compare two different
        # quantities.
        label = rf"{name}: $M_\mathrm{{peak}}$={M[peak]:.2g}$\,M_\odot$"
        curves.append(
            {
                "M": M,
                "f": f,
                "label": label,
                "color": CURVE_COLORS[i % len(CURVE_COLORS)],
            }
        )
        print(f"  {path}  zeta_c={zeta_c}  M_peak={M[peak]:.4e}  peak f={f[peak]:.4e}")

    plot_pbh_abundance_multi(curves, filename=args.output, category="pbh")


if __name__ == "__main__":
    main()
