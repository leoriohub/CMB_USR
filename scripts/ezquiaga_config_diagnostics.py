"""Diagnostics for the Ezquiaga CHI configs: potential and flow parameters.

For each config given on the command line, produce

  * the effective potential V/V0 vs the Jordan-frame field x = phi/mu,
    zoomed on the inflection region, with the inflection point marked;
  * the flow parameters eps_H(N) and eta_H(N) over the last stages of
    inflation, with the pivot and the end of inflation marked;
  * the Hubble rate H(N) in code units.

The three configs are overlaid on each other as well, so the effect of
moving (x_c, c, beta) on the potential shape is visible directly.

Usage
-----
    python scripts/ezquiaga_config_diagnostics.py \
        configs/ezquiaga/fm_curv_stellar.json \
        configs/ezquiaga/fm_curv_subsolar.json \
        configs/ezquiaga/fm_curv_asteroid.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plotting import PAPER_RCPARAMS, TOL, get_path
from scripts.constants import S
import pspectrum_pipeline as psp
from inf_dyn_background import get_derived_quantities, run_background_simulation
from models.ezquiaga_chi import inflection_parameters


COLORS = [TOL["red"], TOL["blue"], TOL["green"], TOL["yellow"], TOL["purple"]]


def load_config(path):
    with open(path) as f:
        return json.load(f)


def label_for(cfg):
    """Label that distinguishes configs.

    x_c, c and beta are shared by the stellar and sub-solar configs, so
    N_star is included; without it the two curves carry the same legend
    entry and one hides the other.
    """
    inf = cfg.get("inflection", {})
    return (
        rf"$x_c$={inf.get('x_c'):.3f}, $c$={cfg['model_params']['c']:.3f}, "
        rf"$\beta$={inf.get('beta'):.1e}, $N_\star$={cfg['pipeline']['N_star']:.0f}"
    )


def V_over_V0(x, a, b, c):
    """Jordan-frame potential shape, Eq. (1) of the paper.

    V(x)/V0 = x^4 (1 + a ln^2 x) / [1 + c (1 + b ln x) x^2]^2,
    with x = phi/mu.  The inflection is defined in this frame, so this is
    where it should be shown: the Einstein-frame field chi smears it out
    through the phi -> chi conformal transformation.
    """
    return x**4 * (1.0 + a * np.log(x) ** 2) / (
        1.0 + c * (1.0 + b * np.log(x)) * x**2
    ) ** 2


def background(cfg, model, bg_steps=20000):
    """Run the background ODE and return the derived quantities.

    run_background_simulation takes the *evaluation grid* as T_span, not a
    (start, end) pair: passing a 2-tuple returns a 2-point solution.
    """
    T_span = np.linspace(0.0, model.T_max, bg_steps)
    sol = run_background_simulation(model, T_span)
    derived = get_derived_quantities(sol, model)
    return sol, derived


def end_index(derived):
    """Index where eps_H first reaches 1 (end of inflation)."""
    epsH = derived["epsH"]
    hits = np.where(epsH >= 1.0)[0]
    if len(hits) == 0:
        return len(epsH) - 1
    return int(hits[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="+", help="Config JSON files")
    ap.add_argument("--prefix", default="ezq_diag", help="Output filename prefix")
    args = ap.parse_args()

    cfgs = [load_config(p) for p in args.configs]
    models = [psp.model_from_config(c) for c in cfgs]
    labels = [label_for(c) for c in cfgs]

    x_cs = [c["inflection"]["x_c"] for c in cfgs]
    x_lo = min(x_cs) * 0.75
    x_hi = max(x_cs) * 1.25

    # ── Figure 1: potentials, full range and zoom ──────────────────────
    with plt.rc_context(PAPER_RCPARAMS):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

        for cfg, lab, col in zip(cfgs, labels, COLORS):
            inf = cfg["inflection"]
            xc, beta = inf["x_c"], inf["beta"]
            c = cfg["model_params"]["c"]
            a, b = inflection_parameters(xc, c, beta)

            xs = np.linspace(x_lo, x_hi, 4000)
            ax1.plot(xs, V_over_V0(xs, a, b, c), "-", color=col, lw=1.3, label=lab)

            # Zoom on the inflection.  Subtract the tangent at x_c so the
            # curvature is visible: for beta -> 0 the residual is cubic and
            # flat at x_c, while beta > 0 leaves a shallow bump.
            xz = np.linspace(xc - 0.02, xc + 0.02, 2000)
            vz = V_over_V0(xz, a, b, c)
            v_c = V_over_V0(np.array([xc]), a, b, c)[0]
            slope = (V_over_V0(np.array([xc + 1e-6]), a, b, c)[0] - v_c) / 1e-6
            ax2.plot(xz, vz - (v_c + slope * (xz - xc)),
                     "-", color=col, lw=1.3, label=lab)
            ax2.axvline(xc, color=col, ls=":", lw=0.8, alpha=0.7)

        ax1.set_xlabel(r"$x=\phi/\mu$")
        ax1.set_ylabel(r"$V/V_0$")
        ax1.set_title("Full potential", fontsize=9)
        ax1.legend(fontsize=6, loc="lower left")

        ax2.set_xlabel(r"$x=\phi/\mu$")
        ax2.set_ylabel(r"$V/V_0 - \mathrm{tangent\ at\ } x_c$")
        ax2.set_title("Inflection region (tangent subtracted)", fontsize=9)
        ax2.legend(fontsize=6, loc="best")

        fig.tight_layout()
        out1 = get_path("diagnostics", f"{args.prefix}_potential.png")
        fig.savefig(out1, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out1}")

    # ── Figure 2: flow parameters and Hubble rate ──────────────────────
    with plt.rc_context(PAPER_RCPARAMS):
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))

        for cfg, model, lab, col in zip(cfgs, models, labels, COLORS):
            sol, derived = background(cfg, model)
            N = derived["N"]
            eps1 = derived["eps1"]
            eps2 = derived["eps2"]
            # eps2 = d ln eps1/dN is enormous during the initial transient,
            # where eps1 climbs from ~1e-8 to ~1e-4 within the first few
            # grid points as the field settles onto the attractor.  That
            # transient is real but is not the slow-roll or USR behaviour
            # of interest, so drop the first ~5 e-folds and clip the rest.
            settle = N >= 5.0
            eps2 = np.where(settle, eps2, np.nan)
            eps2 = np.clip(eps2, -50.0, 50.0)
            z_bg = sol[2]
            i_end = end_index(derived)
            N_end = N[i_end]
            N_std = N_end - N[: i_end + 1]

            axes[0].semilogy(N_std, eps1[: i_end + 1], "-", color=col, lw=1.2, label=lab)
            axes[1].plot(N_std, eps2[: i_end + 1], "-", color=col, lw=1.2, label=lab)
            axes[2].semilogy(N_std, np.abs(z_bg[: i_end + 1]) * S,
                             "-", color=col, lw=1.2, label=lab)

        axes[0].set_xlabel(r"$N$ (e-folds before end)")
        axes[0].set_ylabel(r"$\varepsilon_1$")
        axes[0].axhline(1.0, color="0.5", ls="--", lw=0.8)
        axes[0].legend(fontsize=5.5, loc="best")

        axes[1].set_xlabel(r"$N$ (e-folds before end)")
        axes[1].set_ylabel(r"$\varepsilon_2 = d\ln\varepsilon_1/dN$")
        axes[1].axhline(0.0, color="0.5", ls="--", lw=0.8)

        axes[2].set_xlabel(r"$N$ (e-folds before end)")
        axes[2].set_ylabel(r"$H$")

        for ax in axes:
            ax.invert_xaxis()

        fig.tight_layout()
        out2 = get_path("diagnostics", f"{args.prefix}_flow.png")
        fig.savefig(out2, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out2}")


if __name__ == "__main__":
    main()
