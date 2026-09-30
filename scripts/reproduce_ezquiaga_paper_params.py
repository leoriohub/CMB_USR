#!/usr/bin/env python3
"""Deterministic reproduction of the "wrong parameters" claim about
Ezquiaga, Garcia-Bellido & Ruiz Morales, "Primordial Black Hole production
in Critical Higgs Inflation", Phys. Lett. B 776 (2018) 345-349
(arXiv:1705.04861v3).

Reproduces, from the published text alone, the claims of
CMB_Anomaly/ezquiaga/main.tex (Secs. 2-3, Table 1):

  (i)   The bare RG ratios reported in the paper,
            a = b_lambda/lambda_0 = 1.2e-6 / 2.23e-7 = 5.381166
            b = b_xi/xi_0        = 11.5 / 7.55        = 1.523179
        do NOT satisfy the inflection conditions V'(x_c) = V''(x_c) = 0
        at the paper's own reference point (x_c, c) = (0.784, 0.77):
        V'(x_c) = -5.655e-3 < 0, V''(x_c) = +5.221e-3 > 0  -> local minimum.

  (ii)  The minimum (x ~ 0.8085) is bracketed by a maximum at x ~ 0.7596
        with barrier (Vmax - Vmin)/Vmin = 6.2e-4, and the exact
        background dynamics (canonical Einstein-frame field chi,
        chi0 = 8.0, y0 = -1e-4) never reach epsilon_H = 1: the field
        settles in the basin, inflation never ends (N_total ~ 182.6 at
        the integration cutoff, main.tex: ~182).

  (iii) The inflection-matched parameters from the paper's own formulas
        (a = a(x_c,c), b = (1-beta) b(x_c,c)) roll through and end
        inflation at N_total = 87.36 (beta=1e-5, c=0.77) and 89.29
        (beta=4e-5, c=0.771), matching main.tex's 87.4 and 89.3.

Model conventions are those of pspectrum_pipeline.py for the archived
runs (outputs/Ezquiaga/): the chi(x) conformal spline is built with the
published b_xi = 11.5, and (a, b, v0) are then overridden from the
inflection formulas (same construction that produced main.tex's numbers).

Every input is hardcoded from the published text (arXiv:1705.04861v3,
Eq. "set" and the model-parameter list). No random numbers, no fits.

Usage:  python scripts/reproduce_ezquiaga_paper_params.py
"""

import numpy as np
from scipy.optimize import brentq

from models.ezquiaga_chi import EzquiagaCHIModel, inflection_parameters
import inf_dyn_background as bg

# ---------------------------------------------------------------------------
# Published values (arXiv:1705.04861v3, Sec. II: "The reference point ...")
# ---------------------------------------------------------------------------
LAMBDA_0 = 2.23e-7   # lambda_0
B_LAMBDA = 1.2e-6    # b_lambda
XI_0 = 7.55          # xi_0
B_XI = 11.5          # b_xi
K2MU2 = 0.102        # kappa^2 mu^2
X_C = 0.784          # reference x_c
C_REF = 0.77         # reference c (rounded)
BETA_REF = 1e-5      # reference beta

A_PAPER = B_LAMBDA / LAMBDA_0   # 5.381166
B_PAPER = B_XI / XI_0           # 1.523179
C_IMPLIED = XI_0 * K2MU2        # 0.7701  (what the numbers imply for c)

# main.tex Table 1 targets (last two columns are the V/V0 shape derivatives
# at x_c, in absolute units). Table 1 prints them scaled by 10^-3:
# Paper -5.6552 / +5.2208, Exact 0.0000 / 0.0000, Working +0.0032 / +0.0079.
TAB1 = {
    "Paper":   dict(a=5.381166, b=1.523179, Vp=-5.6552e-3, Vpp=5.2208e-3),
    "Exact":   dict(a=5.335304, b=1.519340, Vp=0.0,        Vpp=0.0),
    "Working": dict(a=5.335304, b=1.519325, Vp=3.2e-6,     Vpp=7.9e-6),
}

T_MAX = 1000.0
BG_STEPS = 5000
CHI0, Y0 = 8.0, -1e-4


def make_model_paper(c=C_REF):
    """Model with the LITERAL published couplings (paper.json path)."""
    return EzquiagaCHIModel(lambda_0=LAMBDA_0, b_lambda=B_LAMBDA,
                            xi_0=XI_0, b_xi=B_XI, c=c)


def make_model_inflection(c, beta):
    """Pipeline-style construction (pspectrum_pipeline.py): constructor
    keeps the published b_xi = 11.5 for the chi(x) conformal spline, then
    (a, b, v0) are overridden from the paper's inflection formulas."""
    m = EzquiagaCHIModel(lambda_0=LAMBDA_0, xi_0=XI_0, c=c)
    a_new, b_new = inflection_parameters(X_C, c, beta)
    m.a, m.b = a_new, b_new
    m.v0 = m._V0 * m.a / (m.b * m.c) ** 2
    return m


def make_model_inflection_consistent(c, beta):
    """Variants: same (a, b), but the conformal spline is built with the
    matched b_xi = b*xi_0 (physically self-consistent ξ running)."""
    a_new, b_new = inflection_parameters(X_C, c, beta)
    return EzquiagaCHIModel(lambda_0=LAMBDA_0, b_lambda=a_new * LAMBDA_0,
                            xi_0=XI_0, b_xi=b_new * XI_0, c=c)


def static_derivs(model):
    vp = float(model._dVdx(X_C))
    vpp = float(model._d2Vdx2(X_C))
    grid = np.linspace(0.50, 1.20, 1401)
    dV = model._dVdx(grid)
    roots = []
    for i in range(len(grid) - 1):
        if dV[i] * dV[i + 1] < 0.0:
            r = brentq(model._dVdx, grid[i], grid[i + 1])
            roots.append((r, float(model._d2Vdx2(r))))
    return vp, vpp, roots


def barrier(model, roots):
    if len(roots) < 2:
        return None
    vals = sorted((float(model._V(r)), r) for r, _ in roots)
    v_min, x_min = vals[0]
    v_max, x_max = vals[-1]
    return (v_max - v_min) / v_min, x_min, x_max


def run_dynamics(model, label):
    """Background EOM in the canonical field chi; returns fractional
    N_total at the eps_H = 1 crossing (pipeline convention) or the
    end-of-run state if the field stalls."""
    model.x0 = CHI0
    model.y0 = Y0
    T = np.linspace(0.0, T_MAX, BG_STEPS)
    sol = bg.run_background_simulation(model, T)
    q = bg.get_derived_quantities(sol, model)
    epsH, N = q["epsH"], q["N"]
    chi = sol[0]
    end_idx = np.where(epsH >= 1.0)[0]
    if end_idx.size:
        j = int(end_idx[0])
        f = 0.0
        if j > 0 and epsH[j - 1] < 1.0:
            f = (1.0 - epsH[j - 1]) / (epsH[j] - epsH[j - 1])
        return dict(ended=True,
                    N_end=float(N[j - 1] + f * (N[j] - N[j - 1])),
                    N_idx=float(N[j]), eps_end=float(epsH[j]))
    x_final = float(model._x_of_chi(chi[-1]))
    return dict(ended=False, N_final=float(N[-1]),
                eps_final=float(epsH[-1]), chi_final=float(chi[-1]),
                x_final=x_final, eps_max=float(np.nanmax(epsH)))


def scan_for_exact_inflection():
    """Is there ANY (x, c) for which the published (a, b) are an exact
    inflection?  Deterministic 2-D grid scan of the relative residual."""
    xs = np.linspace(0.20, 2.00, 181)
    cs = np.linspace(0.10, 5.00, 246)
    X, C = np.meshgrid(xs, cs)
    lnx = np.log(X)
    den = 1 + C * X**2 + 2 * lnx - 4 * lnx**2
    a = 4.0 / den
    b = (2 * (1 + C * X**2 + 4 * lnx + 2 * C * X**2 * lnx)
         / (C * X**2 * den))
    res = ((a - A_PAPER) / A_PAPER)**2 + ((b - B_PAPER) / B_PAPER)**2
    i, j = np.unravel_index(np.argmin(res), res.shape)
    return float(xs[j]), float(cs[i]), float(res[i, j])


def main():
    print("=" * 78)
    print("Deterministic reproduction: 'wrong parameters' claim vs "
          "Ezquiaga et al. 1705.04861")
    print("=" * 78)
    print(f"Published couplings (Eq. 'set'): lambda_0={LAMBDA_0:g}, "
          f"b_lambda={B_LAMBDA:g}, xi_0={XI_0:g}, b_xi={B_XI:g}, "
          f"kappa^2 mu^2={K2MU2:g}")
    print(f"Reference point: x_c={X_C}, c={C_REF}, beta={BETA_REF:g}")
    print(f"Derived bare ratios: a = b_lambda/lambda_0 = {A_PAPER:.6f}, "
          f"b = b_xi/xi_0 = {B_PAPER:.6f}   (main.tex: 5.381166, 1.523179)")
    print(f"Implied c = xi_0*kappa^2*mu^2 = {C_IMPLIED:.6f} "
          f"(paper quotes rounded 0.77)")
    print()

    # ---------------- static: inflection conditions at x_c ----------------
    print("-" * 78)
    print("Static check: V'(x_c), V''(x_c) of the V/V0 shape at "
          "x_c = 0.784")
    print("-" * 78)
    print("{:<9}{:>11}{:>11}{:>14}{:>14}   main.tex Table 1".format(
        "Set", "a", "b", "V'(x_c)", "V''(x_c)"))
    a_ex, b_ex = inflection_parameters(X_C, C_REF, 0.0)
    a_wk, b_wk = inflection_parameters(X_C, C_REF, BETA_REF)
    a_tw, b_tw = inflection_parameters(X_C, 0.771, 4e-5)
    print(f"{'a_exact':<9}{a_ex:>11.6f}{b_ex:>11.6f}"
          f"{'':>14}{'':>14}")
    print(f"{'a_working':<9}{a_wk:>11.6f}{b_wk:>11.6f}"
          f"{'':>14}{'':>14}")
    print(f"{'a_tweaked':<9}{a_tw:>11.6f}{b_tw:>11.6f}"
          f"{'':>14}{'':>14}")

    sets = [
        ("Paper", A_PAPER, B_PAPER, C_REF, TAB1["Paper"]),
        ("Paper(c=0.7701)", A_PAPER, B_PAPER, C_IMPLIED, TAB1["Paper"]),
        ("Exact", a_ex, b_ex, C_REF, TAB1["Exact"]),
        ("Working", a_wk, b_wk, C_REF, TAB1["Working"]),
    ]
    for name, a, b, c, tgt in sets:
        m = EzquiagaCHIModel(lambda_0=LAMBDA_0, b_lambda=a * LAMBDA_0,
                             xi_0=XI_0, b_xi=b * XI_0, c=c)
        vp = float(m._dVdx(X_C))
        vpp = float(m._d2Vdx2(X_C))
        if name not in TAB1:
            note = "  (variant: c from Eq. 8 primitives, not a Table 1 row)"
        elif (np.isclose(vp, tgt["Vp"], rtol=2e-2, atol=1e-12)
              and np.isclose(vpp, tgt["Vpp"], rtol=2e-2, atol=1e-12)):
            note = "  matches main.tex Table 1"
        else:
            note = (f"  MISMATCH vs main.tex Table 1 "
                    f"({tgt['Vp']:.4g}, {tgt['Vpp']:.4g})")
        print(f"{name:<9}{a:>11.6f}{b:>11.6f}{vp:>14.6g}{vpp:>14.6g}"
              f"{note}")

    # extrema + barrier for the Paper set
    m_paper = make_model_paper(C_REF)
    vp, vpp, roots = static_derivs(m_paper)
    print()
    print("Paper set: roots of V'(x) = 0 bracketing x_c:")
    for r, v2 in roots:
        kind = "MIN" if v2 > 0 else "MAX"
        print(f"  x = {r:.6f}   V'' = {v2:+.4e}   -> local {kind}")
    br = barrier(m_paper, roots)
    if br:
        dv, x_min, x_max = br
        print(f"  barrier (Vmax-Vmin)/Vmin = {dv:.4e}   "
              f"(main.tex: ~6.2e-4, min at x~0.808, max at x~0.760)")
    print()
    print(f"Effective beta of published b:  1 - b_paper/b_exact = "
          f"{1 - B_PAPER / b_ex:+.6f}")
    print(f"Effective beta of published a:  1 - a_paper/a_exact = "
          f"{1 - A_PAPER / a_ex:+.6f}")
    print("  (sign negative -> local minimum, as in main.tex ~-2.5e-3; the "
          "printed primitives sit on the wrong side of the inflection)")
    x_best, c_best, res_best = scan_for_exact_inflection()
    print(f"Best exact-inflection fit for (a,b)_paper over x in [0.2,2], "
          f"c in [0.1,5]: x={x_best:.3f}, c={c_best:.3f}, rel. residual "
          f"={res_best:.3e}")
    print("  (0 would mean an exact inflection exists somewhere; the "
          "published values are not an exact inflection for ANY (x,c))")
    print()

    # -------------------------- dynamics ---------------------------------
    print("-" * 78)
    print(f"Dynamics: exact background EOM, canonical chi, chi0={CHI0}, "
          f"y0={Y0}, T_max={T_MAX} (pipeline conventions)")
    print("-" * 78)
    print(f"{'Config':<30}{'ends?':<7}{'N_total':>10}"
          f"{'eps_final':>11}{'chi_final':>10}{'x_final':>9}")
    cases = [
        ("Paper (published)", make_model_paper(C_REF),
         "stalls, N~182 (main.tex)"),
        ("Exact beta=0", make_model_inflection(C_REF, 0.0),
         "long USR, then ends"),
        ("fig3 beta=1e-5, c=0.77", make_model_inflection(C_REF, BETA_REF),
         "N_total=87.4 (main.tex)"),
        ("tweaked beta=4e-5, c=0.771",
         make_model_inflection(0.771, 4e-5),
         "N_total=89.3 (main.tex)"),
        ("fig3, consistent b_xi", make_model_inflection_consistent(C_REF,
                                                                   BETA_REF),
         "sensitivity: 84.3"),
        ("tweaked, consistent b_xi",
         make_model_inflection_consistent(0.771, 4e-5),
         "sensitivity: 85.0"),
    ]
    for label, model, expect in cases:
        r = run_dynamics(model, label)
        if r["ended"]:
            print(f"{label:<30}{'yes':<7}{r['N_end']:>10.3f}"
                  f"{r['eps_end']:>11.4g}{'':>10}{'':>9}"
                  f"   expected: {expect}")
        else:
            print(f"{label:<30}{'NO':<7}{r['N_final']:>10.2f}"
                  f"{r['eps_final']:>11.4g}{r['chi_final']:>10.4f}"
                  f"{r['x_final']:>9.4f}   expected: {expect}")
    print()
    print("Verdict: 'NO' = epsilon_H never reached 1 within the "
          "integration -> inflation never ends (field trapped at the "
          "local minimum x ~ 0.808).")


if __name__ == "__main__":
    main()
