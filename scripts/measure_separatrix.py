"""Distance from the reference trajectory to the USR separatrix, in units of H/2pi.

For each scale at which the delta-N machinery differentiates, measure how far the
background state (x, y) at the delta-N slice can be displaced before the
separate-universe patch no longer reaches the end of inflation, and compare that
distance with the size of a quantum fluctuation, H/(2 pi).

Motivation. f_NL is a ratio of derivatives of N(x, y), so the finite-difference
step must be small compared with any structure in N. Near an inflection N grows
without bound as the field approaches the stalling boundary, so the surface is
effectively singular there. If the separatrix sits within a few H/(2 pi) of the
reference trajectory, the derivative expansion is not the right tool at that
scale and the distribution has to be sampled instead of differentiated.

The search doubles the offset from ``--h-min`` until a patch fails, then bisects
to a relative tolerance. A returned distance equal to ``--h-max`` means no
boundary was found below that value (the trajectory is far from the separatrix).

Usage:
    python -m scripts.measure_separatrix \\
        --config configs/ezquiaga/a1_stellar.json --nexits 48,50,52,54
"""

import argparse
import json

import numpy as np

from inf_dyn_background import run_background_simulation, get_derived_quantities
from pspectrum_pipeline import find_end_of_inflation
from scripts.compute_fnl_usr import model_factory, run_patch_N_total
from scripts.constants import S


def patch_ends(make_model, x_p, y_p, T_span, max_doublings):
    """True if the patch reaches the end of inflation inside the extended window."""
    try:
        run_patch_N_total(make_model, x_p, y_p, T_span, max_doublings=max_doublings)
        return True
    except RuntimeError:
        return False


def separatrix_distance(make_model, x_s, y_s, T_span, direction='x',
                        h_min=1e-6, h_max=0.5, tol_rel=0.1, max_doublings=2):
    """Distance in ``direction`` from (x_s, y_s) at which patches stop ending.

    Both signs are scanned (doubling from ``h_min`` until a patch fails, then
    bisecting) and the smaller distance is returned, since either side can be
    the one that stalls. Returns ``h_max`` if neither side fails below it.
    """
    def ends_at(d):
        x_p = x_s + d if direction == 'x' else x_s
        y_p = y_s + d if direction == 'y' else y_s
        return patch_ends(make_model, x_p, y_p, T_span, max_doublings)

    if not ends_at(0.0):
        raise RuntimeError(
            f"reference state (x={x_s:.6g}, y={y_s:.6g}) does not end inflation")

    best = float(h_max)
    for sign in (+1.0, -1.0):
        d = h_min
        hi = None
        while d < h_max:
            if not ends_at(sign * d):
                hi = d
                break
            d *= 2.0
        if hi is None:
            continue
        lo = 0.0
        while (hi - lo) > tol_rel * hi:
            mid = 0.5 * (lo + hi)
            if ends_at(sign * mid):
                lo = mid
            else:
                hi = mid
        best = min(best, hi)
    return float(best)


def main():
    parser = argparse.ArgumentParser(
        description='Distance to the USR separatrix vs the quantum fluctuation '
                    'size H/2pi, per delta-N slice')
    parser.add_argument('--config', type=str, default=None,
                        help='Config JSON resolved through models.registry')
    parser.add_argument('--x0', type=float, default=5.75)
    parser.add_argument('--y0', type=float, default=-0.170)
    parser.add_argument('--nexits', type=str, default='48,50,52,54',
                        help='Comma-separated N_exit values (e-folds from start)')
    parser.add_argument('--slice-offset', type=float, default=1.0,
                        help='E-folds after horizon exit where the delta-N slice sits')
    parser.add_argument('--bg-steps', type=int, default=8000)
    parser.add_argument('--max-doublings', type=int, default=2,
                        help='Integration-window doublings allowed before a patch '
                             'counts as stalling')
    parser.add_argument('--h-min', type=float, default=1e-6)
    parser.add_argument('--h-max', type=float, default=0.5)
    parser.add_argument('--tol', type=float, default=0.1,
                        help='Relative tolerance of the bisection')
    parser.add_argument('--json', type=str, default=None)
    args = parser.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        ics = cfg.get('ics', {})
        args.x0 = ics.get('x0', args.x0)
        args.y0 = ics.get('y0', args.y0)

    make_model = model_factory(cfg if args.config else None, 0.13, 15000.0)

    model = make_model(quiet=False)
    model.x0 = args.x0
    model.y0 = args.y0
    T_span = np.linspace(0.0, model.T_max, args.bg_steps)

    bg = run_background_simulation(model, T_span)
    der = get_derived_quantities(bg, model)
    end_idx = find_end_of_inflation(der['epsH'])
    if end_idx == -1:
        raise SystemExit('reference trajectory never ends inflation')
    N_total = der['N'][end_idx]
    print(f"  reference: N_total={N_total:.4f}, x0={args.x0}, y0={args.y0}")

    header = (f"{'N_exit':>7} {'x_slice':>10} {'y_slice':>11} {'H/2pi':>10} "
              f"{'dx_sep':>10} {'dx/(H/2pi)':>11} {'dy_sep':>10} {'dy/|y_s|':>9}")
    print(header)
    print('-' * len(header))

    results = []
    for N_exit in (float(v) for v in args.nexits.split(',')):
        N_slice = min(N_exit + args.slice_offset, N_total - 1.5)
        x_s = float(np.interp(N_slice, der['N'], bg[0]))
        y_s = float(np.interp(N_slice, der['N'], bg[1]))
        z_s = float(np.interp(N_slice, der['N'], bg[2]))
        h_quantum = z_s * S / (2.0 * np.pi)

        dx = separatrix_distance(make_model, x_s, y_s, T_span, 'x',
                                 args.h_min, args.h_max, args.tol, args.max_doublings)
        dy = separatrix_distance(make_model, x_s, y_s, T_span, 'y',
                                 args.h_min, args.h_max, args.tol, args.max_doublings)

        ratio_x = dx / h_quantum if h_quantum > 0 else float('nan')
        ratio_y = dy / abs(y_s) if y_s != 0 else float('nan')
        print(f"{N_exit:7.2f} {x_s:10.5f} {y_s:11.4e} {h_quantum:10.3e} "
              f"{dx:10.3e} {ratio_x:11.3g} {dy:10.3e} {ratio_y:9.3g}")
        results.append({
            'N_exit': N_exit, 'N_slice': N_slice,
            'x_slice': x_s, 'y_slice': y_s, 'z_slice': z_s,
            'H_over_2pi': h_quantum,
            'dx_separatrix': dx, 'dy_separatrix': dy,
            'dx_over_H2pi': ratio_x, 'dy_over_absy': ratio_y,
            'hit_h_max': bool(dx >= args.h_max or dy >= args.h_max),
        })

    print("\nInterpretation: dx/(H/2pi) >> 1 means the physical fluctuation is far "
          "from the separatrix (derivatives meaningful); order unity or below means "
          "the separatrix is inside the quantum spread (sample, do not differentiate).")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'config_file': args.config, 'results': results}, f,
                      indent=2, default=str)
        print(f"Results saved to {args.json}")


if __name__ == '__main__':
    main()
