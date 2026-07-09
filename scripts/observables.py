"""
Observables extracted from the primordial scalar power spectrum P_S(k).

Single source of truth for n_s extraction and PBH-peak characterization.

The pivot ``k_pivot`` is the SAME wavenumber used for A_s normalization in
the pipeline (``pspectrum_pipeline.k_pivot_phys``) - caller passes it
explicitly so As normalization and n_s extraction always share one pivot.

n_s extraction has two methods:

    - ``'lsq'`` (default): least-squares fit over
      ``[k_pivot/ns_window, k_pivot*ns_window]``.  This is a window-averaged
      slope that is less sensitive to local numerical noise in P_S(k).

    - ``'derivative'``: exact logarithmic derivative via cubic spline at
      k_pivot.  This is the theoretical definition but can be noisy on
      discrete MS output.

Two additional n_s methods exist elsewhere in the project:
  - SR algebraic  n_s = 1 + 2 eta_H - 4 eps_H  (``scripts/background_scan.py``)
  - Local 2-point slope via ±5 bg-timesteps (``scripts/plotting.py``)

These are separate observables that do NOT live in this module.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from scripts.constants import As_planck


def extract_ns(k_phys, P_S, k_pivot, ns_window=4.0, method="lsq"):
    """n_s at k_pivot from P_S(k), via window-averaged LSQ fit by default.

    n_s - 1 = slope of ln P_S vs ln k averaged over
    ``[k_pivot/ns_window, k_pivot*ns_window]``.

    ``k_pivot`` MUST be the same value used for A_s normalization in the
    pipeline - this is the single-pivot invariant of the project.

    Parameters
    ----------
    k_phys : array, physical wavenumbers [Mpc^-1]
    P_S : array, primordial scalar power spectrum values
    k_pivot : float, pivot wavenumber [Mpc^-1]
    ns_window : float, fit half-width used for ``'lsq'`` method.
                Default 4.0. Ignored for ``'derivative'``.
    method : str, ``'lsq'`` (default) or ``'derivative'``.
             ``'lsq'`` fits a straight line over
             ``[k_pivot/ns_window, k_pivot*ns_window]``.
             ``'derivative'`` computes the theoretical n_s via
             logarithmic derivative at k_pivot.

    Returns
    -------
    n_s : float or None
    meta : dict with keys {k_pivot, method, ...}
    """
    k_phys = np.asarray(k_phys, dtype=float)
    P_S = np.asarray(P_S, dtype=float)

    if method == "derivative":
        # Logarithmic derivative: n_s - 1 = d ln P / d ln k at k_pivot.
        valid = np.isfinite(P_S) & (P_S > 1e-30) & np.isfinite(k_phys)
        k_v, p_v = k_phys[valid], P_S[valid]
        if len(k_v) < 4:
            return None, dict(k_pivot=float(k_pivot), n_modes=len(k_v),
                              method=method, error="too few points")
        order = np.argsort(k_v)
        k_v, p_v = k_v[order], p_v[order]
        lk, lp = np.log(k_v), np.log(p_v)
        uniq = np.concatenate(([True], np.diff(lk) > 0))
        lk, lp = lk[uniq], lp[uniq]
        if len(lk) < 4:
            return None, dict(k_pivot=float(k_pivot), n_modes=len(lk),
                              method=method, error="too few unique points")
        spl = CubicSpline(lk, lp, bc_type="not-a-knot", extrapolate=False)
        lk_pivot = float(np.log(k_pivot))
        if lk_pivot < lk[0] or lk_pivot > lk[-1]:
            return None, dict(k_pivot=float(k_pivot), n_modes=len(lk),
                              method=method, error="k_pivot outside grid")
        n_s = 1.0 + float(spl(lk_pivot, nu=1))
        meta = dict(k_pivot=float(k_pivot), ns_window=None, n_modes=int(len(lk)),
                    k_range=[float(np.exp(lk[0])), float(np.exp(lk[-1]))],
                    method=method)
        return n_s, meta

    lo = float(k_pivot) / float(ns_window)
    hi = float(k_pivot) * float(ns_window)
    idx = (k_phys >= lo) & (k_phys <= hi) & np.isfinite(P_S) & (P_S > 1e-30)
    k_fit = k_phys[idx]
    ps_fit = P_S[idx]
    if len(k_fit) < 3:
        return None, dict(k_pivot=float(k_pivot), ns_window=float(ns_window),
                          n_modes=0, k_range=[float(lo), float(hi)], method=method)
    coeffs = np.polyfit(np.log(k_fit), np.log(ps_fit), 1)
    n_s = float(coeffs[0] + 1.0)
    meta = dict(k_pivot=float(k_pivot), ns_window=float(ns_window),
                n_modes=int(len(k_fit)), k_range=[float(lo), float(hi)],
                method=method)
    return n_s, meta


def interpolate_As(k_phys, P_S, k_pivot):
    """Interpolated value of P_S at k_pivot (the A_s amplitude implied by
    the spectrum). Useful for sanity-checking normalization.

    Returns float or None if k_pivot is outside the finite data range.
    """
    k_phys = np.asarray(k_phys, dtype=float)
    P_S = np.asarray(P_S, dtype=float)
    valid = np.isfinite(P_S) & (P_S > 1e-30)
    if valid.sum() < 2:
        return None
    k_v, p_v = k_phys[valid], P_S[valid]
    order = np.argsort(k_v)
    k_v, p_v = k_v[order], p_v[order]
    lk_pivot = float(np.log(k_pivot))
    lk = np.log(k_v)
    if lk_pivot < lk[0] or lk_pivot > lk[-1]:
        return None
    return float(np.exp(np.interp(lk_pivot, lk, np.log(p_v))))


def extract_pbh_peak(k_phys, P_S, As=As_planck):
    """Peak of P_S(k) at small scales - a PBH observable, NOT an n_s.

    The PBH peak lives at k ~ 10^9 to 10^18 Mpc^-1, far from any CMB pivot,
    so a spectral index is meaningless there. This function returns the
    peak location, amplitude, and amplitude relative to A_s.

    Parameters
    ----------
    k_phys : array, physical wavenumbers [Mpc^-1]
    P_S : array, primordial scalar power spectrum values
    As : float, scalar amplitude at the CMB pivot (for the ratio)

    Returns
    -------
    (k_peak, P_S_peak, P_S_peak / As) or (None, None, None)
    """
    k_phys = np.asarray(k_phys, dtype=float)
    P_S = np.asarray(P_S, dtype=float)
    valid = np.isfinite(P_S) & (P_S > 1e-30)
    if valid.sum() < 2:
        return None, None, None
    k_v, p_v = k_phys[valid], P_S[valid]
    i = int(np.argmax(p_v))
    return float(k_v[i]), float(p_v[i]), float(p_v[i] / As)


def model_from_params(x_c, c, beta):
    """Construct Ezquiaga CHI model from inflection parameters."""
    from models.ezquiaga_chi import EzquiagaCHIModel, inflection_parameters

    m = EzquiagaCHIModel(c=c)
    a, b = inflection_parameters(x_c, c, beta=beta)
    m.a = a
    m.b = b
    m.v0 = m._V0 * m.a / (m.b * m.c) ** 2
    return m
