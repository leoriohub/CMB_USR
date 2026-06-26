"""
Drop-in replacement for numba_ms_solver.numba_run_ms.
Delegates to the compiled Fortran module ms_solver_fort.
Falls back to Numba if Fortran module is not compiled.
"""
import numpy as np
import os
import sys

# Ensure the project root and fortran directory are in the import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fortran'))

try:
    import ms_solver_fort as _fort
    HAVE_FORTRAN = True
except ImportError:
    try:
        from fortran import ms_solver_fort as _fort
        HAVE_FORTRAN = True
    except ImportError:
        HAVE_FORTRAN = False

def fortran_run_ms_grid(bg_sol, T_span_bg, end_idx, k_codes, model,
                        k_start_factor=100.0, S=None, bg_coefs=None):
    if not HAVE_FORTRAN:
        raise RuntimeError("Fortran module not compiled. Run: cd fortran && make")
    if S is None:
        S = model.S
    from numba_ms_solver import build_numba_splines
    if bg_coefs is None:
        bg_coefs = build_numba_splines(bg_sol, T_span_bg, model=model)

    # Pack spline coefs into a flat Fortran-friendly array
    bc_arr = _pack_spline_coefs(bg_coefs)

    alpha = float(getattr(model, 'alpha', 0.0))
    v0 = float(model.v0)

    # 1 if model needs spline-based potential derivatives (e.g. Ezquiaga/Punctuated)
    use_spline = 1 if len(bg_coefs) >= 7 else 0

    k_codes = np.asarray(k_codes, dtype=np.float64)
    x_bg = np.asarray(bg_sol[0], dtype=np.float64)
    y_bg = np.asarray(bg_sol[1], dtype=np.float64)
    z_bg = np.asarray(bg_sol[2], dtype=np.float64)
    n_bg = np.asarray(bg_sol[3], dtype=np.float64)
    T_span_bg = np.asarray(T_span_bg, dtype=np.float64)

    P_S, P_T, start_idx_arr = _fort.solve_ms_grid(
        k_codes, bc_arr, x_bg, y_bg, z_bg, n_bg, T_span_bg,
        int(end_idx), float(k_start_factor), float(S), float(v0), float(alpha), int(use_spline)
    )
    return P_S, P_T, start_idx_arr

def _pack_spline_coefs(bg_coefs):
    """Convert list-of-tuples spline coefs to (n_var, 5, n_pts) Fortran array."""
    n_var = len(bg_coefs)
    n_pts = len(bg_coefs[0][0])
    bc = np.zeros((n_var, 5, n_pts), dtype=np.float64, order='F')
    for i, (t, a, b, c, d) in enumerate(bg_coefs):
        bc[i, 0, :] = t
        bc[i, 1, :n_pts-1] = a
        bc[i, 2, :n_pts-1] = b
        bc[i, 3, :n_pts-1] = c
        bc[i, 4, :n_pts-1] = d
    return bc
