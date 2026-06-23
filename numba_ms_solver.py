"""
Numba-accelerated Mukhanov-Sasaki solver.

Provides numba_run_ms(), a drop-in replacement for
inf_dyn_MS_full.run_ms_simulation() with ~2x pipeline speedup.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from functools import partial

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    prange = range
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def wrapper(f):
            return f
        return wrapper


@njit(cache=True)
def _spline_eval(t, t_grid, a, b, c, d):
    i = np.searchsorted(t_grid, t) - 1
    i = max(0, min(i, len(t_grid) - 2))
    dt = t - t_grid[i]
    return a[i] + dt * (b[i] + dt * (c[i] + dt * d[i]))


@njit(cache=True)
def _spline_eval_at_index(t, t_grid, i, a, b, c, d):
    dt = t - t_grid[i]
    return a[i] + dt * (b[i] + dt * (c[i] + dt * d[i]))


def _spline_eval_py(t, t_grid, a, b, c, d):
    """Pure-Python cubic spline eval (no Numba). For scipy.solve_ivp use."""
    i = int(np.searchsorted(t_grid, t)) - 1
    i = max(0, min(i, len(t_grid) - 2))
    dt = t - t_grid[i]
    return a[i] + dt * (b[i] + dt * (c[i] + dt * d[i]))


def build_numba_splines(bg_sol, T_span_bg, model=None):
    n_vars = bg_sol.shape[0]
    splines = []
    for i in range(n_vars):
        s = CubicSpline(T_span_bg, bg_sol[i], bc_type='not-a-knot', extrapolate=True)
        splines.append((s.x, s.c[3].copy(), s.c[2].copy(), s.c[1].copy(), s.c[0].copy()))
    # Pre-compute potential + derivatives for non-native models (e.g. Ezquiaga)
    if model is not None:
        x_bg = bg_sol[0]
        for vals in ([model.f(x) for x in x_bg],
                     [model.dfdx(x) for x in x_bg],
                     [model.d2fdx2(x) for x in x_bg]):
            arr = np.asarray(vals, dtype=float)
            s = CubicSpline(T_span_bg, arr, bc_type='not-a-knot', extrapolate=True)
            splines.append((s.x, s.c[3].copy(), s.c[2].copy(), s.c[1].copy(), s.c[0].copy()))
    return splines


def _extract_potential(model):
    from models.higgs import HiggsModel, FullHiggsModel
    from models.punctuated import PunctuatedInflationModel

    if isinstance(model, HiggsModel):
        alpha = model.alpha
        @njit
        def _f(x): return (1.0 - np.exp(-alpha * x)) ** 2
        @njit
        def _dfdx(x): return 2.0 * alpha * np.exp(-alpha * x) * (1.0 - np.exp(-alpha * x))
        @njit
        def _d2fdx2(x): return 2.0 * alpha**2 * np.exp(-alpha * x) * (2.0 * np.exp(-alpha * x) - 1.0)
        return _f, _dfdx, _d2fdx2

    if isinstance(model, PunctuatedInflationModel):
        m2 = model.m**2; a = model._alpha; l = model.lam; v0 = model.v0
        @njit
        def _f(x): return (0.5*m2*x**2 - a/3.0*x**3 + 0.25*l*x**4) / v0
        @njit
        def _dfdx(x): return (m2*x - a*x**2 + l*x**3) / v0
        @njit
        def _d2fdx2(x): return (m2 - 2.0*a*x + 3.0*l*x**2) / v0
        return _f, _dfdx, _d2fdx2

    return None, None, None


_POTENTIAL_CACHE = {}
_NUMBA_CALL_COUNT = 0

def _get_potential_cached(model):
    """Cached version of _extract_potential. Same model params → same @njit fns."""
    key = (type(model).__name__, getattr(model, 'alpha', None),
           getattr(model, 'm', None), getattr(model, 'lam', None),
           getattr(model, 'xi_val', None), getattr(model, 'v0', None),
           getattr(model, '_alpha', None))
    if key not in _POTENTIAL_CACHE:
        _POTENTIAL_CACHE[key] = _extract_potential(model)
    return _POTENTIAL_CACHE[key]


def make_rhs(f, df, d2f, S, v0):
    """Create a @njit MS ODE RHS with captured potential functions."""
    Si2 = 1.0 / (S*S)
    @njit(cache=True)
    def rhs(vars_8, T, bc, k_rel, ni):
        t_grid = bc[0][0]
        i = np.searchsorted(t_grid, T) - 1
        i = max(0, min(i, len(t_grid) - 2))
        
        x = _spline_eval_at_index(T, t_grid, i, bc[0][1], bc[0][2], bc[0][3], bc[0][4])
        y = _spline_eval_at_index(T, t_grid, i, bc[1][1], bc[1][2], bc[1][3], bc[1][4])
        z = _spline_eval_at_index(T, t_grid, i, bc[2][1], bc[2][2], bc[2][3], bc[2][4])
        n_rel = _spline_eval_at_index(T, t_grid, i, bc[3][1], bc[3][2], bc[3][3], bc[3][4]) - ni
        
        v0_dfdx = v0 * df(x) * Si2
        dydT = -3.0*z*y - v0_dfdx
        k2a2 = k_rel*k_rel * np.exp(-2.0*n_rel)
        m2 = (2.5*y*y + 2.0*y*dydT/z + 2.0*z*z
              + 0.5*y*y*y*y/(z*z) - v0*d2f(x)*Si2 - k2a2)
        v, vT, u, uT, h, hT, g, gT = vars_8
        return np.array([vT, -z*vT+v*m2, uT, -z*uT+u*m2,
                         hT, -z*hT-h*(k2a2-2.0*z*z+0.5*y*y),
                         gT, -z*gT-g*(k2a2-2.0*z*z+0.5*y*y)])
    return rhs


def make_rhs_spline(S, v0):
    """Create a @njit MS ODE RHS reading potential derivatives from bc[4-6]."""
    Si2 = 1.0 / (S*S)
    @njit(cache=True)
    def rhs(vars_8, T, bc, k_rel, ni):
        t_grid = bc[0][0]
        i = np.searchsorted(t_grid, T) - 1
        i = max(0, min(i, len(t_grid) - 2))
        
        x = _spline_eval_at_index(T, t_grid, i, bc[0][1], bc[0][2], bc[0][3], bc[0][4])
        y = _spline_eval_at_index(T, t_grid, i, bc[1][1], bc[1][2], bc[1][3], bc[1][4])
        z = _spline_eval_at_index(T, t_grid, i, bc[2][1], bc[2][2], bc[2][3], bc[2][4])
        n_rel = _spline_eval_at_index(T, t_grid, i, bc[3][1], bc[3][2], bc[3][3], bc[3][4]) - ni
        df_val = _spline_eval_at_index(T, t_grid, i, bc[5][1], bc[5][2], bc[5][3], bc[5][4])
        d2f_val = _spline_eval_at_index(T, t_grid, i, bc[6][1], bc[6][2], bc[6][3], bc[6][4])
        
        v0_dfdx = v0 * df_val * Si2
        dydT = -3.0*z*y - v0_dfdx
        k2a2 = k_rel*k_rel * np.exp(-2.0*n_rel)
        m2 = (2.5*y*y + 2.0*y*dydT/z + 2.0*z*z
              + 0.5*y*y*y*y/(z*z) - v0*d2f_val*Si2 - k2a2)
        v, vT, u, uT, h, hT, g, gT = vars_8
        return np.array([vT, -z*vT+v*m2, uT, -z*uT+u*m2,
                         hT, -z*hT-h*(k2a2-2.0*z*z+0.5*y*y),
                         gT, -z*gT-g*(k2a2-2.0*z*z+0.5*y*y)])
    return rhs


def make_rhs_scipy(S, v0):
    """Python-callable RHS for scipy.solve_ivp. Uses _spline_eval_py (no Numba)."""
    Si2 = 1.0 / (S*S)
    def rhs(t, y, bc, k_rel, ni):
        v, vt, u, ut, h, ht, g, gt = y
        y_bg = _spline_eval_py(t, *bc[1])
        z = _spline_eval_py(t, *bc[2])
        n_rel = _spline_eval_py(t, *bc[3]) - ni
        df = _spline_eval_py(t, *bc[5])
        d2f = _spline_eval_py(t, *bc[6])
        v0_dfdx = v0 * df * Si2
        dydt = -3.0*z*y_bg - v0_dfdx
        k2a2 = k_rel*k_rel * np.exp(-2.0*n_rel)
        m2 = (2.5*y_bg*y_bg + 2.0*y_bg*dydt/z + 2.0*z*z
              + 0.5*y_bg*y_bg*y_bg*y_bg/(z*z) - v0*d2f*Si2 - k2a2)
        return np.array([vt, -z*vt+v*m2, ut, -z*ut+u*m2,
                         ht, -z*ht-h*(k2a2-2.0*z*z+0.5*y_bg*y_bg),
                         gt, -z*gt-g*(k2a2-2.0*z*z+0.5*y_bg*y_bg)])
    return rhs


# ── Dormand-Prince 5(4) Butcher tableau ──────────────────────────────
# Hairer, Norsett, Wanner (1993), Solving ODEs I, Table 5.2
# Standard DP5(4)7M coefficients. These define a 6-stage embedded
# Runge-Kutta method giving 5th-order solution + 4th-order error.
# Never modify these — they define the method the way digits define π.

_INTEGRATOR_CACHE = {}

def _get_integrator(model, S, v0, use_spline=False, method='dp5'):
    key = (type(model).__name__, S, v0, use_spline, method)
    if key in _INTEGRATOR_CACHE:
        return _INTEGRATOR_CACHE[key]

    solvers = {'rk45': 'RK45', 'dop853': 'DOP853', 'radau': 'Radau',
               'bdf': 'BDF', 'lsoda': 'LSODA'}

    def integrate(y0, T_start, T_end, output_t, bc, k_rel, ni,
                  h_init=None, rtol=1e-8, atol=1e-10, max_steps=200000):
        rhs = make_rhs_scipy(S, v0)
        method_name = solvers.get(method, method)
        sol = solve_ivp(
            lambda t, y: rhs(t, y, bc, k_rel, ni),
            [T_start, T_end], y0, method=method_name,
            t_eval=output_t, rtol=rtol, atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"SciPy solve_ivp failed: {sol.message}")
        return sol.y

    _INTEGRATOR_CACHE[key] = integrate
    return integrate


def numba_run_ms(bg_sol, T_span_bg, T_ms, ni, k_code, model, S=None,
                 bg_coefs=None, method='dp5'):
    if S is None:
        S = model.S
    v0 = float(model.v0)
    k_rel = k_code * np.exp(-ni)

    if method == 'dp5':
        from models.higgs import HiggsModel
        if isinstance(model, HiggsModel):
            model_type = 0
            alpha = float(model.alpha)
            if bg_coefs is None or len(bg_coefs) < 4:
                bg_coefs = build_numba_splines(bg_sol, T_span_bg)
        else:
            model_type = 1
            alpha = 0.0
            if bg_coefs is None or len(bg_coefs) < 7:
                bg_coefs = build_numba_splines(bg_sol, T_span_bg, model=model)

        bc = bg_coefs
        zc = bc[2]
        zi = _spline_eval(T_ms[0], *zc)
        yv = zi / k_rel
        vi = 1.0 / np.sqrt(2.0*k_rel)
        y0 = np.array([vi, k_rel/np.sqrt(2.0*k_rel)*yv, yv*vi, -k_rel/np.sqrt(2.0*k_rel)*(1-yv*yv),
                       vi, k_rel/np.sqrt(2.0*k_rel)*yv, yv*vi, -k_rel/np.sqrt(2.0*k_rel)*(1-yv*yv)])

        return _integrate_dp5(y0, T_ms[0], T_ms[-1], T_ms, bc, k_rel, ni, S, v0, model_type, alpha)

    f_nb, dfdx_nb, d2fdx2_nb = _get_potential_cached(model)
    use_spline = f_nb is None
    if use_spline:
        if bg_coefs is None or len(bg_coefs) < 7:
            bg_coefs = build_numba_splines(bg_sol, T_span_bg, model=model)
        integrate = _get_integrator(model, S, v0, use_spline=True, method=method)
    else:
        bg_coefs = bg_coefs if bg_coefs is not None else build_numba_splines(bg_sol, T_span_bg)
        integrate = _get_integrator(model, S, v0, use_spline=False, method=method)
    bc = bg_coefs

    zc = bc[2]
    zi = _spline_eval(T_ms[0], *zc)
    yv = zi / k_rel
    vi = 1.0 / np.sqrt(2.0*k_rel)
    y0 = np.array([vi, k_rel/np.sqrt(2.0*k_rel)*yv, yv*vi, -k_rel/np.sqrt(2.0*k_rel)*(1-yv*yv),
                   vi, k_rel/np.sqrt(2.0*k_rel)*yv, yv*vi, -k_rel/np.sqrt(2.0*k_rel)*(1-yv*yv)])

    return integrate(y0, T_ms[0], T_ms[-1], T_ms, bc, k_rel, ni)


@njit(cache=True)
def _rhs_eval(vars_8, T, bc, k_rel, ni, S, v0, model_type, alpha):
    t_grid = bc[0][0]
    i = np.searchsorted(t_grid, T) - 1
    i = max(0, min(i, len(t_grid) - 2))
    
    x = _spline_eval_at_index(T, t_grid, i, bc[0][1], bc[0][2], bc[0][3], bc[0][4])
    y = _spline_eval_at_index(T, t_grid, i, bc[1][1], bc[1][2], bc[1][3], bc[1][4])
    z = _spline_eval_at_index(T, t_grid, i, bc[2][1], bc[2][2], bc[2][3], bc[2][4])
    n_rel = _spline_eval_at_index(T, t_grid, i, bc[3][1], bc[3][2], bc[3][3], bc[3][4]) - ni
    
    if model_type == 0:
        # Higgs analytical
        df_val = 2.0 * alpha * np.exp(-alpha * x) * (1.0 - np.exp(-alpha * x))
        d2f_val = 2.0 * alpha**2 * np.exp(-alpha * x) * (2.0 * np.exp(-alpha * x) - 1.0)
    else:
        # Spline-based
        df_val = _spline_eval_at_index(T, t_grid, i, bc[5][1], bc[5][2], bc[5][3], bc[5][4])
        d2f_val = _spline_eval_at_index(T, t_grid, i, bc[6][1], bc[6][2], bc[6][3], bc[6][4])
        
    Si2 = 1.0 / (S*S)
    v0_dfdx = v0 * df_val * Si2
    dydT = -3.0*z*y - v0_dfdx
    k2a2 = k_rel*k_rel * np.exp(-2.0*n_rel)
    m2 = (2.5*y*y + 2.0*y*dydT/z + 2.0*z*z
          + 0.5*y*y*y*y/(z*z) - v0*d2f_val*Si2 - k2a2)
    v, vT, u, uT, h, hT, g, gT = vars_8
    return np.array([vT, -z*vT+v*m2, uT, -z*uT+u*m2,
                     hT, -z*hT-h*(k2a2-2.0*z*z+0.5*y*y),
                     gT, -z*gT-g*(k2a2-2.0*z*z+0.5*y*y)])


@njit(cache=True)
def _integrate_dp5(y0, T_start, T_end, output_t, bc, k_rel, ni, S, v0, model_type, alpha,
                   h_init=1e-2, rtol=1e-8, atol=1e-10, max_steps=200000):
    a21=0.2; a31=3/40; a32=9/40
    a41=44/45; a42=-56/15; a43=32/9
    a51=19372/6561; a52=-25360/2187; a53=64448/6561; a54=-212/729
    a61=9017/3168; a62=-355/33; a63=46732/5247; a64=49/176; a65=-5103/18656
    b1=35/384; b2=0; b3=500/1113; b4=125/192; b5=-2187/6784; b6=11/84
    c2=0.2; c3=0.3; c4=0.8; c5=8/9
    d1=5179/57600; d2=0; d3=7571/16695; d4=393/640; d5=-92097/339200; d6=187/2100; d7=1/40

    n_out = len(output_t)
    out = np.zeros((8, n_out))
    y = y0.copy()
    t = T_start
    h = min(h_init, (T_end - T_start) / 10.0)
    step = 0
    err_prev = 0.0
    oi = 0
    yp = y.copy()
    tp = t
    f0 = _rhs_eval(y, t, bc, k_rel, ni, S, v0, model_type, alpha)

    while t < T_end and step < max_steps and oi < n_out:
        if t + h > T_end:
            h = T_end - t

        k1 = f0
        k2 = _rhs_eval(y + h*a21*k1, t + h*c2, bc, k_rel, ni, S, v0, model_type, alpha)
        k3 = _rhs_eval(y + h*(a31*k1 + a32*k2), t + h*c3, bc, k_rel, ni, S, v0, model_type, alpha)
        k4 = _rhs_eval(y + h*(a41*k1 + a42*k2 + a43*k3), t + h*c4, bc, k_rel, ni, S, v0, model_type, alpha)
        k5 = _rhs_eval(y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4), t + h*c5, bc, k_rel, ni, S, v0, model_type, alpha)
        k6 = _rhs_eval(y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5), t + h, bc, k_rel, ni, S, v0, model_type, alpha)

        y_new = y + h*(b1*k1 + b2*k2 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
        f0 = _rhs_eval(y_new, t + h, bc, k_rel, ni, S, v0, model_type, alpha)
        y4 = y + h*(d1*k1 + d2*k2 + d3*k3 + d4*k4 + d5*k5 + d6*k6 + d7*f0)

        ym = np.maximum(np.abs(y_new), np.abs(y))
        sc = atol + rtol * ym
        err = np.sqrt(np.mean(((y_new - y4) / sc) ** 2))

        if err <= 1.0:
            yp, tp = y, t
            y, t = y_new, t + h

            while oi < n_out and output_t[oi] <= t:
                θ = (output_t[oi] - tp) / h
                out[:, oi] = yp + θ * (y - yp)
                oi += 1

            if err > 0:
                fac = ((1/err)**0.14 * (err_prev/err)**0.08) if err_prev > 0 else (1/err)**0.2
                h *= min(5.0, max(0.1, 1.0 * fac))
            err_prev = err
        else:
            if err > 0:
                h *= max(0.1, 0.8 * (1/err)**0.25)

        h = max(h, 1e-8)
        step += 1

    while oi < n_out:
        out[:, oi] = y
        oi += 1
    return out


@njit(parallel=True, cache=True)
def _solve_grid_parallel_numba(
    k_codes, bg_sol_0, bg_sol_1, bg_sol_2, bg_sol_3, T_span_bg, end_idx, k_start_factor,
    bc, S, v0, model_type, alpha
):
    n_modes = len(k_codes)
    P_S_arr = np.zeros(n_modes)
    P_T_arr = np.zeros(n_modes)
    start_idx_arr = np.zeros(n_modes, dtype=np.int64)

    y_end = bg_sol_1[end_idx]
    z_end = bg_sol_2[end_idx]
    n_end = bg_sol_3[end_idx]
    epsH_end = max(y_end**2 / (2.0 * z_end**2), 1e-30)

    log_az = bg_sol_3 + np.log(np.maximum(bg_sol_2, 1e-300))

    for i in prange(n_modes):
        k_code = k_codes[i]

        target_start = np.log(k_code) - np.log(k_start_factor)
        min_val = 1e300
        start_idx = 0
        for idx in range(end_idx):
            val = abs(log_az[idx] - target_start)
            if val < min_val:
                min_val = val
                start_idx = idx

        ni = bg_sol_3[start_idx]
        t_start = T_span_bg[start_idx]
        t_end = T_span_bg[end_idx]

        k_rel = k_code * np.exp(-ni)
        zc = bc[2]

        t_grid = zc[0]
        s_idx = np.searchsorted(t_grid, t_start) - 1
        s_idx = max(0, min(s_idx, len(t_grid) - 2))
        dt = t_start - t_grid[s_idx]
        zi_eval = zc[1][s_idx] + dt * (zc[2][s_idx] + dt * (zc[3][s_idx] + dt * zc[4][s_idx]))

        yv = zi_eval / k_rel
        vi = 1.0 / np.sqrt(2.0*k_rel)
        y0_init = np.array([vi, k_rel/np.sqrt(2.0*k_rel)*yv, yv*vi, -k_rel/np.sqrt(2.0*k_rel)*(1-yv*yv),
                            vi, k_rel/np.sqrt(2.0*k_rel)*yv, yv*vi, -k_rel/np.sqrt(2.0*k_rel)*(1-yv*yv)])

        output_t = np.array([t_end])
        ms_sol = _integrate_dp5(y0_init, t_start, t_end, output_t, bc, k_rel, ni, S, v0, model_type, alpha)

        v_f = ms_sol[0, 0]
        u_f = ms_sol[2, 0]
        h_f = ms_sol[4, 0]
        g_f = ms_sol[6, 0]

        n_rel_end = n_end - ni
        inv_A2 = np.exp(-2.0 * n_rel_end)
        zeta2 = (v_f**2 + u_f**2) * inv_A2 * (S**2) / (2.0 * epsH_end)
        P_S_arr[i] = (k_rel**3 * zeta2) / (2.0 * np.pi**2)
        h2 = (h_f**2 + g_f**2) * inv_A2 * (S**2)
        P_T_arr[i] = 4.0 * (k_rel**3 * h2) / (np.pi**2)
        start_idx_arr[i] = start_idx

    return P_S_arr, P_T_arr, start_idx_arr


def numba_run_ms_grid(bg_sol, T_span_bg, end_idx, k_codes, model, k_start_factor=100.0, S=None, bg_coefs=None, method='dp5'):
    if S is None:
        S = model.S
    
    from models.higgs import HiggsModel
    if isinstance(model, HiggsModel) and method == 'dp5':
        model_type = 0
        alpha = float(model.alpha)
        if bg_coefs is None or len(bg_coefs) < 4:
            bg_coefs = build_numba_splines(bg_sol, T_span_bg)
    else:
        model_type = 1
        alpha = 0.0
        if bg_coefs is None or len(bg_coefs) < 7:
            bg_coefs = build_numba_splines(bg_sol, T_span_bg, model=model)

    v0 = float(model.v0)

    return _solve_grid_parallel_numba(
        k_codes, bg_sol[0], bg_sol[1], bg_sol[2], bg_sol[3], T_span_bg, end_idx, k_start_factor,
        bg_coefs, S, v0, model_type, alpha
    )
