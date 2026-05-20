"""
Numba-accelerated Mukhanov-Sasaki solver.

Provides numba_run_ms(), a drop-in replacement for
inf_dyn_MS_full.run_ms_simulation() with ~2x pipeline speedup.
"""
import numpy as np
from scipy.interpolate import CubicSpline

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
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


def build_numba_splines(bg_sol, T_span_bg):
    n_vars = bg_sol.shape[0]
    splines = []
    for i in range(n_vars):
        s = CubicSpline(T_span_bg, bg_sol[i], bc_type='not-a-knot', extrapolate=True)
        # scipy stores [d, c, b, a] for S(x)=a + b*dx + c*dx^2 + d*dx^3
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
    @njit
    def rhs(vars_8, T, bc, k_rel, ni):
        x = _spline_eval(T, *bc[0])
        y = _spline_eval(T, *bc[1])
        z = _spline_eval(T, *bc[2])
        n_rel = _spline_eval(T, *bc[3]) - ni
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


# ── Dormand-Prince 5(4) Butcher tableau ──────────────────────────────
# Hairer, Norsett, Wanner (1993), Solving ODEs I, Table 5.2
# Standard DP5(4)7M coefficients. These define a 6-stage embedded
# Runge-Kutta method giving 5th-order solution + 4th-order error.
# Never modify these — they define the method the way digits define π.

_INTEGRATOR_CACHE = {}

def _get_integrator(model, S, v0):
    key = type(model).__name__
    if key in _INTEGRATOR_CACHE:
        return _INTEGRATOR_CACHE[key]

    f, df, d2f = _get_potential_cached(model)
    rhs = make_rhs(f, df, d2f, S, v0)

    @njit(cache=True)
    def integrate(y0, T_start, T_end, output_t, bc, k_rel, ni,
                  h_init=1e-2, rtol=1e-8, atol=1e-10, max_steps=200000):
        # ── DP5 coefficients (flat, Numba-compatible) ────────────
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
        f0 = rhs(y, t, bc, k_rel, ni)

        while t < T_end and step < max_steps and oi < n_out:
            if t + h > T_end:
                h = T_end - t

            k1 = f0
            k2 = rhs(y + h*a21*k1, t + h*c2, bc, k_rel, ni)
            k3 = rhs(y + h*(a31*k1 + a32*k2), t + h*c3, bc, k_rel, ni)
            k4 = rhs(y + h*(a41*k1 + a42*k2 + a43*k3), t + h*c4, bc, k_rel, ni)
            k5 = rhs(y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4), t + h*c5, bc, k_rel, ni)
            k6 = rhs(y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5), t + h, bc, k_rel, ni)

            y_new = y + h*(b1*k1 + b2*k2 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
            f0 = rhs(y_new, t + h, bc, k_rel, ni)
            y4 = y + h*(d1*k1 + d2*k2 + d3*k3 + d4*k4 + d5*k5 + d6*k6 + d7*f0)

            # ── Error control ──────────────────────────────────────
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

    _INTEGRATOR_CACHE[key] = integrate
    return integrate


def numba_run_ms(bg_sol, T_span_bg, T_ms, ni, k_code, model, S=5e-5, bg_coefs=None):
    f_nb, dfdx_nb, d2fdx2_nb = _get_potential_cached(model)
    if f_nb is None:
        raise NotImplementedError(
            f"Numba not supported for {type(model).__name__}. "
            f"Use Python solver (inf_dyn_MS_full.run_ms_simulation)."
        )
    v0 = model.v0; k_rel = k_code * np.exp(-ni)
    bc = bg_coefs if bg_coefs is not None else build_numba_splines(bg_sol, T_span_bg)

    zc = bc[2]
    zi = _spline_eval(T_ms[0], *zc)
    yv = zi / k_rel
    vi = 1.0 / np.sqrt(2.0*k_rel)
    y0 = np.array([vi, k_rel/np.sqrt(2.0*k_rel)*yv, yv*vi, -k_rel/np.sqrt(2.0*k_rel)*(1-yv*yv),
                   vi, k_rel/np.sqrt(2.0*k_rel)*yv, yv*vi, -k_rel/np.sqrt(2.0*k_rel)*(1-yv*yv)])

    integrate = _get_integrator(model, S, v0)
    return integrate(y0, T_ms[0], T_ms[-1], T_ms, bc, k_rel, ni)
