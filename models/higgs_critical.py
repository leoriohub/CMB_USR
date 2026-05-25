import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, PchipInterpolator

from .base import InflationModel


class CriticalHiggsModel(InflationModel):
    """
    RGE-improved Higgs inflation near the critical point.

    Level 1 — Bezrukov & Shaposhnikov (1403.6078), Eq. (7):
        U(chi) ~= lambda(z') * mu_bar^4 / (4 * xi^2)
        lambda(z) = lambda_0 + b * ln^2(z)
        xi = constant

    Level 2 — Ezquiaga, Garcia-Bellido & Ruiz Morales (1705.04861),
    following Salvio (1712.04477) Prescription I:
        lambda(z') and xi(z') both run with the RG scale.

    In Planck units (M_P=1), with alpha = sqrt(2/3):
        g(x)  = 1 - exp(-alpha * x)
        u(x)  = ln(sqrt(g(x)) / kappa)
        N(u)  = 1 + a * u^2
        D(u)  = (1 + b * u)^2           (b = b_xi / xi_0)
        f(x)  = N(u) / D(u) * g(x)^2

    When b_xi = 0: D = 1, recovers Level 1 (Bezrukov).
    When a = 0 and b_xi = 0: recovers tree-level Higgs.

    Parameters
    ----------
    lambda_0 : float
        Self-coupling minimum value.
        [Bezrukov 1403.6078, Eq. 1]
    xi_0 : float
        Non-minimal coupling at the critical scale.
        [Ezquiaga 1705.04861, Eq. 3]
    a : float
        Running amplitude = b_lambda / lambda_0.
        a = 0 recovers tree-level Higgs (no running).
        Varies ~16 near critical (Bezrukov) or ~5 (Ezquiaga).
        [Bezrukov 1403.6078, Sec. 2]
    b_xi : float
        Running coefficient for xi: xi(z') = xi_0 + b_xi * ln(z').
        b_xi = 0 recovers Level 1 (Bezrukov, constant xi).
        [Ezquiaga 1705.04861, Eq. 3]
    kappa : float
        O(1) factor controlling the RG subtraction scale.
        [Bezrukov 1403.6078, Eq. 7]
    """
    def __init__(self, lambda_0=0.13, xi_0=15000.0, a=0.0, b_xi=0.0,
                 kappa=1.0):
        super().__init__("Critical Higgs Inflation")
        self.alpha = np.sqrt(2 / 3)
        self.lambda_0 = lambda_0
        self.xi_0 = xi_0
        self.a = a
        self.b_xi = b_xi
        self.b = b_xi / xi_0  # running in D = (1 + b*u)^2
        self.kappa = kappa
        self.v0 = self.lambda_0 / (4 * self.xi_0**2)

        self.x0 = 5.70
        self.y0 = -0.10
        self.T_max = 2000.0
        self.bg_steps = 1000

    # ── helpers: g(x) = 1 - exp(-alpha * x) ────────────────────────────────

    def _g(self, x):
        return 1 - np.exp(-self.alpha * x)

    def _dgdx(self, x):
        return self.alpha * np.exp(-self.alpha * x)

    def _d2gdx2(self, x):
        return -(self.alpha**2) * np.exp(-self.alpha * x)

    # ── helpers: u(x) = ln(sqrt(g(x)) / kappa) ────────────────────────────

    def _u(self, x):
        g = np.clip(self._g(x), 1e-100, None)
        return 0.5 * np.log(g) - np.log(self.kappa)

    def _dudx(self, x):
        g = np.clip(self._g(x), 1e-100, None)
        return 0.5 * self._dgdx(x) / g

    def _d2udx2(self, x):
        g = np.clip(self._g(x), 1e-100, None)
        dg = self._dgdx(x)
        d2g = self._d2gdx2(x)
        return 0.5 * (d2g * g - dg**2) / g**2

    # ── N(u) = 1 + a*u^2  (running lambda contribution) ───────────────────

    def _N(self, u):
        return 1 + self.a * u**2

    def _dNdu(self, u):
        return 2 * self.a * u

    def _d2Ndu2(self, u):
        return 2 * self.a

    # ── D(u) = (1 + b*u)^2  (running xi contribution) ─────────────────────

    def _D(self, u):
        return (1 + self.b * u)**2

    def _dDdu(self, u):
        b = self.b
        return 2 * b * (1 + b * u)

    def _d2Ddu2(self, u):
        return 2 * self.b**2

    # ── h(x) = g(x)^2 ──────────────────────────────────────────────────────

    def _h(self, g):
        return g**2

    def _dhdx(self, g, dg):
        return 2 * g * dg

    def _d2hdx2(self, g, dg, d2g):
        return 2 * (dg**2 + g * d2g)

    # ── f(x) = N(u) / D(u) * g(x)^2  and derivatives ─────────────────────

    def f(self, x):
        u = self._u(x)
        g = self._g(x)
        N = self._N(u)
        D = self._D(u)
        h = self._h(g)
        return N / D * h

    def dfdx(self, x):
        u = self._u(x)
        du = self._dudx(x)
        g = self._g(x)
        dg = self._dgdx(x)

        N = self._N(u)
        dN = self._dNdu(u) * du
        D = self._D(u)
        dD = self._dDdu(u) * du
        h = self._h(g)
        dh = self._dhdx(g, dg)

        dR = (dN * D - N * dD) / D**2
        R = N / D
        return dR * h + R * dh

    def d2fdx2(self, x):
        u = self._u(x)
        du = self._dudx(x)
        d2u = self._d2udx2(x)
        g = self._g(x)
        dg = self._dgdx(x)
        d2g = self._d2gdx2(x)

        N = self._N(u)
        dN = self._dNdu(u) * du
        d2N = self._d2Ndu2(u) * du**2 + self._dNdu(u) * d2u
        D = self._D(u)
        dD = self._dDdu(u) * du
        d2D = self._d2Ddu2(u) * du**2 + self._dDdu(u) * d2u
        h = self._h(g)
        dh = self._dhdx(g, dg)
        d2h = self._d2hdx2(g, dg, d2g)

        R = N / D
        dR = (dN * D - N * dD) / D**2
        d2R = ((d2N * D - N * d2D) * D - 2 * (dN * D - N * dD) * dD) / D**3

        return d2R * h + 2 * dR * dh + R * d2h


class EzquiagaCHIModel(InflationModel):
    """
    Critical Higgs Inflation model from Ezquiaga et al. (1705.04861).

    Implements the exact Einstein-frame effective potential (their Eq. 6)
    with the full phi -> chi transformation (their Eq. 5) computed
    numerically at init.

    The potential is defined in terms of x = phi / mu (Jordan frame field
    divided by the critical scale mu). The transformation dchi/dx is
    integrated to give chi(x), then inverted to give x(chi).  The solver
    works with chi (Einstein-frame canonically normalized field).

    Running couplings [Ezquiaga 1705.04861, Eqs. 2-3]:
        lambda(x) = lambda_0 + b_lambda * ln^2(x)
        xi(x)     = xi_0 + b_xi * ln(x)

    Conformal + field redefinition [Eqs. 4-5]:
        dchi/dphi = sqrt(1 + phi^2 (xi + 6(xi + phi*xi'/2)^2)) / (1 + xi*phi^2)

    Effective potential [Eq. 6]:
        V(x) = V_0 * (1 + a*ln^2(x)) * x^4 / (1 + c*(1 + b*ln(x))*x^2)^2

    Parameters
    ----------
    lambda_0 : float
        Self-coupling at the critical scale. [Eq. 2]
    b_lambda : float
        Running coefficient for lambda. [Eq. 2]
    xi_0 : float
        Non-minimal coupling at the critical scale. [Eq. 3]
    b_xi : float
        Running coefficient for xi. [Eq. 3]
    c : float
        Dimensionless parameter c = xi_0 * mu^2 (in Planck units).
        Reference: c = 0.77. [Eq. 6 and surrounding text]
    n_grid : int
        Number of grid points for the phi->chi spline.
    """
    def __init__(self, lambda_0=2.23e-7, b_lambda=1.2e-6,
                 xi_0=7.55, b_xi=11.5, c=0.77, n_grid=5000):
        super().__init__("Ezquiaga CHI")
        self.lambda_0 = lambda_0
        self.b_lambda = b_lambda
        self.xi_0 = xi_0
        self.b_xi = b_xi
        self.a = b_lambda / lambda_0
        self.b = b_xi / xi_0
        self.c = c
        # Critical scale: mu^2 = c / xi_0 (in Planck units)
        self.mu_sq = c / xi_0
        self.mu = np.sqrt(self.mu_sq)

        # Build the phi -> chi transformation spline (stores internal state)
        self._build_splines(n_grid)

        # Normalize so f -> 1 on the CMB plateau (large x).
        # Plateau value: V_inf = V_0 * a / (b*c)^2
        # [Ezquiaga 1705.04861, Eq. after Eq. 6]
        self._V0 = self.lambda_0 * self.mu**4 / 4
        self._V_asympt = self._V0 * self.a / (self.b * self.c)**2
        self.v0 = self._V_asympt

        # Default ICs (overridden per run)
        self.x0 = 6.0
        self.y0 = -1e-4
        self.T_max = 1000.0
        self.bg_steps = 5000

    # ── build phi(chi) spline ─────────────────────────────────────────────

    def _build_splines(self, n_grid):
        """Compute chi(x) from Eq. 5, store inverse x(chi) splines."""
        # x = phi / mu grid: log-spaced from small to large
        x_min = 1e-10
        x_max = 50000.0
        x_grid = np.logspace(np.log10(x_min), np.log10(x_max), n_grid)

        xi = self.xi_0 + self.b_xi * np.log(x_grid)
        # dxi/dphi = b_xi / phi = b_xi / (mu * x)
        # Combined term inside Eq. 5 in terms of x:
        # phi^2 * (xi + 6*(xi + phi*xi'/2)^2)
        # = mu^2*x^2 * (xi + 6*(xi + b_xi/2)^2)
        mu2 = self.mu_sq
        inner = xi + 6 * (xi + self.b_xi / 2)**2
        numerator = np.sqrt(1 + mu2 * x_grid**2 * inner)
        denominator = np.clip(1 + xi * mu2 * x_grid**2, 1e-100, None)
        dchi_dx = mu2 * x_grid * numerator / denominator

        # dchi/dx = dchi/dphi * dphi/dx = dchi/dphi * mu
        # Actually, let me redo this more carefully:
        # dchi/dphi = sqrt(1 + phi^2*(...)) / (1 + xi*phi^2)
        # dchi/dx = dchi/dphi * dphi/dx = dchi/dphi * mu
        # dphi/dx = mu (since x = phi/mu)
        dphi_dx = self.mu
        dchi_dx = numerator / denominator * dphi_dx

        # Integrate: chi(x) = integral dchi/dx dx from 0 to x
        chi_vals = cumulative_trapezoid(dchi_dx, x_grid, initial=0.0)

        # Monotonic-preserving splines (avoids artificial wiggles from CubicSpline)
        self._chi_spline = PchipInterpolator(x_grid, chi_vals,
                                              extrapolate=False)

        # Compute dchi_dx and d2chi_dx2 analytically from the formula
        # rather than from a spline (more accurate at coarser grid)
        self._dchi_dx_analytic = PchipInterpolator(x_grid, dchi_dx,
                                                    extrapolate=False)

        # Build inverse: x(chi) — monotonic preserving
        x_valid = chi_vals > 0
        if not np.any(x_valid):
            raise RuntimeError("chi(x) did not produce positive values")
        self._x_spline = PchipInterpolator(
            chi_vals[x_valid], x_grid[x_valid], extrapolate=False
        )
        self._chi_max = float(chi_vals[-1])

    def _x_of_chi(self, chi):
        """Jordan frame x = phi/mu as a function of Einstein frame chi."""
        chi_a = np.maximum(np.asarray(chi, dtype=float), 0.0)
        chi_a = np.minimum(chi_a, self._chi_max)
        return self._x_spline(chi_a)

    def _dx_dchi(self, chi, x):
        """dx/dchi = 1 / (dchi/dx)."""
        dchi_dx = self._dchi_dx_analytic(np.asarray(x, dtype=float))
        return 1.0 / np.clip(dchi_dx, 1e-100, None)

    def _d2x_dchi2(self, chi, x, dx_dchi):
        """d2x/dchi2 = -d2chi/dx2 / (dchi/dx)^3."""
        dchi_dx = self._dchi_dx_analytic(np.asarray(x, dtype=float))
        d2chi_dx2 = self._compute_d2chi_dx2(np.asarray(x, dtype=float))
        return -d2chi_dx2 / np.clip(dchi_dx**3, 1e-100, None)

    def _compute_d2chi_dx2(self, x):
        """Second derivative d2chi/dx2 from the transformation formula."""
        xs = np.asarray(x, dtype=float)
        xi = self.xi_0 + self.b_xi * np.log(np.clip(xs, 1e-100, None))
        mu2 = self.mu_sq
        # dxi/dx = b_xi / x
        dxi = self.b_xi / xs
        # Term inside sqrt
        S = 1 + mu2 * xs**2 * (xi + 6 * (xi + self.b_xi / 2)**2)
        dS = mu2 * 2 * xs * (xi + 6 * (xi + self.b_xi / 2)**2) \
             + mu2 * xs**2 * (dxi + 12 * (xi + self.b_xi / 2) * dxi)
        F = 1 + xi * mu2 * xs**2
        dF = dxi * mu2 * xs**2 + xi * mu2 * 2 * xs
        sqrt_S = np.sqrt(np.clip(S, 1e-100, None))
        dpsi_dx = sqrt_S / F
        ddpsi = (dS / (2 * sqrt_S) * F - sqrt_S * dF) / F**2
        return ddpsi * self.mu

    # ── potential V(x) — Ezquiaga Eq. 6 ──────────────────────────────────

    def _V(self, x):
        """Dimensionless V(x)/V_0 from Eq. 6."""
        xs = np.clip(np.asarray(x, dtype=float), 1e-30, None)
        lnx = np.log(xs)
        with np.errstate(divide="ignore", invalid="ignore"):
            num = (1 + self.a * lnx**2) * xs**4
            den = (1 + self.c * (1 + self.b * lnx) * xs**2)**2
            result = num / den
        if np.ndim(result) > 0:
            result[~np.isfinite(result)] = 0.0
        elif not np.isfinite(result):
            result = 0.0
        return result

    def _dVdx(self, x):
        """dV/dx (in units of V_0)."""
        xs = np.clip(np.asarray(x, dtype=float), 1e-30, None)
        lnx = np.log(xs)
        # Precompute common terms
        A = 1 + self.a * lnx**2
        B = 1 + self.b * lnx
        D = 1 + self.c * B * xs**2
        # dA/dx = 2*a*ln(x)/x
        dA = 2 * self.a * lnx / xs
        # dB/dx = b/x
        dB = self.b / xs
        # dD/dx = c*(dB*dx*x^2 + B*2x) = c*(b*x + 2*B*x) = c*x*(2*B + b)
        dD = self.c * xs * (2 * B + self.b)

        dnum = dA * xs**4 + A * 4 * xs**3
        dden = 2 * D * dD

        with np.errstate(divide="ignore", invalid="ignore"):
            result = (dnum * D**2 - A * xs**4 * dden) / D**4
        if np.ndim(result) > 0:
            result[~np.isfinite(result)] = 0.0
        elif not np.isfinite(result):
            result = 0.0
        return result

    def _d2Vdx2(self, x):
        """d2V/dx2 (in units of V_0)."""
        xs = np.clip(np.asarray(x, dtype=float), 1e-30, None)
        lnx = np.log(xs)
        x2 = xs**2
        A = 1 + self.a * lnx**2
        B = 1 + self.b * lnx
        D = 1 + self.c * B * x2

        dA = 2 * self.a * lnx / xs
        d2A = 2 * self.a * (1 - lnx) / x2
        dB = self.b / xs
        d2B = -self.b / x2
        dD = self.c * xs * (2 * B + self.b * xs)
        d2D = self.c * (2 * B + 3 * self.b)

        # num = A * x^4
        num = A * xs**4
        dnum = dA * xs**4 + A * 4 * xs**3
        d2num = d2A * xs**4 + 2 * dA * 4 * xs**3 + A * 12 * x2

        # den = D^2
        den = D**2
        dden = 2 * D * dD
        d2den = 2 * (dD**2 + D * d2D)

        with np.errstate(divide="ignore", invalid="ignore"):
            result = ((d2num * den - num * d2den) * den
                      - 2 * (dnum * den - num * dden) * dden) / den**3
        if np.ndim(result) > 0:
            result[~np.isfinite(result)] = 0.0
        elif not np.isfinite(result):
            result = 0.0
        return result

    # ── f(chi) and derivatives ────────────────────────────────────────────

    def _norm(self):
        """Normalization factor so f -> 1 on plateau."""
        return (self.b * self.c)**2 / self.a

    def f(self, chi):
        x = self._x_of_chi(np.asarray(chi, dtype=float))
        return self._V(x) * self._norm()

    def dfdx(self, chi):
        chi_a = np.asarray(chi, dtype=float)
        x = self._x_of_chi(chi_a)
        dx_dchi = self._dx_dchi(chi_a, x)
        dVdx = self._dVdx(x)
        return dVdx * dx_dchi * self._norm()

    def d2fdx2(self, chi):
        chi_a = np.asarray(chi, dtype=float)
        x = self._x_of_chi(chi_a)
        dx_dchi = self._dx_dchi(chi_a, x)
        d2x_dchi2 = self._d2x_dchi2(chi_a, x, dx_dchi)
        dVdx = self._dVdx(x)
        d2Vdx2 = self._d2Vdx2(x)
        return (d2Vdx2 * dx_dchi**2 + dVdx * d2x_dchi2) * self._norm()

    @staticmethod
    def patch_background_solver():
        """Monkey-patch inf_dyn_background: NaN replaced by last valid value."""
        import inf_dyn_background as bg_mod
        if hasattr(bg_mod, "_patched_by_ezquiaga"):
            return
        _original = bg_mod.run_background_simulation
        def _safe(model, T_span):
            bg = _original(model, T_span)
            out = np.copy(bg)
            for row in out:
                mask = np.isfinite(row)
                if not np.all(mask):
                    last_valid = np.where(mask)[0][-1] if np.any(mask) else 0
                    row[~mask] = row[last_valid]
            return out
        bg_mod.run_background_simulation = _safe
        bg_mod._patched_by_ezquiaga = True
