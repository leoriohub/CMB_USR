import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

from .base import InflationModel


def inflection_parameters(x_c, c, beta=1e-5):
    lnx = np.log(x_c)
    x2 = x_c**2
    denom = 1 + c * x2 + 2 * lnx - 4 * lnx**2
    a = 4 / denom
    b_exact = (
        2 * (1 + c * x2 + 4 * lnx + 2 * c * x2 * lnx) / (c * x2 * denom)
    )
    return a, (1 - beta) * b_exact


class EzquiagaCHIModel(InflationModel):
    """
    Critical Higgs Inflation model from Ezquiaga et al. (1705.04861).

    The potential V(x) (Eq. 6) is defined in terms of x = phi/mu (Jordan frame).
    The ODE integration uses chi (Einstein frame, canonically normalized) to
    avoid the non-canonical kinetic term from the conformal transformation.
    The phi <-> chi mapping is computed numerically at init via the spline.

    Paper convention for plots:
      - Potential: V/V0 vs x = phi/mu (Jordan frame)
      - N evolution: N vs chi (Einstein frame, from ODE)
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
        self.mu_sq = c / xi_0
        self.mu = np.sqrt(self.mu_sq)

        self._build_splines(n_grid)

        self._V0 = self.lambda_0 * self.mu**4 / 4
        self._V_asympt = self._V0 * self.a / (self.b * self.c)**2
        self.v0 = self._V_asympt

        self.x0 = 6.0
        self.y0 = -1e-4
        self.T_max = 1000.0
        self.bg_steps = 5000

    def _build_splines(self, n_grid):
        x_min = 1e-10
        x_max = 50000.0
        x_grid = np.logspace(np.log10(x_min), np.log10(x_max), n_grid)

        xi = self.xi_0 + self.b_xi * np.log(x_grid)
        mu2 = self.mu_sq
        inner = xi + 6 * (xi + self.b_xi / 2)**2
        numerator = np.sqrt(1 + mu2 * x_grid**2 * inner)
        denominator = np.clip(1 + xi * mu2 * x_grid**2, 1e-100, None)
        dchi_dx = numerator / denominator * self.mu

        chi_vals = cumulative_trapezoid(dchi_dx, x_grid, initial=0.0)

        self._chi_spline = PchipInterpolator(x_grid, chi_vals,
                                              extrapolate=False)
        self._dchi_dx_analytic = PchipInterpolator(x_grid, dchi_dx,
                                                    extrapolate=False)

        x_valid = chi_vals > 0
        if not np.any(x_valid):
            raise RuntimeError("chi(x) did not produce positive values")
        self._x_spline = PchipInterpolator(
            chi_vals[x_valid], x_grid[x_valid], extrapolate=False
        )
        self._chi_min = float(chi_vals[x_valid][0])
        self._chi_max = float(chi_vals[-1])

    def _x_of_chi(self, chi):
        chi_a = np.maximum(np.asarray(chi, dtype=float), self._chi_min)
        chi_a = np.minimum(chi_a, self._chi_max)
        return self._x_spline(chi_a)

    def _dx_dchi(self, chi, x):
        dchi_dx = self._dchi_dx_analytic(np.asarray(x, dtype=float))
        return 1.0 / np.clip(dchi_dx, 1e-100, None)

    def _d2x_dchi2(self, chi, x, dx_dchi):
        dchi_dx = self._dchi_dx_analytic(np.asarray(x, dtype=float))
        d2chi_dx2 = self._compute_d2chi_dx2(np.asarray(x, dtype=float))
        return -d2chi_dx2 / np.clip(dchi_dx**3, 1e-100, None)

    def _compute_d2chi_dx2(self, x):
        xs = np.asarray(x, dtype=float)
        xi = self.xi_0 + self.b_xi * np.log(np.clip(xs, 1e-100, None))
        mu2 = self.mu_sq
        dxi = self.b_xi / xs
        S = 1 + mu2 * xs**2 * (xi + 6 * (xi + self.b_xi / 2)**2)
        dS = mu2 * 2 * xs * (xi + 6 * (xi + self.b_xi / 2)**2) \
             + mu2 * xs**2 * (dxi + 12 * (xi + self.b_xi / 2) * dxi)
        F = 1 + xi * mu2 * xs**2
        dF = dxi * mu2 * xs**2 + xi * mu2 * 2 * xs
        sqrt_S = np.sqrt(np.clip(S, 1e-100, None))
        dpsi_dx = sqrt_S / F
        ddpsi = (dS / (2 * sqrt_S) * F - sqrt_S * dF) / F**2
        return ddpsi * self.mu

    def _V(self, x):
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
        xs = np.clip(np.asarray(x, dtype=float), 1e-30, None)
        lnx = np.log(xs)
        A = 1 + self.a * lnx**2
        B = 1 + self.b * lnx
        D = 1 + self.c * B * xs**2
        dA = 2 * self.a * lnx / xs
        dB = self.b / xs
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
        dD = self.c * xs * (2 * B + self.b)
        d2D = self.c * (2 * B + 3 * self.b)

        num = A * xs**4
        dnum = dA * xs**4 + A * 4 * xs**3
        d2num = d2A * xs**4 + 2 * dA * 4 * xs**3 + A * 12 * x2

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

    def _norm(self):
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
