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


