import numpy as np
from .base import InflationModel


class PunctuatedInflationModel(InflationModel):

    def __init__(self, m=1.1323e-7, lam=3.3299e-15, phi0=None, name="Punctuated Inflation"):
        super().__init__(name)
        self.m = m
        self.lam = lam
        self.sqrt_lam = np.sqrt(lam)

        # Exact perfect inflection point: V'(φ) = φ(m - √λ φ)²
        # α = 2√λ m gives V'(φ) = φ(m - √λ φ)² which is never negative,
        # vanishing only at φ = m/√λ with no local minimum.
        self._alpha = 2 * self.sqrt_lam * m
        self.phi0_inflection = m / self.sqrt_lam
        if phi0 is not None:
            self.phi0_inflection = float(phi0)

        # v₀ = V(φ₀) for normalization
        x0 = self.phi0_inflection
        self.v0 = (m**2/2 * x0**2 - self._alpha/3 * x0**3 + lam/4 * x0**4)

        # Paper ICs: φ_ini = 12, φ̇_ini via slow-roll approximation
        self.x0 = 12.0
        self.y0 = 0.0

        # Punctuated inflation spans ~96 e-folds, needs long integration
        self.T_max = 500000.0
        self.bg_steps = 100000

    def f(self, x):
        return (self.m**2 / 2 * x**2
                - self._alpha / 3 * x**3
                + self.lam / 4 * x**4) / self.v0

    def dfdx(self, x):
        return (self.m**2 * x
                - self._alpha * x**2
                + self.lam * x**3) / self.v0

    def d2fdx2(self, x):
        return (self.m**2
                - 2 * self._alpha * x
                + 3 * self.lam * x**2) / self.v0

    def get_jit_funcs(self):
        from numba import njit
        m2 = self.m**2
        a = self._alpha
        l = self.lam
        v0 = self.v0
        @njit(cache=True)
        def _f(x): return (0.5*m2*x**2 - a/3.0*x**3 + 0.25*l*x**4) / v0
        @njit(cache=True)
        def _dfdx(x): return (m2*x - a*x**2 + l*x**3) / v0
        @njit(cache=True)
        def _d2fdx2(x): return (m2 - 2.0*a*x + 3.0*l*x**2) / v0
        return _f, _dfdx, _d2fdx2
