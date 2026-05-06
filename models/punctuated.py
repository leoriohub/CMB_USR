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
        self.phi0 = 12.0
        self.y0 = 0.0

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
