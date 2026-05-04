import numpy as np
from .base import InflationModel


class PunctuatedInflationModel(InflationModel):

    def __init__(self, m=1.1323e-7, lam=3.3299e-15, name="Punctuated Inflation"):
        super().__init__(name)
        self.m = m
        self.lam = lam
        self.sqrt_lam = np.sqrt(lam)
        # Inflection point at φ₀ = m/√λ for n=3
        self.phi0_inflection = m / self.sqrt_lam
        # v₀ = V(φ₀) = m⁴/(12λ)
        self.v0 = m**4 / (12 * lam)
        # Paper ICs: φ_ini = 10, φ̇_ini = 0 (attractor trajectory)
        self.phi0 = 10.0
        self.y0 = 0.0

    def f(self, x):
        return (self.m**2 / 2 * x**2
                - 2 * self.sqrt_lam * self.m / 3 * x**3
                + self.lam / 4 * x**4) / self.v0

    def dfdx(self, x):
        return (self.m**2 * x
                - 2 * self.sqrt_lam * self.m * x**2
                + self.lam * x**3) / self.v0

    def d2fdx2(self, x):
        return (self.m**2
                - 4 * self.sqrt_lam * self.m * x
                + 3 * self.lam * x**2) / self.v0
