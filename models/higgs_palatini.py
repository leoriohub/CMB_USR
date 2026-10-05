"""Tree-level Palatini Higgs inflation in reduced Planck units.

The ODE field x is the canonical Einstein-frame field chi, not the
Jordan-frame Higgs h. For constant couplings and negligible Higgs vev,
h = sinh(sqrt(xi) * chi) / sqrt(xi) and
U(chi) = lambda / (4 * xi**2) * tanh(sqrt(xi) * chi)**4.

Reference: Bauer and Demir, https://arxiv.org/abs/0803.2664.
"""

import numpy as np

from .base import InflationModel


class PalatiniHiggsModel(InflationModel):
    """Canonical Palatini potential with positive constant lambda and xi.

    lam is the inflation-scale quartic coupling in V_J = lam * h**4 / 4.
    Defaults are nominal, not an amplitude-calibrated configuration.
    Initial conditions approximate the slow-roll attractor 70 e-folds
    before epsilon_V = 1. Integration accuracy must be checked separately,
    particularly at large xi; S controls code time T = S * t.
    """

    def __init__(self, lam=0.13, xi=1e9, S=5e-5):
        super().__init__("Palatini Higgs Inflation", S=S)
        self.lam = lam
        self.xi_val = xi
        self.sqrt_xi = np.sqrt(xi)
        self.v0 = self.lam / (4 * self.xi_val**2)

        # Slow-roll N = (sinh^2(u0) - sinh^2(u_end)) / (8 xi).
        n_initial = 70.0
        u_end = 0.5 * np.arcsinh(np.sqrt(32.0 * self.xi_val))
        self.x0 = np.arcsinh(np.sqrt(
            8.0 * self.xi_val * n_initial + np.sinh(u_end)**2
        )) / self.sqrt_xi
        z_sr = np.sqrt(self.v0 * self.f(self.x0) / 3.0) / self.S
        self.y0 = -self.v0 * self.dfdx(self.x0) / (3.0 * self.S**2 * z_sr)
        self.T_max = 2.0 * n_initial / z_sr
        self.bg_steps = 50000

    def f(self, x):
        return np.tanh(self.sqrt_xi * x)**4

    def dfdx(self, x):
        u = self.sqrt_xi * x
        t = np.tanh(u)
        q = np.exp(-2.0 * np.abs(u))
        sech2 = 4.0 * q / (1.0 + q)**2
        return 4.0 * self.sqrt_xi * t**3 * sech2

    def d2fdx2(self, x):
        u = self.sqrt_xi * x
        t = np.tanh(u)
        q = np.exp(-2.0 * np.abs(u))
        sech2 = 4.0 * q / (1.0 + q)**2
        return 4.0 * self.xi_val * t**2 * sech2 * (3.0 - 5.0 * t**2)
