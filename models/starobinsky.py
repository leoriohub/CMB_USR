"""Starobinsky R^2 inflation model in Einstein frame.

Potential: V(phi) = 3/4 M^2 (1 - exp(-sqrt(2/3) phi))^2

The mass scale M (scalaron mass) is determined by the CMB normalization
P_S(k_pivot) = A_s. With the pipeline's A_s normalization, the exact
value of M is not critical — set v0 as a nominal parameter.
"""
import numpy as np
from .base import InflationModel


class StarobinskyModel(InflationModel):
    """Starobinsky R^2 inflation in Einstein frame.

    Parameters
    ----------
    v0 : float
        Potential normalization V0 = 3/4 M^2, where M is the scalaron mass.
        Default 4.5e-11 gives roughly the right P_S amplitude.
    """

    def __init__(self, v0=4.5e-11):
        super().__init__("Starobinsky R^2")
        self.alpha = np.sqrt(2.0 / 3.0)
        self.v0 = v0

        # Default ICs: start on the plateau, slow-roll attractor
        self.x0 = 5.5
        self.y0 = -1e-4

        # Integration range (shorter than Higgs — no USR transient)
        self.T_max = 500.0
        self.bg_steps = 10000

    def f(self, x):
        return (1.0 - np.exp(-self.alpha * x)) ** 2

    def dfdx(self, x):
        a = self.alpha
        ex = np.exp(-a * x)
        return 2.0 * a * ex * (1.0 - ex)

    def d2fdx2(self, x):
        a = self.alpha
        ex = np.exp(-a * x)
        return 2.0 * a * a * ex * (2.0 * ex - 1.0)

    def get_jit_funcs(self):
        from numba import njit
        alpha = self.alpha
        @njit(cache=True)
        def _f(x): return (1.0 - np.exp(-alpha * x)) ** 2
        @njit(cache=True)
        def _dfdx(x): return 2.0 * alpha * np.exp(-alpha * x) * (1.0 - np.exp(-alpha * x))
        @njit(cache=True)
        def _d2fdx2(x): return 2.0 * alpha**2 * np.exp(-alpha * x) * (2.0 * np.exp(-alpha * x) - 1.0)
        return _f, _dfdx, _d2fdx2
