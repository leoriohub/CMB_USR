import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline

from .base import InflationModel


class HiggsModel(InflationModel):

    def __init__(self, lam=0.1, xi=1000.0):
        super().__init__("Higgs Inflation")
        self.alpha = np.sqrt(2/3)
        self.lam = lam
        self.xi_val = xi
   
        self.v0 = self.lam / (4 * self.xi_val**2)
        
        # Default Initial Conditions (USR Exploration Defaults)
        self.phi0 = 5.8
        self.yi = -0.01

    def f(self, x):
        return (1 - np.exp(-self.alpha * x))**2

    def dfdx(self, x):
        return 2 * self.alpha * np.exp(-self.alpha * x) * (1 - np.exp(-self.alpha * x))

    def d2fdx2(self, x):
        return 2 * self.alpha**2 * np.exp(-self.alpha * x) * (2 * np.exp(-self.alpha * x) - 1)


class FullHiggsModel(InflationModel):
    """
    Exact Higgs Inflation potential without the high-field approximation.
    Integrates the exact conformal inversion numerically to retain absolute precision
    throughout reheating down to the true h^4 minimum.
    """
    def __init__(self, lam=0.1, xi=1000.0, v_vev=0.0):
        super().__init__("Full Higgs Inflation")
        
        self.lam = lam
        self.xi_val = xi
        self.v_vev = v_vev # Default to 0, since v/M_P is tiny (~10^-15)
        
        # We scale v0 exactly like the approximated model so the plateau height is ~1.0
        self.v0 = self.lam / (4 * self.xi_val**2) 
        
        # Standard Initial Conditions
        self.phi0 = 5.5
        self.yi = -1

        # Precompute the inverse transformation grid: psi(x) where psi = h/M_P, x = chi/M_P
        self.psi_max = 100.0 / np.sqrt(self.xi_val) # Enough to reach way past the plateau
        self.psi_grid = np.linspace(0, self.psi_max, 5000)
        
        def dx_dpsi_deriv(x, psi): # ODE system for x(psi)
            if psi < 0: psi = 0 
            num = np.sqrt(1 + self.xi_val * psi**2 * (1 + 6 * self.xi_val))
            den = 1 + self.xi_val * psi**2
            return [num / den]

        # Solve x(psi)
        x_sol = odeint(dx_dpsi_deriv, [0.0], self.psi_grid)
        self.x_grid = x_sol[:, 0]
        
        # We need the inverse: psi(x)
        self.psi_spline = CubicSpline(self.x_grid, self.psi_grid)

    def _get_psi(self, x):
        if np.isscalar(x):
            return self.psi_spline(np.clip(x, 0, self.x_grid[-1]))
        return self.psi_spline(np.clip(x, 0, self.x_grid[-1]))

    def _get_dpsi_dx(self, psi):
        # Exact mathematical inverse: dpsi/dx = 1 / (dx/dpsi)
        num = 1 + self.xi_val * psi**2
        den = np.sqrt(1 + self.xi_val * psi**2 * (1 + 6 * self.xi_val))
        return num / den

    def _get_d2psi_dx2(self, psi, dpsi_dx):
        # d/dx (dpsi/dx) = d/dpsi (dpsi/dx) * dpsi/dx
        u = 1 + self.xi_val * psi**2
        v2 = 1 + self.xi_val * psi**2 * (1 + 6 * self.xi_val)
        v = np.sqrt(v2)
        
        du = 2 * self.xi_val * psi
        dv = (self.xi_val * psi * (1 + 6 * self.xi_val)) / v
        
        dG_dpsi = (du * v - u * dv) / v2
        return dG_dpsi * dpsi_dx

    def f(self, x):
        psi = self._get_psi(x)
        # f(psi) = [ xi * (psi^2 - v^2) / (1 + xi*psi^2) ]^2
        num = self.xi_val * (psi**2 - self.v_vev**2)
        den = 1 + self.xi_val * psi**2
        return (num / den)**2

    def dfdx(self, x):
        psi = self._get_psi(x)
        dpsi = self._get_dpsi_dx(psi)
        
        num = self.xi_val * (psi**2 - self.v_vev**2)
        den = 1 + self.xi_val * psi**2
        g = num / den
        
        num_dg = 2 * self.xi_val * psi * (1 + self.xi_val * self.v_vev**2)
        den_dg = (1 + self.xi_val * psi**2)**2
        dg_dpsi = num_dg / den_dg
        
        df_dpsi = 2 * g * dg_dpsi
        return df_dpsi * dpsi

    def d2fdx2(self, x):
        psi = self._get_psi(x)
        dpsi = self._get_dpsi_dx(psi)
        d2psi = self._get_d2psi_dx2(psi, dpsi)
        
        num = self.xi_val * (psi**2 - self.v_vev**2)
        den = 1 + self.xi_val * psi**2
        g = num / den
        
        num_dg = 2 * self.xi_val * psi * (1 + self.xi_val * self.v_vev**2)
        den_dg = (1 + self.xi_val * psi**2)**2
        dg_dpsi = num_dg / den_dg
        
        term1 = 2 * self.xi_val * (1 + self.xi_val * self.v_vev**2)
        term2 = 1 + self.xi_val * psi**2
        term3 = 2 * self.xi_val * psi
        
        dnum_dg = term1
        dden_dg = 2 * term2 * term3
        
        d2g_dpsi2 = (dnum_dg * den_dg - num_dg * dden_dg) / (den_dg**2)
        d2f_dpsi2 = 2 * (dg_dpsi**2 + g * d2g_dpsi2)
        
        return d2f_dpsi2 * (dpsi**2) + (2 * g * dg_dpsi) * d2psi
