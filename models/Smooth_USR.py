import numpy as np
from scipy.special import hyperu, hyp1f1, gamma
from scipy.integrate import cumulative_trapezoid, odeint
from scipy.interpolate import CubicSpline

from .base import InflationModel

class SmoothUSRTransitionModel(InflationModel):
    """
    Numerically reconstructed potential from the analytical smooth SR-USR-SR model.
    Based on arXiv:2603.17465v1
    """
    def __init__(self, alpha=22.63, mu=2.0294, eps_sr1=1e-6, H0=1.0):
        super().__init__(f"Smooth USR (alpha={alpha}, mu={mu})")
        
        self.alpha = alpha
        self.mu = mu
        self.eps_sr1 = eps_sr1
        self.H0 = H0
        self.v0 = H0**2
        self.S = 5e-5 
        
        # Restore original high grid density for exact numerical precision
        N_vals_sr = np.linspace(-35.0, 0, 2000, endpoint=False)
        N_vals_usr = np.linspace(0, 15, 1500)
        
        q_sq = 9/4 + alpha - mu**2
        self.q_sq = q_sq
        self.q = np.sqrt(q_sq)
        q = self.q
        
        def W(kappa, mu_val, z):
            return np.exp(-z/2) * (z**(mu_val + 0.5)) * hyperu(0.5 + mu_val - kappa, 1 + 2*mu_val, z)
            
        def M(kappa, mu_val, z):
            return np.exp(-z/2) * (z**(mu_val + 0.5)) * hyp1f1(0.5 + mu_val - kappa, 1 + 2*mu_val, z)
            
        k0 = alpha / (2*q)
        k1 = k0 + 1
        
        W0 = W(k0, mu, 2*q)
        M0 = M(k0, mu, 2*q)
        W1 = W(k1, mu, 2*q)
        M1 = M(k1, mu, 2*q)
        
        denom = (alpha + q + 2*mu*q)*M1*W0 + 2*q*M0*W1
        B1_prime = ((alpha - 2*q - 2*q**2)*M0 - (alpha + q + 2*mu*q)*M1) / denom
        B2_prime = -((alpha - 2*q - 2*q**2)*W0 + 2*q*W1) / denom
        
        z_arg = 2 * q * np.exp(-N_vals_usr)
        W_val = W(k0, mu, z_arg)
        M_val = M(k0, mu, z_arg)
        
        Z_scaled = B1_prime * W_val + B2_prime * M_val
        eps1_usr = eps_sr1 * np.exp(-2*N_vals_usr) * (Z_scaled)**2
        
        N_vals = np.concatenate((N_vals_sr, N_vals_usr))
        eps1_vals = np.concatenate((np.full_like(N_vals_sr, eps_sr1), eps1_usr))
        
        dphi_dN = np.sqrt(2 * eps1_vals)
        int_dphi = cumulative_trapezoid(dphi_dN, x=N_vals, initial=0.0)

        # We want phi to increase with N and be zero at N=0
        idx_N0 = np.argmin(np.abs(N_vals - 0.0))
        phi_vals = int_dphi - int_dphi[idx_N0]
        
        # Calculate exact H(N) anchored at N=0 so H(tau_*) = H0
        int_eps1 = cumulative_trapezoid(eps1_vals, x=N_vals, initial=0.0)
        int_eps1 = int_eps1 - int_eps1[idx_N0]
        H_vals = H0 * np.exp(-int_eps1)

        V_vals = H_vals**2 * (3 - eps1_vals) 

        self.N_grid = N_vals
        self.eps1_grid = eps1_vals

        # Interp functions need strictly increasing x.
        # Remove any non-strictly increasing points (eps1 = 0)
        phi_uniq, uniq_idx = np.unique(phi_vals, return_index=True)
        V_uniq = V_vals[uniq_idx]
        
        self.phi_grid = phi_uniq
        self.V_grid = V_uniq
        
        self.v_spline = CubicSpline(self.phi_grid, self.V_grid)
        self.dv_spline = self.v_spline.derivative(nu=1)
        self.d2v_spline = self.v_spline.derivative(nu=2)
        
        # Set initial conditions for integration appropriately
        # Suppose we want to start 2 efolds before transition (N=-2).
        idx_i = np.argmin(np.abs(N_vals - (-15.0)))
        self.phi0 = phi_vals[idx_i]
        
        H_initial_val = H_vals[idx_i]
        self.yi = dphi_dN[idx_i] * (H_initial_val / self.S)
    
    def f(self, x):
        return self.v_spline(x) / self.v0

    def dfdx(self, x):
        return self.dv_spline(x) / self.v0

    def d2fdx2(self, x):
        return self.d2v_spline(x) / self.v0

    def _laguerre_non_integer(self, n, b, z):
        """Generalized Laguerre Poly for non-integer n using 1F1."""
        # L_n^b(z) = Gamma(n+b+1)/(Gamma(n+1)*Gamma(b+1)) * 1F1(-n, b+1, z)
        denom = (gamma(n + 1) * gamma(b + 1))
        # Safely handle zero/tiny values in denominator
        if np.any(np.abs(denom) < 1e-300):
            denom = denom + 1e-300
        coeff = gamma(n + b + 1) / denom
        return coeff * hyp1f1(-n, b + 1, z)

    def _get_G_functions(self, N):
        """Implements G1-G4 from Appendix A of arXiv:2603.17465."""
        from scipy.special import hyperu
        z = 2 * self.q * np.exp(-N)
        
        # Hypergeometric arguments based on paper mapping
        arg_u1 = 0.5 - (self.alpha / (2 * self.q)) + self.mu
        arg_u2 = 1.5 - (self.alpha / (2 * self.q)) + self.mu
        
        # G1 (Eq A2)
        term1 = (np.exp(N) * (3 + 2 * self.mu) - 2 * self.q) * hyperu(arg_u1, 1 + 2 * self.mu, z)
        term2 = 2 * (self.alpha - self.q - 2 * self.q * self.mu) * hyperu(arg_u2, 2 + 2 * self.mu, z)
        G1 = term1 + term2
        
        # G2 (Eq A3)
        n1 = self.alpha / (2 * self.q) - self.mu - 1.5
        n2 = self.alpha / (2 * self.q) - self.mu - 0.5
        G2 = 4 * self.q * self._laguerre_non_integer(n1, 1 + 2 * self.mu, z) - \
             (np.exp(N) * (2 * self.mu + 3) - 2 * self.q) * self._laguerre_non_integer(n2, 2 * self.mu, z)
        
        # G3 (Eq A4)
        G3 = hyperu(arg_u1, 1 + 2 * self.mu, z)
        
        # G4 (Eq A5)
        n3 = self.alpha / (2 * self.q) - 0.5 - self.mu
        G4 = self._laguerre_non_integer(n3, 2 * self.mu, z)
        
        return G1, G2, G3, G4

    def _precalculate_transition_coeffs(self):
        """Pre-calculates G functions at N=0 to speed up analytical formulas."""
        self._G0 = self._get_G_functions(0.0)

    def epsilon2_analytical(self, N):
        """Exact epsilon2 solution from Eq A1."""
        G1, G2, G3, G4 = self._get_G_functions(N)
        # Use pre-calculated transition constants
        if not hasattr(self, '_G0'):
            self._precalculate_transition_coeffs()
        G1_0, G2_0, G3_0, G4_0 = self._G0
        
        num = G1_0 * G2 - G2_0 * G1
        den = G1_0 * G4 + G2_0 * G3
        
        return np.exp(-N) * (num / den)

    def get_initial_conditions(self):
        # Multiply f by v0 since the potential function is now dimensionless
        zi = np.sqrt(self.yi**2/6 + (self.v0 * self.f(self.phi0))/(3*self.S**2))
        return self.phi0, self.yi, zi, -15.0
