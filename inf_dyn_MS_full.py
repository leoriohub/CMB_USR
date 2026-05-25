#########################################################################################################
#########################################################################################################
#
# Please refer to <arXiv link> for explaination of variables and instructions for using the code
#
#########################################################################################################
#########################################################################################################

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

# execution block
if __name__ == "__main__":
    import os
    import sys
    
    # It is required to execute the script 'inf_dyn_background.py' and save the data in a text file before this script can be executed
    # We input the initial conditions and horizon exit for various scales from the background data
    # Please change the filename in this line if you have saved the data with a different name or at a different location
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/inf_bg_data.txt'))
    data = np.loadtxt(data_path) # row: T,N,Ne,x,y,z,aH,epsH,etaH,meff,Ps,Pt. 12 columns

    # This function returns the index of the row in the data file where the number of e-folds before the 
    #   end of inflation (Ne) attains a certain value specified by the argument 
    def i(Ne):
        return np.max(np.where(data[:,2]>=Ne))



#########################################################################################################
# The model of inflation is defined in this section
#########################################################################################################

# This term defines one unit of time 
S = 5e-5 


# parameters used in the potential function
M = 5.9e-6 
v0 = 0.5*M**2


# dimensionless potential function and its derivatives
def f(x):
    return x**2

def dfdx(x):
    return 2*x

def d2fdx2(x):
    return 2




#########################################################################################################
# In this section we set the initial conditions for both background as well as fluctuations
# The background data file is used to input initial conditions for background quantities
# We solve the dynamical equations, including the dimensionless Mukhanov-Sasaki equation using the
#   function scipy.integrate.odeint 
#########################################################################################################


### The dynamical variables are defined as follows:
#
# background:
#
# x : dimensionless field value [ \phi / m_p ]
# y : dimensionless field velocity [ dx/dT or \dot\phi / (m_p ^2 * S) ]
# A : dimensionless scale factor [ a * m_p * S ]
# z : dimensionless hubble parameter [ H / (S * m_p) ]
#
#
# fluctuations:
#
# v : real part of the Mukhanov-Sasaki variable [ v_k ] for scalar fluctuations [ \zeta_k ]
# u : imaginary part of the Mukhanov-Sasaki variable for scalar fluctuations
# h : real part of the Mukhanov-Sasaki variable for tensor fluctuations [ h_k ]
# g : imaginary part of the Mukhanov-Sasaki variable for tensor fluctuations


def build_bg_interpolators(bg_sol, T_span):
    """
    Build interpolation functions from a pre-computed background solution.
    
    Returns functions that give x(T), y(T), z(T), n(T) at any time,
    avoiding the need to re-integrate the background for each k-mode.
    """
    x_interp = interp1d(T_span, bg_sol[0], kind='cubic', fill_value='extrapolate')
    y_interp = interp1d(T_span, bg_sol[1], kind='cubic', fill_value='extrapolate')
    z_interp = interp1d(T_span, bg_sol[2], kind='cubic', fill_value='extrapolate')
    n_interp = interp1d(T_span, bg_sol[3], kind='cubic', fill_value='extrapolate')
    return x_interp, y_interp, z_interp, n_interp


def run_ms_simulation(bg_interp, ni, T_span, k, model):
    """
    Integrates the Mukhanov-Sasaki perturbation equations for a specific comoving k-mode.
    
    The background trajectory is provided via interpolation functions (from build_bg_interpolators),
    so only the 8 perturbation variables are integrated — the background is NOT re-solved.
    
    To avoid floating-point overflow during long integrations, both the scale factor 
    and the k-mode are defined relative to the integration start time. 
    Exact finite-time Bunch-Davies initial conditions are applied to avoid numerical artifacts.
    """
    v0 = model.v0
    S = model.S
    x_interp, y_interp, z_interp, n_interp = bg_interp
    
    # Scale everything to a relative scale factor A_rel where A_rel(start) = 1.
    # This prevents e^(-1000) underflow or e^1000 overflow when N_total > 300.
    k_rel = k * np.exp(-ni)
    
    # y = a*H/k = A_rel*H/k_rel
    # At start A_rel = 1, so y = zi/k_rel
    zi = float(z_interp(T_span[0]))
    y = zi / k_rel
    
    # Exact complex phase correction for being at finite time
    vi = (1/np.sqrt(2*k_rel))
    ui = y * vi
    
    # Exact physical time derivatives (converted from conformal tau)
    v_Ti = k_rel / np.sqrt(2*k_rel) * y
    u_Ti = -k_rel / np.sqrt(2*k_rel) * (1 - y**2)
    
    # Tensors share identical vacuum physics
    hi = vi
    gi = ui
    h_Ti = v_Ti
    g_Ti = u_Ti

    def sys(var, T):
        [v, v_T, u, u_T, h, h_T, g, g_T] = var
        
        # Background from interpolation (no re-integration)
        x = float(x_interp(T))
        y_idx = float(y_interp(T))
        z = float(z_interp(T))
        n_rel = float(n_interp(T)) - ni
        
        # Precompute shared terms
        v0_dfdx = v0 * model.dfdx(x) / S**2
        dydT = -3*z*y_idx - v0_dfdx
        
        # safe evaluation of k/a to avoid overflow
        k2_invA2 = k_rel**2 * np.exp(-2*n_rel)

        # Shared effective mass for scalar fluctuations
        m2_scalar = (2.5*y_idx**2 + 2*y_idx*dydT/z + 2*z**2
                     + 0.5*y_idx**4/z**2 - v0*model.d2fdx2(x)/S**2 - k2_invA2)

        # scalar fluctuations
        dvdT = v_T
        dv_TdT = -z*v_T + v*m2_scalar
        dudT = u_T
        du_TdT = -z*u_T + u*m2_scalar
        
        # tensor fluctuations
        dhdT = h_T
        dh_TdT = -z*h_T - h*(k2_invA2 - 2*z**2 + 0.5*y_idx**2)
        dgdT = g_T
        dg_TdT = -z*g_T - g*(k2_invA2 - 2*z**2 + 0.5*y_idx**2)

        return [dvdT, dv_TdT, dudT, du_TdT, dhdT, dh_TdT, dgdT, dg_TdT]

    # Initialize n_rel = 0
    sol = odeint(sys, [vi,v_Ti,ui,u_Ti,hi,h_Ti,gi,g_Ti], T_span, rtol=1e-12, atol=1e-14, mxstep=5000000)
    return np.transpose(sol)


def run_ms_simulation_full(bg_interp, ni, T_span, k, model):
    """
    Legacy wrapper that also returns the background trajectory alongside perturbations.
    
    Use run_ms_simulation() for production — this exists for backward compatibility
    with code that reads background from the MS solution array.
    """
    x_interp, y_interp, z_interp, n_interp = bg_interp
    
    # Reconstruct background arrays from interpolation
    bg_x = x_interp(T_span)
    bg_y = y_interp(T_span)
    bg_z = z_interp(T_span)
    bg_n = n_interp(T_span) - ni
    
    # Run perturbation-only integration
    pert_sol = run_ms_simulation(bg_interp, ni, T_span, k, model)
    
    # Stack: [x, y, z, n_rel, v, v_T, u, u_T, h, h_T, g, g_T]
    return np.vstack([bg_x, bg_y, bg_z, bg_n, pert_sol])


def get_ms_derived_quantities(sol_data, model, k, ni):
    """
    Calculates power spectra for the simulated mode.
    Note: k passed here must be k_code, ni scales it to k_rel matching A_rel variables!
    
    sol_data can be either the 12-var full output (legacy) or the 8-var perturbation-only
    output from run_ms_simulation. If 8-var, background must be provided via bg_interp.
    """
    v0 = model.v0
    S = model.S
    
    k_rel = k * np.exp(-ni)
    
    if sol_data.shape[0] == 12:
        # Legacy full solution: extract background from solution
        x, y_idx, z, n_rel, v, v_T, u, u_T, h, h_T, g, g_T = sol_data
    else:
        # Perturbation-only solution: background not in array
        # This path requires external background — caller should use get_ms_derived_quantities_with_bg
        raise ValueError(
            "8-variable solution requires background. Use get_ms_derived_quantities_with_bg() "
            "or pass the 12-variable output from run_ms_simulation_full()."
        )
    
    with np.errstate(divide='ignore', invalid='ignore'):
        epsH = y_idx**2 / (2 * z**2)
        
        # Power spectra using relative k and A
        inv_A2 = np.exp(-2*n_rel)
        zeta2 = (v**2 + u**2) * inv_A2 * (S**2) / (2*epsH)
        P_S = (k_rel**3 * zeta2)/(2*np.pi**2)
        h2 = (h**2 + g**2) * inv_A2 * (S**2)
        P_T = 4*(k_rel**3 * h2)/(np.pi**2)
    
    return {
        'P_S': P_S,
        'P_T': P_T,
        'aHk': np.exp(n_rel)*z/k_rel
    }


def get_ms_derived_quantities_with_bg(pert_sol, bg_interp, T_span, model, k, ni):
    """
    Calculates power spectra from perturbation-only solution + background interpolation.
    
    Use this with run_ms_simulation() output (8-variable) instead of the legacy 12-variable path.
    """
    v0 = model.v0
    S = model.S
    x_interp, y_interp, z_interp, n_interp = bg_interp
    
    k_rel = k * np.exp(-ni)
    
    v, v_T, u, u_T, h, h_T, g, g_T = pert_sol
    x = x_interp(T_span)
    y_idx = y_interp(T_span)
    z = z_interp(T_span)
    n_rel = n_interp(T_span) - ni
    
    with np.errstate(divide='ignore', invalid='ignore'):
        epsH = y_idx**2 / (2 * z**2)
        
        inv_A2 = np.exp(-2*n_rel)
        zeta2 = (v**2 + u**2) * inv_A2 * (S**2) / (2*epsH)
        P_S = (k_rel**3 * zeta2)/(2*np.pi**2)
        h2 = (h**2 + g**2) * inv_A2 * (S**2)
        P_T = 4*(k_rel**3 * h2)/(np.pi**2)
    
    return {
        'P_S': P_S,
        'P_T': P_T,
        'aHk': np.exp(n_rel)*z/k_rel
    }

# execution block
if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import inf_dyn_background as bg_solver
    
    if not os.path.exists('data/inf_bg_data.txt'):
        print("Required background data file missing. Run inf_dyn_background.py first.")
    else:
        data = np.loadtxt('data/inf_bg_data.txt')
        def i_idx(Ne_val):
            return np.max(np.where(data[:,2]>=Ne_val))

        Nk = 60 
        k_val = data[i_idx(Nk), 6]
        
        # Initial conditions for this mode
        xi_ms = data[i_idx(Nk+5), 3]
        yi_ms = data[i_idx(Nk+5), 4]
        from models import QuadraticModel
        
        # Instantiate real OOP model for stand-alone test
        test_model = QuadraticModel()
        
        zi_ms = np.sqrt(yi_ms**2/6 + (test_model.v0*test_model.f(xi_ms)/(3*test_model.S**2)))
        ni_ms = np.log(1e-3) + 77.4859 - (Nk+5)
        
        T_span = np.linspace(0, 200, 10000)
        
        # Solve background once, then build interpolators
        bg_sol = bg_solver.run_background_simulation(test_model, T_span)
        bg_interp = build_bg_interpolators(bg_sol, T_span)
        
        # Solve only perturbations using interpolated background
        sol_data = run_ms_simulation_full(bg_interp, ni_ms, T_span, k_val, test_model)
        derived = get_ms_derived_quantities(sol_data, test_model, k_val, ni_ms)
        
        plt.plot(derived['aHk'], derived['P_S'], 'r')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

#########################################################################################################
#########################################################################################################
