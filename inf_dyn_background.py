#########################################################################################################
#########################################################################################################
#
# Please refer to <arXiv link> for explaination of variables and instructions for using the code
#
#########################################################################################################
#########################################################################################################

import numpy as np
from scipy.integrate import odeint



#########################################################################################################
# The model of inflation is defined in this section
#########################################################################################################


def run_background_simulation(model, T_span):
    """
    Integrates the background field equations (phi, phi_dot, H) over physical time T.
    
    The system maps the specific potential geometry V(phi) into the
    exact dynamic background evaluation, avoiding analytical slow-roll approximations.
    """
    phi0, y0, zi, Ni = model.get_initial_conditions()
    v0 = model.v0
    S = model.S

    def sys(var, T):
        [x, y, z, n] = var
        dxdT = y
        dydT = -3*z*y - v0*model.dfdx(x)/S**2 
        dzdT = -0.5*y**2
        dndT = z # d(ln A)/dT = H = z
        return [dxdT, dydT, dzdT, dndT]

    # Using tighter tolerances for general stability
    sol = odeint(sys, [phi0, y0, zi, Ni], T_span, rtol=1e-10, atol=1e-12, mxstep=1000000)
    return np.transpose(sol)

def get_derived_quantities(sol_data, model):
    """
    Extracts physical observables and slow-roll parameters from the exact background integration.
    Computes epsH and etaH dynamically to accurately track transient non-slow-roll phases (like USR).
    """
    x, y, z, n = sol_data
    v0 = model.v0
    S = model.S
    Ni = model.get_initial_conditions()[3] # Get Ni

    N = n - Ni
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Slow-roll parameters
        # Exact dynamical parameters
        epsH = y**2 / (2 * z**2)
        yz = y * z
        yz_safe = np.where(yz >= 0, np.maximum(yz, 1e-30), np.minimum(yz, -1e-30))
        etaH = -(-3*z*y - v0*model.dfdx(x)/S**2)/yz_safe
        
        #Slow roll approximations
        # Observables
        ns = 1 + 2*etaH - 4*epsH
        r = 16*epsH
        Ps = (S*z)**2 / (8 * np.pi**2 * epsH)
        Pt = 2*(S*z)**2 / (np.pi**2)
    
    # Scale Mapping
    # aH = A*z = exp(n)*z.  Be careful with exp(n) if n is large.
    # Usually we don't need aH explicitly as a float if it's huge.
    # But if returned, it might overflow.
    # Let's return log_aH = n + log(z)
    
    return {
        'N': N,
        'epsH': epsH,
        'etaH': etaH,
        'ns': ns,
        'r': r,
        'Ps': Ps,
        'Pt': Pt,
        'n': n # Return log scale factor
    }

