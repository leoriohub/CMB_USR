import numpy as np

class InflationModel:
    """Base class for inflationary models."""
    def __init__(self, name, S=5e-5):
        self.name = name
        self.S = S
        self.v0 = None # Potential scale, to be defined in subclasses
        self.phi0 = None
        self.yi = None
        self.Ai = 1e-5

    def f(self, x):
        """Dimensionless potential f(x)"""
        raise NotImplementedError

    def dfdx(self, x):
        """First derivative of potential  x"""
        raise NotImplementedError

    def d2fdx2(self, x):
        """Second derivative of potential  x"""
        raise NotImplementedError

    def get_initial_conditions(self):
        """Returns [phi0, yi, zi, Ai]"""
        # zi depends on potential, calculating here ensures consistency
        zi = np.sqrt(self.yi**2/6 + (self.v0 * self.f(self.phi0) / (3 * self.S**2)))
        # Return Ni (log scale factor) instead of Ai
        # Default Ai was 1e-5. Ni = ln(1e-5) approx -11.51
        Ni = np.log(self.Ai)
        return [self.phi0, self.yi, zi, Ni]
