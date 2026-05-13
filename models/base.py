import numpy as np

class InflationModel:
    """Base class for inflationary models.

    Unit conventions (all quantities in natural Planck units, M_P = 1):
      ODE variables:
        x = phi            — inflaton field value (M_P=1, so dimensionless)
        y = dx/dT          — field velocity in code time T
        z = H / S          — Hubble rate scaled by code unit factor S
        n = ln(a)          — log scale factor
        T = S * t          — code time (physical time t in Planck units)

      Initial conditions:
        x0   = x(0)        — initial field value (in Planck units, M_P=1)
        y0   = y(0)        — initial field velocity dx/dT (code units)

      Potential:
        f(x)               — dimensionless potential shape
        V = v0 * f(x) / S^2 — potential in code energy units

    So x0 = 6.60 means the field starts at phi = 6.60 in Planck units
    (phi ~ 6.6 M_P), and y0 = -0.736 means dx/dT = -0.736 at T=0.

    S = 5e-5 keeps all ODE variables O(1) during inflation.

    NOTE: For backward compatibility, `phi0` is an alias for `x0`.
    Both access the same underlying value.
    """
    def __init__(self, name, S=5e-5):
        self.name = name
        self.S = S
        self.v0 = None  # Potential scale, defined in subclasses
        self._x0 = None  # Internal: initial field value (phi_0 / M_P)
        self.y0 = None    # Initial field velocity (y0 = dx/dT at T=0)
        self.Ai = 1e-5
        self.T_max = 5000.0
        self.bg_steps = 10000

    @property
    def x0(self):
        """Initial field value x0 = phi_0 / M_P (Planck units, M_P=1)."""
        return self._x0

    @x0.setter
    def x0(self, value):
        self._x0 = value

    @property
    def phi0(self):
        """Deprecated: use x0 instead. Same underlying value."""
        return self._x0

    @phi0.setter
    def phi0(self, value):
        self._x0 = value

    def f(self, x):
        """Dimensionless potential f(x). V = v0 * f(x) / S^2 (code units)."""
        raise NotImplementedError

    def dfdx(self, x):
        """First derivative of potential f(x)."""
        raise NotImplementedError

    def d2fdx2(self, x):
        """Second derivative of potential f(x)."""
        raise NotImplementedError

    def get_initial_conditions(self):
        """Returns [x0, y0, zi, Ni] for the ODE solver.

        Maps to ODE variables [x, y, z, n]:
          x0 = x0    (field value in Planck units, M_P=1)
          y0 = y0    (code unit velocity)
          z0 = zi    (Hubble rate from Friedmann constraint)
          n0 = Ni    (initial log scale factor)
        """
        zi = np.sqrt(self.y0**2/6 + (self.v0 * self.f(self.x0) / (3 * self.S**2)))
        Ni = np.log(self.Ai)
        return [self.x0, self.y0, zi, Ni]
