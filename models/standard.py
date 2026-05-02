from .base import InflationModel

class QuadraticModel(InflationModel):
    def __init__(self):
        super().__init__("Quadratic Inflation")
        M = 5.9e-6
        self.v0 = 0.5 * M**2
        self.phi0 = 17.5 # Approx 60 e-folds
        self.yi = -0.05 # Slow-roll initial velocity


    def f(self, x):
        return x**2

    def dfdx(self, x):
        return 2 * x

    def d2fdx2(self, x):
        return 2
