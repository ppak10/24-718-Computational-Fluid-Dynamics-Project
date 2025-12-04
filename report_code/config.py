from pint import Quantity
from typing import cast

class Config():
    def __init__(self):
        # Mesh
        self.x_domain = Quantity(3000, "microns") # Lx
        self.y_domain = Quantity(3000, "microns") # Ly

        self.x_center = self.x_domain / 2
        self.y_center = self.y_domain / 2

        self.x_step = Quantity(100, "microns") # dx
        self.y_step = Quantity(100, "microns") # dy

        # Time
        self.time_step_max = Quantity(0.5, "s") # t_max
        self.time_step = Quantity(0.00002, "s") # dt

        # Material
        # k = Quantity(6.7, "W/mK")
        # Q = Quantity(1000, "W")

        self.density = Quantity(4430, "kg/m^3")
        self.mu = Quantity(0.00476, "Pa.s")

        self.temperature_initial = Quantity(500, "K")
        self.temperature_melt = Quantity(700, "K")
        #
        # q = 1
        #
        # alpha = Quantity(2*(10**-6), "m^2/s")
        # sigma = Quantity(0.000025, "m")
        # sigma_d = Quantity(-1.9e-4, "N/m.K")
        #
        # tol = 1e-5
        # solver_id = 5
        #
