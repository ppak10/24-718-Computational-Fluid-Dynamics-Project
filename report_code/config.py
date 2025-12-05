from pint import Quantity
from typing import cast
import numpy as np

class Config():
    def __init__(self):
        # Mesh
        self.x_domain = Quantity(3, "mm")  # Lx
        self.y_domain = Quantity(3, "mm")  # Ly

        self.x_center = self.x_domain / 2
        self.y_center = self.y_domain / 2

        self.x_step = Quantity(50, "microns")  # dx
        self.y_step = Quantity(50, "microns")  # dy

        # Time
        self.time_step_max = Quantity(0.5, "s")
        self.time_step = Quantity(0.00002, "s")  # dt = 20 microseconds

        # Material
        self.density = Quantity(4430, "kg/m^3")  # rho
        self.mu = Quantity(0.00476, "Pa.s")  # dynamic viscosity

        self.temperature_initial = Quantity(500, "K")  # Tpre
        self.temperature_melt = Quantity(700, "K")  # tMelt
        
        # Thermal properties
        self.alpha = Quantity(2e-6, "m^2/s")  # thermal diffusivity
        self.k = Quantity(6.7, "W/m/K")  # thermal conductivity
        
        # Heat source (gaussian) 
        self.Q = Quantity(1000, "W")  # total power
        self.sigma = Quantity(200, "microns")  # Gaussian width
        
        # Marangoni
        self.dSigma_dT = Quantity(-1.9e-4, "N/m/K")  # surface tension gradient

