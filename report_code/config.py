from pint import Quantity

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
        self.viscosity = Quantity(0.00476, "Pa.s")
        self.thermal_diffusivity = Quantity(2e-6, "m^2/s")
        self.thermal_conductivity = Quantity(6.7, "W/m/K")

        # Temperature
        self.temperature_initial = Quantity(500, "K")  # Tpre
        self.temperature_melt = Quantity(700, "K")  # tMelt
        
        # Heat source (gaussian) 
        self.Q = Quantity(1000, "W")  # total power
        self.sigma = Quantity(200, "microns")  # Gaussian width
        
        # Marangoni
        self.dSigma_dT = Quantity(-1.9e-4, "N/m/K")  # surface tension gradient

