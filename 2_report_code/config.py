from pint import Quantity

# Mesh Dimensions
Lx = Quantity(0.003, "m")
Ly = Quantity(0.003, "m")

xcen = Lx/2

dx = Quantity(0.0001, "m")
dy = Quantity(0.0001, "m")

# Time
t_max = Quantity(0.5, "s")
dt = Quantity(0.00002, "s")

# Material
k = Quantity(6.7, "W/mK")
Q = Quantity(1000, "W")
rho = Quantity(4430, "kg/m^3")
mu = Quantity(0.00476, "Pa.s")
nu = mu/rho

temperature_preheat = Quantity(500, "K")
temperature_melt = Quantity(700, "K")

q = 1

alpha = Quantity(2*(10**-6), "m^2/s")
sigma = Quantity(0.000025, "m")
sigma_d = Quantity(-1.9e-4, "N/m.K")

tol = 1e-5
solver_id = 5

