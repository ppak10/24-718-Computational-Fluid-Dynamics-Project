Lx = 0.003  #m
Ly = 0.003  #m

# dx = 0.0001
# dy = 0.0001

# dx = 0.00005
# dy = 0.00005

dx = 0.00002
dy = 0.00002

# t_max = 2.0  #s
# t_max = 0.5  #s
t_max = 1.0 #s
# t_max = 0.1  #s
alpha = 2*(10**-6) #m^2/s
k = 6.7    #W/mK

save_timestep = 10000

# dt = 0.00002
# dt = 0.00001
dt = 0.000004

Q = 1000  #W
q = 1
sigma = 0.000025  #m
xcen = Lx/2
Tpre = 500

tMelt = 700 # K
rho = 4430 # kg/m^3
mu = 0.00476 # Pa.s
dSigma = -1.9e-4 # N/m.K
# dSigma = 1.9e-4 # N/m.K reverse flow

nu = mu/rho

tol = 1e-5
solverID = 5
