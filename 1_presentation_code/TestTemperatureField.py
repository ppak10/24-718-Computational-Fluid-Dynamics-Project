


# Import required python modules
import numpy as np
import time

# Import additional modules
import tempMethods
import tempPlots



if __name__ == "__main__":

    # Input Data ##################################################
    Lx = 0.003  #m
    Ly = 0.003  #m
    dx = 0.0001
    dy = 0.0001
    t_max = 20  #s
    alpha = 2*(10**-6) #m^2/s
    k = 6.7    #W/mK
    dt = 0.0002
    q = 1000   #W
    sigma = 0.0001  #m
    xcen = Lx/2

    tMelt = 1900 # K
    rho = 4430 # kg/m^3
    mu = 4760 # Pa.s
    dSigma = -1.9e-4 # N/m.K

    ###############################################################

    Nx = int(Lx/dx)
    Ny = int(Ly/dy)
    num_timesteps = int(t_max/dt)

    Neu_BC = np.zeros(Nx+1)

    for i in range(Nx+1):
        Neu_BC[i] = (q/k) * (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-(((i*dx)-xcen)**2)/(2*(sigma**2)))

    print(Neu_BC)

    To = np.ones([Nx+1, Ny+1]) * 300

    To[:, Ny] = To[:, Ny-1] + (dt * Neu_BC)

    print(np.transpose(To))
    
    print('Number of points (x-direction): {0:2d} '.format(Nx+1))
    print('Mesh size (dx): {0:.8f} mm'.format(dx))
    print('Mesh size (dx): {0:.8f} mm'.format(dy))
    print('Mesh size (dx): {0:.8f} mm'.format(dt))
    #print('CFL number: {0:2d} '.format(CFL))

    u = np.ones([Nx+1, Ny+1]) * 0
    v = np.ones([Nx+1, Ny+1]) * 0

    # Upwind (advection) and Central differences (diffusion)

    start = time.time()
    TempTestField = tempMethods.TempField(u, v, alpha, dt, dx, dy, Nx, Ny, To, Neu_BC, num_timesteps)
    end = time.time()
    print('time elpased: {0:.8f} s'.format(end - start))

    # Plot results

    print(np.transpose(TempTestField[:,:,-1]))

    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)

    tempPlots.myplots(dt, x, y, TempTestField)