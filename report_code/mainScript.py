# Import required python modules
import numpy as np

# Import additional modules
import tempMethods
import tempPlots
import velocityMethods

from tqdm.rich import tqdm


if __name__ == "__main__":

    # Input Data ##################################################
    Lx = 0.003  #m
    Ly = 0.003  #m
    dx = 0.0001
    dy = 0.0001
    t_max = 2.0  #s
    alpha = 2*(10**-6) #m^2/s
    k = 6.7    #W/mK
    dt = 0.00002
    Q = 1000  #W
    q = 1
    sigma = 0.000025  #m
    xcen = Lx/2
    Tpre = 500

    tMelt = 700 # K
    rho = 4430 # kg/m^3
    mu = 0.00476 # Pa.s
    dSigma = -1.9e-4 # N/m.K

    nu = mu/rho

    tol = 1e-5
    solverID = 5

    ###############################################################

    Nx = int(Lx/dx)
    Ny = int(Ly/dy)
    num_timesteps = int(t_max/dt)

    Neu_BC = np.zeros(Nx+1)
    PDF = np.zeros(Nx+1)

    for i in range(Nx+1):
        PDF[i] = (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-(((i*dx)-xcen)**2)/(2*(sigma**2)))

    for i in range(Nx+1):
        Neu_BC[i] = (Q/dx)*PDF[i]/np.sum(PDF)

    #print(Neu_BC)

    To = np.ones([Nx+1, Ny+1]) * Tpre

    To[:, Ny] = To[:, Ny-1] + (dt * Neu_BC)

    T = np.zeros((Nx+1,Ny+1,num_timesteps+1)) 
    T[:,:,0] = To  

    #print(np.transpose(To))
    
    print('Number of points (x-direction): {0:2d} '.format(Nx+1))
    print('Mesh size (dx): {0:.8f} mm'.format(dx))
    print('Mesh size (dy): {0:.8f} mm'.format(dy))
    print('Mesh size (dt): {0:.8f} mm'.format(dt))
    #print('CFL number: {0:2d} '.format(CFL))


# VELOCITY INITIALIZATION

    psi_t = np.zeros((Nx+1,Ny+1,num_timesteps+1))   

    w_t =np.zeros((Nx+1,Ny+1,num_timesteps+1))      
    
    # STEP 1: Initialize velocity field
    # ------------------------------------------------------
    # u = np.zeros((Nx+1,Ny+1,num_timesteps+1))
    u = np.zeros((Nx+2,Ny+1,num_timesteps+1))
    # v = np.zeros((Nx+1,Ny+1,num_timesteps+1))
    v = np.zeros((Nx+1,Ny+2,num_timesteps+1))

    # STEP 2: Compute w on interior nodes
    # ------------------------------------------------------

    for i in range(Nx - 1):
        for j in range(Ny - 1):
            w_t[i+1,j+1,0] = ((v[i+2,j+1,0]-v[i,j+1,0])/2*dx)-((u[i+1,j+2,0]-u[i+1,j,0])/2*dy)

    # STEP 3: Compute psi on interior nodes
    # ------------------------------------------------------
    
    psi_t[:,:,0] = velocityMethods.PoissonIterativeSolver(psi_t[:,:,0],w_t[:,:,0],tol, dx,solverID)
   
    # STEP 4: Compute BCs for w
    # ------------------------------------------------------

    # Extended trunctation for boundary conditions but maybe should interpolate instead.
    # x = 0: left wall (changed v[0,1:-1,0] to v[0,1:-2,0] to match shapes)
    # w_t[0,1:-1,0] = (2*((psi_t[0,1:-1,0]-psi_t[1,1:-1,0])/dx**2)-(2/dx)*v[0,1:-1,0]-(u[0,2:,0]-u[0,0:-2,0])/(2*dy))
    w_t[0,1:-1,0] = (2*((psi_t[0,1:-1,0]-psi_t[1,1:-1,0])/dx**2)-(2/dx)*v[0,1:-2,0]-(u[0,2:,0]-u[0,0:-2,0])/(2*dy))
    # x = Lx: right wall (changed v[0,1:-1,0] to v[0,1:-2,0] to match shapes)
    # w_t[-1,1:-1,0] = (2*((psi_t[-1,1:-1,0]-psi_t[-2,1:-1,0])/dx**2)+(2/dx)*v[-1,1:-1,0]-(u[-1,2:,0]-u[-1,0:-2,0])/(2*dy))
    w_t[-1,1:-1,0] = (2*((psi_t[-1,1:-1,0]-psi_t[-2,1:-1,0])/dx**2)+(2/dx)*v[-1,1:-2,0]-(u[-1,2:,0]-u[-1,0:-2,0])/(2*dy))
    # y = 0: lower wall (changed u[1:-1,0,0] to u[1:-2,0,0] to match shapes)
    # w_t[1:-1,0,0] = (2*((psi_t[1:-1,0,0]-psi_t[1:-1,1,0])/dy**2)+(2/dy)*u[1:-1,0,0]+(v[2:,0,0]-v[0:-2,0,0])/(2*dx))
    w_t[1:-1,0,0] = (2*((psi_t[1:-1,0,0]-psi_t[1:-1,1,0])/dy**2)+(2/dy)*u[1:-2,0,0]+(v[2:,0,0]-v[0:-2,0,0])/(2*dx))
    # y = Ly: upper wall (changed u[1:-1,0,0] to u[1:-2,0,0] to match shapes)
    # w_t[1:-1,-1,0] = -(2*((psi_t[1:-1,-1,0]-psi_t[1:-1,-2,0])/dy**2)-(2/dy)*u[1:-1,-1,0]+(v[2:,-1,0]-v[0:-2,-1,0])/(2*dx))
    w_t[1:-1,-1,0] = -(2*((psi_t[1:-1,-1,0]-psi_t[1:-1,-2,0])/dy**2)-(2/dy)*u[1:-2,-1,0]+(v[2:,-1,0]-v[0:-2,-1,0])/(2*dx))

# TIME LOOP

# for n in tqdm(range(2)):
for n in tqdm(range(num_timesteps)):
    # w_t[:,:,n+1], psi_t[:,:,n+1], u[:,:,n+1], v[:,:,n+1] = velocityMethods.CavityFlow_SfV(u[:,:,n], v[:,:,n], w_t[:,:,n], psi_t[:,:,n], T[:,:,n], nu, dt, dx, dy, Nx, Ny, q, solverID, tol, mu,dSigma,tMelt,rho)
    psi_t[:,:,n+1], u[:,:,n+1], v[:,:,n+1] = velocityMethods.fractional_step_method(u[:,:,n], v[:,:,n], psi_t[:,:,n], T[:,:,n], nu, dt, dx, dy, Nx, Ny, q, solverID, tol, mu,dSigma,tMelt,rho)
    T[:,:,n+1] = tempMethods.TempFieldTimeStep(u[:,:,n], v[:,:,n], alpha, dt, dx, dy, Nx, Ny, T[:,:,n], Neu_BC, Tpre)

# print(u[:,:,-1])

x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)

u_x = np.linspace(0, Lx, Nx+2)
v_y = np.linspace(0, Ly, Ny+2)

tempPlots.myplots(dt, x, y, T)
tempPlots.VelocityMagnitudes(u[:,:,-1], v[:,:,-1])

print(u[:,:,-1].shape, v[:,:,-1].shape)
u_plot = np.zeros((Nx + 2, Ny + 2))
u_plot[:, :-1] = u[:,:,-1]

v_plot = np.zeros((Nx + 2, Ny + 2))
v_plot[:-1, :] = v[:,:,-1]
print(u_plot.shape, v_plot.shape)

tempPlots.VelocityField(u_x,v_y,u_plot,v_plot,q)
tempPlots.stream_plot(x,y,u[:,:,-1], v[:,:,-1],q)
