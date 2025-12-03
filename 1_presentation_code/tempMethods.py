

import numpy as np
from numba import jit

@jit
def TempFieldTimeStep(u, v, alpha, dt, dx, dy, Nx, Ny, T, Neu_BC, Tpre):
    """
    The Upwind Central Difference Method for 2D advection-diffusion equation

    Input:
    u: x component of advection velocity [m/s]
    v: y component of advection velocity [m/s]
    dt: time step
    dx: mesh size in x-direction
    dy: mesh size in y-direction
    uo: initial condition
    u_dir: Dirichlet boundary condition
    u_neu: Neumann boundary condition
    Nx: number of points in x-direction
    Ny: number of points in y-direction
    num_timesteps: number of time steps
    """

    beta_x = alpha*dt/(dx**2)
    beta_y = alpha*dt/(dy**2)

    cfl_x = np.max(u)*dt/dx
    cfl_y = np.max(v)*dt/dy

    const = cfl_x + cfl_y + 2*beta_x + 2*beta_y

   # if const < 1:
        #print(f"Thermal stability violation, stability = {const}")

    # Initialize T_ref
    # ** add your code here **

    T_ref = T.copy()


    # Impose boundary conditions
    # ** add your code here **
    
    for l in range(1, Nx):
        for j in range(1, Ny):

            # Compute u @ n+1
            nu_x = u[l,j]*dt/dx
            nu_y = v[l,j]*dt/dy

            T[l, j] = (1-nu_x-nu_y-(2*beta_x)-(2*beta_y))*T_ref[l, j]+beta_x*T_ref[l+1, j]+(nu_x+beta_x)*T_ref[l-1, j]+beta_y*T_ref[l, j+1]+(nu_y+beta_y)*T_ref[l, j-1]

    #T[:, 0] = T[:, 1]
    #T[-1, 1:-1] = T[-2, 1:-1]
    #T[0, 1:-1] = T[1, 1:-1]

    T[:, 0] = Tpre
    T[-1, 1:-1] = Tpre
    T[0, 1:-1] = Tpre

    T[:, -1] = T[:, -2] + (dy * Neu_BC)

    return T

@jit
def TempField(u, v, alpha, dt, dx, dy, Nx, Ny, To, Neu_BC, num_timesteps):
    """
    The Upwind Central Difference Method for 2D advection-diffusion equation

    Input:
    u: x component of advection velocity [m/s]
    v: y component of advection velocity [m/s]
    dt: time step
    dx: mesh size in x-direction
    dy: mesh size in y-direction
    uo: initial condition
    u_dir: Dirichlet boundary condition
    u_neu: Neumann boundary condition
    Nx: number of points in x-direction
    Ny: number of points in y-direction
    num_timesteps: number of time steps
    """

    # Initialize T
    # ** add your code here **

    T = np.zeros((Nx+1, Ny+1, num_timesteps+1))

    # Initial conditions
    # ** add your code here **

    T[:, :, 0] = To

    # Time loop
    
    for n in range(num_timesteps):
        T[:, :, n+1] = TempFieldTimeStep(u, v, alpha, dt, dx, dy, Nx, Ny, T[:, :, n], Neu_BC)
    
    #print(u[:,:,num_timesteps])
    return T

@jit
def surfaceVelocity(u,T,mu,dx,dt,dSigma,tMelt,rho):
    Nx = len(u[:,-1])
    u_BC = np.zeros(Nx)
    for i in range(1,Nx-1):
        if T[i,-1] < tMelt:
            u_BC[i] = 0
        else:
            u_BC[i] = u[i, -1] + (dt/(rho*dx))*(dSigma*(T[i+1,-1]-T[i-1,-1])-(mu/dx)*(u[i,-1]-u[i,-2]))
    #print(u_BC)
    return u_BC 