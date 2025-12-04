

import numpy as np
from numba import jit

@jit
def TempFieldTimeStep(u, v, alpha, dt, dx, dy, Nx, Ny, T, Neu_BC):
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

    T[:, 0] = T[:, 1]
    T[Nx, 1:Ny] = T[Nx-1, 1:Ny]
    T[0, 1:Ny] = T[1, 1:Ny]
    T[:, Ny] = T[:, Ny-1] + (dt * Neu_BC)

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
