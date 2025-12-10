import numpy as np
import tempMethods

from numba import jit

from convection import compute_convection_u, compute_convection_v
from laplacian import compute_laplacian

@jit
def fractional_step_method(u, v, psi, T, nu, dt, dx, dy, Nx, Ny, q, solverID, tol, mu,dSigma,tMelt,rho):
    """
    Fractional step method for velocity field computation
    """

    # print("u.shape", u.shape)
    # print("v.shape", v.shape)

    # Compute provisional velocity components
    u_interpolated = 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:]) # (31, 30)
    v_interpolated = 0.25 * (v[:-1, :-1] + v[:-1, 1:] + v[1:, :-1] + v[1:, 1:]) # (30, 31)

    # print("u_interpolated: ", u_interpolated.shape) # (32, 31)
    # print("v_interpolated: ", v_interpolated.shape) # (31, 32)

    conv_u = compute_convection_u(u, v_interpolated, dx, dy) # (32, 31)
    conv_v = compute_convection_v(v, u_interpolated, dx, dy) # (31, 32)

    # print("conv_u: ", conv_u.shape)
    # print("conv_v: ", conv_v.shape)

    lap_u = compute_laplacian(u, dx, dy) # (32, 31)
    lap_v = compute_laplacian(v, dx, dy) # (31, 32)

    # print("lap_u: ", lap_u.shape)
    # print("lap_v: ", lap_v.shape)

    provisional_u = u + dt * (-conv_u + nu * lap_u) # (32, 31)
    provisional_v = v + dt * (-conv_v + nu * lap_v) # (31, 32)

    # print("provisional_u: ", provisional_u.shape)
    # print("provisional_v: ", provisional_v.shape)

    # Solve for pressure

    # Equation 6
    du_dx = (provisional_u[1:, :] - provisional_u[:-1, :]) / dx # (31, 31)
    dv_dy = (provisional_v[:, 1:] - provisional_v[:, :-1]) / dy # (31, 31)

    # print("du_dx", du_dx.shape)
    # print("dv_dy", dv_dy.shape)

    rhs = (rho / dt) * (du_dx + dv_dy) # (31, 31)
    # print("rhs", rhs.shape)

    rhs_internal = rhs[1:-1, 1:-1] # (31, 31)
    # print("rhs", rhs.shape)

    p = psi.copy() # (31, 31)
    # print("p", p.shape)
    factor = 2.0 / (dx * dx) + 2.0 / (dy * dy)

    for _ in range(1000):
        term_1 = (p[2:, 1:-1] + p[:-2, 1:-1]) / (dx * dx) # (29, 29)
        # term_1 = (p[2:, :] + p[:-2, :]) / (dx * dx) # (29, 31)
        # print("term_1", term_1.shape)
        term_2 = (p[1:-1, 2:] + p[1:-1, :-2]) / (dy * dy) # (29, 29)
        # term_2 = (p[:, 2:] + p[:, :-2]) / (dy * dy) # (31, 29)
        # print("term_2", term_2.shape)

        lap = term_1 + term_2 - rhs_internal
        # print("lap", lap.shape)

        psi_new = p.copy()
        psi_new[1:-1, 1:-1] = lap / factor

        # Boundary conditions
        psi_new[0, :] = psi_new[1, :]      # Left
        psi_new[-1, :] = psi_new[-2, :]    # Right
        psi_new[:, 0] = psi_new[:, 1]      # Bottom
        psi_new[:, -1] = psi_new[:, -2]    # Top

        p = psi_new

    # print("p: ", p.shape)

    # Apply correct velocity
    # u = np.zeros_like(provisional_u) # (32, 31)
    u = provisional_u.copy() # (32, 31)
    # print("u", u.shape)
    # v = np.zeros_like(provisional_v) # (31, 32)
    v = provisional_v.copy() # (31, 32)
    # print("v", v.shape)
    
    # Pressure gradient for u
    dpdx = (p[1:, 1:-1] - p[:-1, 1:-1]) / dx # (30, 29)
    # print("dpdx", dpdx.shape)
    u[1:-1, 1:-1] = provisional_u[1:-1, 1:-1] - (dt / rho) * dpdx
    
    # Pressure gradient for v
    dpdy = (p[1:-1, 1:] - p[1:-1, :-1]) / dy # (29, 30)
    # print("dpdy", dpdy.shape)
    v[1:-1, 1:-1] = provisional_v[1:-1, 1:-1] - (dt / rho) * dpdy

    # Marangoni surface velocity
    u[:, -1] = tempMethods.surfaceVelocity(u, T, mu, dx, dt, dSigma, tMelt, rho)

    return p, u, v 


@jit
def CavityFlow_SfV(u, v, w, psi, T, nu, dt, dx, dy, Nx, Ny, q, solverID, tol, mu,dSigma,tMelt,rho):
    """
    The lid-driven cavity flow problem solved using Stream Function-Vorticity formulation

    Input:
    u_init: initial x velocity field, contains boundary conditions
    v_init: initial y velocity field, contains boundary conditions 
    nu: kinematic viscosity
    dt: time step
    dx: grid spacing in x
    dy: grid spacing in y
    Nx: number of grid points in x
    Ny: number of grid points in y
    num_timesteps: number of time steps
    q: upwind scheme parameter
    solverID: Poisson solver ID
    BC: boundary condition ID
    """

    psi_t = np.zeros((Nx+1,Ny+1))   
    w_t = np.zeros((Nx+1,Ny+1))      
    
    # STEP 1: Initialize velocity field
    # ------------------------------------------------------
    u_t = np.zeros((Nx+1,Ny+1))
    v_t = np.zeros((Nx+1,Ny+1))
    boolMask = T > tMelt

   

    # STEP 5 & 6: Solve
    # ------------------------------------------------------

    # STEP 5: Solve Vorticity Transport Equation, we provide the upwind scheme implementation for you

    for i in range(Nx-1):
        for j in range(Ny-1):
            #if T[i+1,j+1] > tMelt:
            if i>0 and i<(Nx-2) and j>0 and j<(Ny-2):
                upwind_x = max(u[i+1,j+1],0)*(w[i-1,j+1] - 3*w[i,j+1] +3*w[i+1,j+1] - w[i+2,j+1])/(3*dx)+min(u[i+1,j+1],0)*(-w[i+3,j+1] + w[i,j+1] -3*w[i+1,j+1] + 3*w[i+2,j+1])/(3*dx)
                upwind_y= max(v[i+1,j+1],0)*(w[i+1,j-1] - 3*w[i+1,j] +3*w[i+1,j+1] - w[i+1,j+2])/(3*dy)+min(v[i+1,j+1],0)*(-w[i+1,j+3] + w[i+1,j] -3*w[i+1,j+1] + 3*w[i+1,j+2])/(3*dy)
            else:
                upwind_x = 0
                upwind_y = 0

                w_t[i+1,j+1] = w[i+1,j+1]-(u[i+1,j+1]*dt/(2*dx))*(w[i+2,j+1]-w[i,j+1])-q*dt*upwind_x-(v[i+1,j+1]*dt/(2*dy))*(w[i+1,j+2]-w[i+1,j])-q*dt*upwind_y
                w_t[i+1,j+1] += nu*dt*((w[i+2,j+1]-2*w[i+1,j+1]+w[i,j+1])/(dx**2)+(w[i+1,j+2]-2*w[i+1,j+1]+w[i+1,j])/(dy**2))
            #else:
                #w_t[i+1,j+1] = 0
                
    
    # STEP 6: Solve Poisson Equation

    psi_t[:,:] = PoissonIterativeSolver(psi[:,:], w_t[:,:], tol, dx, solverID)

    # update velocity BC's for next time step

    u_t[:,-1] = tempMethods.surfaceVelocity(u,T,mu,dx,dt,dSigma,tMelt,rho)
    
    # Compute velocity field on interior nodes

    u_t[1:-1,1:-1] = (psi_t[1:-1,2:] - psi_t[1:-1,0:-2])/2.0/dy
    v_t[1:-1,1:-1] = -(psi_t[2:,1:-1] - psi_t[0:-2,1:-1])/2.0/dx

    #u_t = u_t * boolMask
    #v_t = v_t * boolMask

    # BC's for w

    # x = 0: left wall
    w_t[0,1:-1] = (2*((psi_t[0,1:-1]-psi_t[1,1:-1])/dx**2)-(2/dx)*v_t[0,1:-1]-(u_t[0,2:]-u_t[0,0:-2])/(2*dy))
    # x = Lx: right wall
    w_t[-1,1:-1] = (2*((psi_t[-1,1:-1]-psi_t[-2,1:-1])/dx**2)+(2/dx)*v_t[-1,1:-1]-(u_t[-1,2:]-u_t[-1,0:-2])/(2*dy))
    # y = 0: lower wall
    w_t[1:-1,0] = (2*((psi_t[1:-1,0]-psi_t[1:-1,1])/dy**2)+(2/dy)*u_t[1:-1,0]-(v_t[2:,0]-v_t[0:-2,0])/(2*dx))
    # y = Ly: upper wall
    w_t[1:-1,-1] = (2*((psi_t[1:-1,-1]-psi_t[1:-1,-2])/dy**2)-(2/dy)*u_t[1:-1,-1]-(v_t[2:,-1]-v_t[0:-2,-1])/(2*dx))

    u_t *= boolMask
    v_t *= boolMask
    w_t *= boolMask
    psi_t *= boolMask

    # return w_t, psi_t, u_t, v_t
    return w_t, psi_t, u_t, v_t

@jit
def PoissonIterativeSolver(T, omega, tol, dx, solverID):
    """
    The Poisson Iterative Solver for 2D Temperature distribution

    Input:
    T_dir: Dirichlet boundary condition
    T_neu: Neumann boundary condition
    dx: mesh size in x-direction
    dy: mesh size in y-direction
    Nx: number of points in x-direction
    Ny: number of points in y-direction
    tol: tolerance
    w: SOR relaxation factor
    solverID: 3 for Point Jacobi, 4 for Gauss-Seidel, 5 for SOR
    """
    w = 1.75

    Tref = np.copy(T)
    Nx = len(T[:,0])-1
    Ny = len(T[0,:])-1

    # Initialize error
    error = 1 
    numIts = 0
    
    # Solver

    if solverID == 3:

        while error > tol:

            for j in range(Ny-1):
                for i in range(Nx-1):
                    T[i+1,j+1] = 0.25*(Tref[i,j+1]+Tref[i+2,j+1]+Tref[i+1,j]+Tref[i+1,j+2])

            numIts += 1

            # Compute error
            error = np.linalg.norm(T-Tref)
           
            Tref = np.copy(T)
        
    elif solverID == 4:

        while error > tol:

            for j in range(Ny-1):
                for i in range(Nx-1):
                    T[i+1,j+1] = 0.25*(T[i,j+1]+Tref[i+2,j+1]+T[i+1,j]+Tref[i+1,j+2])

            numIts += 1

            # Compute error
            error = np.linalg.norm(T-Tref)

            Tref = np.copy(T)
        

    elif solverID == 5:

        while error > tol and numIts < 1000:

            for j in range(Ny-1):
                for i in range(Nx-1):
                    TTilde = 0.25*(T[i,j+1]+Tref[i+2,j+1]+T[i+1,j]+Tref[i+1,j+2]) + 0.25*omega[i+1,j+1]*dx**2
                    T[i+1,j+1] = Tref[i+1,j+1] + w*(TTilde - Tref[i+1,j+1])

            numIts += 1

            # Compute error
            error = np.linalg.norm(T-Tref)
            Tref = np.copy(T)
        

    return T

