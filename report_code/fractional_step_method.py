import numpy as np

from numba import jit
from velocityMethods import PoissonIterativeSolver

@jit
def surfaceVelocity(u,T,mu,dx,dt,dSigma,tMelt,rho):
    Nx = len(u[:,-1])
    u_BC = np.zeros(Nx)
    for i in range(1,Nx-1):
        if T[i,-1] < tMelt:
            u_BC[i] = 0
        else:
            u_BC[i] = u[i, -1] + (dt/(rho*dx))*(dSigma*(T[i+1,-1]-T[i-1,-1])-(mu/dx)*(u[i,-1]-u[i,-2]))

    return u_BC 

@jit
def compute_advection_u(u, v_interpolated, dx, dy):
    """
    Computes the advection term for u-momentum (Equation A13).
    """
    conv = np.zeros_like(u)

    # Internal nodes
    dudx = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx) # (30, 29)
    dudy = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dy) # (30, 29)

    # print("dudx", dudx.shape)
    # print("dudy", dudx.shape)
    # print("u[1:-1, 1:-1]", u[1:-1, 1:-1].shape) # (30, 29)

    u_conv_x = u[1:-1, 1:-1] * dudx # (30, 29)
    # print("u_conv_x", u_conv_x.shape)
    #
    # print("v_interpolated[1:-1, 1:-1]", v_interpolated[1:-1, 1:-1].shape)  # (28, 29)
    # print("v_interpolated[:, 1:-1]", v_interpolated[:, 1:-1].shape)  # (28, 29)
    v_conv_y = v_interpolated[:, 1:-1] * dudy
    # print("v_conv_y", v_conv_y.shape) 
    #
    # print("conv[1:-1, 1:-1]", conv[1:-1, 1:-1].shape)
    conv[1:-1, 1:-1] = u_conv_x + v_conv_y

    return conv # (32, 31)

@jit
def compute_advection_v(v, u_interpolated, dx, dy):
    """
    Computes the advection term for v-momentum (Equation A13).
    """
    conv = np.zeros_like(v)

    # Internal nodes
    dvdx = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dx)
    dvdy = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dy)

    u_conv_x = u_interpolated[1:-1, :] * dvdx
    v_conv_y = v[1:-1, 1:-1] * dvdy

    conv[1:-1, 1:-1] = u_conv_x + v_conv_y

    return conv # (31, 32)

@jit
def compute_diffusion(v, dx, dy):
    diffusion = np.zeros_like(v)

    d2vdx2 = (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx**2
    d2vdy2 = (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / dy**2
    diffusion[1:-1, 1:-1] = d2vdx2 + d2vdy2

    return diffusion

@jit
def fsm(u, v, p, T, nu, dt, dx, dy, mu,dSigma,tMelt,rho):
    """
    Fractional step method for velocity field computation
    """

    T_x = np.zeros_like(u)
    T_y = np.zeros_like(v)

    T_x[1:-1, :] = 0.5 * (T[:-1, :] + T[1:, :])
    T_y[:, 1:-1] = 0.5 * (T[:, :-1] + T[:, 1:])

    boolMask_x = T_x > tMelt
    boolMask_y = T_y > tMelt
    boolMask_p = T > tMelt

    # print("u.shape", u.shape)
    # print("v.shape", v.shape)

    # Compute provisional velocity components
    u_interpolated = 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:]) # (31, 30)
    v_interpolated = 0.25 * (v[:-1, :-1] + v[:-1, 1:] + v[1:, :-1] + v[1:, 1:]) # (30, 31)

    # print("u_interpolated: ", u_interpolated.shape)
    # print("v_interpolated: ", v_interpolated.shape)

    adv_u = compute_advection_u(u, v_interpolated, dx, dy) # (32, 31)
    adv_v = compute_advection_v(v, u_interpolated, dx, dy) # (31, 32)

    # print("conv_u: ", conv_u.shape)
    # print("conv_v: ", conv_v.shape)

    dif_u = compute_diffusion(u, dx, dy) # (32, 31)
    dif_v = compute_diffusion(v, dx, dy) # (31, 32)

    # print("lap_u: ", lap_u.shape)
    # print("lap_v: ", lap_v.shape)

    provisional_u = u + dt * (-adv_u + nu * dif_u) # (32, 31)
    provisional_v = v + dt * (-adv_v + nu * dif_v) # (31, 32)

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

    rhs_internal = rhs[1:-1, 1:-1] # (29, 29)
    # print("rhs", rhs.shape)

    # _p = p.copy() # (31, 31)

    solver_id = 5
    # p = PoissonIterativeSolver(p, rhs, 1e-5, dx, solver_id)

    # print("p", p.shape)
    factor = 2.0 / (dx * dx) + 2.0 / (dy * dy)

    iterations = 0
    error = 1
    tol = 1e-4
    while error > tol and iterations < 2000:
    # while error > tol:
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
        # psi_new[0, :] = psi_new[1, :]      # Left
        # psi_new[-1, :] = psi_new[-2, :]    # Right
        # psi_new[:, 0] = psi_new[:, 1]      # Bottom
        psi_new[:, -1] = psi_new[:, -2]    # Top

        iterations += 1

        error = np.linalg.norm(psi_new - p)

        p = np.copy(psi_new)
    print(error, iterations)

    # for _ in range(1000):
    #     term_1 = (p[2:, 1:-1] + p[:-2, 1:-1]) / (dx * dx) # (29, 29)
    #     # term_1 = (p[2:, :] + p[:-2, :]) / (dx * dx) # (29, 31)
    #     # print("term_1", term_1.shape)
    #     term_2 = (p[1:-1, 2:] + p[1:-1, :-2]) / (dy * dy) # (29, 29)
    #     # term_2 = (p[:, 2:] + p[:, :-2]) / (dy * dy) # (31, 29)
    #     # print("term_2", term_2.shape)
    #
    #     lap = term_1 + term_2 - rhs_internal
    #     # print("lap", lap.shape)
    #
    #     psi_new = p.copy()
    #     psi_new[1:-1, 1:-1] = lap / factor
    #
    #     # Boundary conditions
    #     psi_new[0, :] = psi_new[1, :]      # Left
    #     psi_new[-1, :] = psi_new[-2, :]    # Right
    #     psi_new[:, 0] = psi_new[:, 1]      # Bottom
    #     psi_new[:, -1] = psi_new[:, -2]    # Top
    #
    #     p = psi_new

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
    u[:, -1] = surfaceVelocity(u, T, mu, dx, dt, dSigma, tMelt, rho)

    u = u * boolMask_x
    v = v * boolMask_y
    p = p * boolMask_p

    return p, u, v 
