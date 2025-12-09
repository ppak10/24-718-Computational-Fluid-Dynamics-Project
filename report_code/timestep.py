import jax.numpy as jnp

from functools import partial
from jax import jit, lax

from .convection import compute_convection_u, compute_convection_v
from .laplacian import compute_laplacian_u, compute_laplacian_v

MAX_ITER_POISSON = 1000

@jit
def compute_provisional_velocity(u, v, nu, dx, dy, dt):
    """
    Computes the velocity components without pressure gradient.
    """

    Nx = u.shape[0] - 1
    Ny = v.shape[1] - 1

    # u*
    # Interpolate v to u-location
    v_at_u = interpolate_velocity(v, "u", Nx, Ny)

    # Compute convection and diffusion
    conv_u = compute_convection_u(u, v_at_u, dx, dy)
    lap_u = compute_laplacian_u(u, dx, dy)

    # Provisional velocity (explicit)
    u_star = u + dt * (-conv_u + nu * lap_u)

    # v*
    # Interpolate u to v-location
    u_at_v = interpolate_velocity(u, "v", Nx, Ny)

    # Compute convection and diffusion
    conv_v = compute_convection_v(v, u_at_v, dx, dy)
    lap_v = compute_laplacian_v(v, dx, dy)

    # Provisional velocity (explicit)
    v_star = v + dt * (-conv_v + nu * lap_v)

    return u_star, v_star

@jit
def solve_pressure(u_star, v_star, p_init, rho, dx, dy, dt):
    """
    Solves the poisson equation for pressure with jacobi iteration

    Args:
        u_star: (Nx, Ny + 1)
        v_star: (Nx + 1, Ny)
        p_init: (Nx + 1, Ny + 1)
        rho: density
        dx: grid spacing
        dy: grid spacing
        dt: time step
        max_iterations
    """

    div_u = (u_star[1:, :] - u_star[:-1, :]) / dx
    div_v = (v_star[:, 1:] - v_star[:, :-1]) / dy

    # rhs = rhs.at[1:-1, 1:-1].set((rho / dt) * (div_u + div_v))
    rhs_internal = (rho / dt) * (div_u + div_v)

    dx2 = dx * dx
    dy2 = dy * dy

    factor = 2.0 / dx2 + 2.0 / dy2

    def jacobi_iteration(p, _):
        """
        Single jacobi iteration for internal pressure cells
        """

        p_new = p

        print(((p[2:, 1:-1] + p[:-2, 1:-1]) / dx2 + (p[1:-1, 2:] + p[1:-1, :-2]) / dy2).shape, rhs_internal.shape)


        laplacian = ((p[2:, 1:-1] + p[:-2, 1:-1]) / dx2 + 
                     (p[1:-1, 2:] + p[1:-1, :-2]) / dy2) - rhs_internal

        p_new = p_new.at[1:-1, 1:-1].set(laplacian / factor)

        
        # Boundary conditions
        p_new = p_new.at[0, :].set(p_new[1, :])      # Left
        p_new = p_new.at[-1, :].set(p_new[-2, :])    # Right
        p_new = p_new.at[:, 0].set(p_new[:, 1])      # Bottom
        p_new = p_new.at[:, -1].set(p_new[:, -2])    # Top
        
        return p_new, None
    
    p_final, _ = lax.scan(jacobi_iteration, p_init, None, length=MAX_ITER_POISSON)
    
    return p_final

@jit
def correct_velocity(u_star, v_star, p, rho, dx, dy, dt):
    """
    correct velocities using pressure gradient
    
    Args:
        u_star: (Nx, Ny+1) - provisional u velocity
        v_star: (Nx+1, Ny) - provisional v velocity
        p: (Nx+1, Ny+1) - pressure field
        rho: density
        dx, dy, dt: grid spacing and timestep
    """
    u = jnp.zeros_like(u_star)
    v = jnp.zeros_like(v_star)
    
    # Pressure gradient for u
    dpdx = (p[1:, 1:-1] - p[:-1, 1:-1]) / dx
    u = u.at[:, :].set(u_star - (dt / rho) * dpdx)
    
    # Pressure gradient for v
    dpdy = (p[1:-1, 1:] - p[1:-1, :-1]) / dy
    v = v.at[:, :].set(v_star - (dt / rho) * dpdy)
    
    return u, v

@partial(jit, static_argnums=(1, 2, 3))
def interpolate_velocity(velocity, interpolation_direction: str, Nx: int, Ny: int):
    """
    Interpolates velocity, Applies equation A9
    """

    velocity_avg = 0.25 * (velocity[:-1, :-1] + velocity[:-1, 1:] + velocity[1:, :-1] + velocity[1:, 1:])
    if interpolation_direction == "u":
        velocity_interpolated = jnp.zeros((Nx + 1, Ny))
        velocity_interpolated = velocity_interpolated.at[1:-1, :].set(velocity_avg)

    elif interpolation_direction == "v":
        velocity_interpolated = jnp.zeros((Nx, Ny + 1))
        velocity_interpolated = velocity_interpolated.at[:, 1:-1].set(velocity_avg)

    return velocity_interpolated

