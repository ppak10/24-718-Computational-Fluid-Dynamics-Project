import jax.numpy as jnp

from functools import partial
from jax import jit
from typing import cast

from .config import Config
from .convection import compute_convection_u, compute_convection_v
from .laplacian import compute_laplacian_u, compute_laplacian_v
from .utils import interpolate_velocity

def run():
    config = Config()

    Lx = cast(float, config.x_domain.to("m").magnitude)
    Ly = cast(float, config.y_domain.to("m").magnitude)

    dx = cast(float, config.x_step.to("m").magnitude)
    dy = cast(float, config.y_step.to("m").magnitude)
    dt = cast(float, config.time_step.to("s").magnitude)

    t_max = cast(float, config.time_step_max.to("s").magnitude)

    Nx = int(Lx / dx)
    Ny = int(Ly / dy)
    Nt = int(t_max / dt)

    rho = cast(float, config.density.to("kg/m^3").magnitude)
    mu = cast(float, config.mu.to("Pa.s").magnitude)
    nu = mu / rho

    T_melt = cast(float, config.temperature_melt.to("K").magnitude)
    T_initial = cast(float, config.temperature_initial.to("K").magnitude)

    print(Lx, Ly, dx, dy, dt, Nx, Ny, Nt, rho, mu, nu)

    # Pressure at cell centers
    p = jnp.zeros((Nx + 1, Ny + 1))
    print(p.shape)

    # Temperature field at cell centers
    T = jnp.ones((Nx + 1, Ny + 1)) * T_initial

    # u-velocity along vertical faces (i + 1/2, j)
    u = jnp.zeros((Nx, Ny + 1))
    u_no_p = jnp.zeros((Nx, Ny + 1))
    print(u.shape)

    # v-velocity along vertical faces (i, j + 1/2)
    v = jnp.zeros((Nx + 1, Ny))
    v_no_p = jnp.zeros((Nx + 1, Ny))
    print(v.shape)


@partial(jit, static_argnums=(6, 7, 8, 9, 10, 11))
def compute_provisional_velocity(u, v, nu, T, T_melt, rho, dx, dy, dt, Nx, Ny):
    """
    Computes the velocity components without pressure gradient.

    Args:
        u: (Nx, Ny + 1)
        v: (Nx + 1, Ny)
        nu: viscosity
        T: (Nx + 1, Ny + 1)
        T_melt: Kelvin
        rho: density
        dx: grid spacing
        dy: grid spacing
        dt: time step
    """

    Nx = u.shape[0]
    Ny = v.shape[1]

    is_liquid = T > T_melt

    # u*
    # Interpolate v to u-location
    v_at_u = interpolate_velocity(v, "v", Nx, Ny)

    # Compute convection and diffusion
    conv_u = compute_convection_u(u, v_at_u, dx, dy)
    lap_u = compute_laplacian_u(u, dx, dy)

    # Provisional velocity (explicit)
    u_star = u + dt * (-conv_u + nu * lap_u)

    # v*
    # Interpolate u to v-location
    u_at_v = interpolate_velocity(u, "u", Nx, Ny)

    # Compute convection and diffusion
    conv_v = compute_convection_v(v, u_at_v, dx, dy)
    lap_v = compute_laplacian_v(v, dx, dy)

    # Provisional velocity (explicit)
    v_star = v + dt * (-conv_v + nu * lap_v)

    return u_star, v_star

