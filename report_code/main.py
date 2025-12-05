import jax.numpy as jnp

from functools import partial
from jax import lax, jit
from typing import cast
from tqdm.rich import tqdm

from .boundary_conditions import apply_boundary_conditions
from .config import Config
from .convection import compute_convection_u, compute_convection_v
from .laplacian import compute_laplacian_u, compute_laplacian_v
from .mask import apply_liquid_mask
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

    STEPS = min(1000, Nt)



    rho = cast(float, config.density.to("kg/m^3").magnitude)
    mu = cast(float, config.mu.to("Pa.s").magnitude)
    nu = mu / rho

    T_melt = cast(float, config.temperature_melt.to("K").magnitude)
    T_initial = cast(float, config.temperature_initial.to("K").magnitude)

    # Stability check
    CFL = max(nu * dt / dx**2, nu * dt / dy**2)
    print(f"Diffusion CFL number: {CFL:.4f} (should be < 0.5 for stability)")
    if CFL > 0.5:
        print("WARNING: Time step too large for stability!")

    print(Lx, Ly, dx, dy, dt, Nx, Ny, Nt, rho, mu, nu)

    # Pressure at cell centers
    p = jnp.zeros((Nx + 1, Ny + 1))
    print(p.shape)

    # # Temperature field at cell centers
    # T = jnp.ones((Nx + 1, Ny + 1)) * T_initial

    # Temperature field with hot spot
    x = jnp.linspace(0, Lx, Nx+1)
    y = jnp.linspace(0, Ly, Ny+1)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    
    x0, y0 = Lx/2, Ly/2
    sigma = 0.0005  # 0.5 mm
    T_peak = 1500.0  # K
    
    T = T_initial + (T_peak - T_initial) * jnp.exp(-((X-x0)**2 + (Y-y0)**2) / (2*sigma**2))

    # u-velocity along vertical faces (i + 1/2, j)
    u = jnp.zeros((Nx, Ny + 1))
    u_no_p = jnp.zeros((Nx, Ny + 1))
    print(u.shape)

    # v-velocity along vertical faces (i, j + 1/2)
    v = jnp.zeros((Nx + 1, Ny))
    v_no_p = jnp.zeros((Nx + 1, Ny))
    print(v.shape)

    max_iter_poisson = 1000

    print("Starting simulation...")
    for n in tqdm(range(STEPS)):
        # Single timestep
        u, v, p = timestep(u, v, p, T, rho, nu, T_melt, dx, dy, dt, max_iter_poisson)

        # TODO: Update temperature (thermal solver)
        # T = update_temperature(T, u, v, ...)

    print("Simulation complete!")

    # Visualize results
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Temperature
    im0 = axes[0,0].imshow(T.T, origin='lower', cmap='hot')
    axes[0,0].set_title('Temperature (K)')
    plt.colorbar(im0, ax=axes[0,0])

    # Pressure
    im1 = axes[0,1].imshow(p.T, origin='lower', cmap='RdBu_r')
    axes[0,1].set_title('Pressure (Pa)')
    plt.colorbar(im1, ax=axes[0,1])

    # Velocity magnitude
    # Interpolate to common grid for visualization
    u_vis = (u[:, :-1] + u[:, 1:]) / 2
    v_vis = (v[:-1, :] + v[1:, :]) / 2
    vel_mag = jnp.sqrt(u_vis**2 + v_vis**2)
    im2 = axes[1,0].imshow(vel_mag.T, origin='lower', cmap='viridis')
    axes[1,0].set_title('Velocity Magnitude (m/s)')
    plt.colorbar(im2, ax=axes[1,0])

    # Quiver
    x = jnp.linspace(0, Lx, Nx)
    y = jnp.linspace(0, Ly, Ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    skip = max(1, Nx // 20)
    axes[1,1].quiver(X[::skip, ::skip], Y[::skip, ::skip],
                     u_vis[::skip, ::skip], v_vis[::skip, ::skip])
    axes[1,1].set_title('Velocity Field')
    axes[1,1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('cfd_results.png', dpi=150)
    print("Results saved to cfd_results.png")

    return u, v, p, T


@partial(jit, static_argnums=(10,))
def timestep(u, v, p, T, rho, nu, T_melt, dx, dy, dt, max_iter_poisson):
    u_star, v_star = compute_provisional_velocity(u, v, nu, T, T_melt, rho, dx, dy, dt)
    u_star, v_star = apply_liquid_mask(u_star, v_star, T, T_melt)
    u_star, v_star = apply_boundary_conditions(u_star, v_star, T, T_melt)

    p_new = solve_pressure(u_star, v_star, p, rho, dx, dy, dt, max_iter_poisson)

    u_new, v_new = correct_velocity(u_star, v_star, p_new, rho, dx, dy, dt)
    u_new, v_new = apply_liquid_mask(u_new, v_new, T, T_melt)
    u_new, v_new = apply_boundary_conditions(u_new, v_new, T, T_melt)

    return u_new, v_new, p_new


@jit
def compute_provisional_velocity(u, v, nu, T, T_melt, rho, dx, dy, dt):
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

@partial(jit, static_argnums=(7,))
def solve_pressure(u_star, v_star, p_init, rho, dx, dy, dt, max_iterations=1000):
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

    Nx = u_star.shape[0]
    Ny = v_star.shape[1]

    rhs = jnp.zeros((Nx + 1, Ny + 1))

    div_u = (u_star[1:, 1:-1] - u_star[:-1, 1:-1]) / dx
    div_v = (v_star[1:-1, 1:] - v_star[1:-1, :-1]) / dy

    rhs = rhs.at[1:-1, 1:-1].set((rho / dt) * (div_u + div_v))

    dx2 = dx * dx
    dy2 = dy * dy

    factor = 2.0 / dx2 + 2.0 / dy2

    def jacobi_iteration(p, _):
        """
        single jacobi iteration
        """

        p_new = jnp.zeros_like(p)


        laplacian = ((p[2:, 1:-1] + p[:-2, 1:-1]) / dx2 + 
                     (p[1:-1, 2:] + p[1:-1, :-2]) / dy2 - 
                     rhs[1:-1, 1:-1])
        
        p_new = p_new.at[1:-1, 1:-1].set(laplacian / factor)
        
        # Boundary conditions
        p_new = p_new.at[0, :].set(p_new[1, :])      # Left
        p_new = p_new.at[-1, :].set(p_new[-2, :])    # Right
        p_new = p_new.at[:, 0].set(p_new[:, 1])      # Bottom
        p_new = p_new.at[:, -1].set(p_new[:, -2])    # Top
        
        return p_new, None
    
    p_final, _ = lax.scan(jacobi_iteration, p_init, None, length=max_iterations)
    
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
    u = u.at[:, 1:-1].set(u_star[:, 1:-1] - (dt / rho) * dpdx)
    
    # Pressure gradient for v
    dpdy = (p[1:-1, 1:] - p[1:-1, :-1]) / dy
    v = v.at[1:-1, :].set(v_star[1:-1, :] - (dt / rho) * dpdy)
    
    return u, v
