import jax.numpy as jnp
import sys

from typing import cast
from tqdm.rich import tqdm

from .boundary_conditions import apply_boundary_conditions
from .config import Config
from .gaussian import compute_gaussian_heat_flux
from .mask import apply_liquid_mask
from .plot import create_plots
from .temperature import update_temperature
from .timestep import compute_provisional_velocity, solve_pressure, correct_velocity


def run_simulation():
    config = Config()

    # Domain
    Lx = cast(float, config.x_domain.to("m").magnitude)
    Ly = cast(float, config.y_domain.to("m").magnitude)

    t_max = cast(float, config.time_step_max.to("s").magnitude)

    # Step Sizes
    dx = cast(float, config.x_step.to("m").magnitude)
    dy = cast(float, config.y_step.to("m").magnitude)
    dt = cast(float, config.time_step.to("s").magnitude)

    # Counts
    Nx = int(Lx / dx)
    Ny = int(Ly / dy)
    Nt = int(t_max / dt)

    # Material Properties
    alpha = cast(float, config.thermal_diffusivity.to("m^2/s").magnitude)
    k = cast(float, config.thermal_conductivity.to("W/m/K").magnitude)
    rho = cast(float, config.density.to("kg/m^3").magnitude)
    mu = cast(float, config.viscosity.to("Pa.s").magnitude)
    nu = mu / rho

    # Laser Properties
    Q = cast(float, config.Q.to("W").magnitude)
    sigma = cast(float, config.sigma.to("m").magnitude)
    dSigma_dT = cast(float, config.dSigma_dT.to("N/m/K").magnitude)
    Neu_BC = compute_gaussian_heat_flux(Lx, Nx, dx, Q, sigma)

    # Temperature
    T_melt = cast(float, config.temperature_melt.to("K").magnitude)
    T_init = cast(float, config.temperature_initial.to("K").magnitude)

    # Staggered Grid
    p = jnp.zeros((Nx + 2, Ny + 2))
    T = jnp.ones((Nx + 2, Ny + 2)) * T_init

    print(f"T array shape after initialization: {T.shape}")
    u = jnp.zeros((Nx + 1, Ny))
    v = jnp.zeros((Nx, Ny + 1))

    # Apply initial top BC
    # T = T.at[:, -1].set(T[:, -2] + dy * Neu_BC)

    # Stability check
    CFL = max(nu * dt / dx**2, nu * dt / dy**2)
    print(f"Diffusion CFL number: {CFL:.4f} (should be < 0.5 for stability)")
    if CFL > 0.5:
        print("WARNING: Time step too large for stability!")

    # Running timesteps
    for n in tqdm(range(Nt)):
        # Single timestep
        # Step 1: Provisional velocity
        u_star, v_star = compute_provisional_velocity(u, v, nu, dx, dy, dt)
        u_star, v_star = apply_liquid_mask(u_star, v_star, T, T_melt)

        # Step 2: Pressure
        p = solve_pressure(u_star, v_star, p, rho, dx, dy, dt)
        
        # Step 3: Velocity correction
        u, v = correct_velocity(u_star, v_star, p, rho, dx, dy, dt)
        u, v = apply_liquid_mask(u, v, T, T_melt)

        # Step 4: Apply marangoni to final velocity
        u, v = apply_boundary_conditions(
            u, v, T, T_melt, rho, mu, dx, dy, dt, dSigma_dT
        )

        T = update_temperature(T, u, v, alpha, dt, dx, dy, Neu_BC, T_init, k)

    print("Simulation complete!")

    # Save results
    jnp.save('simulation_u.npy', u)
    jnp.save('simulation_v.npy', v)
    jnp.save('simulation_p.npy', p)
    jnp.save('simulation_T.npy', T)
    jnp.save('simulation_params.npy', jnp.array([Lx, Ly, Nx, Ny]))
    print("Results saved to simulation_*.npy files")

    return u, v, p, T


def load_and_plot():
    """Load saved simulation results and create plots"""
    print("Loading simulation results...")
    u = jnp.load('simulation_u.npy')
    v = jnp.load('simulation_v.npy')
    p = jnp.load('simulation_p.npy')
    T = jnp.load('simulation_T.npy')
    params = jnp.load('simulation_params.npy')
    Lx, Ly, Nx, Ny = float(params[0]), float(params[1]), int(params[2]), int(params[3])

    print("Creating plots...")
    create_plots(u, v, Lx, Ly, Nx, Ny, T, p)
    print("Plots saved!")


def run():
    """Main entry point that handles command-line arguments"""
    if len(sys.argv) > 1 and sys.argv[1] == 'plot':
        load_and_plot()
    else:
        run_simulation()

