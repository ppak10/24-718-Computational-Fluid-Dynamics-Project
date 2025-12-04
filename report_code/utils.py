import jax.numpy as jnp

from functools import partial
from jax import jit

@partial(jit, static_argnums=(1, 2, 3))
def interpolate_velocity(velocity, direction: str, Nx: int, Ny: int):
    """
    Interpolates v from (i, j+1/2) to (i+1/2, j)
    Applies equation A9
    Args:
        velocity: (Nx + 1, Ny) or (Nx, Ny + 1)
        direction: "u" or "v"
        Nx: Grid dimensions
        Ny: Grid dimensions

    Returns:
        velocity_interpolated (Nx, Ny + 1) or (Nx + 1, Ny)
    """

    velocity_avg = 0.25 * (velocity[:-1, :-1] + velocity[:-1, 1:] + velocity[1:, :-1] + velocity[1:, 1:])
    if direction == "v":
        velocity_interpolated = jnp.zeros((Nx, Ny + 1))
        velocity_interpolated = velocity_interpolated.at[:, 1:-1].set(velocity_avg)

    elif direction == "u":
        velocity_interpolated = jnp.zeros((Nx + 1, Ny))
        velocity_interpolated = velocity_interpolated.at[1:-1, :].set(velocity_avg)

    return velocity_interpolated

