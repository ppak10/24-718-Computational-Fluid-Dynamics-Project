import jax.numpy as jnp

from jax import jit

@jit
def compute_laplacian_u(u, dx, dy):
    """
    Computes laplacian for u-momentum

    Args:
        u: (Nx, Ny + 1)
        dx: grid spacing in x
        dy: grid spacing in y
    """

    laplacian = jnp.zeros_like(u)

    d2udx2 = (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
    d2udy2 = (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    laplacian = laplacian.at[1:-1, 1:-1].set(d2udx2 + d2udy2)

    return laplacian

@jit
def compute_laplacian_v(v, dx, dy):
    """
    Computes laplacian for v-momentum

    Args:
        v: (Nx + 1, Ny)
        dx: grid spacing in x
        dy: grid spacing in y
    """

    laplacian = jnp.zeros_like(v)

    d2vdx2 = (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx**2
    d2vdy2 = (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / dy**2
    laplacian = laplacian.at[1:-1, 1:-1].set(d2vdx2 + d2vdy2)

    return laplacian

