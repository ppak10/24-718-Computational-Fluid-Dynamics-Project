import jax.numpy as jnp

from jax import jit

@jit
def compute_convection_u(u, v_at_u, dx, dy):
    """
    Computes the convection term for u-momentum (Equation A13).

    Args:
        u: (Nx, Ny + 1)
        v_at_u: Interpolated terms for v (Nx, Ny + 1)
        dx: grid spacing
        dy: grid spacing
    """
    conv = jnp.zeros_like(u)

    # Internal nodes
    dudx = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx)
    dudy = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dy)

    u_conv_x = u[1:-1, 1:-1] * dudx
    v_conv_y = v_at_u[1:-1, 1:-1] * dudy

    conv = conv.at[1:-1, 1:-1].set(u_conv_x + v_conv_y)

    return conv

@jit
def compute_convection_v(v, u_at_v, dx, dy):
    """
    Computes the convection term for v-momentum (Equation A13).

    Args:
        v: (Nx + 1, Ny)
        v_at_u: Interpolated terms for v (Nx + 1, Ny)
        dx: grid spacing
        dy: grid spacing
    """
    conv = jnp.zeros_like(v)

    # Internal nodes
    dvdx = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dx)
    dvdy = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dy)

    u_conv_x = u_at_v[1:-1, 1:-1] * dvdx
    v_conv_y = v[1:-1, 1:-1] * dvdy

    conv = conv.at[1:-1, 1:-1].set(u_conv_x + v_conv_y)

    return conv
