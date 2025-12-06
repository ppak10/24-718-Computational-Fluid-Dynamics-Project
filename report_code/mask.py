import jax.numpy as jnp

from jax import jit

@jit
def apply_liquid_mask(u, v, T, T_melt):
    """
    Set velocities to zero in solid regions (T < T_melt)
    
    Args:
        u: (Nx, Ny+1) - u velocity
        v: (Nx+1, Ny) - v velocity
        T: (Nx+1, Ny+1) - temperature at cell centers
        T_melt: melting temperature
        
    Returns:
        u, v with solid regions masked out
    """
    # Create liquid mask at cell centers
    is_liquid = T > T_melt
    
    # Interpolate mask to u-locations (i+1/2, j)
    # u is liquid if either neighbor cell is liquid
    mask_u = (is_liquid[:-1, :] | is_liquid[1:, :])
    u_masked = jnp.where(mask_u[:, 1:-1], u[:, 1:-1], 0.0)
    u = u.at[:, 1:-1].set(u_masked)

    # Interpolate mask to v-locations (i, j+1/2)
    # v is liquid if either neighbor cell is liquid
    mask_v = (is_liquid[:, :-1] | is_liquid[:, 1:])
    v_masked = jnp.where(mask_v[1:-1, :], v[1:-1, :], 0.0)
    v = v.at[1:-1, :].set(v_masked)
    
    return u, v
