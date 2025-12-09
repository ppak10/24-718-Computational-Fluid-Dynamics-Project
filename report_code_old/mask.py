import jax.numpy as jnp
from jax import jit

@jit
def apply_liquid_mask(u, v, T, T_melt):
    is_liquid_internal = T[1:-1, 1:-1] > T_melt
    
    mask_u_interior = (is_liquid_internal[:-1, :] | is_liquid_internal[1:, :])
    
    u_masked = jnp.where(mask_u_interior, u[1:-1, :], 0.0)
    u = u.at[1:-1, :].set(u_masked)

    mask_v_interior = (is_liquid_internal[:, :-1] | is_liquid_internal[:, 1:])
    
    v_masked = jnp.where(mask_v_interior, v[:, 1:-1], 0.0)
    v = v.at[:, 1:-1].set(v_masked)
    
    return u, v
