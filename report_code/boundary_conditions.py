import jax.numpy as jnp

from jax import jit

@jit
def apply_boundary_conditions_with_marangoni(u, v, T, T_melt, rho, mu, dx, dy, dt, dSigma_dT):
    """
    Apply boundary conditions with Marangoni effect at top surface
    """

    Nx = u.shape[0]
    Ny = v.shape[1]
    
    # Bottom, left, right walls: no-slip
    u = u.at[:, 0].set(0.0)
    u = u.at[0, :].set(0.0)
    v = v.at[:, 0].set(0.0)
    v = v.at[0, :].set(0.0)
    v = v.at[-1, :].set(0.0)
    
    # === TOP SURFACE: Marangoni + Viscous Balance ===
    is_liquid_top = T[:, -1] > T_melt
    
    # Temperature difference (matches presentation)
    dT = jnp.zeros(Nx+1)
    dT = dT.at[1:-1].set(T[2:, -1] - T[:-2, -1])
    
    # Marangoni term
    marangoni_term = (dt / (rho * dx)) * dSigma_dT * dT[:-1]
    
    # Viscous term: du/dy at surface
    # Corrected to use dy (not dx)
    viscous_term = -(dt * mu / (rho * dy**2)) * (u[:, -1] - u[:, -2])
    
    # Combined surface BC
    u_top_new = u[:, -1] + marangoni_term + viscous_term
    u_top_new = jnp.where(is_liquid_top[:-1], u_top_new, 0.0)
    u = u.at[:, -1].set(u_top_new)
    
    return u, v
