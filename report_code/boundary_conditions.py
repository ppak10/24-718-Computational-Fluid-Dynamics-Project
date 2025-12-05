import jax.numpy as jnp

from jax import jit

from functools import partial

@jit
def apply_boundary_conditions(u, v, T, T_melt):
    """
    Apply boundary conditions for velocity field
    
    Args:
        u: (Nx, Ny+1) - u velocity
        v: (Nx+1, Ny) - v velocity
        T: (Nx+1, Ny+1) - temperature field
        T_melt: melting temperature
        
    Returns:
        u, v with boundary conditions applied
    """
    # No-slip at walls (u=v=0)
    # Bottom wall (j=0)
    u = u.at[:, 0].set(0.0)
    v = v.at[:, 0].set(0.0)
    
    # Left wall (i=0)
    u = u.at[0, :].set(0.0)
    v = v.at[0, :].set(0.0)
    
    # Right wall (i=Nx)
    v = v.at[-1, :].set(0.0)
    # u already at boundary

    u_lid = 0.01
    u = u.at[:, -1].set(u_lid)
    
    return u, v

@jit
def apply_boundary_conditions_with_marangoni(u, v, T, T_melt, rho, mu, dx, dy, dt, dSigma_dT):
    """
    Apply boundary conditions with Marangoni effect at top surface
    
    Args:
        u: (Nx, Ny+1) - u velocity
        v: (Nx+1, Ny) - v velocity
        T: (Nx+1, Ny+1) - temperature field
        T_melt: melting temperature
        rho, mu: material properties
        dx, dy, dt: discretization
        dSigma_dT: surface tension gradient (N/m/K)
    """
    Nx = u.shape[0]
    Ny = v.shape[1]
    
    # Bottom, left, right walls: no-slip
    u = u.at[:, 0].set(0.0)   # Bottom
    u = u.at[0, :].set(0.0)   # Left
    v = v.at[:, 0].set(0.0)   # Bottom
    v = v.at[0, :].set(0.0)   # Left
    v = v.at[-1, :].set(0.0)  # Right
    
    # === TOP SURFACE: Marangoni BC ===
    # Only where liquid (T > T_melt)
    is_liquid_top = T[:, -1] > T_melt
    
    # Compute Marangoni stress: dσ/dx = (dσ/dT) * (dT/dx)
    # Temperature gradient at top surface (central difference)
    dT_dx = jnp.zeros(Nx+1)
    dT_dx = dT_dx.at[1:-1].set((T[2:, -1] - T[:-2, -1]) / (2 * dx))
    
    # Marangoni force term
    marangoni_term = (dt / (rho * dx)) * dSigma_dT * dT_dx[:-1]  # for u locations
    
    # Viscous stress term: -mu * du/dy at surface
    viscous_term = -(mu / (rho * dx)) * dt * (u[:, -1] - u[:, -2])
    
    # Update top u-velocity where liquid
    u_top_new = u[:, -1] + marangoni_term + viscous_term
    u_top_new = jnp.where(is_liquid_top[:-1], u_top_new, 0.0)
    u = u.at[:, -1].set(u_top_new)
    
    return u, v
