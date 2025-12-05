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
