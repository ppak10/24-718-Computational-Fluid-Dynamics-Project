import jax.numpy as jnp
from jax import jit

@jit
def apply_boundary_conditions(u, v, T, T_melt, rho, mu, dx, dy, dt, dSigma_dT):
    
    Nx_cells = u.shape[0] - 1
    
    # No-Slip Boundary Conditions (Bottom, Left, Right, v-Top)
    u = u.at[:, 0].set(0.0) 
    v = v.at[:, 0].set(0.0)
    u = u.at[0, :].set(0.0)
    v = v.at[0, :].set(0.0)
    u = u.at[-1, :].set(0.0)
    v = v.at[-1, :].set(0.0)
    v = v.at[:, -1].set(0.0)
    
    # Marangoni Stress Balance at Top Surface (u-velocity)
    
    # T[1:-1, -1] is the top T-ghost cell row, T[1:-1, -2] is the internal row (60 elements)
    is_liquid_top_T = T[1:-1, -2] > T_melt # Mask based on internal cells (60,)

    # Viscous term: du/dy (uses u_i, j+1 and u_i, j)
    # This must be defined on all u-nodes (61,)
    viscous_term = (dt * mu / (rho * dy)) * (u[:, -1] - u[:, -2])

    # Marangoni Term: Calculate dT/dx on u-nodes (i+1/2)
    marangoni_term = jnp.zeros(u.shape[0]) # Shape (61,)
    
    # FIX: Use 1st-order difference on the internal T cell row T[1:-1, -2] (60 elements)
    # This yields Nx_cells - 1 = 59 interior elements, aligning with u[1:-1]
    T_top_internal_row = T[1:-1, -2]
    dT_dx_interior = (T_top_internal_row[1:] - T_top_internal_row[:-1]) / dx # Shape (59,)
    
    marangoni_interior = (dt / rho) * dSigma_dT * dT_dx_interior
    
    # Set the interior u-nodes (index 1 to Nx_cells-1) (61 -> 59 elements)
    marangoni_term = marangoni_term.at[1:-1].set(marangoni_interior)
    
    # Combined surface BC
    u_top_new = u[:, -1] + marangoni_term + viscous_term
    
    # Apply liquid mask to the interior u-nodes being corrected
    # Mask is derived from adjacent internal T-cells (60 -> 59 elements)
    is_liquid_u_top = (is_liquid_top_T[:-1] | is_liquid_top_T[1:]) 
    
    # u_top_new[1:-1] are the 59 interior u-nodes
    u_masked = jnp.where(is_liquid_u_top, u_top_new[1:-1], 0.0)

    # Update the interior u-nodes on the top boundary
    u = u.at[1:-1, -1].set(u_masked)
    
    return u, v
