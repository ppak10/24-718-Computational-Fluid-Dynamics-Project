import jax.numpy as jnp
from jax import jit

@jit
def update_temperature(T, u, v, alpha, dt, dx, dy, Neu_BC, T_initial, k):
    """
    Update temperature using upwind scheme for advection + central for diffusion
    
    Args:
        T: (Nx+1, Ny+1) - temperature at time n
        u: (Nx, Ny+1) - u velocity
        v: (Nx+1, Ny) - v velocity
        alpha: thermal diffusivity
        dt, dx, dy: discretization
        Neu_BC: (Nx+1,) heat flux at top
        T_initial: temperature for Dirichlet BCs
        k: thermal conductivity
        
    Returns:
        T_new: temperature at time n+1
    """
    Nx = u.shape[0]
    Ny = v.shape[1]
    
    beta_x = alpha * dt / dx**2
    beta_y = alpha * dt / dy**2
    
    T_new = T.copy()
    
    # Interpolate velocities to cell centers for advection
    u_center = (u[:, :-1] + u[:, 1:]) / 2  # (Nx, Ny)
    v_center = (v[:-1, :] + v[1:, :]) / 2  # (Nx, Ny)
    
    # Pad to match T dimensions for easier indexing
    u_pad = jnp.zeros((Nx+1, Ny+1))
    v_pad = jnp.zeros((Nx+1, Ny+1))
    u_pad = u_pad.at[:-1, :-1].set(u_center)
    v_pad = v_pad.at[:-1, :-1].set(v_center)
    
    # Interior points (vectorized)
    # Upwind advection terms
    nu_x = u_pad[1:-1, 1:-1] * dt / dx
    nu_y = v_pad[1:-1, 1:-1] * dt / dy
    
    # Upwind scheme using jnp.where (not Python if/else)
    advection_x = jnp.where(nu_x > 0, nu_x * T[:-2, 1:-1], nu_x * T[1:-1, 1:-1])
    advection_y = jnp.where(nu_y > 0, nu_y * T[1:-1, :-2], nu_y * T[1:-1, 1:-1])
    
    # Diffusion (central differences)
    diffusion_x = beta_x * (T[2:, 1:-1] + T[:-2, 1:-1])
    diffusion_y = beta_y * (T[1:-1, 2:] + T[1:-1, :-2])
    
    # Update (vectorized)
    T_interior = ((1 - 2*beta_x - 2*beta_y) * T[1:-1, 1:-1] + 
                  diffusion_x + diffusion_y + advection_x + advection_y)
    
    T_new = T_new.at[1:-1, 1:-1].set(T_interior)
    
    # Dirichlet BCs (sides and bottom at T_initial)
    T_new = T_new.at[:, 0].set(T_initial)   # Bottom
    T_new = T_new.at[0, :].set(T_initial)   # Left
    T_new = T_new.at[-1, :].set(T_initial)  # Right
    
    # T_new = T_new.at[:, -1].set(T_new[:, -2] + dy * Neu_BC / k)
    T_new = T_new.at[:, -1].set(T_new[:, -2] + dt * Neu_BC)
    
    return T_new
