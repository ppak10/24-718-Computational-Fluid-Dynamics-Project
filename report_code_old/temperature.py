import jax.numpy as jnp
from jax import jit

@jit
def update_temperature(T, u, v, alpha, dt, dx, dy, Neu_BC, T_initial, k):
    beta_x = alpha * dt / dx**2
    beta_y = alpha * dt / dy**2
    
    T_new = T.copy()
    
    u_center = (u[:-1, :] + u[1:, :]) / 2
    v_center = (v[:, :-1] + v[:, 1:]) / 2
    
    # Interior points (vectorized)
    nu_x = u_center * dt / dx
    nu_y = v_center * dt / dy
    
    # Upwind scheme
    advection_x = jnp.where(nu_x > 0, nu_x * T[:-2, 1:-1], nu_x * T[2:, 1:-1])
    advection_y = jnp.where(nu_y > 0, nu_y * T[1:-1, :-2], nu_y * T[1:-1, 2:])
    
    # Diffusion (central differences)
    diffusion_x = beta_x * (T[2:, 1:-1] + T[:-2, 1:-1])
    diffusion_y = beta_y * (T[1:-1, 2:] + T[1:-1, :-2])
    
    # Update (vectorized)
    T_interior = ((1 - 2*beta_x - 2*beta_y) * T[1:-1, 1:-1] + 
                    diffusion_x + diffusion_y - advection_x - advection_y)
    
    T_new = T_new.at[1:-1, 1:-1].set(T_interior)
    
    # Dirichlet BCs (sides and bottom at T_initial)
    T_new = T_new.at[:, 0].set(T_initial)
    T_new = T_new.at[0, :].set(T_initial)
    T_new = T_new.at[-1, :].set(T_initial)
    
    # FIX: Neumann BC applied only to interior T nodes (T[1:-1, -1])
    # T[1:-1, -1] is (60,). Neu_BC is (61,). Slice Neu_BC[1:] to get (60,).
    Neu_BC_interior = Neu_BC[1:]
    
    T_new = T_new.at[1:-1, -1].set(T_new[1:-1, -2] + dy * Neu_BC_interior)
    
    return T_new
