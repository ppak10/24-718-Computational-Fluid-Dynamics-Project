import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

def create_plots(u, v, Lx, Ly, Nx, Ny, T, p):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Temperature
    im0 = axes[0,0].imshow(T.T, origin='lower', cmap='hot')
    axes[0,0].set_title('Temperature (K)')
    plt.colorbar(im0, ax=axes[0,0])

    # Pressure
    im1 = axes[0,1].imshow(p.T, origin='lower', cmap='RdBu_r')
    axes[0,1].set_title('Pressure (Pa)')
    plt.colorbar(im1, ax=axes[0,1])

    # Velocity magnitude
    # Interpolate to common grid for visualization
    u_vis = (u[:, :-1] + u[:, 1:]) / 2
    v_vis = (v[:-1, :] + v[1:, :]) / 2
    vel_mag = jnp.sqrt(u_vis**2 + v_vis**2)
    im2 = axes[1,0].imshow(vel_mag.T, origin='lower', cmap='viridis')
    axes[1,0].set_title('Velocity Magnitude (m/s)')
    plt.colorbar(im2, ax=axes[1,0])

    # Quiver
    x = jnp.linspace(0, Lx, Nx)
    y = jnp.linspace(0, Ly, Ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    skip = max(1, Nx // 20)
    axes[1,1].quiver(X[::skip, ::skip], Y[::skip, ::skip],
                     u_vis[::skip, ::skip], v_vis[::skip, ::skip])
    axes[1,1].set_title('Velocity Field')
    axes[1,1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('cfd_results.png', dpi=150)

