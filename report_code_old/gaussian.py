# gaussian.py (or heat_source.py)
import jax.numpy as jnp

def compute_gaussian_heat_flux(Lx, Nx, dx, Q, sigma):
    """
    Compute Gaussian heat flux for top boundary
    
    Args:
        Lx: domain length
        Nx: number of cells
        dx: grid spacing
        Q: total power (W)
        sigma: Gaussian width (m)
        
    Returns:
        Neu_BC: (Nx+1,) array of heat flux at each point
    """
    xcen = Lx / 2
    x = jnp.linspace(0, Lx, Nx+1)
    
    # Gaussian PDF
    PDF = (1 / jnp.sqrt(2 * jnp.pi * sigma**2)) * jnp.exp(-((x - xcen)**2) / (2 * sigma**2))
    
    # Normalize and scale by power
    Neu_BC = (Q / dx) * PDF / jnp.sum(PDF)
    
    return Neu_BC
