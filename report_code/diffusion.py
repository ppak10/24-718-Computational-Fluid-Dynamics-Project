import numpy as np
from numba import jit

@jit
def compute_diffusion(v, dx, dy):
    diffusion = np.zeros_like(v)

    d2vdx2 = (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx**2
    d2vdy2 = (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / dy**2
    diffusion[1:-1, 1:-1] = d2vdx2 + d2vdy2

    return diffusion

