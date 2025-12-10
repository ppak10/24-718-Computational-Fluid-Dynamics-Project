import numpy as np
from numba import jit

@jit
def compute_laplacian(v, dx, dy):
    """
    Computes laplacian
    """

    laplacian = np.zeros_like(v)

    d2vdx2 = (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx**2
    d2vdy2 = (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / dy**2
    laplacian[1:-1, 1:-1] = d2vdx2 + d2vdy2

    return laplacian

