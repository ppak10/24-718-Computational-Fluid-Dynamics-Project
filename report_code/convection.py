import numpy as np
from numba import jit

@jit
def compute_convection_u(u, v_interpolated, dx, dy):
    """
    Computes the convection term for u-momentum (Equation A13).
    """
    conv = np.zeros_like(u)

    # Internal nodes
    dudx = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx) # (30, 29)
    dudy = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dy) # (30, 29)

    # print("dudx", dudx.shape)
    # print("dudy", dudx.shape)
    # print("u[1:-1, 1:-1]", u[1:-1, 1:-1].shape) # (30, 29)

    u_conv_x = u[1:-1, 1:-1] * dudx # (30, 29)
    # print("u_conv_x", u_conv_x.shape)
    #
    # print("v_interpolated[1:-1, 1:-1]", v_interpolated[1:-1, 1:-1].shape)  # (28, 29)
    # print("v_interpolated[:, 1:-1]", v_interpolated[:, 1:-1].shape)  # (28, 29)
    v_conv_y = v_interpolated[:, 1:-1] * dudy
    # print("v_conv_y", v_conv_y.shape) 
    #
    # print("conv[1:-1, 1:-1]", conv[1:-1, 1:-1].shape)
    conv[1:-1, 1:-1] = u_conv_x + v_conv_y

    return conv # (32, 31)

@jit
def compute_convection_v(v, u_interpolated, dx, dy):
    """
    Computes the convection term for v-momentum (Equation A13).
    """
    conv = np.zeros_like(v)

    # Internal nodes
    dvdx = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dx)
    dvdy = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dy)

    u_conv_x = u_interpolated[1:-1, :] * dvdx
    v_conv_y = v[1:-1, 1:-1] * dvdy

    conv[1:-1, 1:-1] = u_conv_x + v_conv_y

    return conv # (31, 32)
