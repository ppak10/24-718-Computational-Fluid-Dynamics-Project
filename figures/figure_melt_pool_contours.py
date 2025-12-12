import numpy as np
import matplotlib.pyplot as plt
import os

# Grid dimensions
Lx, Ly = 0.003, 0.003

# Melt pool temperature threshold
t_melt = 700

# Load temperature data from different cases
data_3_fine = np.load("results/3_fine/250000/T.npy")
data_3_fine_no_vel = np.load("results/3_fine/250000/T_no_velocity.npy")
data_3_fine_reverse = np.load("results/3_fine_reverse/250000/T.npy")

# Grid dimensions
Nx, Ny = data_3_fine.shape[0] - 1, data_3_fine.shape[1] - 1
x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
ax.set_box_aspect(1)

# Plot melt pool contours for each case
contour_fine = ax.contour(x, y, np.transpose(data_3_fine), [t_melt], colors='blue', linewidths=2, linestyles='-')
contour_no_vel = ax.contour(x, y, np.transpose(data_3_fine_no_vel), [t_melt], colors='green', linewidths=2, linestyles='--')
contour_reverse = ax.contour(x, y, np.transpose(data_3_fine_reverse), [t_melt], colors='red', linewidths=2, linestyles='-.')

# Create legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='blue', linewidth=2, linestyle='-', label='Forward Flow'),
    Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='Forward Flow (no velocity component)'),
    Line2D([0], [0], color='red', linewidth=2, linestyle='-.', label='Reverse Flow')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=12)

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Melt Pool Contours', fontsize=24, pad=25)
ax.set_ylim([0, Ly])
ax.set_xlim([0, Lx])
ax.set_aspect('equal')
ax.tick_params(direction='in')

plt.savefig('figures/figure_melt_pool_contours.png', dpi=300, bbox_inches='tight')
