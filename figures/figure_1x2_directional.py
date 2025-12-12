import numpy as np
import matplotlib.pyplot as plt
import os

# Flow configurations
flow_configs = [
    {'name': '3_fine', 'last_timestep': '250000', 'title': 'Forward Flow', 'dSigma': -1.9e-4},
    {'name': '3_fine_reverse', 'last_timestep': '250000', 'title': 'Reverse Flow', 'dSigma': 1.9e-4}
]

# Grid dimensions
Lx, Ly = 0.003, 0.003

# Melt pool temperature threshold
t_melt = 700

# Create 1x2 subplot
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# Make all subplots have the same physical size
for ax in axes:
    ax.set_box_aspect(1)

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='red', linewidth=2, label='Melt Pool (700K)')]

# First pass: determine global temperature range across both flows
T_min = float('inf')
T_max = float('-inf')
for config in flow_configs:
    data_dir = f"results/{config['name']}/{config['last_timestep']}"
    T = np.load(os.path.join(data_dir, "T.npy"))
    T_min = min(T_min, T.min())
    T_max = max(T_max, T.max())

for idx, config in enumerate(flow_configs):
    ax = axes[idx]

    # Load data
    data_dir = f"results/{config['name']}/{config['last_timestep']}"
    u = np.load(os.path.join(data_dir, "u.npy"))
    v = np.load(os.path.join(data_dir, "v.npy"))
    T = np.load(os.path.join(data_dir, "T.npy"))

    # Interpolate velocities to cell centers
    u_interpolated = 0.5 * (u[1:, :] + u[:-1, :])
    v_interpolated = 0.5 * (v[:, 1:] + v[:, :-1])

    # Transpose for plotting
    u_T = u_interpolated.T
    v_T = v_interpolated.T

    # Grid dimensions
    Nx, Ny = T.shape[0] - 1, T.shape[1] - 1
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)

    xx, yy = np.meshgrid(x, y)

    # Temperature field with consistent colormap range and more contour levels
    contour_T = ax.contourf(x, y, np.transpose(T), levels=25, cmap='plasma', vmin=T_min, vmax=T_max, zorder=1)
    # Only add colorbar to the rightmost subplot
    if idx == len(flow_configs) - 1:
        cbar = plt.colorbar(contour_T, ax=ax, label='Temperature [K]')

    # Streamlines (overlayed on top)
    ax.streamplot(xx, yy, u_T, v_T, color='black', density=1.5, linewidth=1.0, zorder=2)

    # Melt pool contour
    ax.contour(x, y, np.transpose(T), [t_melt], colors='red', linewidths=2, zorder=3)

    ax.set_xlabel(config['title'], fontsize=18)
    # ax.set_xlabel(f"{config['title']} (dσ/dT = {config['dSigma']:.1e} N/m·K)", fontsize=18)
    ax.set_ylabel('y [m]')
    ax.set_ylim([0, Ly])
    ax.set_xlim([0, Lx])
    ax.set_aspect('equal')
    ax.tick_params(direction='in')
    ax.legend(handles=legend_elements, loc='lower left')

# Add super title with padding
fig.suptitle('Directional Stream Comparison', fontsize=32, y=1.08)

plt.savefig('figures/figure_1x2_directional.png', dpi=300, bbox_inches='tight')
