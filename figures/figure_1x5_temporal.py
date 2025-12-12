import numpy as np
import matplotlib.pyplot as plt
import os

# Timestep configurations (dt = 0.000004 s)
timestep_configs = [
    {'folder': '050001', 'time': 0.2},
    {'folder': '100001', 'time': 0.4},
    {'folder': '150001', 'time': 0.6},
    {'folder': '200001', 'time': 0.8},
    {'folder': '250000', 'time': 1.0}
]

# Grid dimensions
Lx, Ly = 0.003, 0.003

# Melt pool temperature threshold
t_melt = 700

# Create 1x5 subplot
fig, axes = plt.subplots(1, 5, figsize=(25, 5), constrained_layout=True)

# Make all subplots have the same physical size
for ax in axes:
    ax.set_box_aspect(1)

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='red', linewidth=2, label='Melt Pool (700K)')]

# First pass: determine global temperature range across all timesteps
T_min = float('inf')
T_max = float('-inf')
for config in timestep_configs:
    data_dir = f"results/3_fine/{config['folder']}"
    T = np.load(os.path.join(data_dir, "T.npy"))
    T_min = min(T_min, T.min())
    T_max = max(T_max, T.max())

for idx, config in enumerate(timestep_configs):
    ax = axes[idx]

    # Load data
    data_dir = f"results/3_fine/{config['folder']}"
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
    if idx == len(timestep_configs) - 1:
        cbar = plt.colorbar(contour_T, ax=ax, label='Temperature [K]')

    # Streamlines (overlayed on top)
    ax.streamplot(xx, yy, u_T, v_T, color='black', density=1.5, linewidth=1.0, zorder=2)

    # Melt pool contour
    ax.contour(x, y, np.transpose(T), [t_melt], colors='red', linewidths=2, zorder=3)

    ax.set_xlabel(f"t = {config['time']:.1f} s", fontsize=18)
    ax.set_ylabel('y [m]')
    ax.set_ylim([0, Ly])
    ax.set_xlim([0, Lx])
    ax.set_aspect('equal')
    ax.tick_params(direction='in')
    ax.legend(handles=legend_elements, loc='lower left')

# Add super title with padding
fig.suptitle('Melt Pool Evolution', fontsize=32, y=1.02)

plt.savefig('figures/figure_1x5_temporal.png', dpi=300, bbox_inches='tight')
