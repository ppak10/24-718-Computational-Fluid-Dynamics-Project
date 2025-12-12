import numpy as np
import matplotlib.pyplot as plt
import os

# Mesh configurations (dx in meters, converted to microns for title)
mesh_configs = [
    {'name': '1_coarse', 'last_timestep': '49999', 'dx': 0.0001, 'title': 'Coarse Mesh'},
    {'name': '2_medium', 'last_timestep': '99999', 'dx': 0.00005, 'title': 'Medium Mesh'},
    {'name': '3_fine', 'last_timestep': '250000', 'dx': 0.00002, 'title': 'Fine Mesh'}
]

# Grid dimensions
Lx, Ly = 0.003, 0.003

# Melt pool temperature threshold
t_melt = 700

# Create 3x1 subplot
fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# Make all subplots have the same physical size
for ax in axes:
    ax.set_box_aspect(1)

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='red', linewidth=2, label='Melt Pool (700K)')]

for idx, config in enumerate(mesh_configs):
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

    # Add mesh grid overlay (set grid first so it appears in background)
    dx = config['dx']
    dy = config['dx']  # dy = dx for square meshes
    x_grid = np.arange(0, Lx + dx, dx)
    y_grid = np.arange(0, Ly + dy, dy)
    ax.set_xticks(x_grid, minor=False)
    ax.set_yticks(y_grid, minor=False)
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.8, alpha=0.4, zorder=0)

    # Streamlines (plotted on top of grid)
    ax.streamplot(xx, yy, u_T, v_T, color=np.sqrt(u_T*u_T + v_T*v_T),
                   density=1.5, linewidth=1.5, cmap='viridis', zorder=2)
    ax.contour(x, y, np.transpose(T), [t_melt], colors='red', linewidths=2, zorder=3)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    # Convert dx from meters to microns (1 m = 1e6 microns)
    dx_microns = config['dx'] * 1e6
    ax.set_title(f"{config['title']} (dx, dy = {dx_microns:.0f} µm)", fontsize=24, pad=25)
    ax.set_ylim([0, Ly])
    ax.set_xlim([0, Lx])
    ax.set_aspect('equal')
    ax.tick_params(direction='in')

    # Show only first and last tick labels
    ax.set_xticklabels([''] * len(x_grid))
    ax.set_yticklabels([''] * len(y_grid))
    xtick_labels = ax.get_xticklabels()
    ytick_labels = ax.get_yticklabels()
    if len(xtick_labels) > 0:
        xtick_labels[0].set_text(f'{x_grid[0]:.4f}')
        xtick_labels[-1].set_text(f'{x_grid[-1]:.4f}')
    if len(ytick_labels) > 0:
        ytick_labels[0].set_text(f'{y_grid[0]:.4f}')
        ytick_labels[-1].set_text(f'{y_grid[-1]:.4f}')
    ax.set_xticklabels(xtick_labels)
    ax.set_yticklabels(ytick_labels)

    ax.legend(handles=legend_elements, loc='lower left')

plt.savefig('figures/figure_3x1_streamlines.png', dpi=300, bbox_inches='tight')
