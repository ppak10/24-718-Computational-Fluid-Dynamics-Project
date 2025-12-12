import numpy as np
import matplotlib.pyplot as plt
import os

# Load data from results/2_medium last timestep
data_dir = "results/2_medium/99999"

p = np.load(os.path.join(data_dir, "p.npy"))
u = np.load(os.path.join(data_dir, "u.npy"))
v = np.load(os.path.join(data_dir, "v.npy"))
T = np.load(os.path.join(data_dir, "T.npy"))

# Interpolate velocities to cell centers
u_interpolated = 0.5 * (u[1:, :] + u[:-1, :])
v_interpolated = 0.5 * (v[:, 1:] + v[:, :-1])

# Transpose for plotting
u_T = u_interpolated.T
v_T = v_interpolated.T

# Calculate velocity magnitude
vel_mag = np.sqrt(u_interpolated**2 + v_interpolated**2)

# Grid dimensions
Nx, Ny = T.shape[0] - 1, T.shape[1] - 1
Lx, Ly = 0.003, 0.003
x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)

xx, yy = np.meshgrid(x, y)

# Melt pool temperature threshold
t_melt = 700

# Create 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

# Make all subplots have the same physical size
for ax in axes.flat:
    ax.set_box_aspect(1)

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='red', linewidth=2, label='Melt Pool (700K)')]

# 1. Temperature (top-left)
ax = axes[0, 0]
contour_T = ax.contourf(x, y, np.transpose(T), cmap='plasma')
plt.colorbar(contour_T, ax=ax, label='Temperature [K]')
melt_contour = ax.contour(x, y, np.transpose(T), [t_melt], colors='red', linewidths=2)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Temperature', fontsize=32)
ax.set_aspect('equal')
ax.tick_params(direction='in')
ax.legend(handles=legend_elements, loc='lower left')

# 2. Pressure (top-right)
ax = axes[0, 1]
contour_p = ax.contourf(x, y, np.transpose(p))
plt.colorbar(contour_p, ax=ax, label='Pressure [Pa]')
ax.contour(x, y, np.transpose(T), [t_melt], colors='red', linewidths=2)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Pressure', fontsize=32)
ax.set_aspect('equal')
ax.tick_params(direction='in')
ax.legend(handles=legend_elements, loc='lower left')

# 3. Streamlines (bottom-left)
ax = axes[1, 0]
ax.streamplot(xx, yy, u_T, v_T, color=np.sqrt(u_T*u_T + v_T*v_T),
               density=1.5, linewidth=1.5, cmap='viridis')
ax.contour(x, y, np.transpose(T), [t_melt], colors='red', linewidths=2)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Streamlines', fontsize=32)
ax.set_ylim([0, Ly])
ax.set_xlim([0, Lx])
ax.set_aspect('equal')
ax.tick_params(direction='in')
ax.legend(handles=legend_elements, loc='lower left')

# 4. Velocity Magnitude (bottom-right)
ax = axes[1, 1]
im = ax.imshow(vel_mag.T, origin='lower', cmap='viridis', extent=[0, Lx, 0, Ly])
plt.colorbar(im, ax=ax, label='velocity [m/s]')
ax.contour(x, y, np.transpose(T), [t_melt], colors='red', linewidths=2)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Velocity Magnitude', fontsize=32)
ax.set_aspect('equal')
ax.tick_params(direction='in')
ax.legend(handles=legend_elements, loc='lower left')

plt.savefig('figures/figure_2x2.png', dpi=300, bbox_inches='tight')
