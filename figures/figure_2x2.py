import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Load data from results/2_medium last timestep
data_dir = "results/2_medium/99999"

p = np.load(f"{data_dir}/p.npy")
u = np.load(f"{data_dir}/u.npy")
v = np.load(f"{data_dir}/v.npy")
T = np.load(f"{data_dir}/T.npy")

# Calculate velocity magnitude
# u has shape (Nx+2, Ny+1), v has shape (Nx+1, Ny+2)
# Need to interpolate to cell centers for proper velocity magnitude
u_center = 0.5 * (u[1:, :] + u[:-1, :])  # Average adjacent u values
v_center = 0.5 * (v[:, 1:] + v[:, :-1])  # Average adjacent v values
vel_mag = np.sqrt(u_center**2 + v_center**2)

# Grid dimensions
Nx, Ny = p.shape[0] - 1, p.shape[1] - 1
Lx, Ly = 0.003, 0.003  # From main.py
x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

# Melt pool temperature threshold
tMelt = 700  # K

# Create 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Pressure (top-left)
ax = axes[0, 0]
contour_p = ax.contourf(X, Y, p, levels=50, cmap='RdBu_r')
plt.colorbar(contour_p, ax=ax, label='Pressure (Pa)')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Pressure Field')
ax.set_aspect('equal')

# 2. Streamlines (top-right)
ax = axes[0, 1]
# Create streamline plot on cell centers
x_center = 0.5 * (x[1:] + x[:-1])
y_center = 0.5 * (y[1:] + y[:-1])
X_center, Y_center = np.meshgrid(x_center, y_center, indexing='ij')
ax.streamplot(X_center, Y_center, u_center, v_center, density=2, color='k', linewidth=1)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Streamlines')
ax.set_aspect('equal')
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)

# 3. Velocity Magnitude (bottom-left)
ax = axes[1, 0]
contour_vel = ax.contourf(X_center, Y_center, vel_mag, levels=50, cmap='viridis')
plt.colorbar(contour_vel, ax=ax, label='Velocity Magnitude (m/s)')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Velocity Magnitude')
ax.set_aspect('equal')

# 4. Melt Pool Contour (bottom-right)
ax = axes[1, 1]
temp_plot = ax.contourf(X, Y, T, levels=50, cmap='hot')
plt.colorbar(temp_plot, ax=ax, label='Temperature (K)')
# Overlay melt pool contour
contour_melt = ax.contour(X, Y, T, levels=[tMelt], colors='cyan', linewidths=2)
ax.clabel(contour_melt, inline=True, fontsize=10, fmt='Melt Pool (700K)')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Temperature & Melt Pool Contour')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('figures/2x2_plots.png', dpi=300, bbox_inches='tight')
plt.show()
