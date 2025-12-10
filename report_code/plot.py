import glob
import os
import matplotlib.pyplot as plt
import numpy as np

from multiprocessing import Pool
from functools import partial

def plot_timestep(timestep_dir, Lx, Ly):
    """Generate plots for a single timestep."""
    # Load data
    u = np.load(os.path.join(timestep_dir, "u.npy"))
    v = np.load(os.path.join(timestep_dir, "v.npy"))
    T = np.load(os.path.join(timestep_dir, "T.npy"))

    u_interpolated = 0.5 * (u[1:, :] + u[:-1, :])
    v_interpolated = 0.5 * (v[:, 1:] + v[:, :-1])

    u_T = u_interpolated.T
    v_T = v_interpolated.T

    Nx, Ny = T.shape[0] - 1, T.shape[1] - 1
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)

    xx, yy = np.meshgrid(x, y)

    # Create plots subdirectory
    plots_dir = os.path.join(timestep_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Temperature contour plot
    plt.figure()
    plt.contourf(x, y, np.transpose(T))
    plt.colorbar(label='Temperature [K]')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.savefig(os.path.join(plots_dir, "contours.png"))
    plt.close()

    # Velocity magnitude plot
    vel_mag = np.sqrt(u_interpolated**2 + v_interpolated**2)

    plt.figure()
    im = plt.imshow(vel_mag.T, origin='lower', cmap='viridis', extent=[0, Lx, 0, Ly])
    plt.title('Velocity Magnitude')
    plt.colorbar(im)
    plt.xlabel('x [m]', fontsize=14)
    plt.ylabel('y [m]', fontsize=14)
    plt.tick_params(labelsize=12)
    plt.savefig(os.path.join(plots_dir, "velocity_magnitude.png"))
    plt.close()

    # Quiver
    plt.figure()
    plt.quiver(xx, yy, u_T, v_T, scale=0.5, cmap='viridis')
    plt.xlabel('x [m]', fontsize=14)
    plt.ylabel('y [m]', fontsize=14)
    plt.tick_params(labelsize=12)
    plt.ylim([0, Ly])
    plt.xlim([0, Lx])
    plt.savefig(os.path.join(plots_dir, "quiver.png"))
    plt.close()

    # Stream
    plt.figure()
    plt.streamplot(xx, yy, u_T, v_T, color=np.sqrt(u_T*u_T + v_T*v_T),
                   density=1.5, linewidth=1.5, cmap='viridis')
    plt.colorbar(label='velocity [m/s]')
    plt.xlabel('x [m]', fontsize=14)
    plt.ylabel('y [m]', fontsize=14)
    plt.tick_params(labelsize=12)
    plt.ylim([0, Ly])
    plt.xlim([0, Lx])
    plt.savefig(os.path.join(plots_dir, "streamplot.png"))
    plt.close()

    return os.path.basename(timestep_dir)


if __name__ == "__main__":
    Lx = 0.003  # m
    Ly = 0.003  # m

    # Find latest run directory
    runs_dir = "runs"
    run_dirs = sorted([d for d in glob.glob(os.path.join(runs_dir, "*")) if os.path.isdir(d)])

    latest_run = run_dirs[-1]

    # Find all timestep directories
    timestep_dirs = sorted([d for d in glob.glob(os.path.join(latest_run, "*")) if os.path.isdir(d)])

    print(f"Processing {len(timestep_dirs)} timesteps")

    # Create partial function with fixed parameters
    plot_func = partial(plot_timestep, Lx=Lx, Ly=Ly)

    # Process timesteps in parallel
    with Pool() as pool:
        results = pool.map(plot_func, timestep_dirs)

    print(f"Plots saved in '{latest_run}'")
