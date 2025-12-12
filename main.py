import os
import numpy as np

from datetime import datetime
from tqdm.rich import tqdm

from fractional_step_method import fsm 
from temperature_method import temperature_field_timestep 

if __name__ == "__main__":

    # Input Data ##################################################
    Lx = 0.003  #m
    Ly = 0.003  #m

    # dx = 0.0001
    # dy = 0.0001

    dx = 0.00005
    dy = 0.00005

    dx = 0.00002
    dy = 0.00002

    # t_max = 2.0  #s
    # t_max = 0.5  #s
    # t_max = 1.0 #s
    t_max = 0.1  #s
    alpha = 2*(10**-6) #m^2/s
    k = 6.7    #W/mK

    save_timestep = 10000

    dt = 0.00125
    # dt = 0.0003125
    # dt = 0.00005

    Q = 1000  #W
    q = 1
    sigma = 0.000025  #m
    xcen = Lx/2
    Tpre = 500

    tMelt = 700 # K
    rho = 4430 # kg/m^3
    mu = 0.00476 # Pa.s
    # dSigma = -1.9e-4 # N/m.K
    dSigma = 1.9e-4 # N/m.K reverse flow

    nu = mu/rho

    tol = 1e-5
    solverID = 5

    ###############################################################

    Nx = int(Lx/dx)
    Ny = int(Ly/dy)
    num_timesteps = int(t_max/dt)

    Neu_BC = np.zeros(Nx+1)
    PDF = np.zeros(Nx+1)

    for i in range(Nx+1):
        PDF[i] = (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-(((i*dx)-xcen)**2)/(2*(sigma**2)))

    for i in range(Nx+1):
        Neu_BC[i] = (Q/dx)*PDF[i]/np.sum(PDF)

    #print(Neu_BC)

    To = np.ones([Nx+1, Ny+1]) * Tpre

    To[:, Ny] = To[:, Ny-1] + (dt * Neu_BC)

    T = np.zeros((Nx+1,Ny+1)) 
    T[:,:] = To  

    T_no_velocity = np.zeros((Nx+1,Ny+1)) 
    T_no_velocity[:,:] = To  

    #print(np.transpose(To))
    
    print('Number of points (x-direction): {0:2d} '.format(Nx+1))
    print('Mesh size (dx): {0:.8f} mm'.format(dx))
    print('Mesh size (dy): {0:.8f} mm'.format(dy))
    print('Mesh size (dt): {0:.8f} mm'.format(dt))
    #print('CFL number: {0:2d} '.format(CFL))


# VELOCITY INITIALIZATION

    p = np.zeros((Nx+1,Ny+1))   
    
    # Initialize velocity field
    u = np.zeros((Nx+2,Ny+1))
    v = np.zeros((Nx+1,Ny+2))

    # Initialize velocity field
    u_conduction = np.zeros((Nx+2,Ny+1))
    v_conduction = np.zeros((Nx+1,Ny+2))

# TIME LOOP

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("runs", timestamp)
os.makedirs(run_dir, exist_ok=True)

zfill_width = len(str(num_timesteps))

for n in tqdm(range(num_timesteps)):
    p_new, u_new, v_new = fsm(u, v, p, T, nu, dt, dx, dy, mu, dSigma, tMelt, rho)
    T_new = temperature_field_timestep(u, v, alpha, dt, dx, dy, Nx, Ny, T, Neu_BC, Tpre)
    u, v, T, p = u_new.copy(), v_new.copy(), T_new.copy(), p_new.copy()
    
    T_no_velocity = temperature_field_timestep(u_conduction, v_conduction, alpha, dt, dx, dy, Nx, Ny, T_no_velocity, Neu_BC, Tpre)

    if n % save_timestep == 0 or n == num_timesteps - 1:
        timestep_dir = os.path.join(run_dir, str(n+1).zfill(zfill_width))
        os.makedirs(timestep_dir, exist_ok=True)

        np.save(os.path.join(timestep_dir, "u.npy"), u)
        np.save(os.path.join(timestep_dir, "v.npy"), v)
        np.save(os.path.join(timestep_dir, "T.npy"), T)
        np.save(os.path.join(timestep_dir, "T_no_velocity.npy"), T_no_velocity)
        np.save(os.path.join(timestep_dir, "p.npy"), p)

