
import matplotlib.pyplot as plt
import numpy as np

def myplots(dt, x, y, T):

    # ** add your code here **
    plt.figure()
    plt.contourf(x, y, np.transpose(T[:, :, -1]))
    plt.colorbar(label='Temperature [K]')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    #plt.title('Gaussian Heat Test')
    plt.savefig("report_code/contours.png")
    plt.close()

def VelocityMagnitudes(u,v):

    # u (61, 60) averaged over X (Axis 0) -> (60, 60)
    u_vis = (u[:-1, :] + u[1:, :]) / 2

    # v (60, 61) averaged over Y (Axis 1) -> (60, 60)
    v_vis = (v[:, :-1] + v[:, 1:]) / 2

    vel_mag = np.sqrt(u_vis**2 + v_vis**2)
    plt.figure()
    im = plt.imshow(vel_mag.T, origin='lower', cmap='viridis', extent=[0, 0.003, 0, 0.003])
    plt.title('Velocity Magnitude')
    plt.colorbar(im)
    plt.xlabel('x [m]', fontsize = 14 )
    plt.ylabel('y [m]', fontsize = 14 )
    plt.tick_params(labelsize=12)

    # plt.show()
    plt.savefig("report_code/velocity_magnitude.png")
    plt.close()

def VelocityField(x,y,u,v,q):

    xx, yy = np.meshgrid(x, y)
    u = np.transpose(u) * 10
    v = np.transpose(v) * 10

    plt.figure()
    plt.quiver(xx,yy,u,v, scale= 0.5, cmap='viridis')
    plt.xlabel('x [m]', fontsize = 14 )
    plt.ylabel('y [m]', fontsize = 14 )
    #plt.title(f'Velocity Vectors at t = 20s, q = {q}', fontsize = 14)
    plt.tick_params(labelsize=12)
    plt.ylim([0,0.003])
    plt.xlim([0,0.003])

    # plt.show()
    plt.savefig("report_code/quiver.png")
    plt.close()

def stream_plot(x, y, u, v, q):
    xx, yy = np.meshgrid(x, y)
    u = 0.5 * (u[1:, :] + u[:-1, :])
    v = 0.5 * (v[:, 1:] + v[:, -1:])

    u = np.transpose(u)
    v = np.transpose(v)

    plt.figure()
    plt.streamplot(xx,yy,u,v, color=np.sqrt(u*u + v*v),density=1.5,linewidth=1.5, cmap='viridis')
    plt.colorbar(label = 'velocity [m/s]')
    plt.xlabel('x [m]', fontsize = 14 )
    plt.ylabel('y [m]', fontsize = 14 )
    #plt.title(f'Streamlines at t= 20s, q = {q}', fontsize = 14)
    plt.tick_params(labelsize=12)
    plt.ylim([0,0.003])
    plt.xlim([0,0.003])

    plt.savefig("report_code/streamplot.png")
    plt.close()
