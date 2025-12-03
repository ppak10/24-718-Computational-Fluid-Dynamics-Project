
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
    plt.show()

def VelocityField(x,y,u,v,q):

    xx, yy = np.meshgrid(x, y)
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
    
    plt.figure()
    plt.quiver(xx,yy,u,v, scale= 10, cmap='viridis')
    plt.xlabel('x [m]', fontsize = 14 )
    plt.ylabel('y [m]', fontsize = 14 )
    #plt.title(f'Velocity Vectors at t = 20s, q = {q}', fontsize = 14)
    plt.tick_params(labelsize=12)
    plt.ylim([0,0.003])
    plt.xlim([0,0.003])

    plt.show()