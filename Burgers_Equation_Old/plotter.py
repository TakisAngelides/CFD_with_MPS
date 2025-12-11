import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import curve_fit

N = 8
nu = 0.01
force_amplitude = 0.01
n_steps = 20000

with h5py.File(f"Data/N_{N}_nu_{nu}_fAmp_{force_amplitude}_steps_{n_steps}.h5", "r") as f:
    
    x = {}
    y = {}
    
    for name, item in f.items():
        
        if "x" in name:
            x[int(name[2:])] = np.array(item)
        else:
            y[int(name[2:])] = np.array(item)
            
    x = sorted(x.items())
    x_data = [value for key, value in x]
    
    y = sorted(y.items())
    y_data = [value for key, value in y] # y_data has two indices (t, x)

def create_animation():  
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    line, = ax.plot([], [], marker = None, linestyle = '-', color = 'skyblue')

    # Set axis limits
    ax.set_xlim(0, max(map(max, x_data)))
    ax.set_ylim(-max(map(max, list(-np.array(y_data)))), max(map(max, y_data)))

    # Function to initialize the plot
    def init():
        line.set_data([], [])
        return line,

    # Function to update the plot in each frame
    def update(frame):
        line.set_data(x_data[frame], y_data[frame])
        return line,

    # Create the animation
    num_frames = 100
    animation = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)
    plt.show()

def get_average_velocity_fixed_r_n(r, n):

    x, y = np.array(x_data), np.array(y_data) # y are the velocities (t, x)

    t_range, x_range = y.shape

    y_diff = []
    
    for t_idx in range(t_range):
        tmp = np.abs(np.roll(y[t_idx], -r) - y[t_idx])**n
        y_diff.append(np.average(tmp[:-1]))
        
    y_diff = np.average(y_diff)
        
    return y_diff

def get_average_velocity_fixed_n(n):

    x, y = np.array(x_data), np.array(y_data) # y are the velocities (t, x)
        
    t_range, x_range = y.shape
    
    avg_vel = []
    
    r_range = np.arange(1, 15)
    
    for r in r_range:
        
        avg_vel.append(get_average_velocity_fixed_r_n(r, n))
    
    # avg_vel_der = []
    # r_range_der = range(1, len(avg_vel)-1)
    # for i in r_range_der:
    #     avg_vel_der.append(avg_vel[i+1]-avg_vel[i-1])
    
    # plt.xscale('log', base = 10)
    # plt.yscale('log', base = 10)
    # plt.plot(r_range, avg_vel, '-o')
    # plt.title(f'N = {N}, $\\nu$ = {nu}, Force Amplitude = {force_amplitude}, n = {n}')
    # plt.show()
    
    # plt.plot(r_range_der, avg_vel_der)
    # plt.show()
    
    # plt.xscale('log', base = 10)
    # plt.yscale('log', base = 10)
    # plt.scatter(r_range[30:40], avg_vel[30:40])
    # plt.show()
    
    return r_range[:5], avg_vel[:5]

def fit_for_scaling_exponent(x_fit, y_fit):

    def power_function(x, z, a):
        return a * x ** z 
    
    params, covariance = curve_fit(power_function, x_fit, y_fit)

    # Extract the fitted parameter (z)
    fitted_z = params[0]

    # Generate fitted curve using the fitted parameter
    fitted_curve = power_function(x_fit, fitted_z, params[1])
    
    plt.xscale('log', base = 10)
    plt.yscale('log', base = 10)
    plt.plot(x_fit, y_fit)
    plt.plot(x_fit, fitted_curve, label = 'fit', linestyle = '--')
    plt.legend()
    plt.show()
    
    return fitted_z

# n_list = np.linspace(0.5, 5, 20)
n_list = np.linspace(0.5, 5, 10) # np.arange(1, 6)
fitted_z_list = []
for n in n_list:
    
    r_range, avg_vel = get_average_velocity_fixed_n(n)

    res = fit_for_scaling_exponent(r_range, avg_vel)

    print('n =', n, ', exponent =', res)

    fitted_z_list.append(res)

plt.plot(n_list, fitted_z_list, '-x', c = 'k')
plt.ylabel('$\\zeta_n$')
plt.xlabel('$n$')
plt.title(f'N = {N}, $\\nu$ = {nu}, Force Amplitude = {force_amplitude}')
# plt.show()
plt.savefig('zeta_vs_n.png', bbox_inches = 'tight')
