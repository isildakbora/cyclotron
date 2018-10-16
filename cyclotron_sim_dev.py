# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as math
from numba import vectorize, jit
import time
# import seaborn as sns
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import pandas as pd
import itertools
import math 

# sns.set()
beginning_time = time.time()

# SPEED OF LIGHT is 299792458 m/s
speed_of_light = 3.0E08

# number pi
pi = math.pi

number_of_points = 200

class Particle:
    def __init__(self, pos, vel, mass, charge, energy=0):
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.charge = charge
        self.energy = energy


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='|'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:."+str(decimals)+"f}").format(100*(iteration/float(total)))
    filledLength = int(length*iteration // total)
    bar = fill*filledLength+'-'*(length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


# ## Set the Design Energy of The Cyclotron
# <br>As an example, let's set the desired energy of the cyclotron as 2 MeV. Simulation will stop when this energy is achieved. Also, we will set the separation of the dees (in meter) and the voltage (in V) applied between the plates. Then we will create our particle object which is defined before in the code and name it proton. <br>


# DESIRED ENERGY OF THE OUTCOMING IONS
desired_energy = 4E5
dee_sep = 1.5E-2
HV = 5.E3
# construct a particle named as "proton"
proton = Particle((1.4E-3, 0.0, 0.0), [0.0, 0.0, 0.0], 1.67E-27, +1.60E-19, )

# ## Create the Magnetic and Electric Fields
# <br> Our dipole magnet will create a uniform (hypothetically) magnetic field in the $z$-direction. So we must create a vector as
# <h3 align="center"> $\mathbf{B}=0\mathbf{i}+0\mathbf{j}+B_z\mathbf{k}$ </h3>
#
# where $B_z$ is -1.1 Tesla


# MAGNETIC FIELD
df = pd.read_csv('/Users/boraisildak/Documents/Github/cyclotron/mag_field_test.csv', na_filter=False, delimiter=",")


x_mesh = np.arange(-230, 230, 5)
y_mesh = np.arange(-230, 230, 5)
z_mesh = np.arange(-20, 20, 5)

@jit
def get_bounding_cube(point, x_mesh, y_mesh, z_mesh):
    x_bound_up = x_mesh.searchsorted(1E3*point[0])
    x_bound = [x_mesh[x_bound_up-1], x_mesh[x_bound_up]]
    y_bound_up = y_mesh.searchsorted(1E3*point[1])
    y_bound = [y_mesh[y_bound_up-1], y_mesh[y_bound_up]]
    z_bound_up = z_mesh.searchsorted(1E3*point[2])
    z_bound = [z_mesh[z_bound_up-1], z_mesh[z_bound_up]]
    
    bounding_cube = np.asarray(list(itertools.product(x_bound, y_bound, z_bound)))
    bounding_cube = bounding_cube.astype(float)
    return bounding_cube
@jit
def get_B_IDW(datasample, point, bounding_points, p=2, metric=[1,1,1]):
    # get B values wrt Inverse Distance Weighting (IDW)
    #https://en.wikipedia.org/wiki/Inverse_distance_weighting
    B_x_idw = 0
    B_y_idw = 0
    B_z_idw = 0
    w_sum = 0
    for r in bounding_points:    
        del_x = abs(1E3*point[0]-r[0])
        del_y = abs(1E3*point[1]-r[1])
        del_z = abs(1E3*point[2]-r[2])
        d = (metric[0]*del_x**2 + metric[1]*del_y**2 + metric[2]*del_z**2)**0.5
        
        if(d==0):
            d=1.

        w = float(1./d**p)
        mask = (datasample.x == r[0]) & (datasample.y == r[1]) & (datasample.z == r[2])
        #print(float(df[mask].Bz))
        B_x_idw += w*float(df[mask].Bx)
        B_y_idw += w*float(df[mask].By)
        B_z_idw += w*float(df[mask].Bz)
        w_sum += w
    #print(B_idw/w_sum)
    if(w_sum==0):
        w_sum = 8.
    B = np.array([B_x_idw/w_sum, B_y_idw/w_sum, B_z_idw/w_sum])
    return B

# ELECTRIC FIELD
B_mean = float(df[(abs(df.x) < 20) & (abs(df.y) < 20) & (df.z == 0)].Bz.mean())
print(B_mean)
q, B, m = proton.charge, B_mean, proton.mass
w = q*B/m
phase = 0.0*(pi)

print("Gyro-Frequency = %2.2f rad/ns" % (1E-9*(q*B/m)))
print("Frequency = %2.2f MHz" % (1E-6*w/(2*pi)))

@jit
def e_field(t, phi=phase):
    # if(np.cos(w*t+phi)<0):
    #     E = [-(HV/dee_sep), 0.0, 0.0]
    # else:
    #     E = [(HV/dee_sep), 0.0, 0.0]
    E = [(HV/dee_sep)*np.sin(w*t+phi), 0.0, 0.0]
    return E

#  Returns the acceleration vector due to an electromagnetic field ( from Lorentz force )
@jit
def em_acceleration(q_over_m, position, velocity, magnetic_field, t):
    #     calculated for a particle at position, with velocity

    if abs(position[0]) >= dee_sep:
        a = q_over_m*np.cross(velocity, magnetic_field)
    else:
        a = q_over_m*(np.array(e_field(t))+np.cross(velocity, magnetic_field))

    return a

v_i = np.linalg.norm(proton.vel)
expected_radius = v_i/((q*B/m))
print("Expected initial radius = %2.2f mm" % (1E3*expected_radius))

expected_period = 2.0*pi/(B*(proton.charge/proton.mass))
print("Expected period = %2.2f ns" % (1E9*expected_period))

delta_t = expected_period/number_of_points
print("delta_t = %2.2f ns" % (1E9*delta_t))
# Count how many times the particle jumps from 1 D to the other
jumps = 0
jumps_max = int(desired_energy/(proton.charge*HV))



# MODIFIED EULER METHOD
# 2nd and the 3rd orders corrections due to Taylor's expansion
def mdf_euler(particle, desired_energy, delta_t):
    print("Calculating the ion path by using a modified Euler's method.")
    q_over_m = particle.charge/particle.mass
    results = []
    energy = []

    i = 0
    t = 0
    p0 = np.array(particle.pos)
    v0 = np.array(particle.vel)

    # Distance traveled
    s = 0

    aux_index = 0

    while 0.5*(particle.mass)*(np.linalg.norm(v0)**2)/proton.charge < (desired_energy):
        # for i in range(int(4E+4)):
        bounding_cube = get_bounding_cube(p0,x_mesh,y_mesh,z_mesh)
        B  = get_B_IDW(df, p0, bounding_cube)
        a = em_acceleration(q_over_m, p0, v0, B, t)

        delta_a = em_acceleration(q_over_m, p0, v0, B, t+delta_t) - em_acceleration(q_over_m, p0, v0, B, t)

        p0 = p0+delta_t*v0+0.5*(delta_t**2)*a+(1/6)*(delta_t**2)*delta_a
        v0 = v0+delta_t*a+0.5*delta_t*delta_a
        results.append(p0)
        i += 1
        t += delta_t
        s += delta_t*np.linalg.norm(v0)
        energy.append(0.5*(particle.mass)*(np.linalg.norm(v0)**2)/proton.charge)

        if int(100*energy[-1]/desired_energy) > aux_index:
            printProgressBar(int(100*energy[-1]/desired_energy), 100, prefix='Accelerating the Ion:',
                             suffix='Complete', length=50)
            aux_index += 1

    print("Euler method finished!")
    return s, results, energy


# 4-th ORDER RUNGE-KUTTA METHOD
@jit
def rk4(particle, desired_energy, delta_t):
    q_over_m = q/m
    results = []

    #  Initial conditions
    i = 0
    p0 = np.array(particle.pos)
    v0 = np.array(particle.vel)
    t = 0
    energy = []

    # Distance traveled
    s = 0

    aux_index = 0

    while 0.5*particle.mass*(np.linalg.norm(v0)**2)/proton.charge < (desired_energy):
        # for i in range(int(4E+4)):

        #if(i%100==0):
        #   print(i)
        if i > int(1E5):
            print("Ion path is probably not stable!")
            break
        bounding_cube = get_bounding_cube(p0,x_mesh,y_mesh,z_mesh)
        B  = get_B_IDW(df, p0, bounding_cube)
        #print(p0, B)
        #B = non_uniform_magnetic_field
        
        p1 = p0
        v1 = v0
        a1 = delta_t*em_acceleration(q_over_m, p1, v1, B, t)
        v1 = delta_t*v1

        p2 = p0+(v1*0.5)
        #bounding_cube = get_bounding_cube(p2,x_mesh,y_mesh,z_mesh)
        #B  = get_B_IDW(df, p2, bounding_cube)
        v2 = v0+(a1*0.5)
        a2 = delta_t*em_acceleration(q_over_m, p2, v2, B, t)
        v2 = delta_t*v2

        p3 = p0+(v2*0.5)
        #bounding_cube = get_bounding_cube(p3,x_mesh,y_mesh,z_mesh)
        #B  = get_B_IDW(df, p3, bounding_cube)
        v3 = v0+(a2*0.5)
        a3 = delta_t*em_acceleration(q_over_m, p3, v3, B, t)
        v3 = delta_t*v3

        p4 = p0+v3
        #bounding_cube = get_bounding_cube(p4,x_mesh,y_mesh,z_mesh)
        #B  = get_B_IDW(df, p4, bounding_cube)
        v4 = v0+a3
        a4 = delta_t*em_acceleration(q_over_m, p4, v4, B, t)
        v4 = delta_t*v4

        dv = (a1+2.0*(a2+a3)+a4)
        v0 = v0+dv/6.0

        dp = (v1+2.0*(v2+v3)+v4)
        p0 = p0+dp/6.0

        results.append(p0.copy())
        t += delta_t
        i += 1
        s += np.linalg.norm(dp)/6
        energy.append(0.5*(particle.mass)*(np.linalg.norm(v0)**2)/proton.charge)

        if int(100*energy[-1]/desired_energy) > aux_index:
            # print(int(100*energy/desired_energy), aux_index)
            printProgressBar(int(100*energy[-1]/desired_energy), 100, prefix='Accelerating the Ion:',
                             suffix='Complete', length=50)
            aux_index += 1

    print("Runge-Kutta method finished!")
    return s, results, energy

s, results, energy = rk4(proton, desired_energy, delta_t)
print("Distance traveled by the ion:%8.2f m" % s)

# PLOTTING
plt.style.use('seaborn')
# fig1 = plt.figure()
# ax[0 ,0] = fig1.add_subplot(1, 1, 1)

# fig2 = plt.figure()
# ax[0, 1] = fig2.add_subplot(1, 1, 1)

# fig3 = plt.figure()
# ax[1, 0] = fig3.add_subplot(1, 1, 1)

# fig4 = plt.figure()
# ax[1, 1] = fig4.add_subplot(1, 1, 1)

# fig5 = plt.figure()
# ax[0, 2] = fig5.add_subplot(1, 1, 1)

width = 15
fig, axs = plt.subplots(2, 3, figsize=(width, 0.6*width))


@jit
def part_plot(particle, max_iter, method, delta_t):
    #  Mark the original position with a blue mark
    x = []
    y = []
    z = []

    x.append(particle.pos[0])
    y.append(particle.pos[1])
    z.append(particle.pos[2])
    axs[0, 0].scatter(x, y, color='blue')

    vz = []
    v0 = np.linalg.norm(particle.vel)
    #  use z array to save the velocity
    vz.append(v0)

    r = [np.linalg.norm(particle.pos[0:2])]

    energy[0:0] = [0.]

    print("initial position", x[0], y[0], vz[0])
    #   print " initial velocity", v0/speed_of_light, 'the speed of light'
    print("initial velocity %1.4f the speed of light" % (v0/speed_of_light))

    # save the positions when in the spacing in a separate array
    #    so that w can change the color to red
    xc = []
    yc = []
    x = []
    y = []

    i = 0
    length = len(results)

    for p in results:

        vz.append(p[2])
        r.append(np.linalg.norm(p[0:2]))

        if p[0] >= dee_sep or p[0] <= -dee_sep:
            #inside the Dee's
            if len(xc):
                axs[0, 0].plot(xc, yc, color='red', linewidth=0.95)
                xc = []
                yc = []
            x.append(p[0])
            y.append(p[1])
            z.append(p[2])
        else:
            # inside the spacing
            if len(xc):
                axs[0, 0].plot(x, y, color='blue', linewidth=0.95)
                x = []
                y = []
            xc.append(p[0])
            yc.append(p[1])
            z.append(p[2])
        printProgressBar(i+1, length, prefix='Plotting Ion Path:', suffix='Complete', length=50)
        i += 1

    if len(xc):
        axs[0, 0].plot(xc, yc, color='red', linewidth=0.95)
        xc = []
        yc = []
    if len(x):
        axs[0, 0].plot(x, y, color='blue', linewidth=0.95)
        x = []
        y = []

    print("number of jumps between D's is", jumps)
    num_points = len(vz)
    #   print "final position", x[num_points-1],y[num_points-1],z[num_points-1]
    print('number of points is', num_points, '*delta_t is total time = ', delta_t*num_points)

    axs[0, 0].set_title("Ion Position - Cyclotron")
    axs[0, 0].set_xlabel("Dimension-X (m)")
    axs[0, 0].set_ylabel("Dimension-Y (m)")

    t = np.linspace(0, len(z)*delta_t, len(vz))
    axs[0, 1].plot(np.multiply(r, 1E3), np.multiply(z, 1E3))

    axs[0, 1].set_xlabel("Radius (mm)")
    axs[0, 1].set_ylabel("z (mm)")

    axs[1, 0].plot(np.multiply(t, 1E6), np.multiply(r, 1E3))
    axs[1, 0].set_title("Time vs. Radius")
    axs[1, 0].set_xlabel("Time ("+chr(956)+")s")
    axs[1, 0].set_ylabel("Radius (mm)")

    axs[1, 1].plot(np.multiply(t, 1E6), np.multiply(z, 1E3))
    axs[1, 1].set_title("Time vs. z")
    axs[1, 1].set_xlabel("Time ("+chr(956)+"s)")
    axs[1, 1].set_ylabel("z (mm)")

    axs[0, 2].plot(np.multiply(t, 1E6), np.multiply(energy,1E-3))
    axs[0, 2].set_title("Time vs. Energy")
    axs[0, 2].set_xlabel("Time ("+chr(956)+"s)")
    axs[0, 2].set_ylabel("Energy (keV)")

    results_to_save = np.array(results).T
    x = np.insert(results_to_save[0], 0, 0.0)
    y = np.insert(results_to_save[1], 0, 0.0)
    z = np.insert(results_to_save[2], 0, 0.0)

    df = pd.DataFrame({"t" : t, "x" : x, "y" : y, "z" : z, "E" : energy})
    df.to_csv("cyclotron_path.csv", index=False)

    #print(len(t), len(x), len(y), len(z), len(energy))

t1 = time.time()
part_plot(proton, desired_energy, 'rk4', delta_t)
print(time.time() - beginning_time)
plt.tight_layout()
plt.show()
plt.savefig("plots.pdf")
