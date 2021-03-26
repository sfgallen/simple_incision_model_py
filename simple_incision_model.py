
"""
simple_incision_model.py is a basic script to solve the transient stream power
model using explicit finite difference techniques. In addition to typical 
python packages included with many python platforms (e.g. Anaconda), the
script calls spim_ftfd.py, which contains to functions for specific 
calculations

Author: Sean F. Gallen
Date modified: 03/26/2021
Contact: sean.gallen[at]colostate.edu
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import spim_ftfd as spm


# model run time and plotting info
run_time = 0.25e6
stabil = 3
n_plots = 6

# define the domain
l_crit = 1e3
l_max = 1e5
dx = 100
L = np.arange(l_crit,l_max,dx)

# define hack's constants and drainage area
ka = 6.7
h = 1.7
A = ka*L**h

# define the uplift rates and stream power parameters

Ui = 0.5e-3 # initial uplift rate
Uf = 1e-3   # final uplift rate

Ki = 1e-5   # initial erodibility
Kf = 1e-5   # final erodibility

m = 0.5     # DA exponent
n = 1.0     # slope exponent

mn = m/n    # concavity

## Calculate the initial steady-state elevtion of the river
[Z,S,X] = spm.steady_state_profile(Ui,Ki,m,n,A,L)

# calculate chi
Achi = A**-mn
Achi = np.flipud(Achi)
chi = np.zeros(np.size(Achi))

for i in range(1,len(chi)):
    dchi = ((Achi[i]+Achi[i-1])/2)*(X[i-1]-X[i])
    chi[i] = chi[i-1] + dchi
    
chi = np.flipud(chi)

# Calculate the finali steady state elevation
[Zf,Sf,_] = spm.steady_state_profile(Uf,Kf,m,n,A,L)

# CLF criterion to determine stable timestep
vi = dx/(Ki*A**m*S**(n-1))
vf = dx/(Kf*A**m*S**(n-1))
vel = np.concatenate((vi,vf))
dt = min(vel)

## plot the initial and final conditions
# calculate initial erosion rate
E = Ki*A**m*S**n

# plot the initial conditions
ax1 = plt.subplot(2,2,1)
ax1.plot(L/1000,Z,'k-')
ax1.set_xlabel('Distance (km)')
ax1.set_ylabel('Elevation (m)')

ax2 = plt.subplot(2,2,2)
ax2.plot(A,S,'k-')
ax2.set_xlabel('Drainage area (m^2)')
ax2.set_ylabel('Slope')
ax2.set_xscale('log')
ax2.set_yscale('log')


ax3 = plt.subplot(2,2,3)
ax3.plot(L/1000,E*1000,'k-')
ax3.set_xlabel('Distance (km)')
ax3.set_ylabel('Erosion rate (mm/yr)')

ax4 = plt.subplot(2,2,4)
ax4.plot(chi,Z,'k-')
ax4.set_xlabel('$\chi$')
ax4.set_ylabel('Elevation (m)')
ax4.invert_xaxis()

# calculate final erosion rate
Ef = Kf*A**m*Sf**n

# plot the initial conditions
ax1.plot(L/1000,Zf,'r-')

ax2.plot(A,Sf,'r-')

ax3.plot(L/1000,Ef*1000,'r-')

ax4.plot(chi,Zf,'r-')


## prepare for the forloop
# time loop prep
t_steps = int(np.ceil(run_time/dt))
t_plots = int(np.floor(t_steps/n_plots))

# allocate memory to tack time varying variables
E_mean = np.empty([t_steps,1])
Z_mean = E_mean
mod_time = np.zeros([t_steps,1])

E_mean[0] = np.mean(E)
Z_mean[0] = np.mean(Z)

# set up waitbar
pbar = tqdm(total=t_steps)

# Initiate the time forloop
for t in range(t_steps):
    
    # Calculate the erosion rate
    E = Kf*A**m*S**n
    E_mean[t] = np.mean(E)
    mod_time[t] = (t)*dt
    
    # evolve the profile
    Z_cur = Z + (Uf*dt) - (E*dt)
    
    # deal with the lower boundary condition
    Z_cur[len(Z_cur)-1] = 0
    
    # update the slope
    S = spm.calc_slope(Z_cur,L)
    
    # set Z to Z_cur
    Z = Z_cur
    
    # update Z_mean
    Z_mean[t] = np.mean(Z)
    
    # plot the results when needed
    if np.remainder(t,t_plots) == 0:
        ax1.plot(L/1000,Z,'b-',linewidth = 0.5)

        ax2.plot(A,S,'b-',linewidth = 0.5)

        ax3.plot(L/1000,E*1000,'b-',linewidth = 0.5)

        ax4.plot(chi,Z,'b-',linewidth = 0.5)
        
    # update the waitbar
    pbar.update(n=1)



ax1.plot(L/1000,Z,'g--',linewidth = 2)

ax2.plot(A,S,'g--',linewidth = 2)

ax3.plot(L/1000,E*1000,'g--',linewidth = 2)

ax4.plot(chi,Z,'g--',linewidth = 2)