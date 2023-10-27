# all physical and numerical parameters are set here.
import numpy as np
import meta_params as mp

# main parameters to set
# these are set in the Jupyter notebook (or the meta_params module)
H0 = mp.H0                                    # initial column height (m)
w0 = mp.w0                                    # compaction rate scale (m/s)
phi0 = mp.phi0                                # initial porosity 
eta = mp.eta                                  # Newtonian viscosity (Pa s)
k0 = mp.k0                                    # permeability pre-factor (m^2)
Pi0 = mp.Pi0                                  # Plastic yield stress scale

# dimensional parameters:
g = 9.81                                      # gravitational acceleration (m/s^2)
rho_s = 917                                   # ice density (kg/m^3)
rho_f = 1.293                                 # density of dry air at O oC (kg/m^3)
mu = 1.7e-5                                   # air viscosity (Pa s)
zeta = eta/phi0                               # initial bulk viscosity (Pa s)
a,b = 3,2                                     # permeability exponents
m,n = 2,2                                     # plasticity exponents
delta = np.sqrt((k0/mu)*((4./3.)*eta + zeta)) # compaction length (m)

# main nondimensional parameters:
beta = k0*g*(rho_s - rho_f)/(mu*w0)
gamma = k0*Pi0/(mu*w0*H0)
eps = H0**2/delta**2 

# domain parameters:
nz = 500                                      # Number of elements in z direction
dz = 1/(nz+1)

# time-stepping parameters:
t_f = mp.t_f/(H0/w0)                          # Final time (relative to time scale)
nt = 5000 #2*int(t_f/dz)                      # Number of time steps
dt = t_f/nt                                   # Timestep size
theta = 0.5                                   # time-integration parameter 
                                              # (0=forward euler, 0.5 = trapezoid, 1 = backward euler)

# misc:
phi_min = 1e-3                                # minimum porosity constraint
rho_b = (1-phi0)*rho_s + phi0*rho_f           # initial bulk density (kg/m^3)

