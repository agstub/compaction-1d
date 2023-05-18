# all physical and numerical parameters are set here.
import numpy as np
import meta_params as mp

# main parameters to set:
H0 = mp.H0                                     # initial column height (m)
w0 = mp.w0                                     # prescribed compaction rate (m/s)
phi0 = mp.phi0                                 # initial porosity 
eta = mp.eta                                   # Newtonian viscosity (Pa s)
k0 = mp.k0                                     # permeability pre-factor (m^2)

# dimensional parameters:
g = 9.81                                      # gravitational acceleration (m/s^2)
rho_s = 917                                   # ice density (kg/m^3)
rho_f = 1.293                                 # density of dry air at O oC (kg/m^3)
mu = 1.7e-5                                   # air viscosity (Pa s)
zeta = eta/phi0                               # initial bulk viscosity (Pa s)
a,b = 3,2                                     # permeability exponents
delta = np.sqrt((k0/mu)*((4./3.)*eta + zeta)) # compaction length (m)

# main nondimensional parameters:
gamma = g*(rho_s - rho_f)*(H0**2)/(np.abs(w0)*((4./3.)*eta + zeta))
eps = H0/delta

# domain parameters:
nz = 500                                      # Number of elements in z direction
dz = 1/(nz+1)

# time-stepping parameters:
t_f = mp.t_f/(H0/w0)                           # Final time (relative to time scale)
nt = 2*int(t_f/dz)                            # Number of time steps
dt = t_f/nt                                   # Timestep size
theta = 0.5                                   # time-integration parameter 
                                              # (0=forward euler, 0.5 = trapezoid, 1 = backward euler)

# misc:
phi_min = 1e-2                                # minimum porosity constraint
rho_b = (1-phi0)*rho_s + phi0*rho_f           # initial bulk density (kg/m^3)

