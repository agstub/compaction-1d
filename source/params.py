# All model/numerical parameters are set here.
import numpy as np

# import parameters to set
w0 = 111/3.154e7                            # prescribed compaction rate (abs. value) (m/s)
phi0 = 0.65                                  # reference (initial) porosity 
H0 = 0.02                                   # column height dimensional (m)
phi_min = 1e-2

eta_s = 1e9                                 # Newtonian solid viscosity (Pa s)

# dimensional parameters
g = 9.81                                    # gravitational acceleration (m/s^2)
rho_s = 917                                 # ice density (kg/m^3)
rho_f = 1.293                               # density of dry air at O oC (kg/m^3)
mu = 1.7e-5                                 # air viscosity (Pa s)
eta = (1-phi0)*eta_s                        # solid viscosity * reference porosity 
k0 = 1e-7                                   # solid permeability pre-factor (m^2)
zeta = eta/phi0                             # bulk viscosity (Pa s)
rho_b = (1-phi0)*rho_s + phi0*rho_f         # bulk density (kg/m^3)

a,b = 3,2                                   # permeability exponents

K0 = k0*(phi0**a)/((1-phi0)**b)             # permeability at phi=phi0

delta = (K0/mu)*((4./3.)*eta + zeta)        # compaction length

# main non dimensional parameter:
alpha = g*(rho_s - rho_f)*(H0**2)/(np.abs(w0)*((4./3.)*eta + zeta))
eps = H0/delta


# Domain parameters
nz = 500                                    # Number of elements in z direction
dz = 1/(nz+1)

# Time-stepping parameters
t_f = 30*60/(H0/w0)                         # Final time (relative to time scale)
nt = 2*int(t_f/dz)                          # Number of time steps
dt = t_f/nt                                 # Timestep size
theta = 0.5                                 # time-integration parameter 
                                            # (0=forward euler, 0.5 = trapezoid, 1 = backward euler)

print(rho_b)