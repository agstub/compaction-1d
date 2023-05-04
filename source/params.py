# All model/numerical parameters are set here.
import numpy as np

# universal parameters
g0 = 9.81                          # Gravitational acceleration m/s^2

# boundary condition parameters
sigma0 = 30*1e3                    # Normal stress at lower boundary (Pa)
u0 = 100/3.154e7                   # Ice flow speed prescribed at top boundary

# ice properties
A0 = 1e-24                         # Glen's law coefficient "ice softness" (Pa^{-n}/s)
n = 3.0                            # Glen's law exponent
B0 = A0**(-1/n)                    # Ice hardness (Pa s^{1/n})
B = (2**((n-1.0)/(2*n)))*B0        # "2*Viscosity" constant in weak form (Pa s^{1/n})
rm2 = 1 + 1.0/n - 2.0              # Exponent in weak form: r-2
rho_i = 917.0                      # Density of ice
eta0_i = 5e13                      # viscosity at zero deviatoric stress 
eps_v = (2*eta0_i/B)**(2.0/rm2)    # Flow law regularization parameter

# water properties
rho_w = 1000.0                     # Density of water kg / m^3
eta_w = 1.8e-3                     # Water viscosity Pa s

# sediment properties
eta0_s = 1e20                      # sediment viscosity (infinity results in perfect plasticity)
rho_s = 2000.0                     # Density of water (kg/m^3)       
mu = 0.5                           # Friction coefficient (dimensionless)
Ks = 1e-12                         # till permeability (m^2)
Ki = 1e-12                         # ice permeability (m^2)
a = 2                              # permeability exponent

# Domain parameters
L = 2.0                            # Length of the domain (m)
H = 2.0                            # Height of the domain (m)
Nx = 20                            # Number of elements in x direction
Nz = 20                            # Number of elements in z direction

# Time-stepping parameters
t_f = (1.0/12.0)*3.154e7           # Final time s
nt = 100                           # Number of time steps
t = np.linspace(0,t_f,nt)          # time array
dt = t_f/nt                        # Timestep size
theta = 0.5                        # time-integration parameter (0=forward euler, 0.5 = trapezoid, 1 = backward euler)

# output parameters
save_vtk = False                   # Flag for saving solutions in VTK format
