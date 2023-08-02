# these are the main physical parameters that determine the parameters in params.py
# I made this file so that it is easier to set all parameters from a notebook
H0 = 0.02                                     # initial column height (m)
w0 = 100/3.154e7                              # compaction rate scale (m/s)
phi0 = 0.5                                    # initial porosity 
eta = 1e7                                     # Newtonian viscosity (Pa s)
k0 = 1e-14                                    # permeability pre-factor (m^2)
t_f = 100*60                                  # Final time (s)
Pi0 = 10*1000                                 # Plastic stress scale