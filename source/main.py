#------------------------------------------------------------------------------------
# This program solves a nonlinear Stokes problem describing ice-shelf response to 
# sub-ice-shelf melting or freezing anomalies. The code relies on FEniCSx-see README
#------------------------------------------------------------------------------------
import os

import matplotlib.pyplot as plt
import numpy as np
from dolfinx import io
from dolfinx.fem import Expression, Function, FunctionSpace
from dolfinx.mesh import create_rectangle, refine
from mesh_routine import get_surfaces, get_vel, move_mesh
from mpi4py import MPI
from params import H, L, Nx, Nz, nt, save_vtk,dt
from stokes import stokes_solve,eta


def solve(m,sol_0):
    # solve the mixture model given:
    # m: melting/freezing rate field 
    # sol_0: initial conditions
    #
    # the solution sol = (u,phii,phiw,pe,pw) returns:
    # phii: ice vol. fraction 
    # phiw: water vol. fraction
    # u: solid velocity (ice+sediment)
    # pw: water pressure
    # pe: effective pressure = p_m - p_w (p_m is solid pressure)
    #
    # the sediment vol. fraction phis is recovered via:
    # phis = 1-phiw-phii

    # generate mesh
    p0 = [-L/2.0,0.0]
    p1 = [L/2.0,H]
    domain = create_rectangle(MPI.COMM_WORLD,[p0,p1], [Nx, Nz])

    # # Begin time stepping
    t=0
    for i in range(nt):

        sol_n = sol_0

        print('Iteration '+str(i+1)+' out of '+str(nt)+' \r',end='')

        # Solve the mixture problem for sol = (phi_i,phi_w,u,p_w,p_e)
        sol = stokes_solve(domain,m,sol_n)
     
  
        # save the stokes solution 
        if save_vtk == True:
            with io.VTXWriter(domain.comm, "../results/output.bp", [sol.sub(0).sub(1)]) as vtx:
                vtx.write(0.0)

        t += dt
 
    return sol