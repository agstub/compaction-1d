#------------------------------------------------------------------------------------
# This program solves a Darcy-Stokes mixture model describe glacial ice flow over a 
# deformable and permeabel till layer. The code relies on FEniCSx-see README.
#------------------------------------------------------------------------------------

from dolfinx import io
from params import  nt, save_vtk,dt
from stokes import stokes_solve


def solve(domain,m,initial):
    # solve the mixture model given:
    # domain: the computational domain
    # m: melting/freezing rate field 
    # initial: initial conditions 
    # *see example.ipynb for an example of how to set these
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

    # # Begin time stepping
    t=0
    for i in range(nt):

        sol_n = initial

        print('Iteration '+str(i+1)+' out of '+str(nt)+' \r',end='')

        # Solve the Darcy-Stokes problem for sol = (phi_i,phi_w,u,p_w,p_e)
        sol = stokes_solve(domain,m,sol_n)
     
  
        # # save the stokes solution 
        # if save_vtk == True:
        #     with io.VTXWriter(domain.comm, "../results/output.bp", [sol.sub(0).sub(0)]) as vtx:
        #         vtx.write(0.0)

        sol_n = sol

        t += dt
 
    return sol