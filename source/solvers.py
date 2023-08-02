# This file contains the functions needed for solving the compaction problem.
import numpy as np
from constitutive import K, Pi, alpha
from dolfinx.fem import (Function, FunctionSpace, dirichletbc,
                         locate_dofs_topological)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import locate_entities_boundary
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from params import beta, dt, eps, gamma, nt, nz, phi_min, theta
from petsc4py import PETSc
from post_process import get_stress, interp
from ufl import Dx, FiniteElement, TestFunction, TestFunctions, ds, dx, split


def weak_form(w,w_t,w_n,phi,phi_t,phi_n,bc_top):
    # Weak form of the residual for the compaction problem

    # weak form of momentum balance:
    F_w =  weak_form_vel(w,w_t,phi,bc_top)

    # define theta variables for time integration:  
    w_theta = theta*w + (1-theta)*w_n
    phi_theta = theta*phi + (1-theta)*phi_n

    # weak form of porosity evolution:
    F_phi = (phi-phi_n)*phi_t*dx + dt*w_theta*Dx(phi_theta,0)*phi_t*dx - dt*(1-phi_theta)*Dx(w_theta,0)*phi_t*dx 

    # add constraint phi>phi_min: phi=max(phi,phi_min)
    F_phi += (phi-0.5*(phi+phi_min + ((phi-phi_min)**2)**0.5))*phi_t*dx
    return F_w + F_phi

def weak_form_vel(w,w_t,phi,bc_top):
    # Weak form of the momentum balance
    F_w =  (eps/K(phi))*w*w_t*dx + alpha(phi)*Dx(w,0)*Dx(w_t,0)*dx  + eps*beta*(1-phi)*w_t*dx
    F_w += -eps*gamma*Pi(phi)*Dx(w_t,0)*dx

    # add stress BC if w is not prescribed at top boundary:
    if bc_top['type'] == 'stress':
        F_w += bc_top['value']*w_t*ds 

    return F_w 


def solve_pde(domain,sol_n,bc_top):
        # solves the compaction PDE problem at a given time step

        # Define function space
        P1 = FiniteElement('P',domain.ufl_cell(),1)     
        element = P1*P1
        V = FunctionSpace(domain,element)       

        sol = Function(V)
        (w,phi) = split(sol)
        (w_n,phi_n) = split(sol_n)
        (w_t,phi_t) = TestFunctions(V)

        # Mark bounadries of mesh 
        H = domain.geometry.x.max()
        facets_t = locate_entities_boundary(domain, domain.topology.dim-1, lambda x: np.isclose(x[0],H))
        facets_b = locate_entities_boundary(domain, domain.topology.dim-1, lambda x: np.isclose(x[0],0))
        dofs_t = locate_dofs_topological(V.sub(0), domain.topology.dim-1, facets_t)
        dofs_b = locate_dofs_topological(V.sub(0), domain.topology.dim-1, facets_b)
       
       # set w = 0 at base  
        bc_b = dirichletbc(PETSc.ScalarType(0), dofs_b,V.sub(0))      

        if bc_top['type'] == 'velocity':
             # set w = -1 at top
            bc_t = dirichletbc(PETSc.ScalarType(bc_top['value']), dofs_t,V.sub(0))    
            bcs = [bc_b,bc_t]
        else:
            # else Neumann condition at top is prescribed in weak form
            bcs = [bc_b]    

        # # Define weak form:
        F = weak_form(w,w_t,w_n,phi,phi_t,phi_n,bc_top)

        # set initial guess for Newton solver to be the solution 
        # from the previous time step:
        sol.sub(0).interpolate(sol_n.sub(0))
        sol.sub(1).interpolate(sol_n.sub(1))
 

        # Solve for sol = (w,phi):
        problem = NonlinearProblem(F, sol, bcs=bcs)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.solve(sol)

        return sol

def full_solve(domain,initial,bc_top):
    # solve the compaction problem given:
    # domain: the computational domain
    # initial: initial conditions 
    # bc_top: boundary condition at top (stress or velocity)
    # *see example.ipynb for an example of how to set these
    #
    # The solution sol = (u,phi) returns:
    # w: vertical velocity 
    # phi: porosity
    #
    # We also save:
    # z: domain coordinates (change over time due to compaction)
    # sigma: the effective stress (dimensional)

    w_arr = np.zeros((nt,nz+1))
    phi_arr = np.zeros((nt,nz+1))
    z_arr = np.zeros((nt,nz+1))
    sigma_arr = np.zeros((nt,nz+1))

    sol_n = initial
    phi_i = phi_arr

    # time-stepping loop
    for i in range(nt):
        print('time step '+str(i+1)+' out of '+str(nt)+' \r',end='')

        # solve the compaction problem for sol = (w,phi)
        sol = solve_pde(domain,sol_n,bc_top)
        
        # displace the mesh according to the velocity solution
        V = FunctionSpace(domain, ("CG", 1))
        w = Function(V)
        w.interpolate(sol.sub(0))
        w = w.x.array
        z = domain.geometry.x
        z[:,0] += dt*w

        # set the solution at the previous time step
        sol_n.sub(0).interpolate(sol.sub(0))
        sol_n.sub(1).interpolate(sol.sub(1))

        # save the solution as numpy arrays
        z_i,w_i = interp(sol_n.sub(0),domain)
        z_i,phi_i = interp(sol_n.sub(1),domain)
        sigma_i = get_stress(sol_n,domain)

        w_arr[i,:] = w_i
        phi_arr[i,:] = phi_i
        z_arr[i,:] = z_i
        sigma_arr[i,:] = sigma_i

    return w_arr,phi_arr, sigma_arr, z_arr


def vel_solve(domain,phi,bc_top):
        # Solve the momentum balance for a fixed porosity
        # **this is done to determine the initial velocity 
        #   given an initial porosity field**

        # Define function space
        V = FunctionSpace(domain, ("CG", 1))
   
        w = Function(V)
        w_t = TestFunction(V)
 
        # Mark bounadries of mesh and define a measure for integration
        H = domain.geometry.x.max()
        facets_t = locate_entities_boundary(domain, domain.topology.dim-1, lambda x: np.isclose(x[0],H))
        facets_b = locate_entities_boundary(domain, domain.topology.dim-1, lambda x: np.isclose(x[0],0))
         
        dofs_t = locate_dofs_topological(V, domain.topology.dim-1, facets_t)
        dofs_b = locate_dofs_topological(V, domain.topology.dim-1, facets_b)

        # set w = 0 at base  
        bc_b = dirichletbc(PETSc.ScalarType(0), dofs_b,V)    

        if bc_top['type'] == 'velocity':
            # set w = -1 at top
            bc_t = dirichletbc(PETSc.ScalarType(bc_top['value']), dofs_t,V)  
            bcs = [bc_b,bc_t]
        else:
            # else set Neumann condition in weak form
            bcs = [bc_b]   
    
        # Define weak form
        F = weak_form_vel(w,w_t,phi,bc_top)

        # set initial guess for Newton solver
        w.interpolate(lambda x: -x[0]/H)
        
        # Solve for w
        problem = NonlinearProblem(F, w, bcs=bcs)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
      
        solver.solve(w)
      
        return w