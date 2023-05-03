# This file contains the functions needed for solving the nonlinear Stokes problem.
from bdry_conds import LeftBoundary, RightBoundary, mark_boundary
from dolfinx.fem import (Constant, Function, dirichletbc,
                         locate_dofs_topological)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.log import LogLevel, set_log_level
from dolfinx.mesh import locate_entities_boundary
from dolfinx.nls.petsc import NewtonSolver
from fem_spaces import mixed_space
from mpi4py import MPI
from params import B, eps_v, eta0_s, eta_w, g0, mu, rho_i, rho_s, rho_w, rm2
from petsc4py import PETSc
from ufl import (FacetNormal, Measure, SpatialCoordinate, TestFunctions, div,
                 dx, grad, inner, split, sym)


def eta_i(u):
      # ice viscosity 
      return 0.5*B*((inner(sym(grad(u)),sym(grad(u)))+eps_v)**(rm2/2.0))

def eta_s(u,pe):
      # sediment viscosity
      f = 1/eta0_s + (2/np.sqrt(2))*(inner(sym(grad(u)),sym(grad(u)))**(1.0/2.0))/(mu*pe)
      return 1/f

def q(pw,g):
      # water flux
      return -(k/eta_w)*(grad(pw)-rho_w*g)

def xi(phiw):
      return (1-phiw)*(1-(2./3.)*phiw)

def eta_m(u,pe,phii,phiw):
      return phii*eta_i(u) + (1-phii-phiw)*eta_s(u,pe)

def weak_form(u,p,v,q,f,ds,nu):
    # Weak form residual of the ice-shelf problem
    F = 2*eta_m(u,pe,phii,phiw)*inner(sym(grad(u)),sym(grad(v)))*dx
    F += (- div(v)*p + q*div(u))*dx - inner(f, v)*dx
    return F

def stokes_solve(domain,m,sol_n):
        # Stokes solver for the ice-shelf problem using Taylor-Hood elements

        # Define function spaces
        V = mixed_space(domain)

        #---------------------Define variational problem------------------------
        w = Function(V)
        (u,phii,phiw,pe,pw) = split(w)
        (u_t,phii_t,phiw_t,pe_t,pw_t) = TestFunctions(V)
      
        # Neumann condition at ice-water boundary
        x = SpatialCoordinate(domain)
     
        # Body force
        rho_m = phii*rho_i + phiw*rho_w + (1-phii-phiw)*rho_s
        g = Constant(domain,PETSc.ScalarType((0,-g0)))  
        f = rho_m*g    

        # Outward-pointing unit normal to the boundary  
        nu = FacetNormal(domain)           

        # Mark bounadries of mesh and define a measure for integration
        facet_tag = mark_boundary(domain)
        ds = Measure('ds', domain=domain, subdomain_data=facet_tag)

        # Define boundary conditions on the inflow/outflow boundary
        facets_1 = locate_entities_boundary(domain, domain.topology.dim-1, LeftBoundary)        
        facets_2 = locate_entities_boundary(domain, domain.topology.dim-1, RightBoundary)
        dofs_1 = locate_dofs_topological(W.sub(0).sub(0), domain.topology.dim-1, facets_1)
        dofs_2 = locate_dofs_topological(W.sub(0).sub(0), domain.topology.dim-1, facets_2)
        bc1 = dirichletbc(PETSc.ScalarType(0), dofs_1,W.sub(0).sub(0))
        bc2 = dirichletbc(PETSc.ScalarType(0), dofs_2,W.sub(0).sub(0))
        bcs = [bc1,bc2]

        # Define weak form
        F = weak_form(u,p,v,q,f,ds,nu)

        # Solve for (u,p)
        problem = NonlinearProblem(F, w, bcs=bcs)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)

        set_log_level(LogLevel.WARNING)
        n, converged = solver.solve(w)
        assert(converged)

        return w