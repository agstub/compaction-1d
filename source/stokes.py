# This file contains the functions needed for solving the nonlinear Stokes problem.
from boundary import TopBoundary, mark_boundary
from dolfinx.fem import (Constant, Function, dirichletbc,
                         locate_dofs_topological)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.log import LogLevel, set_log_level
from dolfinx.mesh import locate_entities_boundary
from dolfinx.nls.petsc import NewtonSolver
from fem_spaces import mixed_space
from mpi4py import MPI
from params import (B, Ki, Ks, a, dt, eps_v, eta0_s, eta_w, g0, mu, rho_i,
                    rho_s, rho_w, rm2, sigma0, theta, u0)
from petsc4py import PETSc
from ufl import (FacetNormal, Measure, SpatialCoordinate, TestFunctions, div,
                 dx, grad, inner, split, sym)


def eta_i(u):
      # ice shear-thinning (Glen's law) viscosity 
      return 0.5*B*((inner(sym(grad(u)),sym(grad(u)))+eps_v)**(rm2/2.0))

def eta_s(u,pe):
      # sediment viscoplastic viscosity
      f = 1/eta0_s + (2/np.sqrt(2))*(inner(sym(grad(u)),sym(grad(u)))**(1.0/2.0))/(mu*pe)
      return 1/f

def q(pw,phii,phiw,g):
      # water flux
      return -(K(phii,phiw)/eta_w)*(grad(pw)-rho_w*g)

def q0(phii,phiw):
      # water flux in absence of (dynamic) pressure gradient
      return rho_w*g*K(phii,phiw)/eta_w

def K(phii,phiw):
      return (phiw**a)*Ki*(phii/(1-phiw)) + (phiw**a)*Ks*((1-phii-phiw)/(1-phiw))

def xi(phiw):
      return (1-phiw)*(1-(2./3.)*phiw)

def eta_m(u,pe,phii,phiw):
      return phii*eta_i(u) + (1-phii-phiw)*eta_s(u,pe)

def weak_form(domain,m,sol_n,t):
      x = SpatialCoordinate(domain)
      V = mixed_space(domain)
      sol = Function(V)
      (u,phii,phiw,pe,pw) = split(sol)
      (u_t,phii_t,phiw_t,pe_t,pw_t) = TestFunctions(V)
      (u_n,phii_n,phiw_n,pe_n,pw_n) = split(sol_n)

      Q = q(pw,phii,phiw,g)
      Q_n = q(pw_n,phii_n,phiw_n,g)
      Q0 = q0(phii,phiw)

      phiw_theta = (1-theta)*phiw_n + theta*phiw
      phii_theta = (1-theta)*phii_n + theta*phii
      u_theta = (1-theta)*u_n + theta*u
      m_theta = (1-theta)*m(x,t-dt) + theta*m(x,t)
      Q_theta = (1-theta)*Q_n + theta*Q
     
      nu = FacetNormal(domain) 
      facet_tag = mark_boundary(domain)          
      ds = Measure('ds', domain=domain, subdomain_data=facet_tag)

      rho_m = phii*rho_i + phiw*rho_w + (1-phii-phiw)*rho_s
      g = Constant(domain,PETSc.ScalarType((0,-g0)))  

      # Weak form of the residual PDEs
      F = 2*eta_m(u,pe,phii,phiw)*inner(sym(grad(u)),sym(grad(u_t)))*dx
      F += (-div(u_t)*(pw+xi(phiw)*pe)  - inner(rho_m*g, u_t))*dx
      F += sigma0*inner(nu,u_t)*ds(3) 
      F += (p_e+(eta_m(u,pe,phii,phiw)/phi_w)*div(u))*pe_t*dx
      F += (-div(u)-(1/rho_i-1/rho_w)*m(x,t))*pw_t*dx + inner(Q,grad(pw_t))*dx 
      F +=  -Q0*pw_t*ds(3) - Q0*pw_t*ds(4)
      F += dt*Q0*phiw_t*ds(3) + dt*Q0*phiw_t*ds(4)
      F += (phiw-phiw_n)*phiw_t*dx + dt*(div(phiw_theta*u_theta)-m_theta/rho_w)*phiw_t*dx
      F += -dt*inner(Q_theta,grad(phiw))*dx
      F += (phii-phii_n)*phii_t*dx + dt*(div(phii_theta*u_theta)+m_theta/rho_i)*phii_t*dx
      return F

def stokes_solve(domain,m,sol_n,t):
      # Stokes solver for the ice-shelf problem using Taylor-Hood elements

      # Define boundary conditions on the inflow/outflow boundary
      V = mixed_space(domain)
      facets_1 = locate_entities_boundary(domain, domain.topology.dim-1, TopBoundary)        
      dofs_1 = locate_dofs_topological(V.sub(0).sub(0), domain.topology.dim-1, facets_1)
      dofs_2 = locate_dofs_topological(V.sub(0).sub(1), domain.topology.dim-1, facets_1)
      bc1 = dirichletbc(PETSc.ScalarType(u0), dofs_1,V.sub(0).sub(0))
      bc2 = dirichletbc(PETSc.ScalarType(0), dofs_2,V.sub(0).sub(1))
      bcs = [bc1,bc2]

      # Define weak form
      F = weak_form(domain,m,sol_n,t)

      # Solve for (u,p)
      problem = NonlinearProblem(F, w, bcs=bcs)
      solver = NewtonSolver(MPI.COMM_WORLD, problem)

      set_log_level(LogLevel.WARNING)
      n, converged = solver.solve(w)
      assert(converged)

      return w