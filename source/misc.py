# this file contains some helper functions for the compaction problem solver

import numpy as np
from dolfinx.fem import (Constant, Expression, Function, FunctionSpace,
                         dirichletbc, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary
from params import a, b, dt, nz, phi0
from petsc4py import PETSc
from scipy.interpolate import griddata
from ufl import Dx, TestFunction, TrialFunction, dx


def K(phi):
      # scaled permeability
      return (phi**a)/((1-phi)**b)

def max(f1,f2):
     # max function, used to enforce positive porosity 
     return 0.5*(f1+f2 + ((f1-f2)**2)**0.5)

def eta_eff(w,phi):
     # Viscosity as a function of dw/dz and porosity
     # f = (Dx(w,0)**2)**((1.0/0.15-1.0)/2)     # shear thickening (hmmm...)
     # f = 1                                    # Newtonian
     f = 1+6e2*(1-phi)**4                    # viscosity decreases with porosity
     return f

def alpha(w,phi):
     # coefficient on dw/dz in weak form
     return (1-phi)*(4./3. + 1/phi)/(4./3. + 1/phi0)*eta_eff(w,phi)

def interp(f,domain):
    # returns a numpy array of a (dolfinx) function f that has been
    # evaluated at the mesh nodes
    V = FunctionSpace(domain, ("CG", 1))
    u = Function(V)
    u.interpolate(Expression(f, V.element.interpolation_points()))

    z = domain.geometry.x[:,0]
    vals = u.x.array

    Z = np.linspace(z.min(),z.max(),nz+1)

    points = (z)
    values = vals
    points_i = Z

    F = griddata(points, values, points_i, method='linear')    

    return points_i,F


def move_mesh(domain,sol):
    # this function computes the surface displacements and moves the mesh
    # by solving Laplace's equation for a smooth displacement function
    # defined for all mesh vertices. (*Solving Laplace is not really necessary 
    # for this 1d problem but would be helpful for higher dimensions)

    V = FunctionSpace(domain, ("CG", 1))
    H = domain.geometry.x.max()

    w_top = Function(V)
    w_top.interpolate(sol.sub(0))
    w_top = w_top.x.array[-1]

    facets_t = locate_entities_boundary(domain, domain.topology.dim-1, lambda x: np.isclose(x[0],H))
    facets_b = locate_entities_boundary(domain, domain.topology.dim-1, lambda x: np.isclose(x[0],0))
        
    dofs_t = locate_dofs_topological(V, domain.topology.dim-1, facets_t)
    dofs_b = locate_dofs_topological(V, domain.topology.dim-1, facets_b)

    bc_top = dirichletbc(PETSc.ScalarType(w_top), dofs_t,V)  # displacement = w at top
    bc_base = dirichletbc(PETSc.ScalarType(0), dofs_b,V)     # w = 0 at base  

    bcs = [bc_top,bc_base]

    # # solve Laplace's equation for a smooth displacement field on all vertices,
    # # given the boundary displacement bc's
    disp = TrialFunction(V)
    v = TestFunction(V)
    a = Dx(disp,0)*Dx(v,0)*dx 
    f = Constant(domain, PETSc.ScalarType(0.0))
    L = f*v*dx

    problem = LinearProblem(a,L, bcs=bcs)
    sol = problem.solve()

    disp_vv = sol.x.array

    z = domain.geometry.x

    z[:,0] += dt*disp_vv

    return domain


def get_stress(sol,domain):
    # compute the effective stress:
    # (1-phi)*alpha(phi)*(dw/dz)
    V = FunctionSpace(domain, ("CG", 1))
    sigma = Function(V)
    phi = Function(V)
    w_z = Function(V)
    phi.interpolate(sol.sub(1))
    w_z_ = Dx(sol.sub(0),0)
    w_z.interpolate(Expression(w_z_, V.element.interpolation_points()))

    f = alpha(sol.sub(0),phi)*w_z
    sigma.interpolate(Expression(f, V.element.interpolation_points()))
 
    return sigma.x.array