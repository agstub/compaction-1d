# thise file contains some post processing functions for calculating 
# the stress from the velocity, and interpolating the solutions 
# onto a uniform grid for plotting 

import numpy as np
from constitutive import Pi, alpha
from dolfinx.fem import Expression, Function, FunctionSpace
from params import H0, Pi0, eta, nz, w0, zeta
from scipy.interpolate import griddata
from ufl import Dx


def interp(f,domain):
    # returns a numpy array of a (dolfinx) function f that has been
    # evaluated at the mesh nodes and interpolated onto a uniform grid
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


def get_stress(sol,domain):
    # compute the (dimensional) effective stress
    V = FunctionSpace(domain, ("CG", 1))
    sigma = Function(V)
    phi = Function(V)
    w_z = Function(V)
    phi.interpolate(sol.sub(1))
    w_z_ = Dx(sol.sub(0),0)
    w_z.interpolate(Expression(w_z_, V.element.interpolation_points()))

    f = -(w0/H0)*((4./3.)*eta + zeta)*alpha(phi)*w_z + Pi0*Pi(phi)
    sigma.interpolate(Expression(f, V.element.interpolation_points()))
 
    return sigma.x.array