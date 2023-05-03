#-------------------------------------------------------------------------------
# This file contains functions that:
# (1) define the boundaries of the mesh (ice-air,ice-water,inflow/outflow)
# (2) mark the boundaries of the mesh, AND ...
# (3) possibly create Dirichlet boundary conditions on one or both side walls of the domain.
#-------------------------------------------------------------------------------
import numpy as np
from dolfinx.mesh import locate_entities, meshtags
from params import L, z_max

#-------------------------------------------------------------------------------
# Define SubDomains for ice-water boundary, ice-bed boundary, inflow (x=0) and
# outflow (x=Length of domain). 

def WaterBoundary(x):
# Ice-water boundary    
    return np.less(x[1],z_max)

def TopBoundary(x):
# Ice-air boundary    
    return np.greater(x[1],z_max)

def LeftBoundary(x):
    # Left boundary (inflow/outflow)
    return np.isclose(x[0],-L/2.0)

def RightBoundary(x):
    # Right boundary (inflow/outflow)
    return np.isclose(x[0],L/2.0)

#-------------------------------------------------------------------------------

def mark_boundary(domain):
    # Assign markers to each boundary segment (except the upper surface).
    # This is used at each time step to update the markers.
    #
    # Boundary marker numbering convention:
    # 1 - Left boundary
    # 2 - Right boundary
    # 3 - Ice-water boundary

    boundaries = [(3, WaterBoundary),(4,TopBoundary),(1, LeftBoundary),(2, RightBoundary)]
    facet_indices, facet_markers = [], []
    fdim = domain.topology.dim - 1
    for (marker, locator) in boundaries:
        facets = locate_entities(domain, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

    return facet_tag
