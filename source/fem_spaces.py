from dolfinx.fem import FunctionSpace
from ufl import FiniteElement, MixedElement

def mixed_space(domain):
    P1 = FiniteElement('P',domain.ufl_cell(),1)     
    P2 = FiniteElement('P',domain.ufl_cell(),2)     
    element = MixedElement([[P2,P2],P2,P2,P1,P1])
    V = FunctionSpace(domain,element)             # Function space for (u,phii,phiw,pe,pw)
    return V
