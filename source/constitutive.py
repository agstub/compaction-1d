# this file defines constitutive relations for the viscosity, yield
# stress / plasticity, and permeability

from params import a, b, m, n


def K(phi):
      # scaled permeability
      return (phi**a)/((1-phi)**b)

def Pi(phi):
      # scaled yield stress / plasticity law
      return (1-phi)*((1-phi)**n)/(phi**m)

def alpha(phi):
     # effective viscosity (coefficient on dw/dz in weak form)
     return (1-phi)*(1+6e2*(1-phi)**4)   #
