"""
Sample runs: python vertical_faults.py --refine=4 --order=2
Description: This is example 7.2 in the paper, it is adapted from
             (Herbin and Hubert, 2008 scirp.org/reference/referencespapers?referenceid=3261890)
"""

from ngsolve.meshes import MakeStructured2DMesh
from prox_mixed import *
import numpy as np
import sys
import argparse
import netgen.gui

# Create a parser object
parser = argparse.ArgumentParser(description="inputs:refine level and polynomial degree")
# Add arguments
parser.add_argument('--refine', type=int, default=1, help='Set the refinement value (default: 1)')
parser.add_argument('--order', type=int, default=1, help='Set the order value (default: 1)')
# Parse the arguments
args = parser.parse_args()
# Assign the values
refine = args.refine
order = args.order
print(f'Refinenment Level: {refine}, Order: {order}')

# Proximal hyperparameters
alpha0 = 1
alphaMax = 1e6
fac = 4.0
tol = 1e-10
maxit = 50
maxNewton = 40
maxerr = 1e-10 

# stablization parameters
eps_1 = 0.0
eps_2 = 0.0
dampfactor = 1.0


def U1(phi):
    dE_plus = 1 / (1 + exp(-phi))
    dE_minus = exp(phi) / (1 + exp(phi))
    return IfPos(phi, dE_plus, dE_minus)


def U2(phi):
    return 0.5 + 0.5 * (phi / sqrt(1 + phi**2))

if __name__ == "__main__":

    #### Problem Set up
    # mesh
    mesh = MakeStructured2DMesh(quads=False, nx = 10, ny = 10)
    for i in range(refine):
        mesh.ngmesh.Refine()

    # diffusion tensor & its inverse
    dom1x = IfPos(x-0.5, 0, 1)
    dom2x = 1-dom1x
    dom1y, dom2y = 0, 0
    for i in range(5):
        dom1y += IfPos((y-0.05-0.2*i)*(y-0.15-0.2*i), 0, 1)
        dom2y += IfPos((y-0.2*i)*(y-0.1-0.2*i), 0, 1)
    dom1 = dom1x*dom1y + dom2x*dom2y
    dom2 = 1-dom1

    vtk = VTKOutput(mesh,coefs=[dom1], names=["dom"],subdivision=order, 
            filename="data/domain_checker"+str(order)+"r"+str(refine))
    vtk.Do()

    D1 = CF(((1000,0), (0, 10)), dims=(2,2))
    iD1 = CF(((1e-3,0), (0, 0.1)), dims=(2,2)) 
    D2 = CF(((1e-2,0), (0, 1e-3)), dims=(2,2))
    iD2 = CF(((1e2,0), (0, 1e3)), dims=(2,2))
    iD = dom1 * iD1 + dom2 * iD2


    # boundary data
    uL = 1.0 
    uR = 0.0
    bdry_values = {'left': uL, 'right': uR}
    #ubc = mesh.BoundaryCF(bdry_values, default=0)
    ubc  = 1-x
    dirichlet_flag = ".*"
    source = 0.0

    ### (1): original hybrid-mixed
    gfu_mx = mixed_fem(mesh, order, iD, dirichlet_flag, ubc, source)
    qh_mx, uh_mx, uhath_mx = gfu_mx.components

 
    ### (2): proximal hybrid-mixed
    gfu_pg, phih0, alpha = prox_hm(mesh, order, iD, dirichlet_flag, ubc, source, U2, alpha0, alphaMax, fac, tol, maxit, maxNewton, maxerr, eps_1, eps_2, dampfactor)
    qh_pg, uh_pg, phih, uhath_pg = gfu_pg.components


    #estimate local mass conservation 
    elconv = Integrate(div(qh_pg) - source,mesh,element_wise=True)
    from IPython import embed 
    print(f"mass conservation {max(np.abs(elconv))}")
    
    vtk = VTKOutput(mesh,coefs=[uh_mx, uh_pg, U2(phih)], names=["uh_mx", "uh_pg", "uh_latent"],subdivision=order, 
            filename="data/vertical_faults"+str(order)+"r"+str(refine))
    vtk.Do()

    Draw(U2(phih), mesh, "latent") 
    input("?")