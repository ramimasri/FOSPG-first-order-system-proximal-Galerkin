"""
Sample runs: python oblique_flow.py --refine =4 --order =2
Description: This is example 7.1 in the paper, it is adapted from
             (Herbin and Hubert, 2008 scirp.org/reference/referencespapers?referenceid=3261890)
"""

from ngsolve.meshes import MakeStructured2DMesh
from prox_mixed import *
import netgen.gui
import numpy as np
import sys
import argparse

# Create a parser object
parser = argparse.ArgumentParser(description="inputs:refine level and polynomial degree")
# Add arguments
parser.add_argument("--refine", type=int, default=1, help="Set the refinement value (default: 1)")
parser.add_argument("--order", type=int, default=1, help="Set the order value (default: 1)")
# Parse the arguments
args = parser.parse_args()
# Assign the values
refine = args.refine
order = args.order
print(f"Refinenment Level: {refine}, Order: {order}")

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
    mesh = MakeStructured2DMesh(quads=False, nx=10, ny=10)
    for i in range(refine):
        mesh.ngmesh.Refine()

    # diffusion tensor & its inverse
    lam = 1e-3
    theta = 2 * pi / 9
    Q = CF(((cos(theta), -sin(theta)), (sin(theta), cos(theta))), dims=(2, 2))
    Lam = CF(((1, 0), (0, lam)), dims=(2, 2))
    iLam = CF(((1, 0), (0, 1 / lam)), dims=(2, 2))
    D = Q * Lam * Q.trans
    iD = Q * iLam * Q.trans

    # boundary data
    uB = IfPos(x - 0.2, IfPos(x - 0.3, 0.5, 2.0 - 5 * x), 1)
    uT = IfPos(x - 0.7, IfPos(x - 0.8, 0.0, 4.0 - 5 * x), 1 / 2)
    uL = IfPos(y - 0.2, IfPos(y - 0.3, 0.5, 2.0 - 5 * y), 1)
    uR = IfPos(y - 0.7, IfPos(y - 0.8, 0.0, 4.0 - 5 * y), 1 / 2)

    bdry_values = {"bottom": uB, "top": uT, "left": uL, "right": uR}
    ubc = mesh.BoundaryCF(bdry_values, default=0)
    dirichlet_flag = ".*"
    source = 0.0

    ### (1): original hybrid-mixed
    gfu_mx = mixed_fem(mesh, order, iD, dirichlet_flag, ubc, source)
    qh_mx, uh_mx, uhath_mx = gfu_mx.components

    # project uh to integration rule spaces to compute max & min
    uh_ir = GridFunction(IntegrationRuleSpace(mesh, order=order + 10))
    uh_ir.Set(uh_mx)

    umin_mx, umax_mx = min(uh_ir.vec), max(uh_ir.vec)

    ### (2): proximal hybrid mixed method
    gfu_pg, phih0, alpha = prox_hm(mesh, order, iD, dirichlet_flag, ubc, source, U1, alpha0, alphaMax, fac, tol, maxit, maxNewton, maxerr, eps_1, eps_2, dampfactor)
    qh_pg, uh_pg, phih, uhath_pg = gfu_pg.components

    # compute local mass conservation
    elconv = Integrate(div(qh_pg) - source, mesh, element_wise=True)
    print(f"mass conservation {max(np.abs(elconv))}")
    conserv = GridFunction(L2(mesh, order=0))
    conserv.vec.FV().NumPy()[:] = elconv

    vtk = VTKOutput(mesh, coefs=[uh_mx, uh_pg, U2(phih)], names=["uh_mx", "uh_pg", "uh_pg_latent"], subdivision=order, filename="data/oblique_flow" + str(order) + "r" + str(refine))
    vtk.Do()

    Draw(U2(phih), mesh, "latent")
    input("?")
