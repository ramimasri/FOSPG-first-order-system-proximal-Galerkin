"""
Sample runs: python punctured_domain.py --refine=3 --order=2
Description: This is example 7.3 in the paper, it is adapted from
             (Li and Huang, 2010 https://doi.org/10.1016/j.jcp.2010.07.009)
"""

from netgen.geom2d import *
from prox_mixed import *
import numpy as np
import scipy as sp
import argparse
import netgen.gui
import os

# Create a parser object
parser = argparse.ArgumentParser(description="inputs:refine level and order")
# Add arguments
parser.add_argument("--refine", type=int, default=1, help="Set the refinement value (default: 1)")
parser.add_argument("--order", type=int, default=1, help="Set the order value (default: 1)")
# Parse the arguments
args = parser.parse_args()
# Assign the values
refine = args.refine
order = args.order
print(f"Refinenment Level: {refine}, Order: {order}")

# create a saving directory if nonexistent
save_directory = "data"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Proximal hyperparameters
alpha0 = 1e-04  # initial alpha
alphaMax = 10e10
fac = 1.5  # alpha^k = fac*alpha^{k-1}
tol = 1e-10  # tolerance for the proximal loop
maxit = 1000
maxNewton = 400
maxerr = 1e-06  # tolerance for the Newton solver
# stabilization parameters
eps_1 = 0.0
eps_2 = 0.01
# Newton's damp factor, the default is 1
dampfactor = 1.0


# two different choices of the function U
# here the solution lies between 0 and 1
# a rewrite of U = exp(phi)/(1 + exp(phi))
def U1(phi):
    dE_plus = 1 / (1 + exp(-phi))
    dE_minus = exp(phi) / (1 + exp(phi))
    return IfPos(phi, dE_plus, dE_minus)


def U2(phi):
    return 0.5 + 0.5 * (phi / sqrt(1 + phi**2))


if __name__ == "__main__":
    #### Problem Set up

    # diffusion tensor & its inverse
    theta = pi * sin(x) * sin(y)
    Q = CF(((cos(theta), -sin(theta)), (sin(theta), cos(theta))), dims=(2, 2))

    lam1 = 1000
    lam2 = 1
    Lam = CF(((lam1, 0), (0, lam2)), dims=(2, 2))
    iLam = Inv(Lam)
    D = Q * Lam * Q.trans
    iD = Q * iLam * Q.trans  # inverse

    # manual mesh
    outer = Rectangle(pmin=(0, 0), pmax=(1, 1), bc="outer")
    inner = Rectangle(pmin=(4 / 9, 4 / 9), pmax=(5 / 9, 5 / 9), bc="inner")
    geo = CSG2d()
    geo.Add(outer - inner)
    mesh = Mesh(geo.GenerateMesh(maxh=0.1, quad_dominated=False))

    for i in range(refine):
        mesh.ngmesh.Refine()

    # boundary conditions
    bdry_values = {"inner": 1.0, "outer": 0.0}
    ubc = mesh.BoundaryCF(bdry_values, default=0)
    source = 0.0
    dirichlet_flag = "inner|outer"

    ### (1): original hybrid-mixed
    gfu_mx = mixed_fem(mesh, order, iD, dirichlet_flag, ubc, source)
    qh_mx, uh_mx, uhath_mx = gfu_mx.components

    ### (2): proximal hybrid mixed method
    gfu_pg, phih0, alpha = prox_hm(mesh, order, iD, dirichlet_flag, ubc, source, U2, alpha0, alphaMax, fac, tol, maxit, maxNewton, maxerr, eps_1, eps_2, dampfactor)
    qh_pg, uh_pg, phih, uhath_pg = gfu_pg.components

    # estimate local mass conservation
    elconv = Integrate(div(qh_pg) - source, mesh, element_wise=True)
    print(f"local mass conservation of qh_pg {max(np.abs(elconv))}")

    elconv2 = Integrate(div(qh_mx) - source, mesh, element_wise=True)
    print(f"local mass conservation of qh_mixed {max(np.abs(elconv2))}")

    # output mass conservation for qh_pg for visualization
    conserv = GridFunction(L2(mesh, order=0))
    conserv.vec.FV().NumPy()[:] = elconv

    # project uh_mx, uh, and phih to integration rule spaces to compute max & min
    uh_ir = GridFunction(IntegrationRuleSpace(mesh, order=order))
    uh_ir.Set(uh_mx)
    umin_mx, umax_mx = min(uh_ir.vec), max(uh_ir.vec)

    uh_ir.Set(uh_pg)
    umin_pg, umax_pg = min(uh_ir.vec), max(uh_ir.vec)
    pmin_pg, pmax_pg = min(phih.vec), max(phih.vec)
    Uphi_min, Uphi_max = U2(pmin_pg), U2(pmax_pg)

    print("LVL: %1d mix, max: %.6e, min: %.6e" % (refine, umax_mx, umin_mx))
    print("LVL: %1d prg  max uh: %.6e, min uh: %.6e, max U(psih) %.6e min U(psih) %.6e" % (refine, umax_pg, umin_pg, Uphi_max, Uphi_min))

    vtk = VTKOutput(
        mesh,
        coefs=[uh_mx, qh_pg, qh_mx, uh_pg, phih, conserv],
        names=["uh_mx", "qh_pg", "qh_mx", "uh_pg", "latent", "conserv"],
        subdivision=0,
        filename="data/punctured_domain" + str(eps_1) + "o" + str(order) + "r" + str(refine),
    )
    vtk.Do()

    Draw(U2(phih), mesh, "latent")
    input("?")

    # postprocessing the primal solution uh for p > 0
    from limiters import getVerts2D, bndlimit

    if order > 0:
        quad = False
        # dofs per cell in DG-Pk
        if quad == True:
            npt = (order + 1) ** 2
        else:
            npt = int((order + 1) * (order + 2) / 2)

        uh0 = GridFunction(L2(mesh, order=order, all_dofs_together=False))
        uh0.Set(uh_pg)
        mips = getVerts2D(mesh, order)
        bndlimit(uh0, order, mips, npt, mesh.ne)

        vtk = VTKOutput(
            mesh,
            coefs=[uh_pg, uh0],
            names=["uh_pg", "uh_limited"],
            subdivision=0,
            filename="data/punctured_domain_limited" + "o" + str(order) + "r" + str(refine),
        )
        vtk.Do()
