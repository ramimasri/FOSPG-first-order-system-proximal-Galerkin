from ngsolve import *
from ngsolve.solvers import Newton
from ngsolve.comp import IntegrationRuleSpace
import numpy as np
import csv


def mixed_fem(mesh, order, iD, dirichlet_flag, ubc, source):
    n = specialcf.normal(mesh.dim)
    V = Discontinuous(HDiv(mesh, order=order, RT=True))
    W = L2(mesh, order=order)
    M = FacetFESpace(mesh, order=order, dirichlet=dirichlet_flag)

    fes = V * W * M
    (q, u, uhat), (r, v, vhat) = fes.TnT()
    a = BilinearForm(fes, condense=True)
    a += (iD * q * r - u * div(r) + div(q) * v) * dx
    a += (uhat * r * n - vhat * q * n) * dx(element_boundary=True)
    a += -source * v * dx
    gfu = GridFunction(fes)
    gfu.components[2].Set(ubc, BND)
    Newton(a, gfu, maxit=1, inverse="sparsecholesky", maxerr=1e4, printing=False)
    return gfu


# prox hybrid mixed method with stabilization
def prox_hm(
    mesh,
    order,
    iD,
    dirichlet_flag,
    ubc,
    source,
    U,
    alpha0=1,
    alphaMax=1e6,
    fac=4.0,
    tol=1e-10,
    maxit=100,
    maxNewton=40,
    maxerr=1e-4,
    eps_1=0.01,
    eps_2=0.01,
    dampfactor=1,
    save_directory="",
    obstacle=0.0,
    u_ex=0.0,
    flux=0.0,
    adaptive=False,
):
    # proximal parameter
    alpha = Parameter(alpha0)

    # finite element spaces
    n = specialcf.normal(mesh.dim)
    V = Discontinuous(HDiv(mesh, order=order, RT=True))
    W = L2(mesh, order=order)
    M = FacetFESpace(mesh, order=order, dirichlet=dirichlet_flag)

    uh0 = GridFunction(W)

    fes = V * W * W * M
    gfu = GridFunction(fes)
    qh, uh, phih, uhath = gfu.components
    uhath.Set(ubc, BND)

    # initial guess
    phih0 = GridFunction(W)
    phih0.Set(0)

    (q, u, phi, uhat), (r, v, psi, vhat) = fes.TnT()
    a = BilinearForm(fes, condense=True)
    a += (iD * q * r - u * div(r) + (div(q) - source) * v) * dx
    a += (uhat * r * n - vhat * q * n) * dx(element_boundary=True)
    h = specialcf.mesh_size
    # get the mesh size
    h_elem = GridFunction(L2(mesh, order=0))
    h_elem.Set(h)
    mesh_size = max(h_elem.vec.FV().NumPy()[:])
    print(f"mesh_size is equal to {mesh_size}")
    # Print the stablization parameters eps_1 and eps_2
    print(f"eps_1 is {eps_1} and eps_2 is {eps_2}")
    if order == 0:
        epsilon = eps_1 * h ** (order + 1)
        a += (u * psi - U(phi) * psi - obstacle * psi + (phi - phih0) / alpha * v) * dx
        a += -epsilon * phi * psi * dx
    # stabilization
    elif order == 1:
        epsilon_1 = eps_1 * h ** (order + 1)
        epsilon_2 = eps_2 * h ** (order + 1)
        # use the vertex rule for stabilization
        # using a quadrature rule exact for P2 that includes the vertices suffices for linears
        points = [(0, 0), (1, 0), (0, 1), (0.5, 0.0), (0.5, 0.5), (0, 0.5), (1 / 3, 1 / 3)]
        weights = [1 / 40, 1 / 40, 1 / 40, 1 / 15, 1 / 15, 1 / 15, 9 / 40]
        ir = IntegrationRule(points=points, weights=weights)

        a += (u * psi - U(phi) * psi - obstacle * psi + (phi - phih0) / alpha * v) * dx(intrules={TRIG: ir})
        a += -epsilon_1 * phi * psi * dx(intrules={TRIG: ir})
        a += -epsilon_2 * grad(phi) * grad(psi) * dx(intrules={TRIG: ir})
    else:
        epsilon_1 = eps_1 * h ** (order + 1)
        epsilon_2 = eps_2 * h ** (order + 1)
        a += (u * psi - U(phi) * psi - obstacle * psi + (phi - phih0) / alpha * v) * dx
        a += -epsilon_1 * phi * psi * dx
        a += -epsilon_2 * grad(phi) * grad(psi) * dx

    err = 1.0
    iter = 0
    with TaskManager():
        while err > tol and iter < maxit:
            phih0.vec.data = phih.vec
            uh0.vec.data = uh.vec
            if adaptive:
                maxerr = min(0.1, err)
            it = Newton(a, gfu, maxit=maxNewton, inverse="sparsecholesky", dampfactor=dampfactor, maxerr=maxerr, printing=False)

            iter += 1
            if it[0] == -1:  # newton didn't converge
                alpha.Set(alpha.Get() * fac)
                continue
            # update error
            err = sqrt(Integrate((uh0 - uh) ** 2, mesh))
            print("Iter %3d, Alpha %.2e, Newton %2d Err %.3e" % (iter, alpha.Get(), it[1], err))
            # update alpha
            alpha.Set(min(alpha.Get() * fac, alphaMax))

            # write data on the Newton linear solves
            if save_directory != "":
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)
                mode = "w" if iter == 1 else "a"
                err_uex = sqrt(Integrate((u_ex - U(phih) - obstacle) ** 2, mesh))
                err_qex = sqrt(Integrate((qh + flux) ** 2, mesh))
                with open(f"{save_directory}/solver{mesh_size:.3e}.csv", mode=mode, newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([f"{mesh_size:.3e}", iter, it[1], f"{err_uex:.3e}", f"{err_qex:.3e}"])

    return gfu, phih0, alpha
