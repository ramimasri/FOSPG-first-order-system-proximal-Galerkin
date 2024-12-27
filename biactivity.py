"""
Sample runs: python biactivity.py --refine=4 --order=2 
Description: This is example 7.4 in the paper, it is taken from
            Keith and Surowiec, 2024 https://link.springer.com/article/10.1007/s10208-024-09681-8
"""
from prox_mixed import *
from netgen.geom2d import *
import os 
import argparse
import netgen.gui
from ngsolve.meshes import MakeStructured2DMesh
import sys
import numpy as np 

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
print(f'Refine: {refine}, Order: {order}')

def U3(phi):
    return exp(phi) 

# Proximal hyperparameters
alpha0 = 1
alphaMax = 1e10
fac = 1.5
tol = 1e-12
maxit = 100
maxNewton = 40
maxerr = 1e-10

# diffusion tensor & its inverse
lam  = 1 
D = CF(((1,0), (0, lam)), dims=(2,2))
iD = CF(((1,0), (0, 1/lam)), dims=(2,2))
#
uex    = IfPos(x, x**4, 0)
source = IfPos(x,-12*x**2,0)

### flux 
du  = CF((uex.Diff(x), uex.Diff(y)))
flux = D*du

# dirichlet bc
ubc = uex
dirichlet_flag = ".*"

def solver(refine, order): 
    #### Problem Set up
    # mesh
    # Define the rectangular geometry [-1,1] x [-1,1]
    geo = SplineGeometry()
    geo.AddRectangle((-1, -1), (1, 1))
    # Generate the mesh with a specified maximum element size
    mesh = Mesh(geo.GenerateMesh(maxh=0.6))
    for i in range(refine):
        mesh.ngmesh.Refine()
    eps_1 = 0.0 
    eps_2 = 1e-05
    dampfactor = 1
    gfu_pg, phih0,alpha = prox_hm(mesh,  order, iD, dirichlet_flag, ubc, source, U3,
            alpha0, alphaMax, fac, tol,
            maxit, maxNewton, maxerr,eps_1, eps_2, dampfactor)
    qh_pg, uh_pg, phih, uhath_pg= gfu_pg.components
    err_qh = sqrt(Integrate((qh_pg+ flux)**2*dx,mesh))
    # project entropy variable
    err_pg = sqrt(Integrate((uh_pg-uex)**2*dx, mesh))
    err_pg2 = sqrt(Integrate((U3(phih) - uex)**2*dx,mesh))
    
    err_PG  = [err_qh, err_pg, err_pg2]

    return mesh,qh_pg, uh_pg, phih, err_PG 

if __name__ == "__main__":

    visualize = True        
    rates     = False         
      
    results =[] 
   
    if visualize: 
         
        mesh,qh_pg, uh_pg, phih, err_PG = solver(refine, order)
        err_qh, err_pg, err_pg2 = err_PG 
        print("LVL: %1d prg err q_h: %.6e, err primal:%.6e, err latent %.6e"%(refine, err_qh, err_pg, err_pg2))

        vtk = VTKOutput(mesh,coefs=[uh_pg, U3(phih)], names= ["uh_pg", "phih"], 
         subdivision=order, filename="data/biactivity"+str(order)+"r"+str(refine))
        vtk.Do()
        Draw(U3(phih), mesh, "dE")
        input("?")
        #estimate local mass conservation 
        elconv = Integrate( div(qh_pg) - source,mesh,element_wise=True)
        print(f"local mass conservation of qh_pg {max(np.abs(elconv))}")
    

    if rates: 
        refine_levels = [1, 2,3,4]  # List of refinement levels to test

        errors_flux = []
        errors_pg   = []
        errors_pg2  = []
        mesh_sizes = [] 

        for refine in refine_levels:
            print(f"LVL {refine}, order {order}")

            mesh,qh_pg, uh_pg, phih, err_PG  = solver(refine, order)
            # estimate the mesh-size
            h = specialcf.mesh_size
            h_elem = GridFunction(L2(mesh, order = 0))
            h_elem.Set(h)
            mesh_sizes.append(max(h_elem.vec.FV().NumPy()[:])) 
            Draw(U3(phih), mesh, "dE")
            input("?")

            err_qh, err_pg, err_pg2 = err_PG 
            print("LVL: %1d prg err q_h: %.6e, err primal:%.6e,err latent %.6e"%(refine, err_qh, err_pg, err_pg2))

            errors_flux.append(err_qh)
            errors_pg.append(err_pg)
            errors_pg2.append(err_pg2)
        


        # Create a specific directory if it doesn't exist
        save_directory = 'rates/biactivity'
        if not os.path.exists(save_directory):
             os.makedirs(save_directory)

        npy_file_path = os.path.join(save_directory, f'meshsize_errors_{order}.npy')
        # Stack the data into columns: [meshsize, errors]
        data = np.column_stack((mesh_sizes, errors_flux, errors_pg, errors_pg2))
        np.save(npy_file_path, data)
        
        data_mixed  = np.column_stack((mesh_sizes, errors_flux)) 
        data_primal = np.column_stack((mesh_sizes, errors_pg)) 
        data_latent = np.column_stack((mesh_sizes, errors_pg2)) 

        np.savetxt(f'{save_directory}/flux_error_{order}.csv', data_mixed, delimiter=',', fmt='%.16f')
        np.savetxt(f'{save_directory}/primal_error_{order}.csv', data_primal, delimiter=',', fmt='%.16f')
        np.savetxt(f'{save_directory}/latent_error_{order}.csv', data_latent, delimiter=',', fmt='%.16f')
