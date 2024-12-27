from ngsolve import *
from netgen.geom2d import SplineGeometry
from netgen.meshing import MeshingParameters
import netgen.gui
from prox_mixed import *
import os 
import argparse
import sys
import numpy as np 
import scipy as sp 
import csv 

# python circle.py --refine=2 --order=1 --linearize=0

# Create a parser object
parser = argparse.ArgumentParser(description='FEM inputs')

# Add arguments
parser.add_argument('--refine', type=int, default=1, help='Set the refinement value (default: 1)')
parser.add_argument('--order', type=int, default=1, help='Set the order value (default: 1)')
parser.add_argument('--linearize', type=int, default=0, 
        help='Set the flag value (default: 0)')

# Parse the arguments
args = parser.parse_args()
# Assign the values
refine = args.refine
order = args.order
linearize = args.linearize == 1
print(f'Refine: {refine}, Order: {order}, Linearize: {linearize}')

# Create a specific directory if it doesn't exist
save_directory = 'rates/circle'
if not os.path.exists(save_directory):
        os.makedirs(save_directory)


# Proximal hyperparameters
alpha0 = 1.0
alphaMax = 1e10
fac = 1.0
tol = 1e-06
maxit = 550
# one step Newton (linearized)
if linearize:
    maxNewton = 1
    maxerr = 1e10
else:
    maxNewton = 50
    maxerr = 1e-10
dampfactor = 1

# Create geometry: a circle
geo = SplineGeometry()
geo.AddCircle(c=(0, 0), r=1, bc="circle_boundary")

# Generate mesh
circle = geo.GenerateMesh(maxh=0.4)  
circle.SecondOrder()
circle.Curve(3)  # Curve the elements to order 3

mesh = Mesh(circle)
r0   = 0.5 
beta = 0.9
b    = r0*beta
tmp  = sqrt(r0**2 - b**2)
# obstacle 
obstacle = IfPos(b-sqrt(x**2 + y**2), sqrt(0.25 - x**2 - y**2), tmp + (b**2)/tmp - sqrt(x**2 + y**2)*b/tmp)  
# exact solution 
a = np.real(np.exp(0.5*sp.special.lambertw((-1/(2*np.exp(1)**2)), -1)+1)) # a approx 0.34898 
A = np.sqrt(0.25 - a**2)/np.log(a)  # -0.34012 
u_ex = IfPos(sqrt(x**2 + y**2) - a,  A*log(sqrt(x**2 + y**2)), sqrt(0.25 - x**2 - y**2) )

## flux, source and boundary condition
source = 0.0 
# dirichlet bc
ubc = 0.0 
dirichlet_flag = ".*"
# diffusion 
iD = CF(((1,0), (0, 1)), dims=(2,2))
du  = CF((u_ex.Diff(x), u_ex.Diff(y)))
flux = iD*du

def U3(phi):
    return exp(phi) + obstacle
def U4(phi):
    return log(1+exp(phi)) + obstacle

def solver(refine, order, U,eps_1, eps_2): 
    # Generate mesh
    circle = geo.GenerateMesh(maxh=0.05)  
    mesh = Mesh(circle)
    for i in range(refine):
        mesh.ngmesh.Refine()
    if order > 1:
        mesh.Curve(2)


    ### : proximal hybrid-mixed
    adaptive = True 
    gfu_pg, phih0,alpha = prox_hm(mesh,  order, iD, dirichlet_flag, ubc, source, U,
        alpha0, alphaMax, fac, tol,
        maxit, maxNewton, maxerr,eps_1, eps_2, dampfactor, save_directory= 'rates/circle', u_ex=u_ex, flux=flux, adaptive= adaptive)
    qh_pg, uh_pg, phih, uhath_pg= gfu_pg.components

    err_qh = sqrt(Integrate((qh_pg+ flux)**2*dx,mesh))

   

    # project entropy variable
    err_pg = sqrt(Integrate((uh_pg-u_ex)**2*dx, mesh))
    err_pg2 = sqrt(Integrate((U(phih) -u_ex)**2*dx,mesh))    
    err_PG  = [err_qh, err_pg, err_pg2]
   
    return mesh, qh_pg, uh_pg, phih, err_PG 

if __name__ == "__main__":

    visualize = True        
    rates     = False            
      
    results =[] 
   
    if visualize: 
         
        eps_1 = 0.0 
        eps_2 = 1e-04
        mesh, qh_pg, uh_pg, phih, err_PG = solver(refine, order, U3, eps_1, eps_2)

        err_qh, err_pg, err_pg2 = err_PG 
        print("LVL: %1d prg err q_h: %.6e, err primal:%.6e, err latent %.6e"%(refine, err_qh, err_pg, err_pg2))

        #estimate local mass conservation 
        elconv  = Integrate(div(qh_pg) - source,mesh,element_wise=True)
        conserv = GridFunction(L2(mesh, order= 0)) 
        conserv.vec.FV().NumPy()[:] = elconv    

        vtk = VTKOutput(mesh,  coefs=[uh_pg , U3(phih), conserv], names= ["uh_pg", "uh_latent","conserv"], 
         subdivision=order, filename="data/spherical_obstacle"+str(order)+"r"+str(refine))
        vtk.Do()

        Draw(U3(phih), mesh, "latent")
        input("?")


    if rates: 
        refine_levels = [0,1,2,3]  # List of refinement levels to test

        errors_flux = []
        errors_pg   = []
        errors_pg2  = []
    
        mesh_sizes = [] 
        eps_1 = 0.0 
        eps_2 = 0.0
        for refine in refine_levels:
            print(f"LVL {refine}, order {order}")

            mesh, qh_pg, uh_pg, phih, err_PG = solver(refine, order, U3,eps_1, eps_2)
            # get the mesh size 
            h = specialcf.mesh_size 
            h_elem = GridFunction(L2(mesh, order = 0))
            h_elem.Set(h)
            mesh_sizes.append(max(h_elem.vec.FV().NumPy()[:]))

            err_qh, err_pg, err_pg2, err_pg2_proj = err_PG 
            errors_flux.append(err_qh)
            errors_pg.append(err_pg)
            errors_pg2.append(err_pg2)
            print("LVL: %1d prg err q_h: %.6e, err primal:%.6e, err latent %.6e"%(refine, err_qh, err_pg, err_pg2,err_pg2_proj))

            Draw(obstacle + U3(phih), mesh, "latent")
            input("?")
        
        npy_file_path = os.path.join(save_directory, f'meshsize_errors_{order}.npy')
        # Stack the data into columns: [meshsize, errors]
        data = np.column_stack((mesh_sizes, errors_flux, errors_pg, errors_pg2))
        np.save(npy_file_path, data)
        
        data_mixed  = np.column_stack((mesh_sizes, errors_flux)) 
        data_primal = np.column_stack((mesh_sizes, errors_pg)) 
        data_latent = np.column_stack((mesh_sizes, errors_pg2)) 

        np.savetxt(f'{save_directory}/flux_error_{order}_{eps_2}.csv', data_mixed, delimiter=',', fmt='%.16f')
        np.savetxt(f'{save_directory}/primal_error_{order}_{eps_2}.csv', data_primal, delimiter=',', fmt='%.16f')
        np.savetxt(f'{save_directory}/latent_error_{order}_{eps_2}.csv', data_latent, delimiter=',', fmt='%.16f')
