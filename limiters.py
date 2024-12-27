from netgen.geom2d import *
from prox_mixed import *
import numpy as np 
import scipy as sp 
import csv 


# get visualization points (assume either pure trig or pure quad mesh)
def getVerts2D(mesh, order):
    mpts = mesh([0],[0], [0]) 
    mpt = mpts[0][3]
    # TODO: only implemented order=1,2 
    irt = []
    irq = []
    for i in range(order+1):
        x0 = i/order
        for j in range(order+1):
            y0 = j/order
            irq.append((x0, y0))
            if (x0+y0 < 1.001): # left triangle
                irt.append((x0,y0))
    nt = len(irt)
    nq = len(irq)
    npts = 0
    for i in range(mesh.ne):
        el = mesh[ElementId(VOL, i)]
        if len(el.vertices)==3:
            npts += nt
        else:
            npts += nq

    mips = np.zeros((npts,), dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
                                           ('meshptr', '<f8'), ('VorB', '<i4'), ('nr', '<i4')])            

    cnt = 0
    for i in range(mesh.ne):
        el = mesh[ElementId(VOL, i)]
        if len(el.vertices)==3: # trig mesh
            for j in range(nt):
                mips[cnt] = (irt[j][0], irt[j][1], 0, mpt, 0, i)
                cnt += 1
        else:
            for j in range(nq):
                mips[cnt] = (irq[j][0], irq[j][1], 0, mpt, 0, i)
                cnt += 1
    return mips

# bnd limiter: bounds are 0 & 1
def bndlimit(gf, order, mips, npts, nelems, vmin=0, vmax=1):
    u0 = gf.vec[0:nelems].FV().NumPy()    
    # TODO: npts = total # dofs per element
    view = gf.vec[nelems:].FV().NumPy().reshape(nelems, npts-1)
    
    # LEFT/RIGHT values
    vals = gf(mips).reshape(nelems, npts) 
    
    umin = vals.min(axis=1)
    umax = vals.max(axis=1)
    
    mask_min = umin < vmin
    mask_max = umax > vmax

    maskA = mask_min & mask_max
    maskB = mask_min & (~mask_max)
    maskC = mask_max & (~mask_min)

    # Type 1 cell: both max/min are violated
    tmin = (vmin-u0[maskA])/(umin[maskA]-u0[maskA])
    tmax = (vmax-u0[maskA])/(umax[maskA]-u0[maskA])
    ttA = np.minimum(tmin, tmax)
    
    view[maskA] *= ttA[:, np.newaxis]

    # Type 2 cell: min is violated
    ttB = (vmin-u0[maskB])/(umin[maskB]-u0[maskB])
    view[maskB] *= ttB[:, np.newaxis]

    # Type 3 cell: max is violated
    ttC = (vmax-u0[maskC])/(umax[maskC]-u0[maskC])
    view[maskC] *= ttC[:, np.newaxis]
    
    return None
