#import numpy as np
import scipy.constants as sp
import time
#import h5py as h5

from fdtd.mesh import *

subgrid = False
name = "_sinLTS"
LTS = True

# Delta #
D = 0.1
D_sg = D/2
D_inv = 1./D

delta = D_sg/3
D_ = np.sqrt(D_sg**2 + delta**2)
D_inner = D_sg-delta
D_interf = np.sqrt(11)/4 * D

# Cells #
nx = 4
ny = 4
nz = 10000

nx_alloc = nx+2
ny_alloc = ny+2
nz_alloc = nz+2

# Subgrid #
zi_sg = int(nz/2)
zf_sg = zi_sg+20
nx_sg = 2*nx
ny_sg = 2*ny
nz_sg = 2*(zf_sg-zi_sg)

nx_alloc_sg = nx_sg+2
ny_alloc_sg = ny_sg+2
nz_alloc_sg = nz_sg+2

# Time #
finalTimeStep = 42000 #100000
if LTS:
    cfl = 0.8
else:
    cfl = 0.4
eps = 8.8541878176e-12
mu = np.pi*4e-7
c = 1./np.sqrt(eps*mu)
Z0 = np.sqrt(mu/eps)
Z0_inv = 1/Z0
Dt = cfl * D/(np.sqrt(3)*sp.speed_of_light)
if LTS:
    Dt_sg = Dt/2
else:
    Dt_sg = Dt

# Constants #
Ce = Dt/eps
Ch = Dt/mu

Ce_sg = Dt_sg/eps
Ch_sg = Dt_sg/mu

# Sources #
z_source = zi_sg - 10
gaussianDelay = 0.01e-6 #0.1e-6
gaussianSpread = 1e-9 #4.4092766513658374e-10 #4.5e-9
gaussianDelayH = gaussianDelay - 0.5*D/c
gaussianAmplitudeE = D_inv * Z0_inv * Ce
gaussianAmplitudeH = gaussianAmplitudeE * Z0_inv

# Alloc #
Ex = np.zeros((nx_alloc  , ny_alloc+1, nz_alloc+1))
Ey = np.zeros((nx_alloc+1, ny_alloc  , nz_alloc+1))
Ez = np.zeros((nx_alloc+1, ny_alloc+1, nz_alloc  ))
Hx = np.zeros((nx_alloc+1, ny_alloc  , nz_alloc  ))
Hy = np.zeros((nx_alloc  , ny_alloc+1, nz_alloc  ))
Hz = np.zeros((nx_alloc  , ny_alloc  , nz_alloc+1))

Ex_sg = np.zeros((nx_alloc_sg  , ny_alloc_sg+1, nz_alloc_sg+1))
Ey_sg = np.zeros((nx_alloc_sg+1, ny_alloc_sg  , nz_alloc_sg+1))
Ez_sg = np.zeros((nx_alloc_sg+1, ny_alloc_sg+1, nz_alloc_sg  ))
Hx_sg = np.zeros((nx_alloc_sg+1, ny_alloc_sg  , nz_alloc_sg  ))
Hy_sg = np.zeros((nx_alloc_sg  , ny_alloc_sg+1, nz_alloc_sg  ))
Hz_sg = np.zeros((nx_alloc_sg  , ny_alloc_sg  , nz_alloc_sg+1))

# Geometry #
DEx = np.full(Ex.shape, D)
DEy = np.full(Ey.shape, D)
DEz = np.full(Ez.shape, D)
DHx = np.full(Hx.shape, D)
DHy = np.full(Hy.shape, D)
DHz = np.full(Hz.shape, D)
SEx = np.full(Ex.shape, D*D)
SEy = np.full(Ey.shape, D*D)
SEz = np.full(Ez.shape, D*D)
SHx = np.full(Hx.shape, D*D)
SHy = np.full(Hy.shape, D*D)
SHz = np.full(Hz.shape, D*D)

DEx_sg = np.full(Ex_sg.shape, D_sg)
DEy_sg = np.full(Ey_sg.shape, D_sg)
DEz_sg = np.full(Ez_sg.shape, D_sg)
DHx_sg = np.full(Hx_sg.shape, D_sg)
DHy_sg = np.full(Hy_sg.shape, D_sg)
DHz_sg = np.full(Hz_sg.shape, D_sg)
SEx_sg = np.full(Ex_sg.shape, D_sg*D_sg)
SEy_sg = np.full(Ey_sg.shape, D_sg*D_sg)
SEz_sg = np.full(Ez_sg.shape, D_sg*D_sg)
SHx_sg = np.full(Hx_sg.shape, D_sg*D_sg)
SHy_sg = np.full(Hy_sg.shape, D_sg*D_sg)
SHz_sg = np.full(Hz_sg.shape, D_sg*D_sg)

# D #
DHz_sg[:, :, 1] = D_interf
DHz_sg[:, :, nz_sg+1] = D_interf

DEx_sg[:, 2::2, 1] = D_
DEx_sg[:, 2::2, nz_sg+1] = D_
DEy_sg[2::2, :, 1] = D_
DEy_sg[2::2, :, nz_sg+1] = D_

DEz_sg[2::2, 2::2, 1] = D_inner
DEz_sg[2::2, 2::2, nz_sg] = D_inner

DHx_sg[1:-1:2, :, 0] = D
DHx_sg[1:-1:2, :, nz_sg+1] = D
DHy_sg[:, 1:-1:2, 0] = D
DHy_sg[:, 1:-1:2, nz_sg+1] = D

DHx_sg[2:-2:2, :, 0] = 0
DHx_sg[2:-2:2, :, nz_sg+1] = 0
DHy_sg[:, 2:-2:2, 0] = 0
DHy_sg[:, 2:-2:2, nz_sg+1] = 0

# S #
SHz_sg[:, :, 1] = surf_interface(D_sg, delta)
SHz_sg[:, :, nz_sg+1] = surf_interface(D_sg, delta)

SEx_sg[:, 1::2, 1] = surf_trapeze(D)
SEx_sg[:, 1::2, nz_sg+1] = surf_trapeze(D)
SEy_sg[1::2, :, 1] = surf_trapeze(D)
SEy_sg[1::2, :, nz_sg+1] = surf_trapeze(D)

SEx_sg[:, 2::2, 1] = surf_triangle(D, delta)
SEx_sg[:, 2::2, nz_sg+1] = surf_triangle(D, delta)
SEy_sg[2::2, :, 1] = surf_triangle(D, delta)
SEy_sg[2::2, :, nz_sg+1] = surf_triangle(D, delta)

SHx_sg[2::2, :, 1] = surf_inner(D_sg, delta)
SHx_sg[2::2, :, nz_sg] = surf_inner(D_sg, delta)
SHy_sg[:, 2::2, 1] = surf_inner(D_sg, delta)
SHy_sg[:, 2::2, nz_sg] = surf_inner(D_sg, delta)

# Surface inverse #
SEx_inv = 1/SEx
SEy_inv = 1/SEy
SEz_inv = 1/SEz
SHx_inv = 1/SHx
SHy_inv = 1/SHy
SHz_inv = 1/SHz
SEx_inv_sg = 1/SEx_sg
SEy_inv_sg = 1/SEy_sg
SEz_inv_sg = 1/SEz_sg
SHx_inv_sg = 1/SHx_sg
SHy_inv_sg = 1/SHy_sg
SHz_inv_sg = 1/SHz_sg

# Slice file #
#fout = h5.File("output.h5", "w")
#fout.create_group("data")
#
#fout_sg = h5.File("output_sg.h5", "w")
#fout_sg.create_group("data")
#
#every = 10
#fout_time = []
#n_data = 0

# Nodal file #
if subgrid:
    fout_nodal_0 = open("output_nodal"+name+"_0.dat", "w")
    fout_nodal_1 = open("output_nodal"+name+"_1.dat", "w")
else:
    fout_nodal_0 = open("output_nodal"+name+"_nosubgrid_0.dat", "w")
    fout_nodal_1 = open("output_nodal"+name+"_nosubgrid_1.dat", "w")

z_nodal_0 = z_source - 2
z_nodal_1 = zf_sg + 5
x_nodal = int(nx/2)
y_nodal = int(ny/2)+1

# Time stepping #
t0 = time.time()
increment_t = 5
next_t = increment_t
last_n = 0
last_t = 0
data = np.zeros(finalTimeStep)

if subgrid:
    if LTS:
        print("--- Start simulation with subgrid (LTS: on)- cfl = {:.3f}---".format(cfl))
    else:
        print("--- Start simulation with subgrid (LTS: off) - cfl = {:.3f}---".format(cfl))
else:
    print("--- Start simulation without subgrid - cfl = {:.3f}---".format(cfl))
print(" Printing data every {} seconds.".format(increment_t))

for n in range(finalTimeStep):
    #########
    #   E   #
    #########
    advanceE(Ex, Ey, Ez, Hx, Hy, Hz, nx, ny, nz, Ce, DHx, DHy, DHz, SEx_inv, SEy_inv, SEz_inv)
    PEC(Ey, Ez, nx)
    PEC_z(Ex, Ey, nz)
    if subgrid:
        if LTS:
            advanceE(Ex_sg, Ey_sg, Ez_sg, Hx_sg, Hy_sg, Hz_sg, nx_sg, ny_sg, nz_sg,
                     Ce_sg, DHx_sg, DHy_sg, DHz_sg, SEx_inv_sg, SEy_inv_sg, SEz_inv_sg)
            PEC(Ey_sg, Ez_sg, nx_sg)
            communicate_E(Ex, Ey, Ex_sg, Ey_sg, nz_sg, zi_sg, zf_sg)
            advanceH(Ex_sg, Ey_sg, Ez_sg, Hx_sg, Hy_sg, Hz_sg, nx_sg, ny_sg, nz_sg,
                     Ch_sg, DEx_sg, DEy_sg, DEz_sg, SHx_inv_sg, SHy_inv_sg, SHz_inv_sg)
            mirror_H(Hx_sg, Hz_sg, ny_sg)
        else:
            advanceE(Ex_sg, Ey_sg, Ez_sg, Hx_sg, Hy_sg, Hz_sg, nx_sg, ny_sg, nz_sg,
                     Ce_sg, DHx_sg, DHy_sg, DHz_sg, SEx_inv_sg, SEy_inv_sg, SEz_inv_sg)
            PEC(Ey_sg, Ez_sg, nx_sg)
            communicate_E(Ex, Ey, Ex_sg, Ey_sg, nz_sg, zi_sg, zf_sg)
            
    
    # Source #
    Ex[1:-1, 1:-1, z_source] += gaussianAmplitudeE * gaussian(n*Dt, gaussianDelay, gaussianSpread)
    
    
    #########
    #   H   #
    #########
    advanceH(Ex, Ey, Ez, Hx, Hy, Hz, nx, ny, nz, Ch, DEx, DEy, DEz, SHx_inv, SHy_inv, SHz_inv)
    mirror_H(Hx, Hz, ny)
    communicate_H(Hx, Hy, Hx_sg, Hy_sg, nz_sg, zi_sg, zf_sg)
    if subgrid:
        if LTS:
            advanceE(Ex_sg, Ey_sg, Ez_sg, Hx_sg, Hy_sg, Hz_sg, nx_sg, ny_sg, nz_sg,
                     Ce_sg, DHx_sg, DHy_sg, DHz_sg, SEx_inv_sg, SEy_inv_sg, SEz_inv_sg)
            PEC(Ey_sg, Ez_sg, nx_sg)
            advanceH(Ex_sg, Ey_sg, Ez_sg, Hx_sg, Hy_sg, Hz_sg, nx_sg, ny_sg, nz_sg,
                     Ch_sg, DEx_sg, DEy_sg, DEz_sg, SHx_inv_sg, SHy_inv_sg, SHz_inv_sg)
            mirror_H(Hx_sg, Hz_sg, ny_sg)
        else:
            advanceH(Ex_sg, Ey_sg, Ez_sg, Hx_sg, Hy_sg, Hz_sg, nx_sg, ny_sg, nz_sg,
                     Ch_sg, DEx_sg, DEy_sg, DEz_sg, SHx_inv_sg, SHy_inv_sg, SHz_inv_sg)
            mirror_H(Hx_sg, Hz_sg, ny_sg)
    
    # Source #
    Hy[1:-1, 1:-1, z_source] += gaussianAmplitudeH * gaussian(n*Dt, gaussianDelayH, gaussianSpread)
    
    # Slice probe #
    #if not n%every:
    #    fout["data"][str(n_data)] = Ex[1:-1, 6, 1:-1]
    #    fout_sg["data"][str(n_data)] = Ex_sg[1:-1, 11, 1:-1]
    #    fout_time += [n*Dt]
    #    n_data += 1
    
    # Nodal probe #
    #fout_nodal_0.write("{:.5E} {:.5E}\n".format(n*Dt, Ex[x_nodal, y_nodal, z_nodal_0]))
    #fout_nodal_1.write("{:.5E} {:.5E}\n".format(n*Dt, Ex[x_nodal, y_nodal, z_nodal_1]))
    
    current_t = time.time()-t0
    if current_t > next_t:
        speed = (n-last_n+1)/(current_t-last_t)
        remaining = (finalTimeStep-n-1)/speed
        if not subgrid:
            MCells_sec = speed * (nx*ny*nz) / 1000000
        elif not LTS:
            MCells_sec = speed * (nx*ny*nz + nx_sg*ny_sg*nz_sg) / 1000000
        else:
            MCells_sec = speed * (nx*ny*nz + 2*nx_sg*ny_sg*nz_sg) / 1000000
        print("\n- Iteration {} of {} ({:.2E} of {:.2E} seconds.) -".format(n, finalTimeStep, Dt*n, Dt*finalTimeStep))
        print(" Time remaining: {:.2f} seconds.".format(remaining))
        print(" Processing {} MCells/s".format(MCells_sec))
        next_t += current_t + increment_t
        last_t = current_t
        last_n = n

print("\n--- Done. Time elapsed: {:.2f} seconds".format(time.time()-t0))

# Close file #
#fout.attrs["n_data"] = n_data
#fout_sg.attrs["n_data"] = n_data
#fout_sg.attrs["zi_sg"] = zi_sg
#fout_sg.attrs["zf_sg"] = zf_sg
#
#fout["time"] = fout_time
#fout_sg["time"] = fout_time
#
#fout.close()
#fout_sg.close()

fout_nodal_0.close()
fout_nodal_1.close()
