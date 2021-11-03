import numpy as np
import scipy.constants as sp
import time

from fdtd.mesh import *

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
nz = 100

nx_alloc = nx+2
ny_alloc = ny+2
nz_alloc = nz+2

# Subgrid #
zi_sg = 35
zf_sg = 65
nx_sg = 2*nx
ny_sg = 2*ny
nz_sg = 2*(zf_sg-zi_sg)

nx_alloc_sg = nx_sg+2
ny_alloc_sg = ny_sg+2
nz_alloc_sg = nz_sg+2

# Time #
finalTimeStep = 2000000
cfl = 0.87
eps = 8.8541878176e-12
mu = np.pi * 4e-7
c = 1. / np.sqrt(eps * mu)
Z0 = np.sqrt(mu / eps)
Z0_inv = 1 / Z0
Dt = cfl * D / (np.sqrt(3) * sp.speed_of_light)
Dt_sg = Dt / 2

# Constants #
Ce = Dt / eps
Ch = Dt / mu

Ce_sg = Dt_sg / eps
Ch_sg = Dt_sg / mu

# Sources #
z_source = zi_sg - 2
gaussianDelay = 0.04e-6 #0.1e-6
gaussianSpread = 8e-9 #4.5e-9
gaussianDelayH = gaussianDelay - 0.5*D/c
gaussianAmplitudeE = D_inv * Z0_inv * Ce
gaussianAmplitudeH = gaussianAmplitudeE * Z0_inv

# Alloc #
Ex = np.zeros((nx_alloc, ny_alloc + 1, nz_alloc + 1))
Ey = np.zeros((nx_alloc + 1, ny_alloc, nz_alloc + 1))
Ez = np.zeros((nx_alloc + 1, ny_alloc + 1, nz_alloc))
Hx = np.zeros((nx_alloc + 1, ny_alloc, nz_alloc))
Hy = np.zeros((nx_alloc, ny_alloc + 1, nz_alloc))
Hz = np.zeros((nx_alloc, ny_alloc, nz_alloc + 1))

Ex_sg = np.zeros((nx_alloc_sg, ny_alloc_sg + 1, nz_alloc_sg + 1))
Ey_sg = np.zeros((nx_alloc_sg + 1, ny_alloc_sg, nz_alloc_sg + 1))
Ez_sg = np.zeros((nx_alloc_sg + 1, ny_alloc_sg + 1, nz_alloc_sg))
Hx_sg = np.zeros((nx_alloc_sg + 1, ny_alloc_sg, nz_alloc_sg))
Hy_sg = np.zeros((nx_alloc_sg, ny_alloc_sg + 1, nz_alloc_sg))
Hz_sg = np.zeros((nx_alloc_sg, ny_alloc_sg, nz_alloc_sg + 1))

# Geometry #
DEx = np.full(Ex.shape, D)
DEy = np.full(Ey.shape, D)
DEz = np.full(Ez.shape, D)
DHx = np.full(Hx.shape, D)
DHy = np.full(Hy.shape, D)
DHz = np.full(Hz.shape, D)
SEx = np.full(Ex.shape, D * D)
SEy = np.full(Ey.shape, D * D)
SEz = np.full(Ez.shape, D * D)
SHx = np.full(Hx.shape, D * D)
SHy = np.full(Hy.shape, D * D)
SHz = np.full(Hz.shape, D * D)

DEx_sg = np.full(Ex_sg.shape, D_sg)
DEy_sg = np.full(Ey_sg.shape, D_sg)
DEz_sg = np.full(Ez_sg.shape, D_sg)
DHx_sg = np.full(Hx_sg.shape, D_sg)
DHy_sg = np.full(Hy_sg.shape, D_sg)
DHz_sg = np.full(Hz_sg.shape, D_sg)
SEx_sg = np.full(Ex_sg.shape, D_sg * D_sg)
SEy_sg = np.full(Ey_sg.shape, D_sg * D_sg)
SEz_sg = np.full(Ez_sg.shape, D_sg * D_sg)
SHx_sg = np.full(Hx_sg.shape, D_sg * D_sg)
SHy_sg = np.full(Hy_sg.shape, D_sg * D_sg)
SHz_sg = np.full(Hz_sg.shape, D_sg * D_sg)

# D #
DHz_sg[:, :, 1] = D_interf
DHz_sg[:, :, nz_sg + 1] = D_interf

DEx_sg[:, 2::2, 1] = D_
DEx_sg[:, 2::2, nz_sg + 1] = D_
DEy_sg[2::2, :, 1] = D_
DEy_sg[2::2, :, nz_sg + 1] = D_

DEz_sg[2::2, 2::2, 1] = D_inner
DEz_sg[2::2, 2::2, nz_sg] = D_inner

DHx_sg[1:-1:2, :, 0] = D
DHx_sg[1:-1:2, :, nz_sg + 1] = D
DHy_sg[:, 1:-1:2, 0] = D
DHy_sg[:, 1:-1:2, nz_sg + 1] = D

DHx_sg[2:-2:2, :, 0] = 0
DHx_sg[2:-2:2, :, nz_sg + 1] = 0
DHy_sg[:, 2:-2:2, 0] = 0
DHy_sg[:, 2:-2:2, nz_sg + 1] = 0

# S #
SHz_sg[:, :, 1] = surf_interface(D_sg, delta)
SHz_sg[:, :, nz_sg + 1] = surf_interface(D_sg, delta)

SEx_sg[:, 1::2, 1] = surf_trapeze(D)
SEx_sg[:, 1::2, nz_sg + 1] = surf_trapeze(D)
SEy_sg[1::2, :, 1] = surf_trapeze(D)
SEy_sg[1::2, :, nz_sg + 1] = surf_trapeze(D)

SEx_sg[:, 2::2, 1] = surf_triangle(D, delta)
SEx_sg[:, 2::2, nz_sg + 1] = surf_triangle(D, delta)
SEy_sg[2::2, :, 1] = surf_triangle(D, delta)
SEy_sg[2::2, :, nz_sg + 1] = surf_triangle(D, delta)

SHx_sg[2::2, :, 1] = surf_inner(D_sg, delta)
SHx_sg[2::2, :, nz_sg] = surf_inner(D_sg, delta)
SHy_sg[:, 2::2, 1] = surf_inner(D_sg, delta)
SHy_sg[:, 2::2, nz_sg] = surf_inner(D_sg, delta)

# Surface inverse #
SEx_inv = 1 / SEx
SEy_inv = 1 / SEy
SEz_inv = 1 / SEz
SHx_inv = 1 / SHx
SHy_inv = 1 / SHy
SHz_inv = 1 / SHz
SEx_inv_sg = 1 / SEx_sg
SEy_inv_sg = 1 / SEy_sg
SEz_inv_sg = 1 / SEz_sg
SHx_inv_sg = 1 / SHx_sg
SHy_inv_sg = 1 / SHy_sg
SHz_inv_sg = 1 / SHz_sg

n_data = 0

# Nodal file #
fout = open("output_stability.dat", "w")
z_nodal = zi_sg - 5

# Time stepping #
t0 = time.time()
increment_t = 60
next_t = increment_t
last_n = 0
last_t = 0
data = np.zeros(finalTimeStep)

print("--- Start simulation cfl = {:.3f}---".format(cfl))
print(" Printing data every {} seconds.".format(increment_t))
for n in range(finalTimeStep):
    #########
    #   E   #
    #########
    advanceE(Ex, Ey, Ez, Hx, Hy, Hz, nx, ny, nz, Ce, DHx, DHy, DHz, SEx_inv, SEy_inv, SEz_inv)
    PEC(Ey, Ez, nx)
    PEC_z(Ex, Ey, nz)
    advanceE(Ex_sg, Ey_sg, Ez_sg, Hx_sg, Hy_sg, Hz_sg, nx_sg, ny_sg, nz_sg,
             Ce_sg, DHx_sg, DHy_sg, DHz_sg, SEx_inv_sg, SEy_inv_sg, SEz_inv_sg)
    PEC(Ey_sg, Ez_sg, nx_sg)
    communicate_E(Ex, Ey, Ex_sg, Ey_sg, nz_sg, zi_sg, zf_sg)
    advanceH(Ex_sg, Ey_sg, Ez_sg, Hx_sg, Hy_sg, Hz_sg, nx_sg, ny_sg, nz_sg,
             Ch_sg, DEx_sg, DEy_sg, DEz_sg, SHx_inv_sg, SHy_inv_sg, SHz_inv_sg)
    mirror_H(Hx_sg, Hz_sg, ny_sg)
    
    # Source #
    Ex[1:-1, 1:-1, z_source] += gaussianAmplitudeE * gaussian(n * Dt, gaussianDelay, gaussianSpread)
    
    #########
    #   H   #
    #########
    advanceH(Ex, Ey, Ez, Hx, Hy, Hz, nx, ny, nz, Ch, DEx, DEy, DEz, SHx_inv, SHy_inv, SHz_inv)
    mirror_H(Hx, Hz, ny)
    communicate_H(Hx, Hy, Hx_sg, Hy_sg, nz_sg, zi_sg, zf_sg)
    advanceE(Ex_sg, Ey_sg, Ez_sg, Hx_sg, Hy_sg, Hz_sg, nx_sg, ny_sg, nz_sg,
             Ce_sg, DHx_sg, DHy_sg, DHz_sg, SEx_inv_sg, SEy_inv_sg, SEz_inv_sg)
    PEC(Ey_sg, Ez_sg, nx_sg)
    advanceH(Ex_sg, Ey_sg, Ez_sg, Hx_sg, Hy_sg, Hz_sg, nx_sg, ny_sg, nz_sg,
             Ch_sg, DEx_sg, DEy_sg, DEz_sg, SHx_inv_sg, SHy_inv_sg, SHz_inv_sg)
    mirror_H(Hx_sg, Hz_sg, ny_sg)
    
    # Source #
    Hy[1:-1, 1:-1, z_source] += gaussianAmplitudeH * gaussian(n * Dt, gaussianDelayH, gaussianSpread)
    
    # Nodal probe #
    fout.write("{:.5E} {:.5E}\n".format(n * Dt, Ex[2, 3, z_nodal]))
    
    current_t = time.time() - t0
    if current_t > next_t:
        iters_per_sec = (n - last_n) / (current_t - last_t)
        remaining = (finalTimeStep - n) / iters_per_sec
        MCells_per_sec = iters_per_sec * (nx * ny * nz
            + 2 * nx_sg * ny_sg * nz_sg
        ) / 1000000
        print("\n- Iteration {} of {} ({:.2E} of {:.2E} seconds.) -".format(n, finalTimeStep, Dt * n, Dt * finalTimeStep))
        print(" Time remaining: {:.2f} seconds.".format(remaining))
        print(" Processing {} MCells/s".format(MCells_per_sec))
        next_t += current_t + increment_t
        last_t = current_t
        last_n = n

print("\n--- Done. Time elapsed: {:.2f} seconds".format(time.time() - t0))

# Close file #
fout.close()