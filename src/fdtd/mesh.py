import numpy as np
import numba as nb

@nb.njit(parallel=True, fastmath=True)
def advanceE(Ex, Ey, Ez, Hx, Hy, Hz, nx, ny, nz, Ce, DHx, DHy, DHz, SEx, SEy, SEz):
    Ex[1:nx+1, 1:ny+2, 1:nz+2] = Ex[1:nx+1, 1:ny+2, 1:nz+2] + \
        Ce*SEx[1:nx+1, 1:ny+2, 1:nz+2] * (
            (
                DHz[1:nx+1, 1:ny+2, 1:nz+2]*Hz[1:nx+1, 1:ny+2, 1:nz+2] -
                DHz[1:nx+1, 0:ny+1, 1:nz+2]*Hz[1:nx+1, 0:ny+1, 1:nz+2]
            ) - (
                DHy[1:nx+1, 1:ny+2, 1:nz+2]*Hy[1:nx+1, 1:ny+2, 1:nz+2] -
                DHy[1:nx+1, 1:ny+2, 0:nz+1]*Hy[1:nx+1, 1:ny+2, 0:nz+1]
            )
        )
    
    Ey[1:nx+2, 1:ny+1, 1:nz+2] = Ey[1:nx+2, 1:ny+1, 1:nz+2] + \
        Ce*SEy[1:nx+2, 1:ny+1, 1:nz+2] * (
            (
                DHx[1:nx+2, 1:ny+1, 1:nz+2]*Hx[1:nx+2, 1:ny+1, 1:nz+2] -
                DHx[1:nx+2, 1:ny+1, 0:nz+1]*Hx[1:nx+2, 1:ny+1, 0:nz+1]
            ) - (
                DHz[1:nx+2, 1:ny+1, 1:nz+2]*Hz[1:nx+2, 1:ny+1, 1:nz+2] -
                DHz[0:nx+1, 1:ny+1, 1:nz+2]*Hz[0:nx+1, 1:ny+1, 1:nz+2]
            )
        )
    
    Ez[1:nx+2, 1:ny+2, 1:nz+1] = Ez[1:nx+2, 1:ny+2, 1:nz+1] + \
        Ce*SEz[1:nx+2, 1:ny+2, 1:nz+1] * (
            (
                DHy[1:nx+2, 1:ny+2, 1:nz+1]*Hy[1:nx+2, 1:ny+2, 1:nz+1] -
                DHy[0:nx+1, 1:ny+2, 1:nz+1]*Hy[0:nx+1, 1:ny+2, 1:nz+1]
            ) - (
                DHx[1:nx+2, 1:ny+2, 1:nz+1]*Hx[1:nx+2, 1:ny+2, 1:nz+1] -
                DHx[1:nx+2, 0:ny+1, 1:nz+1]*Hx[1:nx+2, 0:ny+1, 1:nz+1]
            )
        )


@nb.njit(parallel=True, fastmath=True)
def advanceH(Ex, Ey, Ez, Hx, Hy, Hz, nx, ny, nz, Ch, DEx, DEy, DEz, SHx, SHy, SHz):
    Hx[1:nx+2, 1:ny+1, 1:nz+1] = Hx[1:nx+2, 1:ny+1, 1:nz+1] + \
        Ch*SHx[1:nx+2, 1:ny+1, 1:nz+1] * (
            (
                DEy[1:nx+2, 1:ny+1, 2:nz+2]*Ey[1:nx+2, 1:ny+1, 2:nz+2] -
                DEy[1:nx+2, 1:ny+1, 1:nz+1]*Ey[1:nx+2, 1:ny+1, 1:nz+1]
            ) - (
                DEz[1:nx+2, 2:ny+2, 1:nz+1]*Ez[1:nx+2, 2:ny+2, 1:nz+1] -
                DEz[1:nx+2, 1:ny+1, 1:nz+1]*Ez[1:nx+2, 1:ny+1, 1:nz+1]
            )
        )

    Hy[1:nx+1, 1:ny+2, 1:nz+1] = Hy[1:nx+1, 1:ny+2, 1:nz+1] + \
        Ch*SHy[1:nx+1, 1:ny+2, 1:nz+1] * (
            (
                DEz[2:nx+2, 1:ny+2, 1:nz+1]*Ez[2:nx+2, 1:ny+2, 1:nz+1] -
                DEz[1:nx+1, 1:ny+2, 1:nz+1]*Ez[1:nx+1, 1:ny+2, 1:nz+1]
            ) - (
                DEx[1:nx+1, 1:ny+2, 2:nz+2]*Ex[1:nx+1, 1:ny+2, 2:nz+2] -
                DEx[1:nx+1, 1:ny+2, 1:nz+1]*Ex[1:nx+1, 1:ny+2, 1:nz+1]
            )
        )

    Hz[1:nx+1, 1:ny+1, 1:nz+2] = Hz[1:nx+1, 1:ny+1, 1:nz+2] + \
        Ch*SHz[1:nx+1, 1:ny+1, 1:nz+2] * (
            (
                DEx[1:nx+1, 2:ny+2, 1:nz+2]*Ex[1:nx+1, 2:ny+2, 1:nz+2] -
                DEx[1:nx+1, 1:ny+1, 1:nz+2]*Ex[1:nx+1, 1:ny+1, 1:nz+2]
            ) - (
                DEy[2:nx+2, 1:ny+1, 1:nz+2]*Ey[2:nx+2, 1:ny+1, 1:nz+2] -
                DEy[1:nx+1, 1:ny+1, 1:nz+2]*Ey[1:nx+1, 1:ny+1, 1:nz+2]
            )
        )

@nb.njit(parallel=True, fastmath=True)
def mirror_H(Hx, Hz, ny):
    Hx[:, 0, :] = -Hx[:, 1, :]
    Hz[:, 0, :] = -Hz[:, 1, :]
    
    Hx[:, ny+1, :] = -Hx[:, ny, :]
    Hz[:, ny+1, :] = -Hz[:, ny, :]

@nb.njit(parallel=True, fastmath=True)
def PEC(Ey, Ez, nx):
    Ey[1, :, :] = 0
    Ez[1, :, :] = 0
    
    Ey[nx+1, :, :] = 0
    Ez[nx+1, :, :] = 0

@nb.njit(parallel=True, fastmath=True)
def PEC_z(Ex, Ey, nz):
    Ex[:, :, 1] = 0
    Ey[:, :, 1] = 0
    
    Ex[:, :, nz+1] = 0
    Ey[:, :, nz+1] = 0

@nb.njit(parallel=True, fastmath=True)
def communicate_E(Ex, Ey, Ex_sg, Ey_sg, nz_sg, zi_sg, zf_sg):
    Ex[1:-1, 1:-1, zi_sg] = 0.5*(Ex_sg[1:-1:2, 1:-1:2, 1] + Ex_sg[2:-1:2, 1:-1:2, 1])
    Ex[1:-1, 1:-1, zf_sg] = 0.5*(Ex_sg[1:-1:2, 1:-1:2, nz_sg+1] + Ex_sg[2:-1:2, 1:-1:2, nz_sg+1])
    
    Ey[1:-1, 1:-1, zi_sg] = 0.5*(Ey_sg[1:-1:2, 1:-1:2, 1] + Ey_sg[1:-1:2, 2:-1:2, 1])
    Ey[1:-1, 1:-1, zf_sg] = 0.5*(Ey_sg[1:-1:2, 1:-1:2, nz_sg+1] + Ey_sg[1:-1:2, 2:-1:2, nz_sg+1])

def communicate_H(Hx, Hy, Hx_sg, Hy_sg, nz_sg, zi_sg, zf_sg):
    Hx_sg[1:-1:2, 1:-1, 0] = np.repeat(Hx[1:-1, 1:-1, zi_sg-1], 2, 1)
    Hx_sg[1:-1:2, 1:-1, nz_sg+1] = np.repeat(Hx[1:-1, 1:-1, zf_sg], 2, 1)

    Hy_sg[1:-1, 1:-1:2, 0] = np.repeat(Hy[1:-1, 1:-1, zi_sg-1], 2, 0)
    Hy_sg[1:-1, 1:-1:2, nz_sg+1] = np.repeat(Hy[1:-1, 1:-1, zf_sg], 2, 0)

def surf_interface(D_sg, delta):
    return (D_sg**2)/np.sqrt(11) * (3+delta/D_sg)

def surf_trapeze(D):
    return D**2 * 3*np.sqrt(10)/16 * 3/np.sqrt(10)

def surf_triangle(D, delta):
    return D**2 * np.sqrt(10)/16 * (3+2*delta/D)/np.sqrt(10*(1+2*(2*delta/D)**2))

def surf_inner(D_sg, delta):
    return D_sg**2 - D_sg*delta/2

@nb.njit
def gaussian(x, mean, sigma, amplitude=1):
    return amplitude * np.exp(-(x-mean)*(x-mean)/(2*sigma*sigma))

@nb.njit
def gaussian_derivative(x, mean, sigma, amplitude=1):
    return (mean-x)*gaussian(x, mean, sigma, amplitude)
