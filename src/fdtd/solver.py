import numpy as np
import copy
import scipy.constants as sp
import time
import math

from fdtd.common import *
from fdtd.field import *


def gaussian(x, mean, spread):
    return np.exp(-(x-mean)**2/(2*spread**2))


class Solver:
    __timeStepPrint = 5000
    
    def __init__(self, mesh, options, probes, sources):
        self.options = options
        
        self.mesh = copy.deepcopy(mesh)
        self.subsolvers = []
        for submesh in self.mesh.submeshes:
            self.subsolvers.append(Solver(submesh, options, probes, sources))
        
        self.dt = self.options["cfl"] * min(self.mesh.steps) / np.sqrt(3.0)
        
        self.probes = copy.deepcopy(probes)
        for p in self.probes:
            if p["type"] == "slice":
                box = self.mesh.elemIdToBox(p["elemId"])
                box = self.mesh.snap(box)
                ids = self.mesh.toIdx(box)
                Nxyz = abs(np.array(ids[U]) - ids[L])
                p["mesh"] = {"origin": box[L], "steps": self.mesh.steps}
                p["field"] = fieldIndex(p["field"])
                extra = [p["field"][0] if i == p["field"][1] else 1 - p["field"][0] for i in range(3)]
                p["indices"] = np.array(ids) + [[0, 0, 0], extra]
                p["time"] = [0.]
                p["values"] = [np.zeros(tuple(np.array(Nxyz)+extra))]
            elif p["type"] == "nodal":
                if mesh.subgrid_level == p["subgridLevel"]:
                    p["active"] = True
                    p["field"] = fieldIndex(p["field"])
                    p["index"] = mesh.toIdx(mesh.coordinates[mesh.elements[p["elemId"]][0]])
                    p["time"] = []
                    p["values"] = []
                else:
                    p["active"] = False
            else:
                raise ValueError("Invalid probe type :" + p["type"])
        
        self.sources = copy.deepcopy(sources)
        for source in self.sources:
            box = self.mesh.elemIdToBox(source["elemId"])
            if self.mesh.isOverlayed(box):
                ids = mesh.toIdx(box)  # TODO: Hacer una funcion para obtener los ids de los diferentes campos en una box.
            else:
                ids = ([0, 0, 0], [0, 0, 0])
            source["index"] = ids
        
        self.old = Fields(
            ex=np.zeros((mesh.pos[X].size-1, mesh.pos[Y].size, mesh.pos[Z].size)),
            ey=np.zeros((mesh.pos[X].size, mesh.pos[Y].size-1, mesh.pos[Z].size)),
            ez=np.zeros((mesh.pos[X].size, mesh.pos[Y].size, mesh.pos[Z].size-1)),
            hx=np.zeros((mesh.pos[X].size, mesh.pos[Y].size-1, mesh.pos[Z].size-1)),
            hy=np.zeros((mesh.pos[X].size-1, mesh.pos[Y].size, mesh.pos[Z].size-1)),
            hz=np.zeros((mesh.pos[X].size-1, mesh.pos[Y].size-1, mesh.pos[Z].size))
        )

    def getProbes(self):
        res = self.probes
        for p in res:
            p["subprobes"] = []
            for s in self.subsolvers:
                p["subprobes"] = s.getProbes()
    
        return res
    
    # ======================= UPDATE E =============================
    def updateE(self, t, dt, overFields=None):
        eNew = (
            np.zeros(self.old.ex.shape),
            np.zeros(self.old.ey.shape),
            np.zeros(self.old.ez.shape)
        )
        e = self.old.get(E)
        h = self.old.get(H)
        
        d = self.mesh.steps
        d01 = self.mesh.subgridCenterDistance()
        A = self.mesh.areas()
        At, AT, Ae, Ae_mirror = self.mesh.subgridEquivAreasE()  # triangle, trapeze, edge
        
        # Standard Yee Algorithm
        for xyz in range(3):
            lE  = np.roll([(None, None), (1, -1  ), (1, -1  )], xyz, axis=0)
            lH1 = np.roll([(None, None), (1, -1  ), (0, -1  )], xyz, axis=0)
            lH2 = np.roll([(None, None), (1, -1  ), (1, None)], xyz, axis=0)
            lH3 = np.roll([(None, None), (1, None), (1, -1  )], xyz, axis=0)
            lH4 = np.roll([(None, None), (0, -1  ), (1, -1  )], xyz, axis=0)
            eNew[xyz][lE[X, L]:lE[X, U], lE[Y, L]:lE[Y, U], lE[Z, L]:lE[Z, U]] = \
                e[xyz][lE[X, L]:lE[X, U], lE[Y, L]:lE[Y, U], lE[Z, L]:lE[Z, U]] + dt / A[xyz] * (
                    + d[(xyz+1) % 3] * h[(xyz+1) % 3][lH1[X, L]:lH1[X, U], lH1[Y, L]:lH1[Y, U], lH1[Z, L]:lH1[Z, U]]
                    - d[(xyz+1) % 3] * h[(xyz+1) % 3][lH2[X, L]:lH2[X, U], lH2[Y, L]:lH2[Y, U], lH2[Z, L]:lH2[Z, U]]
                    + d[(xyz+2) % 3] * h[(xyz+2) % 3][lH3[X, L]:lH3[X, U], lH3[Y, L]:lH3[Y, U], lH3[Z, L]:lH3[Z, U]]
                    - d[(xyz+2) % 3] * h[(xyz+2) % 3][lH4[X, L]:lH4[X, U], lH4[Y, L]:lH4[Y, U], lH4[Z, L]:lH4[Z, U]]
                )
        
        # Boundary conditions
        for bound in self.mesh.bounds:
            xyz = bound.orientation()
            lu = bound.lu()
            if bound.btype == "pec":
                [lx1, ly1, lz1], [ux1, uy1, uz1] = bound.fieldIds((E, (xyz+1) % 3))
                [lx2, ly2, lz2], [ux2, uy2, uz2] = bound.fieldIds((E, (xyz+2) % 3))
                eNew[(xyz+1) % 3][lx1:ux1, ly1:uy1, lz1:uz1] = 0
                eNew[(xyz+2) % 3][lx2:ux2, ly2:uy2, lz2:uz2] = 0
            
            elif bound.btype == "pmc":
                [lx1, ly1, lz1], [ux1, uy1, uz1] = bound.fieldIds((E, (xyz+1) % 3))
                [lx2, ly2, lz2], [ux2, uy2, uz2] = bound.fieldIds((E, (xyz+2) % 3))
                [lxH, lyH, lzH], [uxH, uyH, uzH] = bound.fieldIds((H, xyz))
                
                if lu == L:
                    lH = [lx1, ly1, lz1][xyz]
                    uH = [ux1, uy1, uz1][xyz]
                    sg = 1.
                else:
                    lH = [lx1-1, ly1-1, lz1-1][xyz]
                    uH = [ux1-1, uy1-1, uz1-1][xyz]
                    sg = -1.

                lE1 = [lx1, ly1, lz1], [ux1, uy1, uz1]
                lE2 = [lx2, ly2, lz2], [ux2, uy2, uz2]
                for plus in (1, 2):
                    if plus == 1:
                        lE = copy.deepcopy(lE1)
                        sg2 = 1.
                    else:
                        lE = copy.deepcopy(lE2)
                        sg2 = -1.
                    lH3 = copy.deepcopy(lE)
                    lH1 = [lxH, lyH, lzH], [uxH, uyH, uzH]
                    lH2 = copy.deepcopy(lH1)
                    lE[L][(xyz+3-plus) % 3] += 1
                    lE[U][(xyz+3-plus) % 3] -= 1
                    lH1[L][(xyz+3-plus) % 3] += 1
                    lH2[U][(xyz+3-plus) % 3] -= 1
                    lH3[L][xyz], lH3[U][xyz] = lH, uH
                    lH3[L][(xyz+3-plus) % 3] += 1
                    lH3[U][(xyz+3-plus) % 3] -= 1
                    eNew[(xyz+plus) % 3][lE[L][X]:lE[U][X], lE[L][Y]:lE[U][Y], lE[L][Z]:lE[U][Z]] = \
                        e[(xyz+plus) % 3][lE[L][X]:lE[U][X], lE[L][Y]:lE[U][Y], lE[L][Z]:lE[U][Z]] + \
                        dt / A[(xyz+plus) % 3] * sg2 * (
                            + d[xyz] * h[xyz][lH1[L][X]:lH1[U][X], lH1[L][Y]:lH1[U][Y], lH1[L][Z]:lH1[U][Z]]
                            - d[xyz] * h[xyz][lH2[L][X]:lH2[U][X], lH2[L][Y]:lH2[U][Y], lH2[L][Z]:lH2[U][Z]]
                            - 2*sg * d[(xyz+3-plus) % 3] * h[(xyz+3-plus) % 3][lH3[L][X]:lH3[U][X],
                                                                               lH3[L][Y]:lH3[U][Y],
                                                                               lH3[L][Z]:lH3[U][Z]]
                        )
            
            elif bound.btype == "overMesh":
                [lx1, ly1, lz1], [ux1, uy1, uz1] = bound.fieldIds((E, (xyz+1) % 3))
                [lx2, ly2, lz2], [ux2, uy2, uz2] = bound.fieldIds((E, (xyz+2) % 3))
                [lxH, lyH, lzH], [uxH, uyH, uzH] = bound.fieldIds((H, xyz))
                [lxH1, lyH1, lzH1], [uxH1, uyH1, uzH1] = bound.fieldIds((H, (xyz+2) % 3))
                [lxH2, lyH2, lzH2], [uxH2, uyH2, uzH2] = bound.fieldIds((H, (xyz+1) % 3))
                [lxo1, lyo1, lzo1], [uxo1, uyo1, uzo1] = bound.overFieldIds((H, (xyz+2) % 3))
                [lxo2, lyo2, lzo2], [uxo2, uyo2, uzo2] = bound.overFieldIds((H, (xyz+1) % 3))
                oh = overFields.get("H")
                sg = 1 - 2*lu

                for plus in (1, 2):
                    if plus == 1:
                        lE = [lx1, ux1, 1], [ly1, uy1, 1], [lz1, uz1, 1]
                        lH3 = [lxH1, uxH1, 1], [lyH1, uyH1, 1], [lzH1, uzH1, 1]
                        lHo = [lxo1, uxo1, 1], [lyo1, uyo1, 1], [lzo1, uzo1, 1]
                        sg2 = 1.
                    else:
                        lE = [lx2, ux2, 1], [ly2, uy2, 1], [lz2, uz2, 1]
                        lH3 = [lxH2, uxH2, 1], [lyH2, uyH2, 1], [lzH2, uzH2, 1]
                        lHo = [lxo2, uxo2, 1], [lyo2, uyo2, 1], [lzo2, uzo2, 1]
                        sg2 = -1.
                    lH1 = [lxH, uxH, 1], [lyH, uyH, 1], [lzH, uzH, 1]
                    lH2 = [lxH, uxH, 1], [lyH, uyH, 1], [lzH, uzH, 1]
                    lE[(xyz+3-plus) % 3][L] += 1
                    lE[(xyz+3-plus) % 3][U] -= 1
                    lE[(xyz+3-plus) % 3][2] = 2
                    lH1[(xyz+3-plus) % 3][L] += 1
                    lH1[(xyz+3-plus) % 3][2] = 2
                    lH2[(xyz+3-plus) % 3][U] -= 1
                    lH2[(xyz+3-plus) % 3][2] = 2
                    lH3[(xyz+3-plus) % 3][L] += 1
                    lH3[(xyz+3-plus) % 3][U] -= 1
                    lH3[(xyz+3-plus) % 3][2] = 2

                    eNew[(xyz+plus) % 3][lE[X][L]:lE[X][U]:lE[X][2], lE[Y][L]:lE[Y][U]:lE[Y][2], lE[Z][L]:lE[Z][U]:lE[Z][2]] = \
                        e[(xyz+plus) % 3][lE[X][L]:lE[X][U]:lE[X][2], lE[Y][L]:lE[Y][U]:lE[Y][2], lE[Z][L]:lE[Z][U]:lE[Z][2]] + \
                        dt / At * sg2 * (
                            + d01 * h[xyz][lH1[X][L]:lH1[X][U]:lH1[X][2], lH1[Y][L]:lH1[Y][U]:lH1[Y][2], lH1[Z][L]:lH1[Z][U]:lH1[Z][2]]
                            - d01 * h[xyz][lH2[X][L]:lH2[X][U]:lH2[X][2], lH2[Y][L]:lH2[Y][U]:lH2[Y][2], lH2[Z][L]:lH2[Z][U]:lH2[Z][2]]
                            - sg * d[(xyz+3-plus) % 3] * h[(xyz+3-plus) % 3][lH3[X][L]:lH3[X][U]:lH3[X][2], lH3[Y][L]:lH3[Y][U]:lH3[Y][2], lH3[Z][L]:lH3[Z][U]:lH3[Z][2]]
                        )

                    lE[(xyz+3-plus) % 3][L] += 1
                    lH1[(xyz+3-plus) % 3][L] += 1
                    lH2[(xyz+3-plus) % 3][L] += 1
                    lH3[(xyz+3-plus) % 3][L] += 1
                    lHo[(xyz+3-plus) % 3][L] += 1
                    lHo[(xyz+3-plus) % 3][U] -= 1

                    eNew[(xyz+plus) % 3][lE[X][L]:lE[X][U]:lE[X][2], lE[Y][L]:lE[Y][U]:lE[Y][2], lE[Z][L]:lE[Z][U]:lE[Z][2]] = \
                        e[(xyz+plus) % 3][lE[X][L]:lE[X][U]:lE[X][2], lE[Y][L]:lE[Y][U]:lE[Y][2], lE[Z][L]:lE[Z][U]:lE[Z][2]] + \
                        dt / AT * sg2 * (
                            + d01 * h[xyz][lH1[X][L]:lH1[X][U]:lH1[X][2], lH1[Y][L]:lH1[Y][U]:lH1[Y][2], lH1[Z][L]:lH1[Z][U]:lH1[Z][2]]
                            - d01 * h[xyz][lH2[X][L]:lH2[X][U]:lH2[X][2], lH2[Y][L]:lH2[Y][U]:lH2[Y][2], lH2[Z][L]:lH2[Z][U]:lH2[Z][2]]
                            - sg * d[(xyz+3-plus) % 3] * h[(xyz+3-plus) % 3][lH3[X][L]:lH3[X][U]:lH3[X][2], lH3[Y][L]:lH3[Y][U]:lH3[Y][2], lH3[Z][L]:lH3[Z][U]:lH3[Z][2]]
                            + 2*sg * d[(xyz+3-plus) % 3] * np.repeat(oh[(xyz+3-plus) % 3][lHo[X][L]:lHo[X][U]:lHo[X][2],
                                                                                          lHo[Y][L]:lHo[Y][U]:lHo[Y][2],
                                                                                          lHo[Z][L]:lHo[Z][U]:lHo[Z][2]],
                                                                2, axis=(xyz+plus) % 3)
                        )

        # Correct edge fields
        for edge in self.mesh.edges():
            if {edge.bound1.btype, edge.bound2.btype} == {"pmc", "overMesh"}:
                if edge.bound1.btype == "pmc":
                    bPmc = edge.bound1
                    bOvr = edge.bound2
                else:
                    bPmc = edge.bound2
                    bOvr = edge.bound1
                
                xyz, lu1, lu2 = edge.orientation()
                oPmc = bPmc.orientation()
                luPmc = bPmc.lu()
                oOvr = bOvr.orientation()
                luOvr = bOvr.lu()
                [lx, ly, lz], [ux, uy, uz] = edge.fieldIds((E, xyz))
                [lxo, lyo, lzo], [uxo, uyo, uzo] = edge.overFieldIds((H, oPmc))
                [lxi, lyi, lzi], [uxi, uyi, uzi] = edge.fieldIds((H, oPmc))
                [lxm, lym, lzm], [uxm, uym, uzm] = edge.fieldIds((H, oOvr))
                oh = overFields.get("H")

                sgPmc = 1 - 2*luPmc
                sgOvr = 1 - 2*luOvr
                sgOr = -1 + 2*int((xyz, oOvr, oPmc) == (X, Y, Z) or \
                                 (oOvr, oPmc, xyz) == (X, Y, Z) or \
                                 (oPmc, xyz, oOvr) == (X, Y, Z))
                
                eNew[xyz][lx:ux, ly:uy, lz:uz] = \
                    e[xyz][lx:ux, ly:uy, lz:uz] + \
                    dt / Ae_mirror * sgOr * (
                        - sgOvr * 2*d[oPmc] * np.repeat(oh[oPmc][lxo:uxo, lyo:uyo, lzo:uzo], 2, axis=xyz)
                        + sgOvr * d[oPmc] * h[oPmc][lxi:uxi, lyi:uyi, lzi:uzi]
                        - 2*sgPmc * d01 * h[oOvr][lxm:uxm, lym:uym, lzm:uzm]
                    )
            
            if {edge.bound1.btype, edge.bound2.btype} == {"overMesh", "overMesh"}:
                b1 = edge.bound1
                b2 = edge.bound2
                xyz = edge.orientation()[0]
                or1 = b1.orientation()
                or2 = b2.orientation()
                lu1 = b1.lu()
                lu2 = b2.lu()
                oh = overFields.get("H")

                [lx, ly, lz], [ux, uy, uz] = edge.fieldIds((E, xyz))
                [lx1, ly1, lz1], [ux1, uy1, uz1] = edge.fieldIds((H, or1))
                [lx2, ly2, lz2], [ux2, uy2, uz2] = edge.fieldIds((H, or2))
                [lxo1, lyo1, lzo1], [uxo1, uyo1, uzo1] = edge.overFieldIds((H, or1))
                [lxo2, lyo2, lzo2], [uxo2, uyo2, uzo2] = edge.overFieldIds((H, or2))

                sg1 = 1 - 2*lu1
                sg2 = 1 - 2*lu2
                sgOr = -1 + 2*int((xyz, or1, or2) == (X, Y, Z) or \
                                  (or1, or2, xyz) == (X, Y, Z) or \
                                  (or2, xyz, or1) == (X, Y, Z))
                
                eNew[xyz][lx:ux, ly:uy, lz:uz] = \
                    e[xyz][lx:ux, ly:uy, lz:uz] + \
                    dt / Ae * sgOr * (
                        - sg2 * d01 * h[or1][lx1:ux1, ly1:uy1, lz1:uz1]
                        + sg2 * 2*d[or1] * np.repeat(oh[or1][lxo1:uxo1, lyo1:uyo1, lzo1:uzo1], 2, axis=xyz)
                        + sg1 * d01 * h[or2][lx2:ux2, ly2:uy2, lz2:uz2]
                        - sg2 * 2*d[or2] * np.repeat(oh[or2][lxo2:uxo2, lyo2:uyo2, lzo2:uzo2], 2, axis=xyz)
                    )

        # Source terms
        for source in self.sources:
            if source["field"] == "E":
                if source["type"] == "dipole":
                    magnitude = source["magnitude"]
                    if magnitude["type"] == "gaussian":
                        c0 = sp.speed_of_light
                        delay = c0 * magnitude["gaussianDelay"]
                        spread = c0 * magnitude["gaussianSpread"]
                        idx = source["index"]
                        magn = np.array(source["direction"])/np.linalg.norm(source["direction"])
                        magn = magn * magnitude["gaussianHeight"]
                        
                        eNew[X][idx[L][X]:idx[U][X]  , idx[L][Y]:idx[U][Y]+1, idx[L][Z]:idx[U][Z]+1] += \
                            gaussian(t, delay, spread) * dt * magn[X]
                        eNew[Y][idx[L][X]:idx[U][X]+1, idx[L][Y]:idx[U][Y]  , idx[L][Z]:idx[U][Z]+1] += \
                            gaussian(t, delay, spread) * dt * magn[Y]
                        eNew[Z][idx[L][X]:idx[U][X]+1, idx[L][Y]:idx[U][Y]+1, idx[L][Z]:idx[U][Z]  ] += \
                            gaussian(t, delay, spread) * dt * magn[Z]
                    else:
                        raise ValueError("Invalid source magnitude type: " + magnitude["type"])
                else:
                    raise ValueError("Invalid source type: " + source["type"])
        
        # Subgridding and updating
        if "localTimeStepping" in self.options and \
                self.options["localTimeStepping"]:
            e[X][:] = eNew[X][:]
            e[Y][:] = eNew[Y][:]
            e[Z][:] = eNew[Z][:]
            for s in self.subsolvers:
                s.updateE(t, dt / 2.0, self.old)
            for s in self.subsolvers:
                s.updateH(t + dt / 4.0, dt / 2.0)
        else:
            for s in self.subsolvers:
                s.updateE(t, dt, self.old)
            e[X][:] = eNew[X][:]
            e[Y][:] = eNew[Y][:]
            e[Z][:] = eNew[Z][:]
    
    # ======================= UPDATE H =============================
    def updateH(self, t, dt):
        hNew = (np.zeros(self.old.hx.shape),
                np.zeros(self.old.hy.shape),
                np.zeros(self.old.hz.shape))
        e = self.old.get(E)
        h = self.old.get(H)
        
        d = self.mesh.steps
        A = self.mesh.areas()
        
        Ab, Aint, Aint_edge = self.mesh.subgridEquivAreasH()
        dA, dB, dC = self.mesh.subgridSteps()
        
        for xyz in range(3):
            lH  = np.roll([(None, None), (None, None), (None, None)], xyz, axis=0)
            lE1 = np.roll([(None, None), (None, None), (1   , None)], xyz, axis=0)
            lE2 = np.roll([(None, None), (None, None), (None, -1  )], xyz, axis=0)
            lE3 = np.roll([(None, None), (None, -1  ), (None, None)], xyz, axis=0)
            lE4 = np.roll([(None, None), (1   , None), (None, None)], xyz, axis=0)
            hNew[xyz][lH[X, L]:lH[X, U], lH[Y, L]:lH[Y, U], lH[Z, L]:lH[Z, U]] = \
                h[xyz][lH[X, L]:lH[X, U], lH[Y, L]:lH[Y, U], lH[Z, L]:lH[Z, U]] + dt / A[xyz] * (
                    + d[(xyz+1) % 3] * e[(xyz+1) % 3][lE1[X, L]:lE1[X, U], lE1[Y, L]:lE1[Y, U], lE1[Z, L]:lE1[Z, U]]
                    - d[(xyz+1) % 3] * e[(xyz+1) % 3][lE2[X, L]:lE2[X, U], lE2[Y, L]:lE2[Y, U], lE2[Z, L]:lE2[Z, U]]
                    + d[(xyz+2) % 3] * e[(xyz+2) % 3][lE3[X, L]:lE3[X, U], lE3[Y, L]:lE3[Y, U], lE3[Z, L]:lE3[Z, U]]
                    - d[(xyz+2) % 3] * e[(xyz+2) % 3][lE4[X, L]:lE4[X, U], lE4[Y, L]:lE4[Y, U], lE4[Z, L]:lE4[Z, U]]
                )
        
        # Updates from subsolvers
        for subsolver in self.subsolvers:
            for bound in subsolver.mesh.bounds:
                if bound.btype == "overMesh":
                    xyz = bound.orientation()
                    lu = bound.lu()

                    [lxo1, lyo1, lzo1], [uxo1, uyo1, uzo1] = bound.overFieldIds((H, (xyz+1) % 3))
                    [lxo2, lyo2, lzo2], [uxo2, uyo2, uzo2] = bound.overFieldIds((H, (xyz+2) % 3))
                    [lx1, ly1, lz1], [ux1, uy1, uz1] = bound.fieldIds((E, (xyz+2) % 3))
                    [lx2, ly2, lz2], [ux2, uy2, uz2] = bound.fieldIds((E, (xyz+1) % 3))
                    lowe = subsolver.old.get("E")

                    sg = 1 - 2*lu
                    lE = [lxo1+lu, lyo1+lu, lzo1+lu][xyz]
                    uE = [uxo1+lu, uyo1+lu, uzo1+lu][xyz]

                    l1 = [lxo1, uxo1],[lyo1, uyo1], [lzo1, uzo1]
                    l2 = [lxo1, uxo1],[lyo1, uyo1], [lzo1, uzo1]
                    l3 = [lx1, ux1, 1],[ly1, uy1, 1], [lz1, uz1, 1]
                    l4 = [lx1, ux1, 1],[ly1, uy1, 1], [lz1, uz1, 1]
                    l1[(xyz+2) % 3][L] += 1
                    l1[(xyz+2) % 3][U] += 1
                    l2[xyz][:] = [lE, uE]
                    l3[(xyz+1) % 3][2] = 2
                    l3[(xyz+2) % 3][2] = 2
                    l4[(xyz+1) % 3][2] = 2
                    l4[(xyz+2) % 3][L] += 1
                    l4[(xyz+2) % 3][2] = 2

                    hNew[(xyz+1) % 3][lxo1:uxo1, lyo1:uyo1, lzo1:uzo1] = \
                        h[(xyz+1) % 3][lxo1:uxo1, lyo1:uyo1, lzo1:uzo1] + \
                        dt / A[(xyz+1) % 3] * (
                            - d[xyz] * e[xyz][l1[X][L]:l1[X][U], l1[Y][L]:l1[Y][U], l1[Z][L]:l1[Z][U]]
                            + d[xyz] * e[xyz][lxo1:uxo1, lyo1:uyo1, lzo1:uzo1]
                            - sg * d[(xyz+2) % 3] * e[(xyz+2) % 3][l2[X][L]:l2[X][U], l2[Y][L]:l2[Y][U], l2[Z][L]:l2[Z][U]]
                            + sg * d[(xyz+2) % 3]/2 * ( lowe[(xyz+2) % 3][l3[X][L]:l3[X][U]:l3[X][2], l3[Y][L]:l3[Y][U]:l3[Y][2], l3[Z][L]:l3[Z][U]:l3[Z][2]] +
                                                        lowe[(xyz+2) % 3][l4[X][L]:l4[X][U]:l4[X][2], l4[Y][L]:l4[Y][U]:l4[Y][2], l4[Z][L]:l4[Z][U]:l4[Z][2]] )
                        )

                    l1 = [lxo2, uxo2],[lyo2, uyo2], [lzo2, uzo2]
                    l2 = [lxo2, uxo2],[lyo2, uyo2], [lzo2, uzo2]
                    l3 = [lx2, ux2, 1],[ly2, uy2, 1], [lz2, uz2, 1]
                    l4 = [lx2, ux2, 1],[ly2, uy2, 1], [lz2, uz2, 1]
                    l1[(xyz+1) % 3][L] += 1
                    l1[(xyz+1) % 3][U] += 1
                    l2[xyz][:] = [lE, uE]
                    l3[(xyz+1) % 3][2] = 2
                    l3[(xyz+2) % 3][2] = 2
                    l4[(xyz+1) % 3][L] += 1
                    l4[(xyz+1) % 3][2] = 2
                    l4[(xyz+2) % 3][2] = 2

                    hNew[(xyz+2) % 3][lxo2:uxo2, lyo2:uyo2, lzo2:uzo2] = \
                        h[(xyz+2) % 3][lxo2:uxo2, lyo2:uyo2, lzo2:uzo2] + \
                        dt / A[(xyz+2) % 3] * (
                            + d[xyz] * e[xyz][l1[X][L]:l1[X][U], l1[Y][L]:l1[Y][U], l1[Z][L]:l1[Z][U]]
                            - d[xyz] * e[xyz][lxo2:uxo2, lyo2:uyo2, lzo2:uzo2]
                            + sg * d[(xyz+1) % 3] * e[(xyz+1) % 3][l2[X][L]:l2[X][U], l2[Y][L]:l2[Y][U], l2[Z][L]:l2[Z][U]]
                            - sg * d[(xyz+1) % 3] / 2 * (lowe[(xyz+1) % 3][l3[X][L]:l3[X][U]:l3[X][2], l3[Y][L]:l3[Y][U]:l3[Y][2], l3[Z][L]:l3[Z][U]:l3[Z][2]] +
                                                         lowe[(xyz+1) % 3][l4[X][L]:l4[X][U]:l4[X][2], l4[Y][L]:l4[Y][U]:l4[Y][2], l4[Z][L]:l4[Z][U]:l4[Z][2]])
                        )
        
        # Updates overmesh values on boundaries
        for bound in self.mesh.bounds:
            if bound.btype == "overMesh":
                xyz = bound.orientation()
                lu = bound.lu()

                # Boundary fields
                [lx, ly, lz], [ux, uy, uz] = bound.fieldIds((H, xyz))
                [lx1, ly1, lz1], [ux1, uy1, uz1] = bound.fieldIds((E, (xyz+1) % 3))
                [lx2, ly2, lz2], [ux2, uy2, uz2] = bound.fieldIds((E, (xyz+2) % 3))

                e1 = e[(xyz+1) % 3][lx1:ux1, ly1:uy1, lz1:uz1]
                e2 = e[(xyz+2) % 3][lx2:ux2, ly2:uy2, lz2:uz2]
                s1 = list(e1.shape)
                s2 = list(e2.shape)
                d1 = np.resize([dA, dB], s1[(xyz+2) % 3])
                d2 = np.resize([dA, dB], s2[(xyz+1) % 3])
                s1[(xyz+1) % 3] = 1
                s2[(xyz+2) % 3] = 1
                d1 = d1.reshape(tuple(s1))
                d2 = d2.reshape(tuple(s2))
                d1 = np.repeat(d1, e1.shape[(xyz+1) % 3], axis=(xyz+1) % 3)
                d2 = np.repeat(d2, e2.shape[(xyz+2) % 3], axis=(xyz+2) % 3)
                e1 = d1 * e1
                e2 = d2 * e2

                l1 = np.roll([(None, None), (None, None), (+1, None)], xyz, axis=0)
                l2 = np.roll([(None, None), (None, None), (None, -1)], xyz, axis=0)
                l3 = np.roll([(None, None), (None, -1), (None, None)], xyz, axis=0)
                l4 = np.roll([(None, None), (+1, None), (None, None)], xyz, axis=0)

                hNew[xyz][lx:ux, ly:uy, lz:uz] = \
                    h[xyz][lx:ux, ly:uy, lz:uz] + \
                    dt/Ab * (
                        + e1[l1[X][L]:l1[X][U], l1[Y][L]:l1[Y][U], l1[Z][L]:l1[Z][U]]
                        - e1[l2[X][L]:l2[X][U], l2[Y][L]:l2[Y][U], l2[Z][L]:l2[Z][U]]
                        + e2[l3[X][L]:l3[X][U], l3[Y][L]:l3[Y][U], l3[Z][L]:l3[Z][U]]
                        - e2[l4[X][L]:l4[X][U], l4[Y][L]:l4[Y][U], l4[Z][L]:l4[Z][U]]
                    )
                    
                # Inner fields
                [lx1, ly1, lz1], [ux1, uy1, uz1] = bound.fieldIds((H, (xyz+1) % 3))
                [lx2, ly2, lz2], [ux2, uy2, uz2] = bound.fieldIds((H, (xyz+2) % 3))
                [lx, ly, lz], [ux, uy, uz] = bound.fieldIds((E, xyz))

                d1 = (1-lu)*dA + lu*dB
                d2 = (1-lu)*dB + lu*dA

                ein = e[xyz][lx:ux, ly:uy, lz:uz]
                sin = ein.shape
                din = np.array([[[dC if (i % 2, j % 2, k % 2) in [(0, 1, 1), (1, 0, 1) or (1, 1, 0)] else dA
                                  for k in range(sin[Z])] for j in range(sin[Y])] for i in range(sin[X])])
                ein = din * ein

                for plus in (1,2):
                    if plus == 1:
                        lH = [lx1, ux1, 1], [ly1, uy1, 1], [lz1, uz1, 1]
                        li1 = np.roll([(None, None, 1), (+1, -1, 2), (None, -1, 1)], xyz, axis=0)
                        li2 = np.roll([(None, None, 1), (+1, -1, 2), (+1, None, 1)], xyz, axis=0)
                        l1 = [lx1, ux1, 1], [ly1, uy1, 1], [lz1, uz1, 1]
                        l2 = [lx1, ux1, 1], [ly1, uy1, 1], [lz1, uz1, 1]
                        l1[xyz][L] += 1
                        l1[xyz][U] += 1
                    else:
                        lH = [lx2, ux2, 1], [ly2, uy2, 1], [lz2, uz2, 1]
                        li1 = np.roll([(None, None, 1), (+1, None, 1), (+1, -1, 2)], xyz, axis=0)
                        li2 = np.roll([(None, None, 1), (None, -1, 1), (+1, -1, 2)], xyz, axis=0)
                        l1 = [lx2, ux2, 1], [ly2, uy2, 1], [lz2, uz2, 1]
                        l2 = [lx2, ux2, 1], [ly2, uy2, 1], [lz2, uz2, 1]
                        l2[xyz][L] += 1
                        l2[xyz][U] += 1
                    lH[(xyz+plus) % 3][L] += 1
                    lH[(xyz+plus) % 3][U] -= 1
                    lH[(xyz+plus) % 3][2] = 2
                    l1[(xyz+plus) % 3][L] += 1
                    l1[(xyz+plus) % 3][U] -= 1
                    l1[(xyz+plus) % 3][2] = 2
                    l2[(xyz+plus) % 3][L] += 1
                    l2[(xyz+plus) % 3][U] -= 1
                    l2[(xyz+plus) % 3][2] = 2
                    d_1 = (2-plus) * d1 + (plus-1) * d2
                    d_2 = (2-plus) * d2 + (plus-1) * d1

                    hNew[(xyz+plus) % 3][lH[X][L]:lH[X][U]:lH[X][2], lH[Y][L]:lH[Y][U]:lH[Y][2], lH[Z][L]:lH[Z][U]:lH[Z][2]] = \
                        h[(xyz+plus) % 3][lH[X][L]:lH[X][U]:lH[X][2], lH[Y][L]:lH[Y][U]:lH[Y][2], lH[Z][L]:lH[Z][U]:lH[Z][2]] + \
                        dt / Aint * (
                            + ein[li1[X][L]:li1[X][U]:li1[X][2], li1[Y][L]:li1[Y][U]:li1[Y][2], li1[Z][L]:li1[Z][U]:li1[Z][2]]
                            - ein[li2[X][L]:li2[X][U]:li2[X][2], li2[Y][L]:li2[Y][U]:li2[Y][2], li2[Z][L]:li2[Z][U]:li2[Z][2]]
                            + d_1 * e[(xyz+3-plus) % 3][l1[X][L]:l1[X][U]:l1[X][2], l1[Y][L]:l1[Y][U]:l1[Y][2], l1[Z][L]:l1[Z][U]:l1[Z][2]]
                            - d_2 * e[(xyz+3-plus) % 3][l2[X][L]:l2[X][U]:l2[X][2], l2[Y][L]:l2[Y][U]:l2[Y][2], l2[Z][L]:l2[Z][U]:l2[Z][2]]
                        )

        # Corrects overmesh edges
        for edge in self.mesh.edges():
            if {edge.bound1.btype, edge.bound2.btype} == {"overMesh", "overMesh"}:
                b1 = edge.bound1
                b2 = edge.bound2
                xyz = edge.orientation()[0]
                or1 = b1.orientation()
                or2 = b2.orientation()
                lu1 = b1.lu()
                lu2 = b2.lu()

                [lx, ly, lz], [ux, uy, uz] = edge.fieldIds((H, xyz))
                [lx1, ly1, lz1], [ux1, uy1, uz1] = edge.fieldIds((E, or1))
                [lx2, ly2, lz2], [ux2, uy2, uz2] = edge.fieldIds((E, or2))
                [lxi1, lyi1, lzi1], [uxi1, uyi1, uzi1] = edge.fieldIds((E, or1))
                [lxi2, lyi2, lzi2], [uxi2, uyi2, uzi2] = edge.fieldIds((E, or2))

                sg1 = 1 - 2*lu1
                sg2 = 1 - 2*lu2
                sgOr = -1 + 2*int((xyz, or1, or2) == (X, Y, Z) or \
                                  (or1, or2, xyz) == (X, Y, Z) or \
                                  (or2, xyz, or1) == (X, Y, Z))
                
                if xyz == X:
                    lx+=1; ux-=1
                    lx1+=1; ux1-=1
                    lxi1+=1; uxi1-=1
                    lx2+=1; ux2-=1
                    lxi2+=1; uxi2-=1
                elif xyz == Y:
                    ly+=1; uy-=1
                    ly1+=1; uy1-=1
                    lyi1+=1; uyi1-=1
                    ly2+=1; uy2-=1
                    lyi2+=1; uyi2-=1
                else:
                    lz+=1; uz-=1
                    lz1+=1; uz1-=1
                    lzi1+=1; uzi1-=1
                    lz2+=1; uz2-=1
                    lzi2+=1; uzi2-=1
                if or2 == X:
                    lxi1 += sg2
                    uxi1 += sg2
                elif or2 == Y:
                    lyi1 += sg2
                    uyi1 += sg2
                else:
                    lzi1 += sg2
                    uzi1 += sg2
                if or1 == X:
                    lxi2 += sg1
                    uxi2 += sg1
                elif or1 == Y:
                    lyi2 += sg1
                    uyi2 += sg1
                else:
                    lzi2 += sg1
                    uzi2 += sg1
                
                # [lx, ly, lz][xyz] += 1
                # [ux, uy, uz][xyz] -= 1
                # [lx1, ly1, lz1][xyz] += 1
                # [ux1, uy1, uz1][xyz] -= 1
                # [lxi1, lyi1, lzi1][xyz] += 1
                # [uxi1, uyi1, uzi1][xyz] -= 1
                # [lx2, ly2, lz2][xyz] += 1
                # [ux2, uy2, uz2][xyz] -= 1
                # [lxi2, lyi2, lzi2][xyz] += 1
                # [uxi2, uyi2, uzi2][xyz] -= 1
                # [lxi1, lyi1, lzi1][or2] += sg2
                # [uxi1, uyi1, uzi1][or2] += sg2
                # [lxi2, lyi2, lzi2][or1] += sg1
                # [uxi2, uyi2, uzi2][or1] += sg1
                
                size = (ux-lx)*(uy-ly)*(uz-lz)
                shape = (ux-lx, uy-ly, uz-lz)
                dB_arr = np.resize([dB, dA], size).reshape(shape)
                dC_arr = np.resize([dC, dA], size).reshape(shape)
                A_arr = np.resize([Aint_edge, dA*dA], size).reshape(shape)
                
                hNew[xyz][lx:ux, ly:uy, lz:uz] = \
                    h[xyz][lx:ux, ly:uy, lz:uz] + \
                    dt / A_arr * sgOr * (
                        - sg2 * dB_arr * e[or1][lx1:ux1, ly1:uy1, lz1:uz1]
                        + sg2 * dC_arr * e[or1][lxi1:uxi1, lyi1:uyi1, lzi1:uzi1]
                        + sg1 * dB_arr * e[or2][lx2:ux2, ly2:uy2, lz2:uz2]
                        - sg1 * dC_arr * e[or2][lxi2:uxi2, lyi2:uyi2, lzi2:uzi2]
                    )
        
        # Corrects overmesh corners
        pass  # TODO

        # Source terms
        for source in self.sources:
            if source["field"] == "H":
                if source["type"] == "dipole":
                    magnitude = source["magnitude"]
                    if magnitude["type"] == "gaussian":
                        c0 = sp.speed_of_light
                        delay = c0 * magnitude["gaussianDelay"]
                        spread = c0 * magnitude["gaussianSpread"]
                        idx = source["index"]
                        magn = np.array(source["direction"])/np.linalg.norm(source["direction"])
                        magn = magn * magnitude["gaussianHeight"]
                        value = gaussian(t, delay, spread)
                        
                        hNew[X][idx[L][X]:idx[U][X]+1, idx[L][Y]:idx[U][Y], idx[L][Z]:idx[U][Z]] += \
                            value * dt * magn[X]
                        hNew[Y][idx[L][X]:idx[U][X], idx[L][Y]:idx[U][Y]+1, idx[L][Z]:idx[U][Z]] += \
                            value * dt * magn[Y]
                        hNew[Z][idx[L][X]:idx[U][X], idx[L][Y]:idx[U][Y], idx[L][Z]:idx[U][Z]+1] += \
                            value * dt * magn[Z]
                    elif magnitude["type"] == "sine":
                        c0 = sp.speed_of_light
                        phi = magnitude["sinePhase"]
                        w = magnitude["sineFrequency"] / c0
                        idx = source["index"]
                        magn = np.array(source["direction"]) / np.linalg.norm(source["direction"])
                        magn = magn * magnitude["sineHeight"]
                        value = np.sin(w*t - phi)
                        
                        hNew[X][idx[L][X]:idx[U][X]+1, idx[L][Y]:idx[U][Y], idx[L][Z]:idx[U][Z]] += \
                            value * dt * magn[X]
                        hNew[Y][idx[L][X]:idx[U][X], idx[L][Y]:idx[U][Y]+1, idx[L][Z]:idx[U][Z]] += \
                            value * dt * magn[Y]
                        hNew[Z][idx[L][X]:idx[U][X], idx[L][Y]:idx[U][Y], idx[L][Z]:idx[U][Z]+1] += \
                            value * dt * magn[Z]
                    else:
                        raise ValueError("Invalid source magnitude type: " + magnitude["type"])
                else:
                    raise ValueError("Invalid source type: " + source["type"])

        if "localTimeStepping" in self.options \
                and self.options["localTimeStepping"]:
            h[X][:] = hNew[X][:]
            h[Y][:] = hNew[Y][:]
            h[Z][:] = hNew[Z][:]
            for s in self.subsolvers:
                s.updateE(t, dt / 2.0, self.old)
            for s in self.subsolvers:
                s.updateH(t + dt / 4.0, dt / 2.0)
        else:
            for s in self.subsolvers:
                s.updateH(t, dt)
            h[X][:] = hNew[X][:]
            h[Y][:] = hNew[Y][:]
            h[Z][:] = hNew[Z][:]
    
    def updateProbes(self, t):
        dimensionalTime = t/sp.speed_of_light
        for p in self.probes:
            if p["type"] == "slice":
                writeStep = "samplingPeriod" not in p \
                    or dimensionalTime/p["samplingPeriod"] >= len(p["time"])
                if writeStep:
                    p["time"].append(dimensionalTime)
                    idx = p["indices"]
                    values = np.zeros(tuple(idx[U]-idx[L]))
                    values[:, :] = self.old.get(p["field"][0])[p["field"][1]][
                        idx[L][X]:idx[U][X], idx[L][Y]:idx[U][Y], idx[L][Z]:idx[U][Z]
                    ]
                    p["values"].append(values)
            elif p["type"] == "nodal":
                if p["active"]:
                    writeStep = "samplingPeriod" not in p \
                        or dimensionalTime/p["samplingPeriod"] >= len(p["time"])
                    if writeStep:
                        p["time"].append(dimensionalTime)
                        p["values"].append(
                            self.old.get(p["field"][0])[p["field"][1]][
                                p["index"][0], p["index"][1], p["index"][2]
                            ]
                        )
            else:
                raise ValueError("Invalid probe type: " + p["type"])
        for s in self.subsolvers:
            s.updateProbes(t)
    
    def solve(self, dimensionalFinalTime):
        tic = time.time()
        t = 0.
        numberOfTimeSteps = int(dimensionalFinalTime * sp.speed_of_light / self.dt)
        
        for n in range(numberOfTimeSteps):
            self.updateE(t, self.dt)
            t += self.dt/2.
            
            self.updateH(t, self.dt)
            t += self.dt/2.
            
            self.updateProbes(t)
            
            if n % self.__timeStepPrint == 0 or n+1 == numberOfTimeSteps:
                remaining = (time.time() - tic) * (numberOfTimeSteps-n) / (n+1)
                mins = math.floor(remaining / 60.0)
                sec = remaining % 60.0
                print("    Step: {0:6d} of {1:6d}. Remaining: {2:2.0f}:{3:2.0f}".format(n, numberOfTimeSteps-1, mins, sec))

        print("    CPU Time: %f [s]" % (time.time() - tic))