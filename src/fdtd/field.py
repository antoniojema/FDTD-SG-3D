import numpy as np

from fdtd.common import *

fieldTypes = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]


def fieldOrientation(field):
    if type(field) == str:
        if field not in fieldTypes:
            raise ValueError("Invalid field label")
        
        if "x" in field:
            return X
        elif "y" in field:
            return Y
        else:
            return Z
    elif type(field) == tuple:
        return fieldOrientation(fieldLabel(field))
    else:
        raise ValueError("Invalid field type")


def fieldIndex(field):
    if field[0] == "E":
        eh = E
    elif field[0] == "H":
        eh = H
    if field[1] == "x":
        xyz = X
    elif field[1] == "y":
        xyz = Y
    elif field[1] == "z":
        xyz = Z
    return eh, xyz


def fieldLabel(field):
    if field[0] == E:
        eh = "E"
    elif field[0] == H:
        eh = "H"
    if field[1] == X:
        xyz = "x"
    elif field[1] == Y:
        xyz = "y"
    elif field[1] == Z:
        xyz = "z"
    return eh+xyz


class Fields:
    def __init__(self, ex, ey, ez, hx, hy, hz):
        self.ex = ex
        self.ey = ey
        self.ez = ez
        self.hx = hx
        self.hy = hy
        self.hz = hz
    
    def get(self, field="all"):
        if field == "all":
            return self.ex, self.ey, self.ez, self.hx, self.hy, self.hz
        elif field in (E, "E"):
            return self.ex, self.ey, self.ez
        elif field in (H, "H"):
            return self.hx, self.hy, self.hz
        
        raise ValueError("Invalid field type")
