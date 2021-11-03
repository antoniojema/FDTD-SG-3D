import numpy as np
import copy

from fdtd.common import *
from fdtd.field import fieldOrientation, fieldLabel


class Bound:
    btypeSet = ["pec", "pmc", "overMesh", None]
    
    def __init__(self, btype, ids=None, overMeshIds=None):
        if btype not in Bound.btypeSet:
            raise ValueError("Invalid boundary type "+str(btype))
        self.btype = btype
        self.ids = ids
        if self.btype == "overMesh":
            self.overBound = Bound(btype=None, ids=overMeshIds)
    
    def orientation(self):
        if self.ids is None:
            raise ValueError("Error getting orientation: boundary ids not defined.")
        return np.where(self.ids[L] == self.ids[U])[0][0]
    
    def lu(self):
        return abs(np.sign(self.ids[L][self.orientation()]))
        
    def idsAs(self, lu, xyz, size=None):
        if size is None:
            lid = np.array([ 0 if i != xyz else -1*lu for i in range(3)], dtype=object)
            uid = np.array([-1 if i != xyz else -1*lu for i in range(3)], dtype=object)
        else:
            lid = np.array([    0     if i != xyz else (size[i]-1)*lu for i in range(3)], dtype=object)
            uid = np.array([size[i]-1 if i != xyz else (size[i]-1)*lu for i in range(3)], dtype=object)
        self.ids = (lid, uid)

    def fieldIds(self, field):
        bOr = self.orientation()
        if type(field) == tuple:
            field = fieldLabel(field)
        fOr = fieldOrientation(field)
        lu = self.lu()
        ids = copy.deepcopy(self.ids)
        if field[0] == "E":
            if fOr == bOr:
                ids[U][(bOr+1) % 3] += 1
                ids[U][(bOr+2) % 3] += 1
                if lu == L:
                    ids[U][bOr] += 1
                else:
                    ids[L][bOr] -= 1
            else:
                aOr = np.delete([X, Y, Z], [bOr, fOr])[0]

                ids[U][aOr] = addId(ids[U][aOr], 1)
                ids[U][bOr] = addId(ids[L][bOr], 1)
        else:
            if fOr == bOr:
                ids[U][bOr] = addId(ids[U][bOr], 1)
            else:
                aOr = np.delete([X, Y, Z], [bOr, fOr])[0]
                ids = self.fieldIds((E, aOr))
                if lu == U:
                    ids[L][bOr] -= 1
                    ids[U][bOr] -= 1
    
        return ids
    
    def overFieldIds(self, field):
        if type(field) == tuple:
            field = fieldLabel(field)
        if field[0] == "E" or self.btype != "overMesh":
            raise ValueError("Cannot request overFieldIds for electric field")
        bOr = self.orientation()
        fOr = fieldOrientation(field)
        if fOr == bOr:
            raise ValueError("Field and boundary orientations must not match for overFieldIds")
        aOr = np.delete([X, Y, Z], [bOr, fOr])[0]
        lu = self.lu()
        ids = copy.deepcopy(self.overBound.ids)
        
        if lu == L:
            ids[L][bOr] -= 1
        else:
            ids[U][bOr] += 1
        ids[U][fOr] += 1
        
        return ids
        
    def overmeshIdsFrom(self, ids):
        if self.ids is None:
            raise ValueError("Overmesh Ids must be set after ids.")
        elif self.btype != "overMesh":
            raise ValueError("Not possible to get overField ids from " + self.btype + " boundary")
        self.overBound = Bound(btype=None, ids=ids)


class Edge:
    def __init__(self, bound1=None, bound2=None, ids=None):
        if (ids is not None) or (bound1 is None) or (bound2 is None):
            self.ids = ids
        else:
            ors = [bound1.orientation(), bound2.orientation()]
            if ors[0] == ors[1]:
                raise ValueError("Error setting edge: both boundaries have same orientation")
            else:
                xyz = np.delete([X, Y, Z], ors)[0]
                self.ids = (np.empty(3, dtype=object), np.empty(3, dtype=object))
    
                if ors[0] == (xyz+1) % 3:
                    b1, b2 = self.bound1, self.bound2 = bound1, bound2
                else:
                    b1, b2 = self.bound1, self.bound2 = bound2, bound1
                
                if b1.ids[L][xyz] != b2.ids[L][xyz] or b1.ids[U][xyz] != b2.ids[U][xyz]:
                    raise ValueError("Boundaries ids don't match")
                
                self.ids[L][xyz] = b1.ids[L][xyz]
                self.ids[U][xyz] = b1.ids[U][xyz]
                self.ids[L][(xyz+1) % 3] = b1.ids[L][(xyz+1) % 3]
                self.ids[U][(xyz+1) % 3] = b1.ids[U][(xyz+1) % 3]
                self.ids[L][(xyz+2) % 3] = b2.ids[L][(xyz+2) % 3]
                self.ids[U][(xyz+2) % 3] = b2.ids[U][(xyz+2) % 3]
                
    def orientation(self):
        xyz = np.where(self.ids[L] != self.ids[U])[0][0]
        lu1 = self.bound1.lu()
        lu2 = self.bound2.lu()
        return xyz, lu1, lu2  # xyz, (xyz+1)%3, (xyz+2)%3
    
    def fieldIds(self, field):
        eOr, lu1, lu2 = self.orientation()
        if type(field) == tuple:
            field = fieldLabel(field)
        fOr = fieldOrientation(field)
        
        if field[0] == "E":
            if fOr == eOr:
                ids = copy.deepcopy(self.ids)
                ids[U][(eOr+1) % 3] += 1
                ids[U][(eOr+2) % 3] += 1
            else:
                if self.bound1.orientation() != fOr:
                    bIds = self.bound1
                    bOther = self.bound2
                else:
                    bIds = self.bound2
                    bOther = self.bound1
                ids = bIds.fieldIds(field)
                if bOther.lu() == L:
                    ids[U][fOr] = ids[L][fOr] + 1
                else:
                    ids[L][fOr] = ids[U][fOr] - 1
        else:
            ids = copy.deepcopy(self.ids)
            if fOr == eOr:
                ids[U][eOr] += 1
                if lu1 == L:
                    ids[U][(eOr+1) % 3] += 1
                else:
                    ids[L][(eOr+1) % 3] -= 1
                if lu2 == L:
                    ids[U][(eOr+2) % 3] += 1
                else:
                    ids[L][(eOr+2) % 3] -= 1
            else:
                if self.bound1.orientation() == fOr:
                    bIds = self.bound1
                    bOther = self.bound2
                else:
                    bIds = self.bound2
                    bOther = self.bound1
                ids = bIds.fieldIds(field)
                if bOther.lu() == L:
                    ids[U][bOther.orientation()] = ids[L][bOther.orientation()] + 1
                else:
                    ids[L][bOther.orientation()] = ids[U][bOther.orientation()] - 1
        
        return ids
    
    def overFieldIds(self, field):
        if type(field) == tuple:
            field = fieldLabel(field)
        if field[0] == "E":
            raise ValueError("Cannot request overFieldIds for electric field")
        eOr = self.orientation()[0]
        fOr = fieldOrientation(field)
        if eOr == fOr:
            raise ValueError("Field and edge orientations must not match")
        
        if self.bound1.orientation() != fOr:
            bIds = self.bound1
            bOther = self.bound2
        else:
            bIds = self.bound2
            bOther = self.bound1
        
        if bIds.btype != "overMesh":
            raise ValueError("Not possible to get overField ids from " + bIds.btype + " boundary")
            
        ids = bIds.overFieldIds(field)
        
        if bOther.lu() == L:
            ids[U][bOther.orientation()] = ids[L][bOther.orientation()] + 1
        else:
            ids[L][bOther.orientation()] = ids[U][bOther.orientation()] - 1
        
        return ids
