
import numpy as np
from skspatial.objects import Vector

class Face(object):
    """A class to define a face on an icosphere
    x,y,z are the coordinates in 3D space of points on an icosphere
    u,v are the coordinates on a 2D plane corresponding to the unfolded icosphere net
    children are the faces at the next layer of magnification (anti coarse graining of a sphere)
    atoms are the atoms that project onto that face"""
    def __init__(self, x=None, y=None, z=None):
        self.x = x # these are vectors from 0,0,0 to x y z etc
        self.y = y
        self.z = z
        self.u = None
        self.v = None
        self.children = []
        self.atoms = []

    def create_children(self, levels_to_do=1):
        """triangles are formed at the midpoints of edges
            we label points in a clockwise manner ALWAYS"""
        if self.children:
            # check we not got no children already
            raise ValueError("create children called on face with children already assigned")
        if levels_to_do == 0:
            return
        # xy is the left hand edge of the triangle
        # finds the mid-point on the line defined by x, y
        # AND PROEJCTS IT TO THE UNIT SPHERE EASY!
        xy = ((self.x + self.y) /2).unit()
        yz = ((self.y + self.z) /2).unit()
        zx = ((self.z + self.x) /2).unit()
        self.children = [ # the four new triangles
            Face(self.x, xy, zx), # 1 bottom left
            Face(xy, self.y, yz), # 2 top
            Face(zx, yz, self.z), # 3 bottom right
            Face(xy, yz, zx)] # 4 middle
        for child in self.children:
            # this should work...
            child.create_children(levels_to_do-1)

        def get_atom_for_point(self):
            pass
            return



