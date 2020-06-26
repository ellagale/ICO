import numpy as np
from skspatial.objects import Vector

class Face(object):
    """A class to define a face on an icosphere
    a,b,c are the points in 3D space of points on an icosphere
    u,v are the coordinates on a 2D plane corresponding to the unfolded icosphere net
    children are the faces at the next layer of magnification (anti coarse graining of a sphere)
    atoms are the atoms that project onto that face"""
    def __init__(self, a=None, b=None, c=None):
        self.a = Vector(a) # these are vectors from 0,0,0 to x y z etc
        self.b = Vector(b)
        self.c = Vector(c)
        self.u = None
        self.v = None
        self.children = []
        self.atoms = []

    def __repr__(self):
        """ Print the Face."""
        return repr("Face: {},{},{}".format(self.a, self.b, self.c))

    def create_children(self, levels_to_do=1):
        """triangles are formed at the midpoints of edges
            we label points in a clockwise manner ALWAYS"""
        if self.children:
            # check we not got no children already
            raise ValueError("create children called on face with children already assigned")
        if levels_to_do == 0:
            return
        # ab is the left hand edge of the triangle
        # finds the mid-point on the line defined by x, y
        # AND PROEJCTS IT TO THE UNIT SPHERE EASY!
        ab = ((self.a + self.b) /2).unit()
        bc = ((self.b + self.c) /2).unit()
        ca = ((self.c + self.a) /2).unit()
        self.children = [ # the four new triangles
            Face(self.a, ab, ca), # 1 bottom left
            Face(ab, self.b, bc), # 2 top
            Face(ca, bc, self.c), # 3 bottom right
            Face(ab, bc, ca)] # 4 middle
        for child in self.children:
            # this should work...
            child.create_children(levels_to_do-1)

        def get_atom_for_point(self):
            pass
            return



