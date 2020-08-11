import numpy as np
from skspatial.objects import Vector

class _FaceIter(object):
    """ Class to iterate over the triangles of the face.
    """
    def __init__(self, face):
        self.face = face
        self.depth = face.get_depth_remaining()
        # how many faces inside this face
        self.face_count = 4 ** self.depth
        self.faces_per_child = self.face_count // 4
        # iteration call
        self.n = 0

    def __next__(self):
        # Handle stop case
        if self.n >= self.face_count:
            raise StopIteration
        # Handle edge case of lone face
        if not self.children:
            self.n += 1
            return self
        # Pick the next child to take from.
        child = self.n % 4
        child_idx = self.n // 4
        self.n += 1
        return self.children[child].get_face(child_idx)
        




class Face(object):
    """A class to define a face on an icosphere
    upwards is whether this face points up or down. Only actually needed for iteration
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
            Face(ab, bc, ca)] # 4 middle.
        for child in self.children:
            # this should work...
            child.create_children(levels_to_do-1)

    def get_atom_for_point(self):
        return
    
    def get_face(self, idx):
        # Handle leaf case
        if not self.children: 
            if idx == 0:
                return self
            else:
                raise ValueError

        # Recurse
        child = idx % 4
        child_idx = idx // 4
        return self.children[child].get_face(child_idx)


    def get_depth_remaining(self):
        """ Returns the depth of this face.
        """
        if not self.children:
            return 0
        return self.children[0].get_depth_remaining()

    def __iter__(self):
        return _FaceIter(self)
