import numpy as np
from skspatial.objects import Point
from scipy.constants import golden
from face import Face

_unit_length = np.sqrt(1+(golden*golden))
# the two distances we need for the coordinates.
_c1 = 1 / _unit_length
_c2 = golden / _unit_length
# The points that make up an icosahedron in space, circumradius 1, centered around 0,0,0
_points = [
        Point([-_c1,  _c2,   0]),
        Point([ _c1,  _c2,   0]),
        Point([ _c1, -_c2,   0]),
        Point([-_c1, -_c2,   0]),
        Point([-_c2,   0, -_c1]),
        Point([ _c2,   0, -_c1]),
        Point([ _c2,   0,  _c1]),
        Point([-_c2,   0,  _c1]),
        Point([  0,  _c1, -_c2]),
        Point([  0,  _c1,  _c2]),
        Point([  0, -_c1,  _c2]),
        Point([  0, -_c1, -_c2])
        ]

class Ico(object):
    """A class to define an icsahedron or icosphere.
    """

    def __init__(self, level=0):
        """ Create a new icosphere.
        Level indicates how many subdivisions to make, 0 (default) will give you an icosohedron.
        All icospheres are centred around the origin with a cicumradius of 1
        """

        


        self.faces = [
                # Top row, around point 1
                Face(_points[0], _points[8], _points[1]),
                Face(_points[8], _points[5], _points[1]),
                Face(_points[5], _points[6], _points[1]),
                Face(_points[6], _points[9], _points[1]),
                Face(_points[9], _points[0], _points[1]),
                # Row 2
                Face(_points[0], _points[4], _points[8]),
                Face(_points[8], _points[11], _points[5]),
                Face(_points[5], _points[2], _points[6]),
                Face(_points[6], _points[10], _points[9]),
                Face(_points[9], _points[7], _points[0]),
                # Row 3
                Face(_points[8], _points[4], _points[11]),
                Face(_points[5], _points[11], _points[2]),
                Face(_points[6], _points[2], _points[10]),
                Face(_points[9], _points[10], _points[7]),
                Face(_points[0], _points[7], _points[4]),
                # Bottom row, around point 3
                Face(_points[4], _points[3], _points[11]),
                Face(_points[11], _points[3], _points[2]),
                Face(_points[2], _points[3], _points[10]),
                Face(_points[10], _points[3], _points[7]),
                Face(_points[7], _points[3], _points[4])
                ]

        # now perform the subdivisions
        # conditional isn't needed, but why waste the time?
        if level==0:
            return

        for face in self.faces:
            face.create_children(level)






