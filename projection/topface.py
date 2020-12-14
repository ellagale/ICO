from operator import attrgetter

import numpy as np
from scipy.constants import golden
from skspatial.objects import Point

from projection.face import Face

# Constants
_unit_length = np.sqrt(1 + (golden * golden))
# the two distances we need for the coordinates.
_c1 = 1 / _unit_length
_c2 = golden / _unit_length
# The points that make up an icosahedron in space, circumradius 1, centered around 0,0,0

_points = [
    Point([-_c1, _c2, 0]),
    Point([_c1, _c2, 0]),
    Point([_c1, -_c2, 0]),
    Point([-_c1, -_c2, 0]),
    Point([-_c2, 0, -_c1]),
    Point([_c2, 0, -_c1]),
    Point([_c2, 0, _c1]),
    Point([-_c2, 0, _c1]),
    Point([0, _c1, -_c2]),
    Point([0, _c1, _c2]),
    Point([0, -_c1, _c2]),
    Point([0, -_c1, -_c2])
]

_edges = [
    frozenset([0, 1]),
    frozenset([8, 1]),
    frozenset([5, 1]),
    frozenset([6, 1]),
    frozenset([9, 1]),
    frozenset([0, 8]),
    frozenset([8, 5]),
    frozenset([5, 6]),
    frozenset([6, 9]),
    frozenset([9, 0]),
    frozenset([0, 4]),
    frozenset([4, 8]),
    frozenset([8, 11]),
    frozenset([11, 5]),
    frozenset([5, 2]),
    frozenset([2, 6]),
    frozenset([6, 10]),
    frozenset([10, 9]),
    frozenset([9, 7]),
    frozenset([7, 0]),
    frozenset([4, 11]),
    frozenset([11, 2]),
    frozenset([2, 10]),
    frozenset([10, 7]),
    frozenset([7, 4]),
    frozenset([4, 3]),
    frozenset([11, 3]),
    frozenset([2, 3]),
    frozenset([10, 3]),
    frozenset([7, 3])
]

# How high to draw triangles for 2D output
_triangle_2d_hypotenuse = 6
_triangle_2d_height = np.sqrt(_triangle_2d_hypotenuse ** 2 - (_triangle_2d_hypotenuse / 2) ** 2)

# Used to make sure the ico will fit in a 3D render
_scale = np.array(_points).flatten('K')


def get_scale():
    return _scale


class TopFace(Face):
    """ A Face that exists at the top level of the icosahedron.
        Contains extra functions  that inner faces do not need.
    """

    def __init__(self, molecule, a_idx, b_idx, c_idx):
        super().__init__(molecule, _points[a_idx], _points[b_idx], _points[c_idx])
        self.edges = {
            (a_idx, b_idx),
            (b_idx, c_idx),
            (c_idx, a_idx)
        }
        self.indices = [a_idx, b_idx, c_idx]
        self.neighbours = []
        # Where in the 2D unwrapped grid to render this face
        self.grid_coords = None

    def get_edges(self):
        """ Returns a set of the edges that this face has.
        """
        return [frozenset(x) for x in self.edges]

    def get_point_indices(self):
        """ Returns a set of the point indices that this face has.
        """
        return self.indices

    def clear_grid(self):
        """ Clear the grid before a fresh unwrap."""
        self.grid_coords = None

    def _set_grid_first_row(self, top_idx, column):
        """ Recurse along the top row of the grid, initialising them
        """
        if top_idx not in self.indices:
            raise ValueError("Invalid point index.")

        if self.grid_coords is not None:
            # we've come all the way around. Stop
            return

        self.grid_coords = (0, column)
        # Now find the faces to update.
        # next point is the point adjacent to the top point on the edge shared with the next
        # face on the top row. It is also a point shared with the adjacent face on the second row
        next_point = [b for a, b in self.edges if a == top_idx][0]
        next_faces = [n for n in self.neighbours if next_point in n.indices]
        assert (len(next_faces) == 2)
        for next_face in next_faces:
            if top_idx in next_face.indices:
                # This is the one at the top of the next column.
                next_face._set_grid_first_row(top_idx, column + 1)
                continue

            # This is the one below us on the second row
            assert (next_face.grid_coords is None)
            next_face.grid_coords = (1, column)

            # Find the one in the third row.
            third_face_candidates = [f for f in next_face.neighbours if (next_point in f.indices and f is not self)]
            assert (len(third_face_candidates) == 1)
            third_face = third_face_candidates[0]
            assert (third_face.grid_coords is None)
            third_face.grid_coords = (2, column)

            # finally the fourth row.
            # two ways we could do it:
            # a. It is the neighbour of third_face that shares a point with next_face
            # b. It is the neighbour of third_face that is on the edge that does not contain next_point
            # Going with a.
            point_2_4 = [b for a, b in next_face.edges if a == next_point][0]
            fourth_face_candidates = [f for f in third_face.neighbours if
                                      (point_2_4 in f.indices and f is not next_face)]
            if len(fourth_face_candidates) != 1:
                print("Third face neighbours:\n{}".format(third_face.neighbours))
                print("Second face:\n{}".format(next_face))
                print("Third face:\n{}".format(third_face))
                print("Second_fourth point: {}".format(point_2_4))
                print(fourth_face_candidates)

            assert (len(fourth_face_candidates) == 1)
            fourth_face = fourth_face_candidates[0]
            assert (fourth_face.grid_coords is None)
            fourth_face.grid_coords = (3, column)

    def set_grid(self, top_idx):
        """ Initialise the grid, with the point marked as top_idx as the top point
            and self as 0,0
        """
        self._set_grid_first_row(self.indices[top_idx], 0)

    def plot2D(self, height=_triangle_2d_height, width=_triangle_2d_hypotenuse):
        """
        Compute the 2D projection of this face
        """
        if self.grid_coords is None:
            raise ValueError("_project2D called with no grid setup")
        y, x = self.grid_coords

        # Time to map from grid space to 2d space
        # If we are on an even numbered row, then we are facing up and our coords will be
        #    v
        #   / \
        #  u - w
        # If we are on an odd numbered row, then we are facing down and our coords will be
        #  u - v
        #   \ /
        #    w

        # first u
        ux = (x + (y // 2) / 2) * width
        uy = (2 - y // 2) * height

        if y % 2 == 0:
            # even row
            vx, vy = ux + (width / 2.0), uy + height
            wx, wy = ux + width, uy
        else:
            # odd row
            vx, vy = ux + width, uy
            wx, wy = ux + (width / 2.0), uy - height

        self._map_coords((ux, uy), (vx, vy), (wx, wy))

    def draw2D(self, plot, height=_triangle_2d_height, width=_triangle_2d_hypotenuse):
        """
        Draw this face onto the supplied plot
        """
        if self.grid_coords is None:
            raise ValueError("draw2D called with no grid setup")

        self._draw(plot)

    def get_ordered_leaf_faces(self):
        """ Returns a list of the lowest level faces, ordered by their locations
                1           1 2 3
               2 3           4 5
              4 5 6           6
              etc.
        """
        if not self.midpoint:
            raise ValueError("get_ordered_leaf_faces called without first plotting.")

        face_list = self.get_leaf_faces()
        partial_list = sorted(face_list, key=attrgetter('midpoint_x'))
        return sorted(partial_list, key=attrgetter('midpoint_y'), reverse=True)

    def __repr2__(self):
        """ Print the Face."""
        output = "Face: {}".format(self.indices)

        if self.midpoint:
            output += " at {}".format(self.midpoint)

        if self.u:
            output += " ({},{},{})".format(self.u, self.v, self.w)

        if self.atoms:
            output += " with {}".format(self.atoms)
        return repr(output)
