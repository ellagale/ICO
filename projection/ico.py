import numpy as np
import matplotlib
import matplotlib.pyplot
import mpl_toolkits.mplot3d
import stl
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


    def draw(self, filename="output.png"):
        """ Draw the Ico in 3d
        I've no idea what parameters, so let's hardcode for now.
        """
        figure = matplotlib.pyplot.figure()
        axes = mpl_toolkits.mplot3d.Axes3D(figure)

    def determining_row_of_upright_triangles(net, row_no, no_of_cols, your_mesh, verbose=True):
        for i in range(no_of_cols):
            if i == no_of_cols - 1:
                # last triangle, you gotta loop it
                top_left = your_mesh.vectors[net[row_no - 2, i]]
                top_right = your_mesh.vectors[net[row_no - 2, 0]]  ## LOOPED!
                bottom_left = your_mesh.vectors[net[row_no - 1, i]]
            else:
                top_left = your_mesh.vectors[net[row_no - 2, i]]
                top_right = your_mesh.vectors[net[row_no - 2, i + 1]]
                bottom_left = your_mesh.vectors[net[row_no - 1, i]]
            chosen_point = np.array(list(
                set([tuple(x) for x in bottom_left]) & set([tuple(x) for x in top_right]) & set(
                    [tuple(x) for x in top_left])))
            neighbours = [x for (x, y) in enumerate(your_mesh.vectors) if np.all(np.isin(chosen_point, y))]
            if verbose:
                print("working off: {}".format(chosen_point))
                print('Between: {},\n {}\n and {}\n'.format(top_left, top_right, bottom_left))
                print("Found neighbours of point are: {} ".format(neighbours))
            next_triangle_index = [x for x in neighbours if x not in net][0]
            if verbose:
                print("Assigned triangle {}".format(next_triangle_index))
            net[row_no, i] = next_triangle_index
            triangle_indices_not_yet_assigned = set(all_triangle_indices) ^ set(net.flatten())
        if verbose:
            print("{} triangles assigned".format(net[row_no, :]))
            print("triangle indices left to assign are:\n{}".format(triangle_indices_not_yet_assigned))
        return net, triangle_indices_not_yet_assigned

    # row_no = 1
    # col_no=0

    def export_for_sphere_CNN(self):
        """

        :return: an array of the values suitible for inputting into SphereCNN
        """
        result = []
        for face in self.faces:
            # TODO: Compact layers 2,3 into one.
            result += face.export_for_sphere_CNN()
        return result


