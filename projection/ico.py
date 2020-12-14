import matplotlib
import functools
import operator
from operator import attrgetter


import matplotlib.pyplot
import mpl_toolkits.mplot3d.art3d

from skspatial.objects import Vector

from projection import topface
from projection.topface import TopFace


class Ico(object):
    """A class to define an icsahedron or icosphere.
    """

    def _add_molecule(self, molecule):
        """ Add a molecule to the ico, marking each face nearest each atom
        """
        for idx, location in enumerate(molecule.coords):
            self._add_atom(Vector(location), idx)

    def __init__(self, molecule, level=0):
        """ Create a new icosphere.
        Level indicates how many subdivisions to make, 0 (default) will give you an icosohedron.
        All icospheres are centred around the origin with a cicumradius of 1
        """
        self.molecule = molecule
        self.faces = [
            # Top row, around point 1
            TopFace(molecule, 0, 1, 8),
            TopFace(molecule, 8, 1, 5),
            TopFace(molecule, 5, 1, 6),
            TopFace(molecule, 6, 1, 9),
            TopFace(molecule, 9, 1, 0),
            # Row 2
            TopFace(molecule, 0, 8, 4),
            TopFace(molecule, 8, 5, 11),
            TopFace(molecule, 5, 6, 2),
            TopFace(molecule, 6, 9, 10),
            TopFace(molecule, 9, 0, 7),
            # Row 3
            TopFace(molecule, 4, 8, 11),
            TopFace(molecule, 11, 5, 2),
            TopFace(molecule, 2, 6, 10),
            TopFace(molecule, 10, 9, 7),
            TopFace(molecule, 7, 0, 4),
            # Bottom row, around point 3
            TopFace(molecule, 4, 11, 3),
            TopFace(molecule, 11, 2, 3),
            TopFace(molecule, 2, 10, 3),
            TopFace(molecule, 10, 7, 3),
            TopFace(molecule, 7, 4, 3)
        ]

        # Now create a dictionary of which faces have which edge and
        # one of which faces have which point
        self.edge_mappings = {}
        self.point_mappings = {}
        for face in self.faces:
            for edge in face.get_edges():
                try:
                    self.edge_mappings[edge].append(face)
                except KeyError:
                    self.edge_mappings[edge] = [face]
            for idx in face.get_point_indices():
                try:
                    self.point_mappings[idx].append(face)
                except KeyError:
                    self.point_mappings[idx] = [face]

        # Use this to tell each face what its neighbours are
        for idx in self.edge_mappings:
            if len(self.edge_mappings[idx]) != 2:
                print("{}:{}".format(idx, self.edge_mappings[idx]))
            assert (len(self.edge_mappings[idx]) == 2)
            l, r = self.edge_mappings[idx]
            l.neighbours.append(r)
            r.neighbours.append(l)

        # now perform the subdivisions
        # conditional isn't needed, but why waste the time?
        if level == 0:
            return

        for face in self.faces:
            face.create_children(level)

        self._add_molecule(self.molecule)

    def get_mesh(self):
        """ return the mesh for this object
        """
        return functools.reduce(operator.add, [f.get_mesh() for f in self.faces])

    def _draw_init(self):
        self._figure = matplotlib.pyplot.figure()
        self._axes = mpl_toolkits.mplot3d.Axes3D(self._figure)
        # self._mesh = i3.get_mesh()
        self._mesh = self.get_mesh()

    def _draw_3d(self, a, b):
        """Draw a single 3d frame
            a,b : angles to rotate the view by
        """
        # Load the stuff
        self._axes.add_collection3d(mpl_toolkits.mplot3d.art3d.Line3DCollection(self._mesh))
        self._axes.scatter(self.molecule.coords[:, 0], self.molecule.coords[:, 1], self.molecule.coords[:, 2],
                           c=self.molecule.colour_list)
        scale = topface.get_scale()
        self._axes.auto_scale_xyz(scale, scale, scale * 1.25)
        self._axes.view_init(a, b)
        # axes.scatter(coords[:,0],coords[:,1],coords[:,2],c=colours)
        matplotlib.pyplot.draw()

    def draw3D(self, a=15, b=30):
        """Quick function to draw a single frame
        """
        self._draw_init()
        self._draw_3d(a, b)

    def plot2D(self, first_face=0, point_idx=0):
        """Unwrap the icosphere, starting with the specified face, with the indicated point on the top
        """
        # Make sure the request makes sense.
        if first_face < 0 or first_face >= 20 or point_idx < 0 or point_idx >= 3:
            raise ValueError("Face needs to be in range 0..20, point 0..2")
        # clear any grid from previous runs
        for f in self.faces:
            f.clear_grid()

        self.faces[first_face].set_grid(point_idx)

        # Now recalculate the grid
        for face in self.faces:
            face.plot2D()

    def draw2D(self, first_face=0, point_idx=0):
        """Unwrap the ico and draw it,
        starting with the specified face, with the demarked point up
        """
        # do the plotting
        self.plot2D(first_face, point_idx)

        # set up the image
        figure = matplotlib.pyplot.figure()
        matplotlib.pyplot.axis('equal')
        # matplotlib.pyplot.axis('off')

        for f in self.faces:
            f.draw2D(figure)
        matplotlib.pyplot.autoscale(enable=True, axis='both')

        matplotlib.pyplot.draw()

    def _add_atom(self, location, idx):
        best_face = None
        best_angle = None
        for face in self.faces:
            angle = location.angle_between(face.normal)
            if best_face is None or angle < best_angle:
                best_face = face
                best_angle = angle
        best_face.add_atom(location, idx)

    def get_face_list(self):
        """ Returns a list of the leaf faces in the Icosphere
        """
        result = []
        partial_list = sorted(self.faces, key=attrgetter('midpoint_x'))
        ordered_faces = sorted(partial_list, key=attrgetter('midpoint_y'), reverse=True)

        for face in ordered_faces:
            result += face.get_ordered_leaf_faces()
        return result
