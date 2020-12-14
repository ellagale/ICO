import functools
import operator

import matplotlib
import matplotlib.pyplot
import matplotlib.tri

import numpy as np
import scipy
import scipy.linalg

from skspatial.objects import Vector, Triangle

from projection.molecule import Molecule


class Face(object):
    """A class to define a face on an icosphere
    a,b,c are the points in 3D space of points on an icosphere
    u,v are the coordinates on a 2D plane corresponding to the unfolded icosphere net
    children are the faces at the next layer of magnification (anti coarse graining of a sphere)
    atoms are the atoms that project onto that face"""

    @staticmethod
    def face_get_colour(face):
        """ hook function to look up the individual colour of a face.
        Mostly implemented as a demonstration of how to do the plugin approach to calculation.
        """
        if not face.atoms:
            # Nothing to see here.
            return 1.0, 1.0, 1.0

        if len(face.atoms) == 1:
            # Only one choice
            return Molecule.get_atom_colour(face.molecule.get_atom(face.atoms[0][1]).GetSymbol())

        atom_list = [face.molecule.get_atom(x).GetSymbol() for _, x in face.atoms]
        atom1 = [a for a in atom_list if a is not "H"]
        atom2 = [a for a in atom1 if a is not "C"]

        if atom2:
            # Something more interesting than Carbon / Hydrogen. Pick the first
            return Molecule.get_atom_colour(atom2[0])
        if atom1:
            # Carbon at least
            return Molecule.get_atom_colour("C")
        # must have been multiple hydrogens
        return Molecule.get_atom_colour("H")

    @staticmethod
    def face_get_masses(face):
        """ Returns the masses of the atoms associated with this face as a 3-tuple:
            1) mass of the atom closest to the centre of the molecule
            2) mass of the atom furthest to the centre of the molecule
            3) total mass of all atoms associated with this face
        """
        min_distance = max_distance = None
        min_mass = max_mass = total_mass = 0.0
        # TODO: this is clunky. FIX IT
        for distance, idx in face.atoms:
            atom_mass = face.molecule.get_atom(idx).GetMass()
            if not min_distance or distance < min_distance:
                min_distance = distance
                min_mass = atom_mass
            if not max_distance or distance > max_distance:
                max_distance = distance
                max_mass = atom_mass
            total_mass += atom_mass

        return min_mass, max_mass, total_mass

    # The function to use when determining the values of a face.
    _lookup_func = face_get_colour

    def __init__(self, molecule, a=None, b=None, c=None):
        self.molecule = molecule  # the molecule we are drawing from.
        self.a = Vector(a)  # these are vectors from 0,0,0 to x y z etc
        self.b = Vector(b)
        self.c = Vector(c)
        self.triangle = Triangle(a, b, c)
        self.normal = self.triangle.normal()
        self.u = None
        self.v = None
        self.w = None
        self.midpoint = None
        self.children = []
        self.atoms = []  # list of (distance from centre, atom_index) tuples

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
        # AND PROJECTS IT TO THE UNIT SPHERE EASY!
        ab = ((self.a + self.b) / 2).unit()
        bc = ((self.b + self.c) / 2).unit()
        ca = ((self.c + self.a) / 2).unit()
        self.children = [  # the four new triangles
            Face(self.molecule, self.a, ab, ca),  # 1 bottom left
            Face(self.molecule, ab, self.b, bc),  # 2 top
            Face(self.molecule, ca, bc, self.c),  # 3 bottom right
            Face(self.molecule, ab, bc, ca)]  # 4 middle
        for child in self.children:
            # this should work...
            child.create_children(levels_to_do - 1)

    def get_mesh(self):
        """ Return the mesh for this face as an array.
        """
        if self.children:
            return functools.reduce(operator.add, [c.get_mesh() for c in self.children])

        # Leaf node
        return [[self.a, self.b, self.c]]

    def get_values(self):
        """ Use the class lookup function to determine the values associated with this face.
        e.g. the colours to render with.
        """
        return self._lookup_func()

    def _get_colour(self):
        """ hard coded version of _get_values that always uses face_get_colour to determine values.
        """
        return self.face_get_colour(self)

    def _map_coords(self, u, v, w):
        """ Inner function for working out the file locations
            uvw need to be specified in clockwise order.
        """
        ux, uy = u
        vx, vy = v
        wx, wy = w
        self.u = u
        self.v = v
        self.w = w

        self.midpoint_x = (ux + vx + wx) / 3
        # only going to approximate Y
        self.midpoint_y = (min(uy, vy, wy) + max(uy, vy, wy)) / 2

        self.midpoint = self.midpoint_x, self.midpoint_y

        if self.children:
            uv = ((ux + vx) / 2, (uy + vy) / 2)
            vw = ((vx + wx) / 2, (vy + wy) / 2)
            wu = ((wx + ux) / 2, (wy + uy) / 2)
            self.children[0]._map_coords(u, uv, wu)
            self.children[1]._map_coords(uv, v, vw)
            self.children[2]._map_coords(wu, vw, w)
            self.children[3]._map_coords(uv, vw, wu)

    def _draw(self, plot):
        """ Inner function for the drawing. Plots this face between the points shown.            
        """
        if not self.midpoint:
            raise ValueError("_draw called before _map_coords")
        if not self.children:
            xs = np.array([x for x, _ in [self.u, self.v, self.w]])
            ys = np.array([y for _, y in [self.u, self.v, self.w]])
            tris = np.array([[0, 1, 2]])
            matplotlib.tri.Triangulation(xs, ys, tris)

            x = matplotlib.pyplot.Polygon(np.array([self.u, self.v, self.w]),
                                          ec='k',
                                          closed=True,
                                          color=self._get_colour())
            plot.gca().add_patch(x)

            return
        for child in self.children:
            child._draw(plot)

    def add_atom(self, location, idx):
        """ Add a specific atom to the face
        """
        self.atoms.append((scipy.linalg.norm(location), idx))
        if not self.children:
            return
        best_child = None
        best_angle = None
        for child in self.children:
            angle = location.angle_between(child.normal)
            if best_child is None or angle < best_angle:
                best_child = child
                best_angle = angle
        best_child.add_atom(location, idx)

    def get_leaf_faces(self):
        """ Returns a list of the lowest level faces
        """
        if self.children:
            if self.children[0].children:
                # recursive case
                result = []
                for child in self.children:
                    result += child.get_leaf_faces()
                return result
            else:
                # this is the parent of a set of leaves
                return self.children
        else:
            raise ValueError("get_leaf_faces reached actual leaf face. Shouldn't happen.")

    def __repr__(self):
        """ Print the Face."""
        # output = "Face: {}".format(self.triangle)
        output = "Face"

        if self.midpoint:
            output += " at {}".format(self.midpoint)

        # if self.u:
        #    output += " ({},{},{})".format(self.u, self.v, self.w)

        if self.atoms:
            output += " with {}".format(self.atoms)
        return repr(output)
