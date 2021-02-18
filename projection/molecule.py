import os

import pandas as pd
import rdkit.Chem
import rdkit.Chem.AllChem
import numpy as np
import random
from projection.helper import rotation_with_quaternion

class Molecule(object):
    """ Holds a molecule
    random_translate will move the molecule by a random amount max given by this (not yet implemented)
    """

    # Load the standard colours and convert into the right format for matplotlib.
    _colour_frame = pd.read_csv(os.path.join("JMolColors-master", "JMolColors-master", "jmolcolors.csv"))
    _colour_list = {}
    for _, a in _colour_frame.iterrows():
        _colour_list[a['atom']] = (a['R'] / 255.0, a['G'] / 255.0, a['B'] / 255.0)
    # for reasons, override carbon and hydrogen
    _colour_list['C'] = (0.0, 0.0, 0.0)
    _colour_list['H'] = (0.0, 0.5, 0.5)

    @classmethod
    def get_atom_colour(cls, atom):
        """ Returns RGB color tuple corresponding to atom
        :param str atom: symbol for atom
        """
        return cls._colour_list[atom]

    @classmethod
    def get_atom_colour_list(cls, atom_list):
        """ Get a list of the corresponding colours for a list of atoms.
        """
        return [cls.get_atom_colour(a) for a in atom_list]

    @staticmethod
    def _atoms_from_mol_file(mol_file, no_of_atoms):
        """Grabs the atoms from a read in version of a molfile
        """
        chopped_mol = mol_file.split('\n')[4:]
        atom_list = []
        for idx in range(no_of_atoms):
            line = chopped_mol[idx].split(' ')
            data = [x for x in line if not x == '']
            atom_list.append(data[3])
        return atom_list

    def _parse_conformer(self, do_random_rotation=False, rotation_vector=[],verbose=False):
        self.coords = self.conformer.GetPositions()
        if do_random_rotation:
            if rotation_vector==[]:
                # assume you want a random rotation
                rotation_vector=random.choices([x for x in range(360)], k=3)
                self.coords=rotation_with_quaternion(rotation_vector[0],
                                         rotation_vector[1],
                                         rotation_vector[2],
                                         self.coords, verbose=verbose)
            else:
                self.coords=rotation_with_quaternion(rotation_vector[0],
                                         rotation_vector[1],
                                         rotation_vector[2],
                                         self.coords, verbose=verbose)
        # self.random_translate
        offset_size = (max(self.coords.flatten()) / 0.75)/100
        if len(np.where(~self.coords.any(axis=1))[0]) > 0:
            # if an atom is at [0,0,0] we just translate the molecule a little bit in one dimension
            offset_axis=random.randint(0,2)
            offset_direction=random.randint(0,1)
            offset = offset_size if offset_direction else -offset_size
            self.coords[:, offset_axis] += offset
        # now scale them to fit the sphere.
        coords_max = max(self.coords.flatten()) / 0.75
        self.coords /= coords_max
        self.coords_x = self.coords[:, 0]
        self.coords_y = self.coords[:, 1]
        self.coords_z = self.coords[:, 2]




    def get_atom(self, idx):
        """ Get the RDKit Atom object by index
        """
        return self.molecule.GetAtoms()[idx]

    def __init__(self, smiles_string, sanitize=True, random_translate=0.0):
        """ Factory method to build a molecule from a smiles string.
        """
        # Parse the molecule
        base_m = rdkit.Chem.MolFromSmiles(smiles_string, sanitize=sanitize)
        if base_m is None:
            raise ImportError(f"Unable to build mol from smile {smiles_string}")
        base_m.UpdatePropertyCache()
        rdkit.Chem.AllChem.EmbedMolecule(base_m)
        # Add the hydrogens
        molecule = rdkit.Chem.AddHs(base_m)
        rdkit.Chem.AllChem.EmbedMolecule(molecule, useRandomCoords=True)
        # use MMFF94 to minimise and make a nice structure
        try:
            rdkit.Chem.AllChem.MMFFOptimizeMolecule(molecule)
        except ValueError:
            raise ImportError(f"Conformer error from smiles: {smiles_string}")

        # get a conformer, any conformer and parse it.
        self.conformer = molecule.GetConformers()[0]
        self.random_translate = random_translate
        self._parse_conformer()

        self.atom_count = self.conformer.GetNumAtoms()

        # get molefile
        mol_file = rdkit.Chem.MolToMolBlock(molecule)
        self.atom_list = Molecule._atoms_from_mol_file(mol_file, self.atom_count)
        self.colour_list = Molecule.get_atom_colour_list(self.atom_list)

        self.molecule = molecule
        self.smiles = smiles_string

    def draw3D(self, target):
        """ Draw the object in 3d
        """
        target.scatter(self.coords_x, self.coords_y, self.coords_z, self.colour_list)

    def __repr__(self):
        return repr("Molecule: {}".format(self.smiles))
