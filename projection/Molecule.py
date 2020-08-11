from rdkit.Chem import AllChem
from rdkit import Chem
import pandas as pd

class Molecule(object):

    _df_colors = pd.read_csv("C:\\Users\\ella_\\Documents\\GitHub\\icosahedron_projection\\JMolColors-master\\JMolColors-master\\jmolcolors.csv")

    def __init__(self, rdkit_molecule, coords, atom_list, colour_list):
        self.rdkit_molecule = rdkit_molecule
        self.coords = coords
        self.atom_list = atom_list
        self.colour_list = colour_list

    @classmethod
    def _get_colour(cls, atom):
        """
        Returns RGB color tuple corresponding to atom

        :param str atom: symbol for atom
        """
        r = cls._df_colors[cls._df_colors['atom'] == atom]['R'].values[0] / 255.0
        g = cls._df_colors[cls._df_colors['atom'] == atom]['G'].values[0] / 255.0
        b = cls._df_colors[cls._df_colors['atom'] == atom]['B'].values[0] / 255.0
        return r, g, b

    @classmethod
    def _get_colour_list(cls, atom_list):
        colour_list = []
        for atom in atom_list:
            if atom == 'H':
                colour = (0, 0.5, 0.5)
            else:
                if atom == 'C':
                    colour = (0, 0, 0)
                else:
                    colour = cls._get_colour(atom)
            colour_list.append(colour)
        return colour_list

    @staticmethod
    def _atoms_from_mol_file(mol_file, no_of_atoms, verbose=True):
        """Grabs the atoms from a read in version of a molfile
        currently this doesn't return a Molecule object
        rewrite a version of this to return Molecules form external molfiles"""
        chopped_mol = mol_file.split('\n')[4:]
        atom_list = []
        for idx in range(no_of_atoms):
            if verbose:
                print(chopped_mol[idx])
            line = chopped_mol[idx].split(' ')
            data = [x for x in line if not x == '']
            if verbose:
                print(data[3])
            atom_list.append(data[3])
        return atom_list

    @classmethod
    def from_smiles(cls, smiles_string, verbose=False):
        """Uses RDKit to add Hydrogens, get 3D coords, do a simple minimisation
        and outputs coords and atom types.
        Repeated use yields slightly different coords"""
        # example - molecules is randomly rotated each time this is called
        # grab a smiles string
        m2 = Chem.MolFromSmiles(smiles_string)
        # embed it what ever that is
        AllChem.EmbedMolecule(m2)
        # add some hydrogens
        m3 = Chem.AddHs(m2)
        # embed it again? I think this makes a molecule object but who knows
        AllChem.EmbedMolecule(m3)
        # use MMFF94 to minimise and make a nice structure
        AllChem.MMFFOptimizeMolecule(m3)
        # get molefile
        mol_file = Chem.MolToMolBlock(m3)
        if verbose:
            print(mol_file)
            # AllChem.Compute2DCoords(m2)
        # coords=[]
        for c in m3.GetConformers():
            if verbose:
                print(c.GetPositions())
        coords = c.GetPositions()
        no_of_atoms = c.GetNumAtoms()
        atom_list = cls._atoms_from_mol_file(mol_file, no_of_atoms, verbose=verbose)
        colour_list = cls._get_colour_list(atom_list)

        return Molecule(m3, coords, atom_list, colour_list)
