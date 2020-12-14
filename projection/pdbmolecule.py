import rdkit.Chem

from projection.molecule import Molecule


class PDBMolecule(Molecule):
    """ Holds a molecule
    """

    def __init__(self, file_location, sanitize=True):
        """ Factory method to build a molecule from a smiles string.
        file_location is the location of hte file to read in on disk
    """
        molecule = rdkit.Chem.rdmolfiles.MolFromPDBFile(file_location, sanitize=sanitize)
        # get a conformer, any conformer and parse it.
        self.conformer = molecule.GetConformer()
        rdkit.Chem.rdMolTransforms.CanonicalizeConformer(self.conformer)
        self._parse_conformer()

        self.atom_count = self.conformer.GetNumAtoms()
        self.atom_list = [x.GetSymbol() for x in molecule.GetAtoms()]

        self.colour_list = Molecule.get_atom_colour_list(self.atom_list)

        self.molecule = molecule
        self.smiles = ''
