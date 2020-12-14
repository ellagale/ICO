import rdkit.Chem

from projection.molecule import Molecule


class Mol2Molecule(Molecule):
    """ Holds a molecule
    """

    def __init__(self, file_location, sanitize=False):
        """ Factory method to build a molecule from a smiles string.
        file_location is the location of hte file to read in on disk
    """
        # Parse the molecule
        # base_m = rdkit.Chem.MolFromSmiles(smiles_string)
        # rdkit.Chem.AllChem.EmbedMolecule(base_m)
        # Add the hydrogens
        # molecule = rdkit.Chem.AddHs(base_m)
        # rdkit.Chem.AllChem.EmbedMolecule(molecule)
        # use MMFF94 to minimise and make a nice structure
        # rdkit.Chem.AllChem.MMFFOptimizeMolecule(molecule)
        molecule = rdkit.Chem.rdmolfiles.MolFromMol2File(
            file_location,
            cleanupSubstructures=False,
            sanitize=sanitize)
        # get a conformer, any conformer and parse it.
        self.conformer = molecule.GetConformer()
        rdkit.Chem.rdMolTransforms.CanonicalizeConformer(self.conformer)
        self._parse_conformer()

        atom_count = self.conformer.GetNumAtoms()

        # get molefile
        mol_file = rdkit.Chem.MolToMolBlock(molecule, kekulize=False)
        self.atom_list = Molecule._atoms_from_mol_file(mol_file, atom_count)
        self.colour_list = Molecule.get_atom_colour_list(self.atom_list)

        self.molecule = molecule
        self.smiles = ''
