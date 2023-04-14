
import h5py
import sys

sys.path.append(r"C:\Users\ella_\Documents\GitHub\graphs_and_topology_for_chemistry")
sys.path.append(r"C:\Users\ella_\Documents\GitHub\icosahedron_projection")
import projection.helper_functions as h

## as using mat files
import scipy.io
from rdkit.Chem import PandasTools
import pandas as pd
import os
from rdkit import RDConfig

#############################################################
# settings
#############################################################

save_dir=r'F:\Nextcloud\science\Datasets\icostar_processed'
data_dir=r'F:\Nextcloud\science\Datasets'
### SMILES strings
input_file='qm7.csv'
input_file_location=os.path.join(data_dir, input_file)
df = pd.read_csv(input_file_location) # does SMILES
df.head()
### 3D coords
test_file='qm7.mat'
sdf_file = 'gdb7.sdf'
sdf_file_location = os.path.join(data_dir, sdf_file)

mat_location = os.path.join(data_dir, test_file)
mat = scipy.io.loadmat(mat_location)
## does 3d coords and some very strange looking SMILES
frame = h.Load_SDF_Files(sdf_file_location)

# use the csv file SMILES strings
df['Compound ID'] = frame['ID']
df.head()

SMILES_array=df['smiles']

##############################################################
#       TEST
##############################################################
DIVISION = 4
do_Hdonors = True
ico_key_name = "icosahedron_level4"
num_of_molecules_override = 5
NUM_MAPS_PER_MOLECULE = 65
sanitize = True
SMILES_array = SMILES_array
num_out_files = 1
extra_augmentation = ''
verbose = True
out_filename = 'TEST5.hdf5'

print('Doing a test of 5 molecules before we do the big one')

#####
h.create_diff_conformer_dataset_from_QM7(
        DIVISION,
        df,
        frame,
        save_dir,
        data_dir,
        out_filename,
        do_Hdonors=do_Hdonors,
        ico_key_name=ico_key_name,
        num_of_molecules_override=num_of_molecules_override,
        NUM_MAPS_PER_MOLECULE=NUM_MAPS_PER_MOLECULE,
        sanitize=True,
        SMILES_array=SMILES_array,
        num_out_files=1,
        extra_augmentation='conformer',
        verbose=False)


################# test the file you just made ############################
print('Test file is: {},{}'.format(save_dir, out_filename))
# infile = "PDBBindTEST_div4.hdf5"
# h5py.File(os.path.join(save_dir,out_filename),"w")
hf = h5py.File(os.path.join(save_dir, out_filename), 'r')
n1 = hf['num_exact_Mol_Wt']
print(n1.value)
print(len(n1))
print(hf['Compound ID'])
print('keys for the created hf file')
print(hf.keys())
hf.close()

print('###############################################################')
print('Now the big dataset')
print('###############################################################')

NUM_MAPS_PER_MOLECULE=120
extra_augmentation='conformer'
print('Doing {} maps per molecule using the {} setting'.format(NUM_MAPS_PER_MOLECULE, extra_augmentation))
# ## THIS DOES THE DATASET BUILDING!

out_filename = 'QM7_augmented_expanded.hdf5'

h.create_diff_conformer_dataset_from_QM7(
        DIVISION,
        df,
        frame,
        save_dir,
        data_dir,
        out_filename,
        do_Hdonors=do_Hdonors,
        ico_key_name=ico_key_name,
        num_of_molecules_override=0,
        NUM_MAPS_PER_MOLECULE=NUM_MAPS_PER_MOLECULE,
        sanitize=True,
        SMILES_array=SMILES_array,
        num_out_files=1,
        extra_augmentation=extra_augmentation,
        rotamer_source='df',
        conformer_source='df',
        verbose=False)

hf.close()

# ## Test stuff
# this allows you to check that the hdf5 has been made correctly
print('###############################################################')
print('Testing the big dataset')
print('###############################################################')
################# test the file you just made ############################
print('Test file is: {},{}'.format(save_dir, out_filename))

hf = h5py.File(os.path.join(save_dir, out_filename), 'r')
n1 = hf['num_exact_Mol_Wt']
print('dataset is {} big'.format(len(n1)))
print(hf['Compound ID'])
print('keys for the created hf file')
print(hf.keys())
hf.close()

sys.exit(0)