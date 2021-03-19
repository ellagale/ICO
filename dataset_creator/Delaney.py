#!/usr/bin/env python
# coding: utf-8

# # Creating the Delaney dataset

# Task: calculating the ESOL (solvation energy) predicted or measured from the data. This dataset comes with some features already. I'll add the extras in, we can chose to use them or not later. Is a nice regression task.
#
# Idea to try: each time you get a conformer, the conformer is in a slightly different place. So lets do 60 unwrappings of the first conformer, and do 60 unwrappings of a different conformer each time? Perhaps as two datasets for ease of use? I think as two datasets atm, for ease of not making a mistake with the dataset.
#
# So, one dataset is standard, 60 unfoldings per molecule, the other is 120.
#
# Hopefully, this task requires some sort of shape information

import numpy as np
import pandas as pd

import os

import h5py

import sys

sys.path.append(r"C:\Users\ella_\Documents\GitHub\graphs_and_topology_for_chemistry")
sys.path.append(r"C:\Users\ella_\Documents\GitHub\icosahedron_projection")

import projection.helper_functions as h

#############################################################
# settings
#############################################################
save_dir = r'F:\Nextcloud\science\Datasets\icostar_processed'
data_dir = r'F:\Nextcloud\science\Datasets'
input_file = 'delaney-processed.csv'
input_file_location = os.path.join(data_dir, input_file)
df = pd.read_csv(input_file_location)
print('Example data')
print(df.head())

SMILES_array = df["smiles"]

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
out_filename = 'TEST2'

print('Doing a test of 5 molecules before we do the big one')

h.Create_Diff_Conformer_Dataset_From_SMILES(DIVISION=4,
                                          df=df,
                                          save_dir=save_dir,
                                          data_dir=data_dir,
                                          out_filename=out_filename,
                                          do_Hdonors=True,
                                          ico_key_name="icosahedron_level4",
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

out_filename = 'Delaney_augmented_expanded.hdf5'
h.Create_Diff_Conformer_Dataset_From_SMILES(DIVISION=4,
                                          df=df,
                                          save_dir=save_dir,
                                          data_dir=data_dir,
                                          out_filename=out_filename,
                                          do_Hdonors=True,
                                          ico_key_name="icosahedron_level4",
                                          num_of_molecules_override=0,
                                          NUM_MAPS_PER_MOLECULE=NUM_MAPS_PER_MOLECULE,
                                          sanitize=True,
                                          SMILES_array=SMILES_array,
                                          num_out_files=1,
                                          extra_augmentation=extra_augmentation,
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


