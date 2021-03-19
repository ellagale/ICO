#!/usr/bin/env python
# coding: utf-8

# # Normalising a dataset
#
# Code to take in a previously created hdf5 dataset and normalise it.
#
# Does L2 normalisation on the input total masses per face.


import numpy as np
import h5py
import os
import sys
from collections import Counter
import matplotlib.pyplot as plt

# grab mi stuff
sys.path.append(r"C:\Users\ella_\Documents\GitHub\graphs_and_topology_for_chemistry")
sys.path.append(r"C:\Users\ella_\Documents\GitHub\icosahedron_projection")

from projection import helper_functions as h

###########################################################
# These should be the only two bits you need to change
###########################################################

save_dir=r'F:\Nextcloud\science\Datasets\icostar_processed'
data_dir=r'F:\Nextcloud\science\Datasets\icostar_processed'

#train_filename = '2021_PDBBind_core_augmented_div4.hdf5'
train_filename = "TEST2.hdf5" #'Delaney_augmented_expanded.hdf5'#
outfile=train_filename
test_filename = ''
do_test = False
field = "icosahedron_level4"
###########################################################
###########################################################
# ### Now lets do the normalisation.

# First compute the maximum value in each element of the row.
#
# _note:_ for finding max, we initialise the values to 0.1 to avoid divide by zero errors. This is fine as no isotope has a mass that low, so if max remains at 0.1 that means there is _never_ an atom in that spot, and so it will always end up mapped to -1 ((2*0/0.1)-1)
# We have to do this in three passes as we need to use data computed in one pass in the next pass.

# **Pass 1:**
# Compute the elementwise mean and max of the data set, storing them in the arrays *mean_val* and *upper* respectively

# **Pass 2:**
# Use the mean to calculate the elementwise **standard deviation** and the **mean normalisation** of the data, storing them in **std_val** and **mean_normalisation** respectively.

# **Pass 3**:
# Use the standard deviation to calculate the elementwise standardisation, storing it in *standardisation*

data_dir=data_dir
print(outfile)
hf = h5py.File(os.path.join(data_dir,outfile), 'r')

hf.keys()

num_of_rows, num_of_molecules, num_of_augments=h.basic_info_hdf5_dataset(
    hf,
    label='molID')

hf.close()

# ## Normalise the training dataset

# #### Open the dataset and create empty columns
print('###############################################################')
print('Normalising dataset')
print('###############################################################')

# name of the new fields that we will add
norm_L2_field = f"{field}_L2_normalised"
norm_mean_field = f"{field}_mean_normalised"
norm_std_field = f"{field}_std_normalised"


# open dataset, create file handle and empty datasets
fh, data, data_L2_out, data_mean_out, data_std_out = h.Open_Train_File_Create_Normalised_Datasets(
                                    data_dir,
                                    outfile,
                                    field,
                                    norm_L2_field,
                                    norm_mean_field,
                                    norm_std_field,
                                    label='molID')

# ## Calculating the icosahedral nets averages etc from the training data

# ### This chunk calculates the invarients (upper, mean_val and std_val) and writes out the normalised datasets to the hdf5 file :)

upper, mean_val, std_val = h.parse_and_normalise_da_data(fh,
                                data,
                                num_of_rows,
                                data_L2_out,
                                data_mean_out,
                                data_std_out,
                                batch_size=10000)


# ##### Look at some example data from the invarients

print('Example invariant data')
print(data[0])
print(upper[0])
print(upper[10:12])
print(mean_val[10:12])
print(std_val[10:12])

print(fh[field].shape)
print(fh[field][0].shape)
moo=fh[field][:,0]
print(moo.shape)

print('Example original')

print(fh['icosahedron_level4'][0][130:170])


print('Example mean normalised')

print(fh['icosahedron_level4_mean_normalised'][0][130:170])


# ## CLOSE FILE!

fh.close()

if do_test:
    print('################################################################')
    print('doing test file, normalising with invariants from training data')
    print('################################################################')
    # # Normalising the test dataset!
    # This is normalised using the icosahedrons upper and lower from the training dataset
    #
    # Not sure that this is finished yet!

    # #### Open the test hdf5 file

    # Open hdf5 file, calc basic details
    outfile = test_filename
    print(outfile)
    fh = h5py.File(os.path.join(data_dir,outfile), 'r+')
    num_of_rows, num_of_molecules, num_of_augments=h.basic_info_hdf5_dataset(fh, label='molID')
    fh.close()

    # #### make some sensible column names

    # name of the new fields that we will add
    norm_L2_field = f"{field}_L2_normalised"
    norm_mean_field = f"{field}_mean_normalised"
    norm_std_field = f"{field}_std_normalised"

    # #### load up the data and create some empty datasets in the test hdf5 file

    # open dataset, create file handle and empty datasets
    # CREATES DATA
    # open dataset, create file handle and empty datasets
    # data_L2_out, data_mean_out, data_std_out are pointers to the dataset in the hdf5 file
    fh, data, data_L2_out, data_mean_out, data_std_out=h.Open_Train_File_Create_Normalised_Datasets(
                                        data_dir,
                                        outfile,
                                        field,
                                        norm_L2_field,
                                        norm_mean_field,
                                        norm_std_field,
                                        label='molID')

    # #### Using the invarients from the training dataset, DO THE NORMALISATION!

    h.parse_and_normalise_da_test_data(
        data,
        upper,
        mean_val,
        std_val,
        data_L2_out,
        data_mean_out,
        data_std_out,
        num_of_rows,
        batch_size=10000)


    record1=fh[norm_L2_field][0]
    record1=record1.flatten()
    print(f'max: {max(record1)}, mean: {np.mean(record1)}, min: {min(record1)}')
    print(f'{Counter(record1)}')
    counted_record1=Counter(record1)

    fh.close()

sys.exit(0)


