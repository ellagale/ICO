import h5py
import numpy as np
import os
from collections import Counter

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
import matplotlib
import matplotlib.pyplot
import matplotlib.tri
import rdkit.Chem
import rdkit.Chem.AllChem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
import mpl_toolkits.mplot3d
import operator
import functools
import os
import gif
import h5py
import scipy.linalg
import slugify
import sys
import random

from skspatial.objects import Point, Vector, Triangle
from operator import attrgetter
from scipy.constants import golden
from IPython.display import HTML

import projection
from projection.molecule import Molecule
import projection.sdf_molecule
from projection.face import Face

from projection.ico import Ico

def basic_info_hdf5_dataset(hf, label='molID'):
    """Calcs some basic data
    hf is the file handle to hdf5 file
    label is the unique ID label per class/molecule in hte dataset"""
    molID_List_orig = hf['molID']
    num_of_rows = len(molID_List_orig)
    print(f'num_of_ rows is:\t{num_of_rows}')
    counted_molID_List=Counter(molID_List_orig)
    molID_List=[x for x in counted_molID_List.keys()]
    #print(molID_List)
    num_of_molecules=len(molID_List)
    print(f'num_of_molecules is:\t {num_of_molecules}')
    egg=Counter(Counter(molID_List_orig).values())
    if not len(egg.keys()) == 1:
        print('Warning: Unbalanced dataset\nMolID: count')
        print(counted_molID_List)
    else:
        num_of_augments=[x for x in egg.keys()][0]
        print(f'num_of_augments is:\t{num_of_augments}')
    return num_of_rows, num_of_molecules, num_of_augments

def Load_SDF_Files(sdf_file):
    """Loads SDF files using pandas and some sensible settings"""
    PandasTools.ChangeMoleculeRendering(renderer='String')
    frame = PandasTools.LoadSDF(
        sdf_file,
        smilesName='SMILES',
        molColName='Molecule',
        includeFingerprints=True,
        removeHs=False,
        strictParsing=True,
        embedProps=True )
    return frame

def normalize(x, axis=-1, order=2):
  """Normalizes a Numpy array. L2 normalisation

  Arguments:
      x: Numpy array to normalize.
      axis: axis along which to normalize.
      order: Normalization order (e.g. 2 for L2 norm).

  Returns:
      A normalized copy of the array.
  """
  l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
  l2[l2 == 0] = 1
  return x / np.expand_dims(l2, axis)

def create_or_recreate_dataset(fh, datasetname, shape, dtype):
    """
    Create a dataset with the given parameters in the file pointed to by fh
    if the dataset already exists then it will be deleted first.
    """
    try:
        del fh[datasetname]
    except KeyError:
        pass
    return fh.create_dataset(datasetname, shape, dtype)


def Open_Train_File_Create_Normalised_Datasets(data_dir,
                                               outfile,
                                               field,
                                               norm_L2_field,
                                               norm_mean_field,
                                               norm_std_field,
                                               label='molID'):
    """Function to open the dataset, make new datasets to populate later,
    grab the data to be normalised and return the file handle
    N.B. does not shut the file
    data_dir is the directory the hdf5 file to be normalised is in
    outfile is the name of the normalised hdf5 file
    field is the icosahedral field to normalise
    norm_L"_field
    norm_mean_field
    norm_std_field as the new feild names for the normalised data"""
    # Open, calc basic details
    print(outfile)
    fh = h5py.File(os.path.join(data_dir, outfile), 'r+')
    num_of_rows, num_of_molecules, num_of_augments = basic_info_hdf5_dataset(fh, label=label)

    # grabs the original data
    data = fh[field]
    row_shape = data.shape[1:]
    # makes a new dataset of the same size
    data_L2_out = create_or_recreate_dataset(fh, norm_L2_field, data.shape, data.dtype)
    data_mean_out = create_or_recreate_dataset(fh, norm_mean_field, data.shape, data.dtype)
    data_std_out = create_or_recreate_dataset(fh, norm_std_field, data.shape, data.dtype)

    row_count = data.shape[0]
    if not num_of_rows == row_count:
        print('Error: num_of_rows from molID is not the same as the number found for {field}')
    print(f'row_shape is {row_shape}')
    return fh, data, data_L2_out, data_mean_out, data_std_out

def pass_1_calc_upper_and_mean(data, num_of_rows, batch_size=10000):
    """loops over the dataset looking for the max seen at each point
    calculates the upper value (max) and the mean stores em somewhere
    returns upper and mean_val which are icosahedral nets of the
    upper and mean values
    """
    lower=0.1
    remaining = num_of_rows
    batch_size = batch_size
    row_shape = data.shape[1:]
    upper = np.full(row_shape, lower, data.dtype) # initialise
    mean_val = np.full(row_shape, 0.0, data.dtype)
    ptr = 0
    while remaining > 0:
        cur_batch_size = batch_size if remaining > batch_size else remaining
        end_ptr = ptr+cur_batch_size
        print(f"going from {ptr} to {end_ptr}")
        values = data[ptr:end_ptr][:]
        candidate = np.amax(values, axis=0)
        batch_mean = np.sum(values, axis=0) / num_of_rows
        ptr = end_ptr
        upper = np.maximum(upper, candidate) # gets the max seen
        mean_val += batch_mean
        remaining -= cur_batch_size
    return upper, mean_val

def pass_2_calc_mean_norm_l2_and_std(mean_val,
                                     upper,
                                     data,
                                     data_mean_out,
                                     data_L2_out,
                                     num_of_rows,
                                     batch_size=10000):
    """loops over the dataset doing the
    standard deviation, L2 and mean normalisations
    this zero centres and whitens your data
    returns mean_normalisation std_val (icosahedral nets)"""
    remaining = num_of_rows
    row_shape = data.shape[1:]
    std_val = np.full(row_shape, 0.0, data.dtype)

    ptr = 0
    while remaining > 0:
        cur_batch_size = batch_size if remaining > batch_size else remaining
        end_ptr = ptr+cur_batch_size
        print(f"going from {ptr} to {end_ptr}")
        value = data[ptr:end_ptr][:]
        data_mean_out[ptr:end_ptr] = (value - mean_val)
        data_L2_out[ptr:end_ptr] = ((2*value) / upper) -1
        batch_std = np.sum(np.square((value - mean_val)), axis=0) / num_of_rows
        ptr = end_ptr
        std_val += batch_std
        remaining -= cur_batch_size
    std_val = np.sqrt(std_val)
    return std_val

def pass_2_write_mean_norm_l2_test(mean_val,
                                     upper,
                                     data,
                                     data_mean_out,
                                     data_L2_out,
                                     num_of_rows,
                                     batch_size=10000):
    """loops over the dataset doing the
    standard deviation, L2 and mean normalisations
    this zero centres and whitens your data
    returns mean_normalisation std_val (icosahedral nets)"""
    remaining = num_of_rows
    row_shape = data.shape[1:]
    #std_val = np.full(row_shape, 0.0, data.dtype)

    ptr = 0
    while remaining > 0:
        cur_batch_size = batch_size if remaining > batch_size else remaining
        end_ptr = ptr+cur_batch_size
        print(f"going from {ptr} to {end_ptr}")
        value = data[ptr:end_ptr][:]
        data_mean_out[ptr:end_ptr] = (value - mean_val)
        data_L2_out[ptr:end_ptr] = ((2*value) / upper) -1
        #batch_std = np.sum(np.square((value - mean_val)), axis=0) / num_of_rows
        ptr = end_ptr
        #std_val += batch_std
        remaining -= cur_batch_size
    #std_val = np.sqrt(std_val)
    return

def pass_3_calc_mean_norm_and_std(std_val,
                                  data,
                                  data_std_out,
                                  data_mean_out,
                                  num_of_rows,
                                  batch_size=10000):
    """Calculates teh standardisation net for the data"""
    remaining = num_of_rows
    ptr = 0
    while remaining > 0:
        cur_batch_size = batch_size if remaining > batch_size else remaining
        end_ptr = ptr+cur_batch_size
        print(f"going from {ptr} to {end_ptr}")
        value = data_mean_out[ptr:end_ptr][:]
        data_std_out[ptr:end_ptr] = value / std_val
        ptr = end_ptr
        remaining -= cur_batch_size

def do_the_norm_and_write_it_out(fh,
                                 data,
                                 data_L2_out,
                                 upper,
                                 mean_val,
                                 std_val):
    """This makes the new data
    fh: file handle of hdf5 file
    data: da data to be normalised
    data_L2_out: dataset handle for l2 normed data
    data_mean_out: dataset handle for mean normalised data
    data_std_out: dataset handle for standardised data
    upper: max net
    mean_val: average net
    std_val: standard deviation net
    """
    invarient_data_upper = create_or_recreate_dataset(fh, "invarient/max", data.shape[1:], upper.dtype)
    invarient_data_mean = create_or_recreate_dataset(fh, "invarient/mean", data.shape[1:], mean_val.dtype)
    invarient_data_std = create_or_recreate_dataset(fh, "invarient/std", data.shape[1:], std_val.dtype)
    invarient_data_upper = upper
    invarient_data_mean = mean_val
    invarient_data_std = std_val
    return


def parse_and_normalise_da_data(fh,
                                data,
                                num_of_rows,
                                data_L2_out,
                                data_mean_out,
                                data_std_out,
                                batch_size=10000):
    """Function loops over the input data and calculates the normalisation nets
    to be used to normalise the data
    this takes 3 passes through the data!:
    1. upper: max per triangle in the icosahedron data
    2. mean_val: mean per triangle
    3. mean_normalisation: data normalised by mean
    4. std_val: standard deviation per triganle
    5. standardisation: standardiation net
    mode=train - will calculate upper and mean"""
    print('Pass 1 through the data, calculating upper and mean_val')
    # This calcs the upper and mean_val from the data - only needed on train
    ## pass 1
    upper, mean_val = pass_1_calc_upper_and_mean(data,
                                                 num_of_rows,
                                                 batch_size)
    print('Pass 2 through the data, calculating std_val and calcs and writes out the mean and L2 normalised data')
    ## pass 2
    std_val = pass_2_calc_mean_norm_l2_and_std(
        mean_val,
        upper,
        data,
        data_mean_out,  # writes out to hdf5
        data_L2_out,  # writes out to hdf5
        num_of_rows,
        batch_size)
    print('Pass 3 through the data, writes out the standardisation data')
    # pass 3
    # this calculates the standardisation net
    pass_3_calc_mean_norm_and_std(std_val,
                                  data,
                                  data_std_out,
                                  data_mean_out,
                                  num_of_rows,
                                  batch_size)

    # Finally write the other values out
    print('Writing out upper, mean_val and std_val invarient nets')
    do_the_norm_and_write_it_out(fh,
                                 data,
                                 data_L2_out,
                                 upper,
                                 mean_val,
                                 std_val)
    return upper, mean_val, std_val


def parse_and_normalise_da_test_data(data,
                                     upper,
                                     mean_val,
                                     std_val,
                                     data_L2_out,
                                     data_mean_out,
                                     data_std_out,
                                     num_of_rows,
                                     batch_size=10000):
    """Function loops over the input data and calculates the normalisation nets
    to be used to normalise the data
    this takes 3 passes through the data!:
    1. upper: max per triangle in the icosahedron data
    2. mean_val: mean per triangle
    3. mean_normalisation: data normalised by mean
    4. std_val: standard deviation per triganle
    5. standardisation: standardiation net
    mode=train - will calculate upper and mean"""
    print('Pass 2 through the data, writing mean_normalisation and l2 norm')
    ## pass 2
    pass_2_write_mean_norm_l2_test(
        mean_val,
        upper,
        data,
        data_mean_out,
        data_L2_out,
        num_of_rows,
        batch_size)
    print('Pass 3 through the data, calculating and writing out standardisation')
    # pass 3
    # this calculates the standardisation net
    pass_3_calc_mean_norm_and_std(std_val,
                                  data,
                                  data_std_out,
                                  data_mean_out,
                                  num_of_rows,
                                  batch_size)

    return

def Create_Diff_Conformer_Dataset_From_SMILES(DIVISION,
                                              df,
                                              save_dir,
                                              data_dir,
                                              out_filename,
                                              do_Hdonors,
                                              ico_key_name,
                                              num_of_molecules_override=0,
                                              NUM_MAPS_PER_MOLECULE=60,
                                              sanitize=True,
                                              SMILES_array=[],
                                              num_out_files=1,
                                              extra_augmentation='conformer',
                                              verbose=False):
    """ Creates the dataset as a hdf5 file
    DIVISION =4 # for the size of icospehre
    save_dir=r'C:\\Users\\ella_\\Nextcloud\\science\\Datasets\\converted_pdbbind\\v2015'
    data_dir=r'C:\\Users\\ella_\\Nextcloud\\science\\Datasets\\pdbbind\\v2015'
    out_filename = "PDBBindLigands_core_augmented_withHs_withHdonors_div4.hdf5"
    do_Hdonors = True/False, whether to do the H donor data or not, don't do it for proteins
    ico_key_name_name = "icosahedron_level4"
    num_of_molecules_override: 0, how many molecules to do, a setting of 0 does all of them
    NUM_MAPS_PER_MOLECULE # how many maps to create, anything over 60 will be augmented
    sanitize=True
    SMILES_array= an array of smiles, set this up from df or fix this
    num_out_files=1 not yet implemented
    extra_augmentation='conformer' other options not yet implemented!
    verbose=False
    If you want to add extra stuff, you'll have to write a new function
    of add it in afterwards
    """

    if num_of_molecules_override == 0:
        # do all smiles woo
        Num_of_molecules = len(SMILES_array)
    else:
        Num_of_molecules = num_of_molecules_override

    smiles_count = Num_of_molecules
    smile_count = Num_of_molecules  # yeah i know, is hacky

    if DIVISION == 1:
        NUM_FACES = 80
    elif DIVISION == 2:
        NUM_FACES = 320
    elif DIVISION == 3:
        NUM_FACES = 1280
    elif DIVISION == 4:
        NUM_FACES = 5120

    if NUM_MAPS_PER_MOLECULE == 1:
        NUM_UNWRAPPINGS_TO_DO = 1
        NUM_DIRECTIONS_TO_DO = 1
        NUM_EXTRA_MAPS_PER_MOLECULE = 0
    elif NUM_MAPS_PER_MOLECULE == 60:
        NUM_UNWRAPPINGS_TO_DO = 20
        NUM_DIRECTIONS_TO_DO = 3
        NUM_EXTRA_MAPS_PER_MOLECULE = 0
    elif NUM_MAPS_PER_MOLECULE > 60:
        # Currently assumes you want 60 plus extra
        NUM_UNWRAPPINGS_TO_DO = 20
        NUM_DIRECTIONS_TO_DO = 3
        NUM_EXTRA_MAPS_PER_MOLECULE = NUM_MAPS_PER_MOLECULE - 60
        if verbose:
            print('Doing {} extra maps using {}s'.format(
                NUM_EXTRA_MAPS_PER_MOLECULE,
                extra_augmentation))

    if sanitize:
        print('Warning! Sanitize seems to muck up the structures, especially for protiens')
        print('Is good for SMILES string input however')

    outfile = h5py.File(os.path.join(save_dir, out_filename), "w")
    if NUM_EXTRA_MAPS_PER_MOLECULE > 0:
        outfile_expanded = h5py.File(os.path.join(save_dir, out_filename + '_extra_' + extra_augmentation), "w")
    string_type = h5py.string_dtype(encoding='utf-8')
    icosahedron_name = ico_key_name

    ##################### set up the out put datasets ################################
    ## this sets up the output datasets
    icosahedron_ds = outfile.create_dataset(icosahedron_name, (smile_count * NUM_MAPS_PER_MOLECULE, NUM_FACES, 3))
    # charge_ds = outfile.create_dataset("charge", (smile_count*NUM_MAPS_PER_MOLECULE,))
    molID_ds = outfile.create_dataset("molID", (smile_count * NUM_MAPS_PER_MOLECULE,))
    ## from the dataset
    compound_id_ds = outfile.create_dataset('Compound ID', (smile_count * NUM_MAPS_PER_MOLECULE,), dtype=string_type)
    esol_predicted_log_solubility_in_mols_per_litre_ds = outfile.create_dataset(
        'ESOL predicted log solubility in mols per litre', (smile_count * NUM_MAPS_PER_MOLECULE,))
    minimum_degree_ds = outfile.create_dataset('Minimum Degree', (smile_count * NUM_MAPS_PER_MOLECULE,))
    molecular_weight_ds = outfile.create_dataset('Molecular Weight', (smile_count * NUM_MAPS_PER_MOLECULE,))
    number_of_h_bond_donors_ds = outfile.create_dataset('Number of H-Bond Donors',
                                                        (smile_count * NUM_MAPS_PER_MOLECULE,))
    number_of_rings_ds = outfile.create_dataset('Number of Rings', (smile_count * NUM_MAPS_PER_MOLECULE,))
    number_of_rotatable_bonds_ds = outfile.create_dataset('Number of Rotatable Bonds',
                                                          (smile_count * NUM_MAPS_PER_MOLECULE,))
    polar_surface_area_ds = outfile.create_dataset('Polar Surface Area', (smile_count * NUM_MAPS_PER_MOLECULE,))
    measured_log_solubility_in_mols_per_litre_ds = outfile.create_dataset('measured log solubility in mols per litre',
                                                                          (smile_count * NUM_MAPS_PER_MOLECULE,))
    smiles_ds = outfile.create_dataset('smiles', (smile_count * NUM_MAPS_PER_MOLECULE,), dtype=string_type)
    ### end from the dataset
    ### start calculated by rdkit
    num_atoms_ds = outfile.create_dataset("num_atoms", (smile_count * NUM_MAPS_PER_MOLECULE,))
    num_bonds_ds = outfile.create_dataset("num_bonds", (smile_count * NUM_MAPS_PER_MOLECULE,))
    num_heavy_atoms_ds = outfile.create_dataset("num_heavy_atoms", (smile_count * NUM_MAPS_PER_MOLECULE,))
    num_exact_Mol_Wt_ds = outfile.create_dataset("num_exact_Mol_Wt", (smile_count * NUM_MAPS_PER_MOLECULE,))
    MolLogP_ds = outfile.create_dataset("MolLogP", (smile_count * NUM_MAPS_PER_MOLECULE,))
    if do_Hdonors:
        num_H_acceptors_ds = outfile.create_dataset("num_H_acceptors", (smile_count * NUM_MAPS_PER_MOLECULE,))
        num_H_donors_ds = outfile.create_dataset("num_H_donors", (smile_count * NUM_MAPS_PER_MOLECULE,))
    num_heteroatoms_ds = outfile.create_dataset("num_ heteroatoms", (smile_count * NUM_MAPS_PER_MOLECULE,))
    num_valence_electrons_ds = outfile.create_dataset("num_valence_electrons", (smile_count * NUM_MAPS_PER_MOLECULE,))
    PMI_1_ds = outfile.create_dataset("PMI_1", (smile_count * NUM_MAPS_PER_MOLECULE,))
    PMI_2_ds = outfile.create_dataset("PMI_2", (smile_count * NUM_MAPS_PER_MOLECULE,))
    PMI_3_ds = outfile.create_dataset("PMI_3", (smile_count * NUM_MAPS_PER_MOLECULE,))
    spherocity_ds = outfile.create_dataset("spherocity", (smile_count * NUM_MAPS_PER_MOLECULE,))
    asphericity_ds = outfile.create_dataset("asphericity", (smile_count * NUM_MAPS_PER_MOLECULE,))
    eccentricity_ds = outfile.create_dataset("eccentricity", (smile_count * NUM_MAPS_PER_MOLECULE,))
    inertial_shape_factor_ds = outfile.create_dataset("inertial_shape_factor", (smile_count * NUM_MAPS_PER_MOLECULE,))
    radius_of_gyration_ds = outfile.create_dataset("radius_of_gyration", (smile_count * NUM_MAPS_PER_MOLECULE,))
    # copied from output of df_maker above sigh
    ### end from the dataset

    ######################### start the loop ###################################
    ## Das Loop
    point_ptr = -1
    for mol_idx in range(Num_of_molecules):
        if mol_idx % 50 == 0:
            print('Got to Molecule no. ', mol_idx)
        ##### grab data from the dataframe
        current_row = df.loc[[mol_idx]]
        ##### grab a molecule! #####################################
        m = Molecule(SMILES_array[mol_idx], sanitize=sanitize)
        tidy_m = m
        # tidy_m.molecule.UpdatePropertyCache() # this is now done in Molecule if you got SMILEs
        ############### put molecule in an icosasphere #############
        # puts the molecule into an icosasphere
        i = Ico(m, DIVISION)
        print(df['Compound ID'].iloc[mol_idx])
        if smiles_count > 0:
            smiles_string = SMILES_array[mol_idx]
        #############################################################################################
        ################################### THIS IS THE FIRST 60 NETS ###############################
        #############################################################################################
        for face_idx in range(NUM_UNWRAPPINGS_TO_DO):
            for point_idx in range(NUM_DIRECTIONS_TO_DO):
                point_ptr += 1
                #### create the map (this does not plot a graphics object)
                i.plot2D(first_face=face_idx, point_idx=point_idx);
                fs = i.get_face_list()
                # i.draw2D()
                #### ####### grab the atom values or colours or whatever############
                Face._lookup_func = Face.face_get_masses
                values = [f.get_values() for f in fs]
                ################ create the measurables you want to record #############
                values_as_array = np.array(values)  # this is hte icosahedron stuff
                num_atoms = tidy_m.molecule.GetNumAtoms()  # number of atoms
                num_bonds = tidy_m.molecule.GetNumBonds()  # number of bonds
                num_heavy_atoms = tidy_m.molecule.GetNumHeavyAtoms()  # number of non-hydrogens
                num_exact_Mol_Wt = Descriptors.ExactMolWt(tidy_m.molecule)  # exact molar weight
                MolLogP = Descriptors.MolLogP(tidy_m.molecule, includeHs=True)  # octanol / water partitian coefficient
                num_heteroatoms = Descriptors.NumHeteroatoms(tidy_m.molecule)
                num_valence_electrons = Descriptors.NumValenceElectrons(tidy_m.molecule)
                if do_Hdonors:
                    num_H_acceptors = Descriptors.NumHAcceptors(tidy_m.molecule)
                    num_H_donors = Descriptors.NumHDonors(tidy_m.molecule)
                PMI_1 = rdMolDescriptors.CalcPMI1(tidy_m.molecule)  # principal moment of inertia 1 (smallest)
                PMI_2 = rdMolDescriptors.CalcPMI2(tidy_m.molecule)  # principal moment of inertia 2
                PMI_3 = rdMolDescriptors.CalcPMI3(tidy_m.molecule)  # principal moment of inertia 3
                spherocity = rdMolDescriptors.CalcSpherocityIndex(tidy_m.molecule)
                asphericity = rdMolDescriptors.CalcAsphericity(tidy_m.molecule)
                eccentricity = rdMolDescriptors.CalcEccentricity(tidy_m.molecule)
                inertial_shape_factor = rdMolDescriptors.CalcInertialShapeFactor(tidy_m.molecule)
                radius_of_gyration = rdMolDescriptors.CalcRadiusOfGyration(tidy_m.molecule)

                ############ assign measurabless to columns ##########################
                ###### assign unfolding net
                icosahedron_ds[point_ptr] = values_as_array
                # charge_ds[point_ptr] = charge
                molID_ds[point_ptr] = mol_idx
                ###### assign stuff from the database
                compound_id = current_row.iloc[0]['Compound ID']
                compound_id_ds[point_ptr] = compound_id
                esol_predicted_log_solubility_in_mols_per_litre = current_row.iloc[0][
                    'ESOL predicted log solubility in mols per litre']
                esol_predicted_log_solubility_in_mols_per_litre_ds[
                    point_ptr] = esol_predicted_log_solubility_in_mols_per_litre
                minimum_degree = current_row.iloc[0]['Minimum Degree']
                minimum_degree_ds[point_ptr] = minimum_degree
                molecular_weight = current_row.iloc[0]['Molecular Weight']
                molecular_weight_ds[point_ptr] = molecular_weight
                number_of_h_bond_donors = current_row.iloc[0]['Number of H-Bond Donors']
                number_of_h_bond_donors_ds[point_ptr] = number_of_h_bond_donors
                number_of_rings = current_row.iloc[0]['Number of Rings']
                number_of_rings_ds[point_ptr] = number_of_rings
                number_of_rotatable_bonds = current_row.iloc[0]['Number of Rotatable Bonds']
                number_of_rotatable_bonds_ds[point_ptr] = number_of_rotatable_bonds
                polar_surface_area = current_row.iloc[0]['Polar Surface Area']
                polar_surface_area_ds[point_ptr] = polar_surface_area
                measured_log_solubility_in_mols_per_litre = current_row.iloc[0][
                    'measured log solubility in mols per litre']
                measured_log_solubility_in_mols_per_litre_ds[point_ptr] = measured_log_solubility_in_mols_per_litre
                smiles = current_row.iloc[0]['smiles']
                smiles_ds[point_ptr] = smiles
                ######## assign stuff you calculated ######
                num_atoms_ds[point_ptr] = num_atoms
                num_bonds_ds[point_ptr] = num_bonds
                num_heavy_atoms_ds[point_ptr] = num_heavy_atoms
                num_exact_Mol_Wt_ds[point_ptr] = num_exact_Mol_Wt
                MolLogP_ds[point_ptr] = MolLogP
                if do_Hdonors:
                    num_H_acceptors_ds[point_ptr] = num_H_acceptors
                    num_H_donors_ds[point_ptr] = num_H_donors
                num_heteroatoms_ds[point_ptr] = num_heteroatoms
                num_valence_electrons_ds[point_ptr] = num_valence_electrons
                PMI_1_ds[point_ptr] = PMI_1
                PMI_2_ds[point_ptr] = PMI_2
                PMI_3_ds[point_ptr] = PMI_3
                spherocity_ds[point_ptr] = spherocity
                asphericity_ds[point_ptr] = asphericity
                eccentricity_ds[point_ptr] = eccentricity
                inertial_shape_factor_ds[point_ptr] = inertial_shape_factor
                radius_of_gyration_ds[point_ptr] = radius_of_gyration
        if verbose:
            print('Finished the 60 standard unfoldings')
        #############################################################################################
        ######################## THE EXTRA AUGMENTATION STARTS HERE !################################
        #############################################################################################
        for extra_idx in range(NUM_EXTRA_MAPS_PER_MOLECULE):
            ## this is it, regen the molecule each time you unwrap to move it about a bit!
            m = Molecule(SMILES_array[mol_idx], sanitize=sanitize)
            tidy_m = m
            for point_idx in range(1):  # hacky cos I didn't want to indent!!!!!
                # we pick the face and direction randomly for this single unfolding
                face_idx = random.choices([x for x in range(NUM_UNWRAPPINGS_TO_DO)], k=1)[0]
                point_idx = random.choices([x for x in range(NUM_DIRECTIONS_TO_DO)], k=1)[0]
                if verbose:
                    print('Doing extra: face {}, direction {}'.format(face_idx, point_idx))
                point_ptr += 1
                #### create the map (this does not plot a graphics object)
                i = Ico(m, DIVISION)
                i.plot2D(first_face=face_idx, point_idx=point_idx);
                fs = i.get_face_list()
                # i.draw2D()
                #### ####### grab the atom values or colours or whatever############
                Face._lookup_func = Face.face_get_masses
                values = [f.get_values() for f in fs]
                ################ create the measurables you want to record #############
                values_as_array = np.array(values)  # this is hte icosahedron stuff
                num_atoms = tidy_m.molecule.GetNumAtoms()  # number of atoms
                num_bonds = tidy_m.molecule.GetNumBonds()  # number of bonds
                num_heavy_atoms = tidy_m.molecule.GetNumHeavyAtoms()  # number of non-hydrogens
                num_exact_Mol_Wt = Descriptors.ExactMolWt(tidy_m.molecule)  # exact molar weight
                MolLogP = Descriptors.MolLogP(tidy_m.molecule, includeHs=True)  # octanol / water partitian coefficient
                num_heteroatoms = Descriptors.NumHeteroatoms(tidy_m.molecule)
                num_valence_electrons = Descriptors.NumValenceElectrons(tidy_m.molecule)
                if do_Hdonors:
                    num_H_acceptors = Descriptors.NumHAcceptors(tidy_m.molecule)
                    num_H_donors = Descriptors.NumHDonors(tidy_m.molecule)
                PMI_1 = rdMolDescriptors.CalcPMI1(tidy_m.molecule)  # principal moment of inertia 1 (smallest)
                PMI_2 = rdMolDescriptors.CalcPMI2(tidy_m.molecule)  # principal moment of inertia 2
                PMI_3 = rdMolDescriptors.CalcPMI3(tidy_m.molecule)  # principal moment of inertia 3
                spherocity = rdMolDescriptors.CalcSpherocityIndex(tidy_m.molecule)
                asphericity = rdMolDescriptors.CalcAsphericity(tidy_m.molecule)
                eccentricity = rdMolDescriptors.CalcEccentricity(tidy_m.molecule)
                inertial_shape_factor = rdMolDescriptors.CalcInertialShapeFactor(tidy_m.molecule)
                radius_of_gyration = rdMolDescriptors.CalcRadiusOfGyration(tidy_m.molecule)

                ############ assign measurabless to columns ##########################
                ###### assign unfolding net
                icosahedron_ds[point_ptr] = values_as_array
                # charge_ds[point_ptr] = charge
                molID_ds[point_ptr] = mol_idx
                ###### assign stuff from the database
                compound_id = current_row.iloc[0]['Compound ID']
                compound_id_ds[point_ptr] = compound_id
                esol_predicted_log_solubility_in_mols_per_litre = current_row.iloc[0][
                    'ESOL predicted log solubility in mols per litre']
                esol_predicted_log_solubility_in_mols_per_litre_ds[
                    point_ptr] = esol_predicted_log_solubility_in_mols_per_litre
                minimum_degree = current_row.iloc[0]['Minimum Degree']
                minimum_degree_ds[point_ptr] = minimum_degree
                molecular_weight = current_row.iloc[0]['Molecular Weight']
                molecular_weight_ds[point_ptr] = molecular_weight
                number_of_h_bond_donors = current_row.iloc[0]['Number of H-Bond Donors']
                number_of_h_bond_donors_ds[point_ptr] = number_of_h_bond_donors
                number_of_rings = current_row.iloc[0]['Number of Rings']
                number_of_rings_ds[point_ptr] = number_of_rings
                number_of_rotatable_bonds = current_row.iloc[0]['Number of Rotatable Bonds']
                number_of_rotatable_bonds_ds[point_ptr] = number_of_rotatable_bonds
                polar_surface_area = current_row.iloc[0]['Polar Surface Area']
                polar_surface_area_ds[point_ptr] = polar_surface_area
                measured_log_solubility_in_mols_per_litre = current_row.iloc[0][
                    'measured log solubility in mols per litre']
                measured_log_solubility_in_mols_per_litre_ds[point_ptr] = measured_log_solubility_in_mols_per_litre
                smiles = current_row.iloc[0]['smiles']
                smiles_ds[point_ptr] = smiles
                ######## assign stuff you calculated ######
                num_atoms_ds[point_ptr] = num_atoms
                num_bonds_ds[point_ptr] = num_bonds
                num_heavy_atoms_ds[point_ptr] = num_heavy_atoms
                num_exact_Mol_Wt_ds[point_ptr] = num_exact_Mol_Wt
                MolLogP_ds[point_ptr] = MolLogP
                if do_Hdonors:
                    num_H_acceptors_ds[point_ptr] = num_H_acceptors
                    num_H_donors_ds[point_ptr] = num_H_donors
                num_heteroatoms_ds[point_ptr] = num_heteroatoms
                num_valence_electrons_ds[point_ptr] = num_valence_electrons
                PMI_1_ds[point_ptr] = PMI_1
                PMI_2_ds[point_ptr] = PMI_2
                PMI_3_ds[point_ptr] = PMI_3
                spherocity_ds[point_ptr] = spherocity
                asphericity_ds[point_ptr] = asphericity
                eccentricity_ds[point_ptr] = eccentricity
                inertial_shape_factor_ds[point_ptr] = inertial_shape_factor
                radius_of_gyration_ds[point_ptr] = radius_of_gyration

    outfile.close()
    # outfile_expanded.close()
    i.draw2D()
    return




def create_diff_conformer_dataset_from_QM7(
        DIVISION,
        df,
        frame,
        save_dir,
        data_dir,
        out_filename,
        do_Hdonors,
         ico_key_name,
        num_of_molecules_override=0,
        NUM_MAPS_PER_MOLECULE=60,
        sanitize=True,
        SMILES_array=[],
        num_out_files=1,
        extra_augmentation='conformer',
        rotamer_source='df',
        conformer_source='df',
        verbose=False):
    """ Creates the dataset as a hdf5 file
    QM7 has both smiles and 3d coords
    DIVISION =4 # for the size of icospehre
    save_dir=r'C:\\Users\\ella_\\Nextcloud\\science\\Datasets\\converted_pdbbind\\v2015'
    data_dir=r'C:\\Users\\ella_\\Nextcloud\\science\\Datasets\\pdbbind\\v2015'
    out_filename = "PDBBindLigands_core_augmented_withHs_withHdonors_div4.hdf5"
    do_Hdonors = True/False, whether to do the H donor data or not, don't do it for proteins
    ico_key_name_name = "icosahedron_level4"
    num_of_molecules_override: 0, how many molecules to do, a setting of 0 does all of them
    NUM_MAPS_PER_MOLECULE # how many maps to create, anything over 60 will be augmented
    sanitize=True
    SMILES_array= an array of smiles, set this up from df or fix this
    num_out_files=1 not yet implemented
    extra_augmentation='conformer' other options not yet implemented!
    verbose=False
    """
    if True:
        if num_of_molecules_override == 0:
            # do all smiles woo
            Num_of_molecules= len(SMILES_array)
        else:
            Num_of_molecules = num_of_molecules_override

        smiles_count = Num_of_molecules
        smile_count = Num_of_molecules # yeah i know, is hacky

        if DIVISION == 1:
            NUM_FACES = 80
        elif DIVISION == 2:
            NUM_FACES = 320
        elif DIVISION == 3:
            NUM_FACES = 1280
        elif DIVISION == 4:
            NUM_FACES = 5120

        if NUM_MAPS_PER_MOLECULE == 1:
            NUM_UNWRAPPINGS_TO_DO = 1
            NUM_DIRECTIONS_TO_DO = 1
            NUM_EXTRA_MAPS_PER_MOLECULE = 0
        elif NUM_MAPS_PER_MOLECULE == 60:
            NUM_UNWRAPPINGS_TO_DO = 20
            NUM_DIRECTIONS_TO_DO = 3
            NUM_EXTRA_MAPS_PER_MOLECULE = 0
        elif NUM_MAPS_PER_MOLECULE > 60:
            # Currently assumes you want 60 plus extra
            NUM_UNWRAPPINGS_TO_DO = 20
            NUM_DIRECTIONS_TO_DO = 3
            NUM_EXTRA_MAPS_PER_MOLECULE = NUM_MAPS_PER_MOLECULE-60
            if verbose:
                print('Doing {} extra maps using {}s'.format(
                    NUM_EXTRA_MAPS_PER_MOLECULE,
                    extra_augmentation))

        if sanitize:
            print('Warning! Sanitize seems to muck up the structures, especially for protiens')
            print('Is good for SMILES string input however')
            print('Will override sanitize option for the 3D structures')

        outfile = h5py.File(os.path.join(save_dir,out_filename),"w")
        #if NUM_EXTRA_MAPS_PER_MOLECULE > 0:
        #    outfile_expanded = h5py.File(os.path.join(save_dir,out_filename + '_extra_' + extra_augmentation),"w")
        string_type = h5py.string_dtype(encoding='utf-8')
        icosahedron_name = ico_key_name

        ##################### set up the out put datasets ################################
        ## this sets up the output datasets
        icosahedron_ds =  outfile.create_dataset(icosahedron_name, (smile_count*NUM_MAPS_PER_MOLECULE, NUM_FACES, 3))
        ## from the dataset
        u0_atom_ds = outfile.create_dataset('u0_atom', (smile_count*NUM_MAPS_PER_MOLECULE,))
        compound_id_ds = outfile.create_dataset('Compound ID', (smile_count*NUM_MAPS_PER_MOLECULE,), dtype=string_type)
        smiles_ds = outfile.create_dataset('smiles', (smile_count*NUM_MAPS_PER_MOLECULE,), dtype=string_type)
        ### end from the dataset
        ### start calculated by rdkit
        num_atoms_ds = outfile.create_dataset("num_atoms", (smile_count*NUM_MAPS_PER_MOLECULE,))
        num_bonds_ds = outfile.create_dataset("num_bonds", (smile_count*NUM_MAPS_PER_MOLECULE,))
        num_heavy_atoms_ds = outfile.create_dataset("num_heavy_atoms", (smile_count*NUM_MAPS_PER_MOLECULE,))
        num_exact_Mol_Wt_ds = outfile.create_dataset("num_exact_Mol_Wt", (smile_count*NUM_MAPS_PER_MOLECULE,))
        MolLogP_ds = outfile.create_dataset("MolLogP", (smile_count*NUM_MAPS_PER_MOLECULE,))
        if do_Hdonors:
            num_H_acceptors_ds = outfile.create_dataset("num_H_acceptors", (smile_count*NUM_MAPS_PER_MOLECULE,))
            num_H_donors_ds = outfile.create_dataset("num_H_donors", (smile_count*NUM_MAPS_PER_MOLECULE,))
        num_heteroatoms_ds = outfile.create_dataset("num_ heteroatoms", (smile_count*NUM_MAPS_PER_MOLECULE,))
        num_valence_electrons_ds = outfile.create_dataset("num_valence_electrons", (smile_count*NUM_MAPS_PER_MOLECULE,))
        PMI_1_ds = outfile.create_dataset("PMI_1", (smile_count*NUM_MAPS_PER_MOLECULE,))
        PMI_2_ds = outfile.create_dataset("PMI_2", (smile_count*NUM_MAPS_PER_MOLECULE,))
        PMI_3_ds = outfile.create_dataset("PMI_3", (smile_count*NUM_MAPS_PER_MOLECULE,))
        spherocity_ds = outfile.create_dataset("spherocity", (smile_count*NUM_MAPS_PER_MOLECULE,))
        asphericity_ds = outfile.create_dataset("asphericity", (smile_count*NUM_MAPS_PER_MOLECULE,))
        eccentricity_ds = outfile.create_dataset("eccentricity", (smile_count*NUM_MAPS_PER_MOLECULE,))
        inertial_shape_factor_ds = outfile.create_dataset("inertial_shape_factor", (smile_count*NUM_MAPS_PER_MOLECULE,))
        radius_of_gyration_ds = outfile.create_dataset("radius_of_gyration", (smile_count*NUM_MAPS_PER_MOLECULE,))
        # copied from output of df_maker above sigh
        ### end from the dataset

        ######################### start the loop ###################################
        ## Das Loop
        point_ptr = -1
        for mol_idx in range(Num_of_molecules):
            if mol_idx % 50 == 0:
                print('Got to Molecule no. ', mol_idx)
            ##### grab data from the dataframe
            current_row = df.loc[[mol_idx]]
            ##### grab a molecule! #####################################
            try:
                if rotamer_source == 'frame':
                    # taking data from the sdf file, good 3D data, bad SMILES
                    m = projection.sdf_molecule.SDFMolecule(
                        molecule=frame['Molecule'][mol_idx],
                        smiles=SMILES_array[mol_idx],
                        do_random_rotation=False,
                        rotation_vector=[])
                elif rotamer_source == 'df':
                    # taking data from the csv file, good SMILES, no 3D data
                    m = Molecule(SMILES_array[mol_idx], sanitize=sanitize)
            except Exception as e:
                print("moo")
                raise e
            tidy_m = m
            #tidy_m.molecule.UpdatePropertyCache() # this is now done in Molecule if you got SMILEs
            ############### put molecule in an icosasphere #############
            # puts the molecule into an icosasphere
            i = Ico(m,DIVISION)
            print(df['Compound ID'].iloc[mol_idx])
            if smiles_count > 0:
                smiles_string = SMILES_array[mol_idx]
            #############################################################################################
            ################################### THIS IS THE FIRST 60 NETS ###############################
            #############################################################################################
            for face_idx in range(NUM_UNWRAPPINGS_TO_DO):
                for point_idx in range(NUM_DIRECTIONS_TO_DO):
                    point_ptr += 1
                    #### create the map (this does not plot a graphics object)
                    i.plot2D(first_face=face_idx, point_idx=point_idx);
                    fs=i.get_face_list()
                    #i.draw2D()
                    #### ####### grab the atom values or colours or whatever############
                    Face._lookup_func = Face.face_get_masses
                    values = [f.get_values() for f in fs]
                    ################ create the measurables you want to record #############
                    values_as_array = np.array(values) # this is hte icosahedron stuff
                    num_atoms = tidy_m.molecule.GetNumAtoms() # number of atoms
                    num_bonds = tidy_m.molecule.GetNumBonds() # number of bonds
                    num_heavy_atoms = tidy_m.molecule.GetNumHeavyAtoms() # number of non-hydrogens
                    num_exact_Mol_Wt = Descriptors.ExactMolWt(tidy_m.molecule) # exact molar weight
                    MolLogP = Descriptors.MolLogP(tidy_m.molecule, includeHs=True) # octanol / water partitian coefficient
                    num_heteroatoms = Descriptors.NumHeteroatoms(tidy_m.molecule)
                    num_valence_electrons = Descriptors.NumValenceElectrons(tidy_m.molecule)
                    if do_Hdonors:
                        num_H_acceptors = Descriptors.NumHAcceptors(tidy_m.molecule)
                        num_H_donors = Descriptors.NumHDonors(tidy_m.molecule)
                    PMI_1 = rdMolDescriptors.CalcPMI1(tidy_m.molecule) # principal moment of inertia 1 (smallest)
                    PMI_2 = rdMolDescriptors.CalcPMI2(tidy_m.molecule) # principal moment of inertia 2
                    PMI_3 = rdMolDescriptors.CalcPMI3(tidy_m.molecule) # principal moment of inertia 3
                    spherocity = rdMolDescriptors.CalcSpherocityIndex(tidy_m.molecule)
                    asphericity = rdMolDescriptors.CalcAsphericity(tidy_m.molecule)
                    eccentricity = rdMolDescriptors.CalcEccentricity(tidy_m.molecule)
                    inertial_shape_factor = rdMolDescriptors.CalcInertialShapeFactor(tidy_m.molecule)
                    radius_of_gyration = rdMolDescriptors.CalcRadiusOfGyration(tidy_m.molecule)

                    ############ assign measurabless to columns ##########################
                    ###### assign unfolding net
                    icosahedron_ds[point_ptr] = values_as_array
                    ###### assign stuff from the database
                    compound_id = current_row.iloc[0]['Compound ID']
                    compound_id_ds[point_ptr] = compound_id
                    u0_atom = current_row.iloc[0]['u0_atom']
                    u0_atom_ds[point_ptr] = u0_atom
                    smiles = current_row.iloc[0]['smiles']
                    smiles_ds[point_ptr] = smiles
                    ######## assign stuff you calculated ######
                    num_atoms_ds[point_ptr] = num_atoms
                    num_bonds_ds[point_ptr] = num_bonds
                    num_heavy_atoms_ds[point_ptr] =  num_heavy_atoms
                    num_exact_Mol_Wt_ds[point_ptr] =  num_exact_Mol_Wt
                    MolLogP_ds[point_ptr] =  MolLogP
                    if do_Hdonors:
                        num_H_acceptors_ds[point_ptr] =  num_H_acceptors
                        num_H_donors_ds[point_ptr] =  num_H_donors
                    num_heteroatoms_ds[point_ptr] =  num_heteroatoms
                    num_valence_electrons_ds[point_ptr] =  num_valence_electrons
                    PMI_1_ds[point_ptr] =  PMI_1
                    PMI_2_ds[point_ptr] =  PMI_2
                    PMI_3_ds[point_ptr] =  PMI_3
                    spherocity_ds[point_ptr] =  spherocity
                    asphericity_ds[point_ptr] =  asphericity
                    eccentricity_ds[point_ptr] =  eccentricity
                    inertial_shape_factor_ds[point_ptr] =  inertial_shape_factor
                    radius_of_gyration_ds[point_ptr] =  radius_of_gyration
            if verbose:
                print('Finished the 60 standard unfoldings')
            #############################################################################################
            ######################## THE EXTRA AUGMENTATION STARTS HERE !################################
            #############################################################################################
            ### !!! Here we use SMILES strings not 3D coordinates ####
            for extra_idx in range(NUM_EXTRA_MAPS_PER_MOLECULE):
                ## this is it, regen the molecule each time you unwrap to move it about a bit!
                if conformer_source == 'frame':
                    # taking data from the sdf file, good 3D data, bad SMILES
                    m = projection.sdf_molecule.SDFMolecule(
                        molecule=frame['Molecule'][mol_idx],
                        smiles=SMILES_array[mol_idx],
                        do_random_rotation=False,
                        rotation_vector=[])
                elif conformer_source == 'df':
                    # taking data from the csv file, good SMILES, no 3D data
                    m = Molecule(SMILES_array[mol_idx], sanitize=sanitize)
                #print(m.molecule)
                tidy_m = m
                for point_idx in range(1): # hacky cos I didn't want to indent!!!!!
                    # we pick the face and direction randomly for this single unfolding
                    face_idx = random.choices([x for x in range(NUM_UNWRAPPINGS_TO_DO)], k=1)[0]
                    point_idx = random.choices([x for x in range(NUM_DIRECTIONS_TO_DO)], k=1)[0]
                    if verbose:
                        print('Doing extra: face {}, direction {}'.format(face_idx, point_idx))
                    point_ptr += 1
                    #### create the map (this does not plot a graphics object)
                    i.plot2D(first_face=face_idx, point_idx=point_idx);
                    fs=i.get_face_list()
                    #i.draw2D()
                    #### ####### grab the atom values or colours or whatever############
                    Face._lookup_func = Face.face_get_masses
                    values = [f.get_values() for f in fs]
                    ################ create the measurables you want to record #############
                    values_as_array = np.array(values) # this is hte icosahedron stuff
                    num_atoms = tidy_m.molecule.GetNumAtoms() # number of atoms
                    num_bonds = tidy_m.molecule.GetNumBonds() # number of bonds
                    num_heavy_atoms = tidy_m.molecule.GetNumHeavyAtoms() # number of non-hydrogens
                    num_exact_Mol_Wt = Descriptors.ExactMolWt(tidy_m.molecule) # exact molar weight
                    MolLogP = Descriptors.MolLogP(tidy_m.molecule, includeHs=True) # octanol / water partitian coefficient
                    num_heteroatoms = Descriptors.NumHeteroatoms(tidy_m.molecule)
                    num_valence_electrons = Descriptors.NumValenceElectrons(tidy_m.molecule)
                    if do_Hdonors:
                        num_H_acceptors = Descriptors.NumHAcceptors(tidy_m.molecule)
                        num_H_donors = Descriptors.NumHDonors(tidy_m.molecule)
                    PMI_1 = rdMolDescriptors.CalcPMI1(tidy_m.molecule) # principal moment of inertia 1 (smallest)
                    PMI_2 = rdMolDescriptors.CalcPMI2(tidy_m.molecule) # principal moment of inertia 2
                    PMI_3 = rdMolDescriptors.CalcPMI3(tidy_m.molecule) # principal moment of inertia 3
                    spherocity = rdMolDescriptors.CalcSpherocityIndex(tidy_m.molecule)
                    asphericity = rdMolDescriptors.CalcAsphericity(tidy_m.molecule)
                    eccentricity = rdMolDescriptors.CalcEccentricity(tidy_m.molecule)
                    inertial_shape_factor = rdMolDescriptors.CalcInertialShapeFactor(tidy_m.molecule)
                    radius_of_gyration = rdMolDescriptors.CalcRadiusOfGyration(tidy_m.molecule)

                    ############ assign measurabless to columns ##########################
                    ###### assign unfolding net
                    icosahedron_ds[point_ptr] = values_as_array
                    #charge_ds[point_ptr] = charge
                    ###### assign stuff from the database
                    compound_id = current_row.iloc[0]['Compound ID']
                    compound_id_ds[point_ptr] = compound_id
                    u0_atom = current_row.iloc[0]['u0_atom']
                    u0_atom_ds[point_ptr] = u0_atom
                    smiles = current_row.iloc[0]['smiles']
                    smiles_ds[point_ptr] = smiles
                    ######## assign stuff you calculated ######
                    num_atoms_ds[point_ptr] = num_atoms
                    num_bonds_ds[point_ptr] = num_bonds
                    num_heavy_atoms_ds[point_ptr] =  num_heavy_atoms
                    num_exact_Mol_Wt_ds[point_ptr] =  num_exact_Mol_Wt
                    MolLogP_ds[point_ptr] =  MolLogP
                    if do_Hdonors:
                        num_H_acceptors_ds[point_ptr] =  num_H_acceptors
                        num_H_donors_ds[point_ptr] =  num_H_donors
                    num_heteroatoms_ds[point_ptr] =  num_heteroatoms
                    num_valence_electrons_ds[point_ptr] =  num_valence_electrons
                    PMI_1_ds[point_ptr] =  PMI_1
                    PMI_2_ds[point_ptr] =  PMI_2
                    PMI_3_ds[point_ptr] =  PMI_3
                    spherocity_ds[point_ptr] =  spherocity
                    asphericity_ds[point_ptr] =  asphericity
                    eccentricity_ds[point_ptr] =  eccentricity
                    inertial_shape_factor_ds[point_ptr] =  inertial_shape_factor
                    radius_of_gyration_ds[point_ptr] =  radius_of_gyration


        outfile.close()
        #outfile_expanded.close()
        #i.draw2D()

        return

def generate_structure_from_smiles(smiles):

    # Generate a 3D structure from smiles

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    status = AllChem.EmbedMolecule(mol)
    status = AllChem.UFFOptimizeMolecule(mol)

    conformer = mol.GetConformer()
    coordinates = conformer.GetPositions()
    coordinates = np.array(coordinates)

    #atoms = get_atoms(mol)

    return coordinates


def Create_Diff_Conformer_Dataset_From_Lipo(DIVISION,
                                            save_dir,
                                            data_dir,
                                            out_filename,
                                            do_Hdonors,
                                            ico_key_name,
                                            num_of_molecules_override=0,
                                            NUM_MAPS_PER_MOLECULE=60,
                                            sanitize=True,
                                            SMILES_array=[],
                                            num_out_files=1,
                                            extra_augmentation='conformer',
                                            verbose=False):
    """ Creates the dataset as a hdf5 file
    DIVISION =4 # for the size of icospehre
    save_dir=r'C:\\Users\\ella_\\Nextcloud\\science\\Datasets\\converted_pdbbind\\v2015'
    data_dir=r'C:\\Users\\ella_\\Nextcloud\\science\\Datasets\\pdbbind\\v2015'
    out_filename = "PDBBindLigands_core_augmented_withHs_withHdonors_div4.hdf5"
    do_Hdonors = True/False, whether to do the H donor data or not, don't do it for proteins
    ico_key_name_name = "icosahedron_level4"
    num_of_molecules_override: 0, how many molecules to do, a setting of 0 does all of them
    NUM_MAPS_PER_MOLECULE # how many maps to create, anything over 60 will be augmented
    sanitize=True
    SMILES_array= an array of smiles, set this up from df or fix this
    num_out_files=1 not yet implemented
    extra_augmentation='conformer' other options not yet implemented!
    verbose=False
    """

    if num_of_molecules_override == 0:
        # do all smiles woo
        Num_of_molecules = len(SMILES_array)
    else:
        Num_of_molecules = num_of_molecules_override

    smiles_count = Num_of_molecules
    smile_count = Num_of_molecules  # yeah i know, is hacky

    if DIVISION == 1:
        NUM_FACES = 80
    elif DIVISION == 2:
        NUM_FACES = 320
    elif DIVISION == 3:
        NUM_FACES = 1280
    elif DIVISION == 4:
        NUM_FACES = 5120

    if NUM_MAPS_PER_MOLECULE == 1:
        NUM_UNWRAPPINGS_TO_DO = 1
        NUM_DIRECTIONS_TO_DO = 1
        NUM_EXTRA_MAPS_PER_MOLECULE = 0
    elif NUM_MAPS_PER_MOLECULE == 60:
        NUM_UNWRAPPINGS_TO_DO = 20
        NUM_DIRECTIONS_TO_DO = 3
        NUM_EXTRA_MAPS_PER_MOLECULE = 0
    elif NUM_MAPS_PER_MOLECULE > 60:
        # Currently assumes you want 60 plus extra
        NUM_UNWRAPPINGS_TO_DO = 20
        NUM_DIRECTIONS_TO_DO = 3
        NUM_EXTRA_MAPS_PER_MOLECULE = NUM_MAPS_PER_MOLECULE - 60
        if verbose:
            print('Doing {} extra maps using {}s'.format(
                NUM_EXTRA_MAPS_PER_MOLECULE,
                extra_augmentation))

    if sanitize:
        print('Warning! Sanitize seems to muck up the structures, especially for protiens')
        print('Is good for SMILES string input however')

    outfile = h5py.File(os.path.join(save_dir, out_filename), "w")
    if NUM_EXTRA_MAPS_PER_MOLECULE > 0:
        outfile_expanded = h5py.File(os.path.join(save_dir, out_filename + '_extra_' + extra_augmentation), "w")
    string_type = h5py.string_dtype(encoding='utf-8')
    icosahedron_name = ico_key_name

    ##################### set up the out put datasets ################################
    ## this sets up the output datasets
    icosahedron_ds = outfile.create_dataset(icosahedron_name, (smile_count * NUM_MAPS_PER_MOLECULE, NUM_FACES, 3))
    # charge_ds = outfile.create_dataset("charge", (smile_count*NUM_MAPS_PER_MOLECULE,))
    molID_ds = outfile.create_dataset("molID", (smile_count * NUM_MAPS_PER_MOLECULE,))
    ## from the dataset
    compound_id_ds = outfile.create_dataset('compound_id', (smile_count * NUM_MAPS_PER_MOLECULE,), dtype=string_type)
    exp_ds = outfile.create_dataset('exp', (smile_count * NUM_MAPS_PER_MOLECULE,))
    smiles_ds = outfile.create_dataset('smiles', (smile_count * NUM_MAPS_PER_MOLECULE,), dtype=string_type)
    ### end from the dataset
    ### start calculated by rdkit
    num_atoms_ds = outfile.create_dataset("num_atoms", (smile_count * NUM_MAPS_PER_MOLECULE,))
    num_bonds_ds = outfile.create_dataset("num_bonds", (smile_count * NUM_MAPS_PER_MOLECULE,))
    num_heavy_atoms_ds = outfile.create_dataset("num_heavy_atoms", (smile_count * NUM_MAPS_PER_MOLECULE,))
    num_exact_Mol_Wt_ds = outfile.create_dataset("num_exact_Mol_Wt", (smile_count * NUM_MAPS_PER_MOLECULE,))
    MolLogP_ds = outfile.create_dataset("MolLogP", (smile_count * NUM_MAPS_PER_MOLECULE,))
    if do_Hdonors:
        num_H_acceptors_ds = outfile.create_dataset("num_H_acceptors", (smile_count * NUM_MAPS_PER_MOLECULE,))
        num_H_donors_ds = outfile.create_dataset("num_H_donors", (smile_count * NUM_MAPS_PER_MOLECULE,))
    num_heteroatoms_ds = outfile.create_dataset("num_ heteroatoms", (smile_count * NUM_MAPS_PER_MOLECULE,))
    num_valence_electrons_ds = outfile.create_dataset("num_valence_electrons", (smile_count * NUM_MAPS_PER_MOLECULE,))
    PMI_1_ds = outfile.create_dataset("PMI_1", (smile_count * NUM_MAPS_PER_MOLECULE,))
    PMI_2_ds = outfile.create_dataset("PMI_2", (smile_count * NUM_MAPS_PER_MOLECULE,))
    PMI_3_ds = outfile.create_dataset("PMI_3", (smile_count * NUM_MAPS_PER_MOLECULE,))
    spherocity_ds = outfile.create_dataset("spherocity", (smile_count * NUM_MAPS_PER_MOLECULE,))
    asphericity_ds = outfile.create_dataset("asphericity", (smile_count * NUM_MAPS_PER_MOLECULE,))
    eccentricity_ds = outfile.create_dataset("eccentricity", (smile_count * NUM_MAPS_PER_MOLECULE,))
    inertial_shape_factor_ds = outfile.create_dataset("inertial_shape_factor", (smile_count * NUM_MAPS_PER_MOLECULE,))
    radius_of_gyration_ds = outfile.create_dataset("radius_of_gyration", (smile_count * NUM_MAPS_PER_MOLECULE,))
    # copied from output of df_maker above sigh
    ### end from the dataset

    ######################### start the loop ###################################
    ## Das Loop
    point_ptr = -1
    for mol_idx in range(Num_of_molecules):
        if mol_idx % 50 == 0:
            print('Got to Molecule no. ', mol_idx)
        ##### grab data from the dataframe
        current_row = df.loc[[mol_idx]]
        ##### grab a molecule! #####################################
        m = Molecule(SMILES_array[mol_idx], sanitize=sanitize)
        tidy_m = m
        # tidy_m.molecule.UpdatePropertyCache() # this is now done in Molecule if you got SMILEs
        ############### put molecule in an icosasphere #############
        # puts the molecule into an icosasphere
        i = Ico(m, DIVISION)
        print(df['CMPD_CHEMBLID'].iloc[mol_idx])
        if smiles_count > 0:
            smiles_string = SMILES_array[mol_idx]
        #############################################################################################
        ################################### THIS IS THE FIRST 60 NETS ###############################
        #############################################################################################
        for face_idx in range(NUM_UNWRAPPINGS_TO_DO):
            for point_idx in range(NUM_DIRECTIONS_TO_DO):
                point_ptr += 1
                #### create the map (this does not plot a graphics object)
                i.plot2D(first_face=face_idx, point_idx=point_idx);
                fs = i.get_face_list()
                # i.draw2D()
                #### ####### grab the atom values or colours or whatever############
                Face._lookup_func = Face.face_get_masses
                values = [f.get_values() for f in fs]
                ################ create the measurables you want to record #############
                values_as_array = np.array(values)  # this is hte icosahedron stuff
                num_atoms = tidy_m.molecule.GetNumAtoms()  # number of atoms
                num_bonds = tidy_m.molecule.GetNumBonds()  # number of bonds
                num_heavy_atoms = tidy_m.molecule.GetNumHeavyAtoms()  # number of non-hydrogens
                num_exact_Mol_Wt = Descriptors.ExactMolWt(tidy_m.molecule)  # exact molar weight
                MolLogP = Descriptors.MolLogP(tidy_m.molecule, includeHs=True)  # octanol / water partitian coefficient
                num_heteroatoms = Descriptors.NumHeteroatoms(tidy_m.molecule)
                num_valence_electrons = Descriptors.NumValenceElectrons(tidy_m.molecule)
                if do_Hdonors:
                    num_H_acceptors = Descriptors.NumHAcceptors(tidy_m.molecule)
                    num_H_donors = Descriptors.NumHDonors(tidy_m.molecule)
                PMI_1 = rdMolDescriptors.CalcPMI1(tidy_m.molecule)  # principal moment of inertia 1 (smallest)
                PMI_2 = rdMolDescriptors.CalcPMI2(tidy_m.molecule)  # principal moment of inertia 2
                PMI_3 = rdMolDescriptors.CalcPMI3(tidy_m.molecule)  # principal moment of inertia 3
                spherocity = rdMolDescriptors.CalcSpherocityIndex(tidy_m.molecule)
                asphericity = rdMolDescriptors.CalcAsphericity(tidy_m.molecule)
                eccentricity = rdMolDescriptors.CalcEccentricity(tidy_m.molecule)
                inertial_shape_factor = rdMolDescriptors.CalcInertialShapeFactor(tidy_m.molecule)
                radius_of_gyration = rdMolDescriptors.CalcRadiusOfGyration(tidy_m.molecule)

                ############ assign measurabless to columns ##########################
                ###### assign unfolding net
                icosahedron_ds[point_ptr] = values_as_array
                # charge_ds[point_ptr] = charge
                molID_ds[point_ptr] = mol_idx
                ###### assign stuff from the database

                compound_id = current_row.iloc[0]['CMPD_CHEMBLID']
                compound_id_ds[point_ptr] = compound_id
                exp = current_row.iloc[0]['exp']
                exp_ds[point_ptr] = exp
                smiles = current_row.iloc[0]['smiles']
                smiles_ds[point_ptr] = smiles

                ######## assign stuff you calculated ######
                num_atoms_ds[point_ptr] = num_atoms
                num_bonds_ds[point_ptr] = num_bonds
                num_heavy_atoms_ds[point_ptr] = num_heavy_atoms
                num_exact_Mol_Wt_ds[point_ptr] = num_exact_Mol_Wt
                MolLogP_ds[point_ptr] = MolLogP
                if do_Hdonors:
                    num_H_acceptors_ds[point_ptr] = num_H_acceptors
                    num_H_donors_ds[point_ptr] = num_H_donors
                num_heteroatoms_ds[point_ptr] = num_heteroatoms
                num_valence_electrons_ds[point_ptr] = num_valence_electrons
                PMI_1_ds[point_ptr] = PMI_1
                PMI_2_ds[point_ptr] = PMI_2
                PMI_3_ds[point_ptr] = PMI_3
                spherocity_ds[point_ptr] = spherocity
                asphericity_ds[point_ptr] = asphericity
                eccentricity_ds[point_ptr] = eccentricity
                inertial_shape_factor_ds[point_ptr] = inertial_shape_factor
                radius_of_gyration_ds[point_ptr] = radius_of_gyration
        if verbose:
            print('Finished the 60 standard unfoldings')
        #############################################################################################
        ######################## THE EXTRA AUGMENTATION STARTS HERE !################################
        #############################################################################################
        for extra_idx in range(NUM_EXTRA_MAPS_PER_MOLECULE):
            ## this is it, regen the molecule each time you unwrap to move it about a bit!
            m = Molecule(SMILES_array[mol_idx], sanitize=sanitize)
            tidy_m = m
            for point_idx in range(1):  # hacky cos I didn't want to indent!!!!!
                # we pick the face and direction randomly for this single unfolding
                face_idx = random.choices([x for x in range(NUM_UNWRAPPINGS_TO_DO)], k=1)[0]
                point_idx = random.choices([x for x in range(NUM_DIRECTIONS_TO_DO)], k=1)[0]
                if verbose:
                    print('Doing extra: face {}, direction {}'.format(face_idx, point_idx))
                point_ptr += 1
                #### create the map (this does not plot a graphics object)
                i = Ico(m, DIVISION)
                i.plot2D(first_face=face_idx, point_idx=point_idx);
                fs = i.get_face_list()
                # i.draw2D()
                #### ####### grab the atom values or colours or whatever############
                Face._lookup_func = Face.face_get_masses
                values = [f.get_values() for f in fs]
                ################ create the measurables you want to record #############
                values_as_array = np.array(values)  # this is hte icosahedron stuff
                num_atoms = tidy_m.molecule.GetNumAtoms()  # number of atoms
                num_bonds = tidy_m.molecule.GetNumBonds()  # number of bonds
                num_heavy_atoms = tidy_m.molecule.GetNumHeavyAtoms()  # number of non-hydrogens
                num_exact_Mol_Wt = Descriptors.ExactMolWt(tidy_m.molecule)  # exact molar weight
                MolLogP = Descriptors.MolLogP(tidy_m.molecule, includeHs=True)  # octanol / water partitian coefficient
                num_heteroatoms = Descriptors.NumHeteroatoms(tidy_m.molecule)
                num_valence_electrons = Descriptors.NumValenceElectrons(tidy_m.molecule)
                if do_Hdonors:
                    num_H_acceptors = Descriptors.NumHAcceptors(tidy_m.molecule)
                    num_H_donors = Descriptors.NumHDonors(tidy_m.molecule)
                PMI_1 = rdMolDescriptors.CalcPMI1(tidy_m.molecule)  # principal moment of inertia 1 (smallest)
                PMI_2 = rdMolDescriptors.CalcPMI2(tidy_m.molecule)  # principal moment of inertia 2
                PMI_3 = rdMolDescriptors.CalcPMI3(tidy_m.molecule)  # principal moment of inertia 3
                spherocity = rdMolDescriptors.CalcSpherocityIndex(tidy_m.molecule)
                asphericity = rdMolDescriptors.CalcAsphericity(tidy_m.molecule)
                eccentricity = rdMolDescriptors.CalcEccentricity(tidy_m.molecule)
                inertial_shape_factor = rdMolDescriptors.CalcInertialShapeFactor(tidy_m.molecule)
                radius_of_gyration = rdMolDescriptors.CalcRadiusOfGyration(tidy_m.molecule)

                ############ assign measurabless to columns ##########################
                ###### assign unfolding net
                icosahedron_ds[point_ptr] = values_as_array
                # charge_ds[point_ptr] = charge
                molID_ds[point_ptr] = mol_idx
                ###### assign stuff from the database
                compound_id = current_row.iloc[0]['CMPD_CHEMBLID']
                compound_id_ds[point_ptr] = compound_id
                exp = current_row.iloc[0]['exp']
                exp_ds[point_ptr] = exp
                smiles = current_row.iloc[0]['smiles']
                smiles_ds[point_ptr] = smiles
                ######## assign stuff you calculated ######
                num_atoms_ds[point_ptr] = num_atoms
                num_bonds_ds[point_ptr] = num_bonds
                num_heavy_atoms_ds[point_ptr] = num_heavy_atoms
                num_exact_Mol_Wt_ds[point_ptr] = num_exact_Mol_Wt
                MolLogP_ds[point_ptr] = MolLogP
                if do_Hdonors:
                    num_H_acceptors_ds[point_ptr] = num_H_acceptors
                    num_H_donors_ds[point_ptr] = num_H_donors
                num_heteroatoms_ds[point_ptr] = num_heteroatoms
                num_valence_electrons_ds[point_ptr] = num_valence_electrons
                PMI_1_ds[point_ptr] = PMI_1
                PMI_2_ds[point_ptr] = PMI_2
                PMI_3_ds[point_ptr] = PMI_3
                spherocity_ds[point_ptr] = spherocity
                asphericity_ds[point_ptr] = asphericity
                eccentricity_ds[point_ptr] = eccentricity
                inertial_shape_factor_ds[point_ptr] = inertial_shape_factor
                radius_of_gyration_ds[point_ptr] = radius_of_gyration

    outfile.close()
    # outfile_expanded.close()
    i.draw2D()
    return

#### This writes out the commands we need, so we can copy and paste below
## there must be a better way of doing this!
def df_maker(column_list, verbose=True):
    for header in column_list:
        # this makes the dfs
        cmd = "%s_ds = %s%s%s" % (
                slugify.slugify(header, separator="_"),
                "outfile.create_dataset('",
                header,
                "', (smile_count*NUM_MAPS_PER_MOLECULE,))")
        if verbose:
            print(cmd)
    return cmd


def df_maker_2(df, verbose=True):
    """writes commands for you and is string aware
    copy this into functions to make the dataset"""
    column_list = df.columns
    for header in column_list:
        # this makes the dfs
        if type(df[header][0]) == str:
            # strings

            cmd = "%s_ds = %s%s%s" % (
                slugify.slugify(header, separator="_"),
                "outfile.create_dataset('",
                header,
                "', (smile_count*NUM_MAPS_PER_MOLECULE,), dtype=string_type )")
            if verbose:
                print(cmd)
        else:
            cmd = "%s_ds = %s%s%s" % (
                slugify.slugify(header, separator="_"),
                "outfile.create_dataset('",
                header,
                "', (smile_count*NUM_MAPS_PER_MOLECULE,))")
            if verbose:
                print(cmd)
    return cmd

def df_writer(column_list):
    verbose=True
    for header in column_list:
        # this makes the dfs
        cmd1 = "%s = %s%s%s" % (
                slugify.slugify(header, separator="_"),
                "current_row.iloc[0]['",
                header,
                "']")
        cmd2 = "%s_ds[point_ptr] = %s" % (
                slugify.slugify(header, separator="_"),
                slugify.slugify(header, separator="_"))

        if verbose:
            print(cmd1)
            print(cmd2)
    return (cmd1, cmd2)