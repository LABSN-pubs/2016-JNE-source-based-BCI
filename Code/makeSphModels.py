"""
makeSphModels.py

@author wronk

Make spherical models and foward solutions for subject in SoP experiment
"""
from os import environ
import os.path as op
import csv

from mne import (make_sphere_model, make_forward_solution, read_source_spaces,
                 convert_forward_solution, pick_info, pick_types)
from mne.minimum_norm import (read_inverse_operator, make_inverse_operator,
                              write_inverse_operator)
from mne.io import read_info
from mne.transforms import read_trans

n_jobs = 6
generic_sph = True
###########################################################
# Load subject info and filenames
###########################################################
# List of subjects to run analysis on
subj_list = ['AKCLEE_107', 'AKCLEE_102', 'AKCLEE_103', 'AKCLEE_104',
             'AKCLEE_105', 'AKCLEE_106', 'AKCLEE_108', 'AKCLEE_110',
             'AKCLEE_113', 'AKCLEE_114']
#subj_list = ['AKCLEE_107']

print 'Running %d Subjects\n' % len(subj_list)

cache_dir = op.join(environ['CODE_ROOT'], 'switchBCIFiles')
subjListFName = op.join(environ['CODE_ROOT'], 'switchBCIFiles', 'SoP_MEG.txt')
struct_dir = op.join(environ['SUBJECTS_DIR'])

# Open/read subject info file into a list of dicts
with open(subjListFName, 'rb') as f:
    subj_d = list(csv.DictReader(f, dialect='excel-tab', delimiter='\t'))

# Get subject indices in the dict list
subj_inds = [i for i, d in enumerate(subj_d) if d['Struct'] in subj_list]
foldNames = [d['Subj#'] for d in [subj_d[i] for i in subj_inds]]

###########################################################
# Create spherical model for each subject
###########################################################
for si, (subj, fold_name) in enumerate(zip(subj_list, foldNames)):
    #fname_save_bem = op.join(struct_dir, subj, 'bem', subj + 'sph.fif')
    fname_save_fwd = op.join(cache_dir, fold_name, 'forward', fold_name +
                             '-SSS-eeg-sph-fwd.fif')
    fname_save_inv = op.join(cache_dir, fold_name, 'inverse', fold_name +
                             '-SSS-eeg-sph-Fixed-inv.fif')

    if generic_sph is True:
        # Use this fsaverage source space for generic spherical head models
        fname_load_src = op.join(struct_dir, 'fsaverage', 'bem',
                                 'fsaverage-5p7-src.fif')
    else:
        # Use this source space for individual subjects
        fname_load_src = op.join(struct_dir, subj, 'bem', subj + '-7-src.fif')

    fname_load_epo = op.join(environ['CODE_ROOT'], 'switchBCIFiles',
                             foldNames[si], 'epochs', 'All_55_sss_EQ_' +
                             foldNames[si] + '-epo.fif')
    fname_load_inv = op.join(cache_dir, fold_name, 'inverse', fold_name +
                             '-55-SSS-eeg-Fixed-inv.fif')
    fname_load_trans = op.join(cache_dir, fold_name, 'trans',
                               'fsaverage-trans.fif')

    # Load measurement info
    subj_info = read_info(fname_load_epo)
    subj_info = pick_info(subj_info, pick_types(subj_info, meg=False,
                                                eeg=True))

    # Read model inverse to get template for cov, mri_head_t
    model_inv = read_inverse_operator(fname_load_inv)
    ###########################################################
    # Create spherical BEM
    sph_bem = make_sphere_model(r0='auto', head_radius='auto', info=subj_info)

    ###########################################################
    # Create foward solution

    # Read source space and trans (mri->head) for foward computation
    src = read_source_spaces(fname_load_src)

    if generic_sph is True:
        # Use this trans if using generic fsaverage head model
        trans = read_trans(fname_load_trans)
    else:
        # Use this trans if using individual's head model
        trans = model_inv['mri_head_t']

    fwd = make_forward_solution(info=subj_info, trans=trans, src=src,
                                bem=sph_bem, fname=fname_save_fwd, meg=False,
                                eeg=True, overwrite=True, n_jobs=n_jobs)

    # Fix orientation
    convert_forward_solution(fwd, surf_ori=True, force_fixed=True, copy=False)

    ###########################################################
    # Create and save inverse solution
    noise_cov = model_inv['noise_cov']

    # EEG only, so depth weighting not needed
    sph_inv = make_inverse_operator(subj_info, fwd, noise_cov, depth=None,
                                    fixed=True)
    write_inverse_operator(fname_save_inv, sph_inv)
