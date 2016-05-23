"""
process_createSTCs.py

@Author: wronk

Create source time course (STC) objects of all trials from epochs object (that
are spit out of 'process_SoP.py') and save the stcs. If desired, calculate stcs
only for single label
"""


import mne
from mne.minimum_norm import apply_inverse_epochs as inv_epochs
from mne.minimum_norm import read_inverse_operator as read_inv
from mne.beamformer import lcmv_epochs
from mne import read_forward_solution as read_fwd
import numpy as np
import sps_fun
reload(sps_fun)

import csv
import glob
from os import environ, mkdir, path as op
from fnmatch import fnmatch
from copy import deepcopy
from scipy.sparse import csr_matrix as csr_mat


class FakeCov(dict):
    def __init__(self, data, info, diag=False):
        self.data = data
        self['data'] = data
        self['bads'] = info['bads']
        self['names'] = info['ch_names']
        self.ch_names = info['ch_names']
        self['eig'] = None
        self['eig_vec'] = None
        self['diag'] = diag

#XXX To check prior to running
#   1) label being used
#   2) Patterns being used and selected
#   3) Inverse method
#   4) Inverse solution (EEG or M/EEG)
########################################
### Load subject information
# List of subjects to run analysis on
subj_list = ['AKCLEE_102', 'AKCLEE_103', 'AKCLEE_104', 'AKCLEE_105',
             'AKCLEE_106', 'AKCLEE_107', 'AKCLEE_108', 'AKCLEE_110',
             'AKCLEE_113', 'AKCLEE_114']
#subj_list = ['AKCLEE_107']

roi = 'LRRon-LRStd_01-rh.label'
#roi = 'UDRon-UDStd_01-lh.label'
lambda2 = 1. / 9.
eq = 'EQ'  # 'DQ' or 'EQ' Just disqualified trials or equalized trials
inv_method = 'MNE'
ftype = 'stc'
bin_width = None
takeMagOfVals = False
inv_model = ['sphere', 'eeg', 'meeg'][0]

# Set patterns for epoch extraction
    # !maintain should be first!
if 'UD' in roi:
    patterns = ['pitch_maintain', 'pitch_switch']
else:
    assert 'LR' in roi
    patterns = ['space_maintain', 'space_switch']
    #patterns = ['space_maintain']

#######################
# Load key linking structural subject numbers to exp subject numbers
subj_info_dir = op.join(environ['CODE_ROOT'], 'switchBCIFiles')
subjListFName = op.join(subj_info_dir, 'SoP_MEG.txt')

struct_dir = op.join(environ['SUBJECTS_DIR'])

# Open/read subject info file into a list of dicts
with open(subjListFName, 'rb') as f:
    subj_d = list(csv.DictReader(f, dialect='excel-tab', delimiter='\t'))

# Get subject indices in the dict list
subj_inds = [i for i, d in enumerate(subj_d) if d['Struct'] in subj_list]

########################################
# Load inv, fwd, and epoch information
########################################
if inv_model == 'sphere':
    forward_end = '-SSS-eeg-sph-fwd.fif'
    inverse_end = '-SSS-eeg-sph-Fixed-inv.fif'
    morph_end = '-sph-fsaverage-morph.npz'
elif inv_model == 'eeg':
    forward_end = '-SSS-fwd.fif'
    inverse_end = '-55-SSS-eeg-Fixed-inv.fif'
    morph_end = '-fsaverage-morph.npz'
else:
    forward_end = '-SSS-fwd.fif'
    inverse_end = '-55-SSS-meg-eeg-Fixed-inv.fif'
    morph_end = '-fsaverage-morph.npz'

foldNames = [d['Subj#'] for d in [subj_d[i] for i in subj_inds]]

#Load all forward, inverse, and epochs
print 'Inverse method: ' + inv_method
print 'Reading inv, epochs, morph matrices for:'
for i, f in zip(subj_inds, foldNames):
    print '\t' + f,
    fname_load_epo = op.join(subj_info_dir, f, 'epochs', 'All_55_sss_' + eq +
                             '_' + f + '-epo.fif')
    subj_d[i]['epo'] = mne.read_epochs(fname_load_epo, preload=False,
                                       verbose=True)
    subj_d[i]['inv'] = read_inv(op.join(subj_info_dir, f, 'inverse',
                                        f + inverse_end), verbose=False)
    if inv_method == 'lcmv':
        if 'meg' not in inverse_end:
            picks = [subj_d[i]['epo'].ch_names[pi] for pi in
                     mne.pick_types(subj_d[i]['epo'].info, meg=False, eeg=True)]
            subj_d[i]['epo'].pick_channels(picks)

        subj_d[i]['fwd'] = read_fwd(op.join(subj_info_dir, f, 'forward',
                                            f + forward_end), verbose=False)

    # Uncomment to specify a single label to calcluate STCs for
    #label_fname = op.join(struct_dir, subj_d[i]['Struct'], 'label', roi)
    #subj_d[i]['lab'] = mne.read_label(label_fname, subject=subj_d[i]['Struct'])

    subj_d[i]['num'] = (f.split('_')[-1])

    # Also get fsaverage morph matrix if not using spherical head inverse
    # (if spherical, head model source space IS fsaverage)
    if inv_model != 'sphere':
        morph_temp = np.load(op.join(subj_info_dir, 'morph_mats',
                                     subj_d[i]['Struct'] + morph_end))
        subj_d[i]['fs_morph'] = csr_mat((morph_temp['data'],
                                         morph_temp['indices'],
                                         morph_temp['indptr']))

    print ' ...  Done'

########################################
# Load fsaverage information
########################################

fsaverage = {}
#fs_roi_fname = op.join(struct_dir, 'fsaverage', 'label', roi)
#fsaverage['lab'] = mne.read_label(fs_roi_fname, 'fsaverage')
fsaverage['src'] = mne.read_source_spaces(op.join(struct_dir, 'fsaverage',
                                                  'bem',
                                                  'fsaverage-5p7-src.fif'),
                                          verbose=False)
fsaverage['vertices'] = [fsaverage['src'][0]['vertno'],
                         fsaverage['src'][1]['vertno']]

########################################
# Calculate source estimates and save
########################################
print '\nInverse: ' + inverse_end
print 'Bin Size: ' + str(bin_width)
print 'Take absolute value: ' + str(takeMagOfVals) + '\n'
print 'Computing source estimate, morphing, and saving stc for:'

sph_add_on = ''  # Folder add-on
if inv_model == 'sphere':
    sph_add_on = '_sph'

for i in subj_inds:
    print '\t' + subj_d[i]['Subj#'],

    #subj_d[i]['epo_stc'] = {}

    for pi, p in enumerate(patterns):

        save_folder_unproc = op.join(subj_info_dir, subj_d[i]['Subj#'], 'stc',
                                     p + '_unprocessed' + sph_add_on)
        if not op.exists(save_folder_unproc):
            mkdir(save_folder_unproc)

        if inv_model != 'sphere':
            save_folder_morph = op.join(subj_info_dir, subj_d[i]['Subj#'],
                                        'stc', p + '_morphedToFsaverage' +
                                        sph_add_on)
            if not op.exists(save_folder_morph):
                mkdir(save_folder_morph)
        ############################################
        # Calculate epochs src estimates
        ############################################
        if inv_method == 'lcmv':
            '''
            fake_noise_cov = FakeCov(subj_d[i]['epo'].get_data(),
                                     subj_d[i]['epo'].info)
            fake_data_cov = FakeCov(subj_d[i]['inv']['source_cov']['data'],
                                    subj_d[i]['inv']['info'])
            '''

            noise_cov = mne.compute_covariance(subj_d[i]['epo'], tmin=-0.2,
                                               tmax=0)
            data_cov = mne.compute_covariance(subj_d[i]['epo'], tmin=0,
                                              tmax=4.5)
            stc_list = lcmv_epochs(subj_d[i]['epo'][p], subj_d[i]['fwd'],
                                   noise_cov, data_cov)
                                   #label=subj_d[i]['lab'])
        else:
            stc_list = inv_epochs(subj_d[i]['epo'][p], subj_d[i]['inv'],
                                  lambda2, inv_method, #label=subj_d[i]['lab'],
                                  verbose=False)

        ######################################################################
        # Save each unprocessed stc file and compute/save morphed stc
        ######################################################################
        for ti, temp_stc in enumerate(stc_list):
            # Save each unprocessed stc file
            save_fname = op.join(save_folder_unproc, subj_d[i]['Subj#'] + '_' +
                                 p + '_' + inv_method + '_t' + str(ti))
            temp_stc.save(save_fname, verbose=False)

            if bin_width is not None:
                temp_stc = temp_stc.bin(bin_width)

            if takeMagOfVals:
                temp_stc._data = np.abs(temp_stc._data)

            # Also morph to fsaverage if not using spherical head inverse
            # (if spherical, head model source space IS fsaverage)
            if inv_model != 'sphere':
                morphed_stc = mne.morph_data_precomputed(
                    subject_from=subj_d[i]['Struct'], subject_to='fsaverage',
                    stc_from=temp_stc, vertices_to=fsaverage['vertices'],
                    morph_mat=subj_d[i]['fs_morph'])

                # Save each morphed stc file
                save_fname = op.join(save_folder_morph, subj_d[i]['Subj#'] +
                                     '_' + p + '_' + inv_method +
                                     '_FsMorphed_bin' + str(bin_width) + '_Abs'
                                     + str(takeMagOfVals) + '_t' + str(ti))
                morphed_stc.save(save_fname, ftype=ftype, verbose=False)

    print ' ...  Done'
