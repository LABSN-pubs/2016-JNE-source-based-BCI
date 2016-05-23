"""
switchPredSrcLoop_all.py

@Author: wronk

Predict attentional switching from activity in source space
using stc data with mean. Iterate over a number of possible parameters
to optimize and implements PCA, ICA, CSP, and mean.

Data use in Figure 5
"""

import os
from os import environ, path as op
from time import strftime

import numpy as np
import cPickle
import csv
import fnmatch

import mne
from mne.io import RawArray
from config import (DATA_PARAMS, CLSF_PARAMS, CSP_PARAMS, ICA_PARAMS,
                    PCA_PARAMS, MEAN_PARAMS)


def roll_raw(data, info):
        """Aux function to rollaxis on data and create raw object."""
        data_rolled = np.rollaxis(data, axis=1)
        data_rolled_shape = data_rolled.shape
        data_raw = data_rolled.reshape(data_rolled.shape[0], -1)
        raw = RawArray(data_raw, info, False)

        return raw, data_rolled_shape

# Choices for decomposition method used in analysis

decomp = 'PCA'  # Choose 'Mean', 'PCA', 'ICA', or 'CSP'
spherical_inv = False  # Whether or not to use a spherical inverse model
save_data = False

debug = False  # Run with simpler parameters to speed up testing
########################################
# Load subject information
########################################
# List of subjects to run analysis on
subj_list = ['AKCLEE_107', 'AKCLEE_102', 'AKCLEE_103', 'AKCLEE_104',
             'AKCLEE_105', 'AKCLEE_106', 'AKCLEE_108', 'AKCLEE_110',
             'AKCLEE_113', 'AKCLEE_114']
#subj_list = ['AKCLEE_102']  # Run just one subject

sph_fname = '_sph' if spherical_inv else ''

roi = DATA_PARAMS['roi']
bin_width = DATA_PARAMS['bin_width']
useAbsInMorph = DATA_PARAMS['useAbsInMorph']
patterns = DATA_PARAMS['patterns']
hemi_idx = DATA_PARAMS['hemi_idx']
hemi = ['lh', 'rh'][hemi_idx]

subj_info_dir = op.join(environ['CODE_ROOT'], 'switchBCIFiles')
subjListFName = op.join(subj_info_dir, 'SoP_MEG.txt')
struct_dir = op.join(environ['SUBJECTS_DIR'])
data_save_dir = op.join(subj_info_dir, 'PickledData', 'SrcPred_ParamLoop')

# Open/read subject info file into a list of dicts
with open(subjListFName, 'rb') as f:
    subj_d = list(csv.DictReader(f, dialect='excel-tab', delimiter='\t'))

# Get subject indices in the dict list
subj_inds = [i for i, d in enumerate(subj_d) if d['Struct'] in subj_list]

########################################
# Load subject information
########################################

# Load fsaverage data as it will be used for morphing
fsaverage = {}
fsaverage['src'] = mne.read_source_spaces(op.join(struct_dir, 'fsaverage',
                                                  'bem',
                                                  'fsaverage-5p7-src.fif'),
                                          verbose=False)
fsaverage['vertices'] = [fsaverage['src'][0]['vertno'],
                         fsaverage['src'][1]['vertno']]
fs_roi_fname = op.join(struct_dir, 'fsaverage', 'label', roi)
fsaverage['lab'] = mne.read_label(fs_roi_fname, 'fsaverage')

foldNames = [d['Subj#'] for d in [subj_d[i] for i in subj_inds]]

#Load all forward, inverse, and epochs
print 'Loading struct/label info for:'
for i, f in zip(subj_inds, foldNames):
    print '\t' + f,

    label_fname = op.join(struct_dir, subj_d[i]['Struct'], 'label', roi)

    subj_d[i]['lab'] = mne.read_label(label_fname, subject=subj_d[i]['Struct'])
    subj_d[i]['num'] = (f.split('_')[-1])

    subj_d[i]['lab_vertno'] = fsaverage['lab'].get_vertices_used()

    print ' ...  Done'

########################################################
# Load source estimates and calculate label time courses
########################################################

print '\nMethod: ' + decomp
print 'Label: ' + fsaverage['lab'].name
print 'Spherical head model: ' + str(spherical_inv)
print 'Loading source estimates:'
for di in subj_inds:
    print '\t' + subj_d[di]['Subj#'],
    subj_d[di]['act_array'] = {}  # Contains all extracted time courses

    for pi, p in enumerate(patterns):

        # Construct pattern for filenames containing the source data
        # There are a number of ways to construct the source data, so we need
        #     to specify which we want (via global params)
        if spherical_inv:
            load_folder = op.join(subj_info_dir, subj_d[di]['Subj#'], 'stc',
                                  p + '_unprocessed' + sph_fname + '/')
            str_pattern = (subj_d[di]['Subj#'] + '_' + p + '_MNE_t')
        else:
            load_folder = op.join(subj_info_dir, subj_d[di]['Subj#'], 'stc',
                                  p + '_morphedToFsaverage' + sph_fname + '/')
            str_pattern = (subj_d[di]['Subj#'] + '_' + p + '_MNE_' +
                           'LRRon-LRStd_01-rh' + '_FsMorphed_binNone' +
                           '_Abs' + str(useAbsInMorph) + '_t')
        stcDataList = []

        # Find all file matching the filename pattern
        f_names = fnmatch.filter(os.listdir(load_folder), str_pattern + '*')
        p_bar = mne.utils.ProgressBar(max_value=len(f_names) / 2 - 1,
                                      spinner=True, mesg='Subject ' + str(di) +
                                      '; Pattern: ' + p)

        # Load source data from each trial and store
        for ti in range(len(f_names) / 2):  # both lh and rh stcs
            load_fname = op.join(load_folder, str_pattern + str(ti) + '-' +
                                 hemi + '.stc')

            stc = mne.read_source_estimate(load_fname)

            if hemi_idx == 0:
                stcDataList.append(stc.bin(bin_width).lh_data[subj_d[di]['lab_vertno'], :])
            else:
                stcDataList.append(stc.bin(bin_width).rh_data[subj_d[di]['lab_vertno'], :])
            p_bar.update(ti)

        subj_d[di]['act_array'][p] = np.array(stcDataList)
        print ''
    print 'Done'

###########################################################################
# Classify
###########################################################################
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA

from mne.decoding import CSP
from mne.preprocessing import ICA
from mne import create_info
from switchPredFun import (cross_val_pca, cross_val_ica, cross_val_csp,
                           cross_val_mean)

t_wind_list = CLSF_PARAMS['t_wind_list']
t_wind_binned = [[int(round(l[0] / bin_width)), int(round(l[1] / bin_width))]
                 for l in t_wind_list]
CLSF_PARAMS.update(t_wind_binned=t_wind_binned)

if debug:
    if len(subj_list) > 3:
        print '\nAre you sure you want to run in debug mode???\n'
    t_wind_list = [(2.9, 3.5)]
    t_wind_binned = [[int(round(l[0] / bin_width)), int(round(l[1] / bin_width))]
                     for l in t_wind_list]
    reg_list = [0.05]
    n_components_list = [3]
    C_range = 10. ** np.arange(-4, 5)
    g_range = 10. ** np.arange(-7, 3)
    save_data = False

######################################################
# Decomposition specific parameters for classification
######################################################


def mean_routine(params, subj_inds, subj_d):
    """Loop over all classification parameters using mean averaging"""

    # Create matrices to hold scores and variance explained
    scores = np.empty((len(params['t_wind_binned']), len(subj_inds),
                       params['cvs'], len(params['kernel_list']),
                       len(params['C_range']), len(params['g_range'])))

    # Iterate over time window, classifier params, subject
    for t_wind_i, t_wind in enumerate(params['t_wind_binned']):
        print '\t' + 'Time window: ' + str(t_wind),

        for si, di in enumerate(subj_inds):
            # Pull data structures from dict and take mean
            temp_m = subj_d[di]['act_array'][patterns[0]][:, :, t_wind[0]:t_wind[1]].mean(2)
            temp_s = subj_d[di]['act_array'][patterns[1]][:, :, t_wind[0]:t_wind[1]].mean(2)

            # Get data and trial labels
            data = np.concatenate((temp_m, temp_s), axis=0)
            target = np.concatenate((np.ones(temp_m.shape[0]) * -1,
                                    np.ones(temp_s.shape[0])), axis=0)

            # Prepare cross-validation
            cv = StratifiedKFold(target, n_folds=params['cvs'])

            for ti, (train_idx, test_idx) in enumerate(cv):
                # Create training and test sets
                data_dict = dict(X_train=data[train_idx],
                                 X_test=data[test_idx],
                                 y_train=target[train_idx],
                                 y_test=target[test_idx])

                # Compute scores over given SVM parameters and store
                try:
                    scores[t_wind_i, si, ti, :, :, :] = \
                        cross_val_mean(data_dict, params['kernel_list'],
                                       params['C_range'], params['g_range'])
                except:
                    'Error raised\nsubj: ' + str(di)
                    raise

        print ' ...  Done.'
    return scores


def pca_routine(params, subj_inds, subj_d):
    """Loop over all classification parameters using PCA"""

    # Create matrices to hold scores and variance explained
    scores = np.empty((len(params['t_wind_binned']),
                       len(params['n_components_list']), len(subj_inds),
                       params['cvs'], len(params['kernel_list']),
                       len(params['C_range']), len(params['g_range'])))
    pca_var_expl = np.empty((len(params['t_wind_binned']),
                             len(params['n_components_list']), len(subj_inds),
                             params['cvs'], len(params['n_components_list'])))
    pca_var_expl[:] = np.nan

    # Iterate over time window, classifier params, subject
    for t_wind_i, t_wind in enumerate(params['t_wind_binned']):
        print '\t' + 'Time window: ' + str(t_wind),

        for comp_i, n_components in enumerate(params['n_components_list']):
            pca = PCA(n_components, whiten=True)
            for si, di in enumerate(subj_inds):
                # Pull data structures from dict
                temp_m = subj_d[di]['act_array'][patterns[0]][:, :, t_wind[0]:t_wind[1]]
                temp_s = subj_d[di]['act_array'][patterns[1]][:, :, t_wind[0]:t_wind[1]]
                temp_m = temp_m.reshape(temp_m.shape[0], -1)
                temp_s = temp_s.reshape(temp_s.shape[0], -1)

                #temp_m = subj_d[di]['act_array'][patterns[0]][:, :, t_wind[0]:t_wind[1]].mean(1)
                #temp_s = subj_d[di]['act_array'][patterns[1]][:, :, t_wind[0]:t_wind[1]].mean(1)

                # Get data and trial labels
                data = np.concatenate((temp_m, temp_s), axis=0)
                target = np.concatenate((np.ones(temp_m.shape[0]) * -1,
                                        np.ones(temp_s.shape[0])), axis=0)

                # Prepare cross-validation
                cv = StratifiedKFold(target, n_folds=params['cvs'])

                for ti, (train_idx, test_idx) in enumerate(cv):
                    # Create training and test sets
                    data_dict = dict(X_train=data[train_idx],
                                     X_test=data[test_idx],
                                     y_train=target[train_idx],
                                     y_test=target[test_idx])

                    # Compute scores over given SVM parameters
                    try:
                        scores[t_wind_i, comp_i, si, ti, :, :, :] = \
                            cross_val_pca(data_dict, pca,
                                          params['kernel_list'],
                                          params['C_range'], params['g_range'])
                        pca_var_expl[t_wind_i, comp_i, si, ti, 0:n_components] = \
                            pca.explained_variance_ratio_
                    except:
                        'Error raised\nsubj: ' + str(di)
                        raise

        print ' ...  Done.'
    return scores, pca_var_expl


def ica_routine(params, subj_inds, subj_d):
    """Loop over all classification parameters using extended infomax ICA"""

    scores = np.empty((len(params['t_wind_binned']),
                       len(params['n_components_list']), len(subj_inds),
                       params['cvs'], len(params['kernel_list']),
                       len(params['C_range']), len(params['g_range'])))

    # Iterate over time window,num ICA components, subject, cross-val
    for t_wind_i, t_wind in enumerate(params['t_wind_binned']):
        print '\t' + 'Time window: ' + str(t_wind),

        for comp_i, n_components in enumerate(params['n_components_list']):

            ica = ICA(n_components=n_components,
                      method=params['ica_init_params']['method'],
                      max_iter=params['ica_init_params']['max_iter'])
            for si, di in enumerate(subj_inds):
                # Pull data structures from dict
                maint_all = subj_d[di]['act_array'][patterns[0]][:, :, t_wind[0]:t_wind[1]]
                switch_all = subj_d[di]['act_array'][patterns[1]][:, :, t_wind[0]:t_wind[1]]
                # Reshape to (n_trials x n_features)
                #temp_m = temp_m.reshape(maint_all.shape[0], -1)
                #temp_s = temp_s.reshape(switch_all.shape[0], -1)

                # Take mean across space
                #maint_all = subj_d[di]['act_array'][patterns[0]][:, :, t_wind[0]:t_wind[1]].mean(1)
                #switch_all = subj_d[di]['act_array'][patterns[1]][:, :, t_wind[0]:t_wind[1]].mean(1)

                # Get data and trial labels
                #data = np.concatenate((temp_m, temp_s), axis=0)
                target = np.concatenate((np.ones(maint_all.shape[0]) * -1,
                                        np.ones(switch_all.shape[0])), axis=0)

                # Construct dummy Raw object to link into MNE's ICA computations
                info = create_info(ch_names=subj_d[di]['act_array'][patterns[0]].shape[1],
                                   sfreq=1. / bin_width, ch_types='eeg')

                info['bads'] = []

                # Prepare cross-validation
                cv = StratifiedKFold(target, n_folds=params['cvs'])

                for ti, (train_idx, test_idx) in enumerate(cv):

                    # Refactor training data to (n_chan x n_features) for ICA
                    data_train = np.concatenate((maint_all,
                                                 switch_all))[train_idx, :, :]
                    raw_train, train_shape = roll_raw(data_train, info)

                    # Fit ICA to training data
                    ica.fit(raw_train, verbose=False)

                    ###########################################################
                    # Construct raw from reshaped data
                    all_data = np.concatenate((maint_all, switch_all), axis=0)
                    raw_all, all_data_shape = roll_raw(all_data, info)

                    # Transform training and testing data
                    all_data_trans = ica._transform_raw(raw_all, None, None)
                    # Un-reshape training and testing data
                    all_data_trans = \
                        all_data_trans.reshape((all_data_trans.shape[0],
                                                all_data_shape[1],
                                                all_data_shape[2]))
                    all_data_trans = np.rollaxis(all_data_trans, axis=1)

                    # Formally construct training and testing data
                    # Reshape to (n_trials x n_features)
                    data_train_trans = all_data_trans[train_idx, :, :]
                    data_train_trans = data_train_trans.reshape(data_train_trans.shape[0], -1)
                    data_test_trans = all_data_trans[test_idx, :, :]
                    data_test_trans = data_test_trans.reshape(data_test_trans.shape[0], -1)

                    # Create training and test sets
                    data_dict = dict(X_train=data_train_trans,
                                     X_test=data_test_trans,
                                     y_train=target[train_idx],
                                     y_test=target[test_idx])

                    # Compute scores over given SVM parameters
                    try:
                        scores[t_wind_i, comp_i, si, ti, :, :, :] = \
                            cross_val_ica(data_dict, params['kernel_list'],
                                          params['C_range'], params['g_range'])
                    except:
                        'Error raised\nsubj: ' + str(di)
                        raise

        print ' ...  Done.'
    return scores


def csp_routine(params, subj_inds, subj_d):
    """Loop over all classification parameters using CSP"""

    scores = np.empty((len(params['t_wind_binned']), len(params['reg_list']),
                       len(params['n_components_list']), len(subj_inds),
                       params['cvs'], len(params['kernel_list']),
                       len(params['C_range']), len(params['g_range'])))

    # Iterate over time window, classifier params, subject
    for t_wind_i, t_wind in enumerate(params['t_wind_binned']):
        print '\t' + 'Time window: ' + str(t_wind),
        for reg_i, reg in enumerate(params['reg_list']):
            for comp_i, n_components in enumerate(params['n_components_list']):
                csp = CSP(n_components=n_components, reg=reg, log=False)
                for si, di in enumerate(subj_inds):
                    # Pull data structures from dict
                    temp_m = subj_d[di]['act_array'][patterns[0]][:, :, t_wind[0]:t_wind[1]]
                    temp_s = subj_d[di]['act_array'][patterns[1]][:, :, t_wind[0]:t_wind[1]]

                    # Get data and trial labels
                    data = np.concatenate((temp_m, temp_s), axis=0)
                    target = np.concatenate((np.ones(temp_m.shape[0]) * -1,
                                            np.ones(temp_s.shape[0])), axis=0)

                    # Prepare cross-validation
                    cv = StratifiedKFold(target, n_folds=params['cvs'])

                    for ti, (train_idx, test_idx) in enumerate(cv):
                        # Create training and test sets
                        data_dict = dict(X_train=data[train_idx],
                                         X_test=data[test_idx],
                                         y_train=target[train_idx],
                                         y_test=target[test_idx])

                        # Compute scores over given SVM parameters
                        try:
                            scores[t_wind_i, reg_i, comp_i, si, ti, :, :, :] = \
                                cross_val_csp(data_dict, csp,
                                              params['kernel_list'],
                                              params['C_range'],
                                              params['g_range'])
                        except:
                            print('Error raised\nsubj: ' + str(di) +
                                  '\nreg: ' + str(reg))
                            raise
        print ' ...  Done.'
    return scores


all_params = CLSF_PARAMS.copy()
print ('Classifying using ' + decomp + ' (@ ' + strftime('%D %H:%M:%S') + ')')
print 'Abs_value: ' + str(useAbsInMorph)
print 'C_range' + str(all_params['C_range'])
print 'g_range' + str(all_params['g_range']) + '\n'

# Update parameter dict and compute scores for each decomposition method
if decomp == 'PCA':
    all_params.update(PCA_PARAMS)
    scores, pca_var_expl = pca_routine(all_params, subj_inds, subj_d)

elif decomp == 'ICA':
    all_params.update(ICA_PARAMS)
    scores = ica_routine(all_params, subj_inds, subj_d)

elif decomp == 'CSP':
    all_params.update(CSP_PARAMS)
    scores = csp_routine(all_params, subj_inds, subj_d)

elif decomp == 'Mean':
    all_params.update(MEAN_PARAMS)
    scores = mean_routine(all_params, subj_inds, subj_d)

else:
    raise ValueError('Decomposition method not valid')


##############
# Saving data
##############
anatom_add_on = '_AnatomRTPJ' if 'RTPJ' in roi else ''
scoreDict = dict(scores=scores,
                 subj_list=subj_list,
                 t_wind_list=t_wind_list,
                 t_wind_binned=t_wind_binned,
                 subj_inds=subj_inds,
                 C_range=all_params['C_range'],
                 g_range=all_params['g_range'],
                 kernel_list=all_params['kernel_list'],
                 finish_time=strftime('%D %H:%M:%S'))

# Update information to be saved and generate file name based on decomp
if decomp == 'Mean':
    order_list = ['Time window', 'Subject index', 'Cross-validation index',
                  'Kernel index', 'C-val index', 'Gamma val index']

    fname_save = op.join(data_save_dir, 'Space', 'mean',
                         'scoreDict_spaceMorphed_mean' + anatom_add_on +
                         sph_fname + '_' + str(int(bin_width * 1000)) +
                         'ms.pkl')
if decomp == 'PCA':
    order_list = ['Time window', 'Number of components', 'Subject index',
                  'Cross-validation index', 'Kernel index', 'C-val index',
                  'Gamma val index']

    scoreDict.update(n_components_list=all_params['n_components_list'])
    fname_save = op.join(data_save_dir, 'Space', 'pca',
                         'scoreDict_spaceMorphed_pca' + anatom_add_on +
                         sph_fname + '_' + str(int(bin_width * 1000)) +
                         'ms.pkl')
elif decomp == 'ICA':
    order_list = ['Time window', 'Number of components', 'Subject index',
                  'Cross-validation index', 'Kernel index', 'C-val index',
                  'Gamma val index']
    scoreDict.update(n_components_list=all_params['n_components_list'])
    fname_save = op.join(data_save_dir, 'Space', 'ica',
                         'scoreDict_spaceMorphed_ica' + anatom_add_on +
                         sph_fname + '_' +
                         ICA_PARAMS['ica_init_params']['method'] + '_' +
                         str(int(bin_width * 1000)) + 'ms.pkl')
elif decomp == 'CSP':
    order_list = ['Time window', 'Regularization', 'Number of components',
                  'Subject index', 'Cross-validation index', 'Kernel index',
                  'C-val index', 'Gamma val index']
    scoreDict.update(reg_list=all_params['reg_list'],
                     n_components_list=all_params['n_components_list'])

    fname_save = op.join(data_save_dir, 'Space', 'csp',
                         'scoreDict_spaceMorphed_csp' + anatom_add_on +
                         sph_fname + '_' + str(int(bin_width * 1000)) +
                         'ms.pkl')
scoreDict.update(score_arr_order=order_list)

# Save pickled data
if save_data:
    pkl_file = open(fname_save, 'wb')
    cPickle.dump(scoreDict, pkl_file)
    pkl_file.close()

print 'Finished ' + decomp + ' processing @ ' + strftime('%D %H:%M:%S')
