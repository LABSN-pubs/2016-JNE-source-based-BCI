"""
switchPredSensLoop_all.py

@Author: wronk

Predict attentional switching from activity in sensor space
using EEG data. Iterate over a number of possible parameters
to optimize. Runs with PCA/ICA/CSP
"""

from os import environ, path as op
from time import strftime

import csv
import cPickle
import numpy as np

import mne
from mne.io import RawArray
from config import DATA_PARAMS, CLSF_PARAMS, CSP_PARAMS, ICA_PARAMS, PCA_PARAMS
from switchPredFun import bin_sens

# Choices for decomposition method used in analysis
decomp = 'ICA'  # Choose 'PCA', 'ICA', or 'CSP'
save_data = False

debug = False  # Run with simpler parameters to speed up testing


def roll_raw(data, info):
        """Aux function to rollaxis on data and create raw object."""
        data_rolled = np.rollaxis(data, axis=1)
        data_rolled_shape = data_rolled.shape
        data_raw = data_rolled.reshape(data_rolled.shape[0], -1)
        raw = RawArray(data_raw, info, False)

        return raw, data_rolled_shape

########################################
# Load subject information
########################################
# List of subjects to run analysis on
subj_list = ['AKCLEE_107', 'AKCLEE_102', 'AKCLEE_103', 'AKCLEE_104',
             'AKCLEE_105', 'AKCLEE_106', 'AKCLEE_108', 'AKCLEE_110',
             'AKCLEE_113', 'AKCLEE_114']

#subj_list = ['AKCLEE_102']  # Run with just one subj for testing

roi = DATA_PARAMS['roi']
bin_width = DATA_PARAMS['bin_width']
useAbsInMorph = DATA_PARAMS['useAbsInMorph']
patterns = DATA_PARAMS['patterns']
hemi_idx = DATA_PARAMS['hemi_idx']
hemi = ['lh', 'rh'][hemi_idx]

cache_dir = op.join(environ['CODE_ROOT'], 'switchBCIFiles')
subjListFName = op.join(cache_dir, 'SoP_MEG.txt')
struct_dir = op.join(environ['SUBJECTS_DIR'])
data_save_dir = op.join(cache_dir, 'PickledData', 'SensPred_ParamLoop')
epo_file_end = op.join('epochs', 'epoch_mats_eeg_unprocessed')
save_data = True

# Open/read subject info file into a list of dicts
with open(subjListFName, 'rb') as f:
    subj_d = list(csv.DictReader(f, dialect='excel-tab', delimiter='\t'))

# Get subject indices in the dict list
subj_inds = [i for i, d in enumerate(subj_d) if d['Struct'] in subj_list]

# Order should be preserved but check periodically
foldNames = [d['Subj#'] for d in [subj_d[i] for i in subj_inds]]

#Load all epochs and subject information
print 'Loading struct info for:'
for i, f in zip(subj_inds, foldNames):
    print '\t' + f,

    subj_d[i]['num'] = (f.split('_')[-1])

    print ' ...  Done'

########################################################
# Load sensor estimates
########################################################
p_bar = mne.utils.ProgressBar(max_value=len(subj_inds), spinner=True,
                              mesg='Subject:')
subj_trial_nums = []

print 'Loading pre-computed sensor data arrays for:'
for si, di in enumerate(subj_inds):
    subj_d[di]['epo_mat'] = {}  # Contains epochs object

    for pi, p in enumerate(patterns):

        load_fname = op.join(cache_dir, subj_d[di]['Subj#'], epo_file_end,
                             p + '.npy')
        orig_mat = np.load(load_fname)

        # Get trials per sub (patterns are equalized, so same # of trials per
        # cond.)
        if pi == 0:
            subj_trial_nums.append(orig_mat.shape[0])

        if bin_width != 0.001:
            subj_d[di]['epo_mat'][p] = bin_sens(orig_mat, bin_width)
        else:
            subj_d[di]['epo_mat'][p] = orig_mat

    p_bar.update(di, mesg='Subject ind: ' + str(di))

print '\nTrials per subject: ' + str(subj_trial_nums)
###########################################################################
# Classify
###########################################################################
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA

from mne.decoding import CSP
from mne.preprocessing import ICA
from mne import create_info
from switchPredFun import cross_val_pca, cross_val_ica, cross_val_csp

# Bashashati et al., 2007 has a review of common decomp techniques

t_wind_list = CLSF_PARAMS['t_wind_list']
t_wind_binned = [[int(round(l[0] / bin_width)), int(round(l[1] / bin_width))]
                 for l in t_wind_list]
CLSF_PARAMS.update(t_wind_binned=t_wind_binned)

debug = False
if debug:
    print 'RUNNING IN DEBUG MODE'
    if len(subj_list) > 3:
        print '\nAre you sure you want to run in debug mode???\n'
    t_wind_list = [(2.5, 3.2)]
    t_wind_list = [[int(round(l[0] / bin_width)), int(round(l[1] / bin_width))]
                   for l in t_wind_list]
    reg_list = ['oas']
    n_components_list = [4]
    C_range = 10. ** np.arange(-2, 2)
    g_range = 10. ** np.arange(-2, 3)
    save_data = True


def pca_routine(params, subj_inds, subj_d):
    """Loop over all classification parameters using PCA"""

    # Create matrices to hold scores and variance explained
    scores = np.empty((len(t_wind_list), len(params['n_components_list']),
                       len(subj_inds), params['cvs'],
                       len(params['kernel_list']), len(params['C_range']),
                       len(params['g_range'])))
    pca_var_expl = np.empty((len(params['t_wind_list']),
                             len(params['n_components_list']), len(subj_inds),
                             params['cvs'], max(params['n_components_list'])))
    scores[:] = np.nan
    pca_var_expl[:] = np.nan
    for t_wind_i, t_wind in enumerate(params['t_wind_binned']):
        print '\t' + 'Time window: ' + str(t_wind),

        for comp_i, n_components in enumerate(params['n_components_list']):

            pca = PCA(n_components=n_components)
            for si, di in enumerate(subj_inds):
                # Pull data structures from dict
                temp_m = subj_d[di]['epo_mat'][patterns[0]][:, :, t_wind[0]:t_wind[1]]
                temp_s = subj_d[di]['epo_mat'][patterns[1]][:, :, t_wind[0]:t_wind[1]]
                temp_m = temp_m.reshape(temp_m.shape[0], -1)
                temp_s = temp_s.reshape(temp_s.shape[0], -1)

                # Construct train and test datasets
                data = np.concatenate((temp_m, temp_s), axis=0)
                target = np.concatenate((np.ones(temp_m.shape[0]) * -1,
                                        np.ones(temp_s.shape[0])), axis=0)

                # Set up cross-validation
                cv = StratifiedKFold(target, n_folds=params['cvs'])

                for ti, (train_idx, test_idx) in enumerate(cv):
                    # Create training and test sets
                    data_dict = dict(X_train=data[train_idx],
                                     X_test=data[test_idx],
                                     y_train=target[train_idx],
                                     y_test=target[test_idx])

                    # Compute scores over given SVM parameters and store
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
    scores[:] = np.nan
    for t_wind_i, t_wind in enumerate(params['t_wind_binned']):
        print '\t' + 'Time window: ' + str(t_wind),

        for comp_i, var_explained in enumerate(params['n_components_list']):

            ica = ICA(n_components=var_explained,
                      method=ICA_PARAMS['ica_init_params']['method'],
                      max_iter=ICA_PARAMS['ica_init_params']['max_iter'])
            for si, di in enumerate(subj_inds):
                # Pull data structures from dict
                maint_all = subj_d[di]['epo_mat'][patterns[0]][:, :, t_wind[0]:t_wind[1]]
                switch_all = subj_d[di]['epo_mat'][patterns[1]][:, :, t_wind[0]:t_wind[1]]
                '''
                temp_m = maint_all.reshape(maint_all.shape[0], -1)
                temp_s = switch_all.reshape(switch_all.shape[0], -1)

                data = np.concatenate((temp_m, temp_s), axis=0)
                '''
                target = np.concatenate((np.ones(maint_all.shape[0]) * -1,
                                        np.ones(switch_all.shape[0])), axis=0)

                # Construct dummy Raw object to use MNE's ICA computations
                info = create_info(ch_names=subj_d[di]['epo_mat'][patterns[0]].shape[1],
                                   sfreq=1. / bin_width, ch_types='eeg')
                info['bads'] = []

                # Set up cross-validation
                cv = StratifiedKFold(target, n_folds=params['cvs'])

                for ti, (train_idx, test_idx) in enumerate(cv):

                    # Refactor training data to (n_chan x n_features) for ICA
                    data_train = np.concatenate((maint_all, switch_all))[train_idx, :, :]
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

                    # Compute scores over given SVM parameters and store
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
                    temp_m = subj_d[di]['epo_mat'][patterns[0]][:, :, t_wind[0]:t_wind[1]]
                    temp_s = subj_d[di]['epo_mat'][patterns[1]][:, :, t_wind[0]:t_wind[1]]

                    # Construct train and test datasets
                    data = np.concatenate((temp_m, temp_s), axis=0)
                    target = np.concatenate((np.ones(temp_m.shape[0]) * -1,
                                            np.ones(temp_s.shape[0])), axis=0)

                    # Set up cross-validation
                    cv = StratifiedKFold(target, n_folds=params['cvs'])

                    for ti, (train_idx, test_idx) in enumerate(cv):
                        # Create training and test sets
                        data_dict = dict(X_train=data[train_idx],
                                         X_test=data[test_idx],
                                         y_train=target[train_idx],
                                         y_test=target[test_idx])

                        # Compute scores over given SVM parameters and store
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
print ('\nClassifying using ' + decomp + ' (@ ' + strftime('%D %H:%M:%S') +
       ')')
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

else:
    raise ValueError('Decomposition method not valid')

##############
# Saving data
##############
scoreDict = dict(scores=scores,
                 subj_list=subj_list,
                 n_components_list=all_params['n_components_list'],
                 t_wind_list=t_wind_list,
                 t_wind_binned=t_wind_binned,
                 subj_inds=subj_inds,
                 C_range=all_params['C_range'],
                 g_range=all_params['g_range'],
                 kernel_list=all_params['kernel_list'],
                 finish_time=strftime('%D %H:%M:%S'))

# Update information to be saved and generate file name based on decomp
if decomp == 'PCA':
    order_list = ['Time window', 'Number of components', 'Subject index',
                  'Cross-validation index', 'Kernel index', 'C-val index',
                  'Gamma val index']

    fname_save = op.join(data_save_dir, 'Space', 'pca',
                         'scoreDict_spaceSens_pca_' +
                         str(int(bin_width * 1000)) + 'ms.pkl')
elif decomp == 'ICA':
    order_list = ['Time window', 'Number of components', 'Subject index',
                  'Cross-validation index', 'Kernel index', 'C-val index',
                  'Gamma val index']
    fname_save = op.join(data_save_dir, 'Space', 'ica',
                         'scoreDict_spaceSens_ica_' +
                         ICA_PARAMS['ica_init_params']['method'] + '_' +
                         str(int(bin_width * 1000)) + 'ms.pkl')
elif decomp == 'CSP':
    order_list = ['Time window', 'Regularization', 'Number of components',
                  'Subject index', 'Cross-validation index', 'Kernel index',
                  'C-val index', 'Gamma val index']
    scoreDict.update(reg_list=all_params['reg_list'])

    fname_save = op.join(data_save_dir, 'Space', 'csp',
                         'scoreDict_spaceSens_csp_' +
                         str(int(bin_width * 1000)) + 'ms.pkl')
scoreDict.update(score_arr_order=order_list)

# Save pickled data
if save_data:
    pkl_file = open(fname_save, 'wb')
    cPickle.dump(scoreDict, pkl_file)
    pkl_file.close()

print 'Finished ' + decomp + ' processing @ ' + strftime('%D %H:%M:%S')
