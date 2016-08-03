"""
switchPredSim_v1.py

@Author: wronk

Predict attentional switching from simulated activity in source space.

Implements PCA, ICA, and CSP
"""

import os
from os import environ, path as op
from time import strftime
from copy import deepcopy

import numpy as np
import cPickle
import csv
import fnmatch

from scipy.sparse import csr_matrix as csr_mat
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA

import mne
from mne import read_forward_solution as read_fwd, read_epochs, create_info
from mne.minimum_norm import (read_inverse_operator as read_inv,
                              apply_inverse_raw)
from mne.simulation import simulate_stc, simulate_raw
from mne.decoding import CSP
from mne.preprocessing import ICA
from mne.io import RawArray

from config import (DATA_PARAMS, CLSF_PARAMS, CSP_PARAMS, ICA_PARAMS,
                    PCA_PARAMS, MEAN_PARAMS)
from switchPredFun import (cross_val_pca, cross_val_ica, cross_val_csp,
                           cross_val_mean, bin_sens, load_fsaverage_dict,
                           get_noise_scale_factor)

# Choices for decomposition method used in analysis
decomp_list = ['pca', 'ica', 'csp']  # Choose 'pca', 'ica', or 'csp'
save_data = True
debug = False  # Run with simpler parameters to speed up testing

SNR = -10
n_jobs = 6
n_trials = 40
trial_len_ms = 600
roi = DATA_PARAMS['roi']
bin_width = DATA_PARAMS['bin_width']
useAbsInMorph = DATA_PARAMS['useAbsInMorph']
patterns = DATA_PARAMS['patterns']
hemi_idx = DATA_PARAMS['hemi_idx']
hemi = ['lh', 'rh'][hemi_idx]

########################################
# Load subject information
########################################
# List of subjects to run analysis on
subj_list = ['AKCLEE_107', 'AKCLEE_102', 'AKCLEE_103', 'AKCLEE_104',
             'AKCLEE_105', 'AKCLEE_106', 'AKCLEE_108', 'AKCLEE_110',
             'AKCLEE_113', 'AKCLEE_114']
#subj_list = ['AKCLEE_113', 'AKCLEE_114']

subj_info_dir = op.join(environ['CODE_ROOT'], 'switchBCIFiles')
subjListFName = op.join(subj_info_dir, 'SoP_MEG.txt')
struct_dir = op.join(environ['SUBJECTS_DIR'])
data_save_dir = op.join(subj_info_dir, 'PickledData', 'SimPred')

with open(subjListFName, 'rb') as f:
    subj_d = list(csv.DictReader(f, dialect='excel-tab', delimiter='\t'))
# Get subject indices in the dict list
subj_inds = [i for i, d in enumerate(subj_d) if d['Struct'] in subj_list]
score_dicts = []

foldNames = [d['Subj#'] for d in [subj_d[i] for i in subj_inds]]
inverse_end = '-55-SSS-eeg-Fixed-inv.fif'
sph_inverse_end = '-SSS-eeg-sph-Fixed-inv.fif'
forward_end = '-SSS-fwd.fif'
epo_beg = 'All_55_sss_EQ_'
morph_end = '-fsaverage-morph.npz'

####################################
# Load all forward, inverse matrices
####################################
print 'Started simulation processing @ ' + strftime('%D %H:%M:%S') + '\n'
print 'Loading struct/label info for fsaverage:'
fsaverage = load_fsaverage_dict()
fs_roi_fname = op.join(struct_dir, 'fsaverage', 'label', roi)
fsaverage['lab'] = mne.read_label(fs_roi_fname, 'fsaverage')

print 'Loading struct/label info for:'
for ii, fold_name in zip(subj_inds, foldNames):
    print '\t%s\n' % fold_name,

    label_fname = op.join(struct_dir, subj_d[ii]['Struct'], 'label', roi)

    subj_d[ii]['num'] = (fold_name.split('_')[-1])
    subj_d[ii]['fwd'] = read_fwd(op.join(subj_info_dir, fold_name, 'forward',
                                         fold_name + forward_end),
                                 force_fixed=True, verbose=False)
    subj_d[ii]['inv'] = read_inv(op.join(subj_info_dir, fold_name, 'inverse',
                                         fold_name + inverse_end),
                                 verbose=False)
    subj_d[ii]['inv_sph'] = read_inv(op.join(subj_info_dir, fold_name,
                                             'inverse', fold_name +
                                             sph_inverse_end), verbose=False)
    morph_temp = np.load(op.join(subj_info_dir, 'morph_mats',
                                 subj_d[ii]['Struct'] + morph_end))
    subj_d[ii]['fs_morph'] = csr_mat((morph_temp['data'],
                                      morph_temp['indices'],
                                      morph_temp['indptr']))

    subj_d[ii]['bem_fname'] = op.join(struct_dir, subj_d[ii]['Struct'], 'bem',
                                      subj_d[ii]['Struct'] +
                                      '-5120-5120-5120-bem-sol.fif')

    temp_epo = read_epochs(op.join(subj_info_dir, fold_name, 'epochs',
                                   epo_beg + fold_name + '-epo.fif'),
                           preload=False, verbose=False)

    # Patch together `info` object from epo object
    info = deepcopy(temp_epo.info)
    info['sfreq'] = 1000.
    info = mne.pick_info(info, sel=mne.pick_types(info, meg=False, eeg=True))
    info['projs'] = [mne.preprocessing.ssp.make_eeg_average_ref_proj(info)]
    subj_d[ii]['info'] = info

    # Get label to use and restrict it to used vertices
    temp_lab = mne.read_label(label_fname, subject=subj_d[ii]['Struct'])
    verts_used = temp_lab.get_vertices_used(subj_d[ii]['inv']['src'][1]['vertno'])
    subj_d[ii]['lab'] = mne.Label(verts_used, hemi=temp_lab.hemi,
                                  name=temp_lab.name, subject=temp_lab.subject)

    print ' ...  Done'

# Simulate activity and classify
sim_times = np.arange(0, 600, 1)
lambda2 = 1. / 9
# XXX Double check if this is an appropriate amplitude
sim_amplitude = 1e-8
sim_fun = lambda x: sim_amplitude * np.ones((x))

p_bar = mne.utils.ProgressBar(max_value=max(1, len(subj_inds) - 1),
                              spinner=True, mesg='Simulation progress')


def get_noise_scale(eeg_est, s_dict):
    eeg_data = eeg_est[:, 0][0]  # Get data for all channels at time zero
    return get_noise_scale_factor(eeg_data, SNR, s_dict['inv']['noise_cov'])


def pca_routine(data_m, data_s, params):
    """Loop over all classification parameters using PCA"""

    # Create matrices to hold scores
    scores = np.empty((len(params['n_components_list']), params['cvs'],
                       len(params['kernel_list']), len(params['C_range']),
                       len(params['g_range'])))

    # Reshape to n_trials x n_features
    temp_m = data_m.reshape((data_m.shape[0], -1))
    temp_s = data_s.reshape((data_s.shape[0], -1))

    data = np.concatenate((temp_m, temp_s), axis=0)
    # Get data and trial labels
    target = np.concatenate((np.ones(temp_m.shape[0]) * -1,
                            np.ones(temp_s.shape[0])), axis=0)

    # Iterate over classifier params
    for comp_i, n_components in enumerate(params['n_components_list']):
        pca = PCA(n_components, whiten=True)

        # Prepare cross-validation
        cv = StratifiedKFold(target, n_folds=params['cvs'])

        for ti, (train_idx, test_idx) in enumerate(cv):
            # Create training and test sets
            data_dict = dict(X_train=data[train_idx],
                             X_test=data[test_idx],
                             y_train=target[train_idx],
                             y_test=target[test_idx])

            # Compute scores over given SVM parameters
            scores[comp_i, ti, :, :, :] = \
                cross_val_pca(data_dict, pca, params['kernel_list'],
                              params['C_range'], params['g_range'])

    return scores


def ica_routine(data_m, data_s, params):
    """Loop over all classification parameters using extended infomax ICA"""

    scores = np.empty((len(params['n_components_list']), params['cvs'],
                       len(params['kernel_list']), len(params['C_range']),
                       len(params['g_range'])))

    # Iterate over time window, ICA components variance, subject, cross-val
    for comp_i, var_explained in enumerate(params['n_components_list']):

        ica = ICA(n_components=var_explained,
                  method=params['ica_init_params']['method'],
                  max_iter=params['ica_init_params']['max_iter'])

        # Get data and trial labels
        target = np.concatenate((np.ones(data_m.shape[0]) * -1,
                                 np.ones(data_s.shape[0])), axis=0)

        # Construct dummy Raw object to link into MNE's ICA computations
        info = create_info(ch_names=data_m.shape[1], sfreq=1. / bin_width,
                           ch_types='eeg')
        info['bads'] = []

        # Prepare cross-validation
        cv = StratifiedKFold(target, n_folds=params['cvs'])

        for ti, (train_idx, test_idx) in enumerate(cv):

            # Refactor training data to (n_chan x n_features) for ICA
            data_train = np.concatenate((data_m, data_s), axis=0)[train_idx, :, :]
            data_train = np.rollaxis(data_train, 1)
            raw_train = RawArray(data_train.reshape(data_train.shape[0], -1),
                                 info, verbose=False)

            # Fit ICA to training data
            ica.fit(raw_train, verbose=False)

            ###########################################################
            # Construct raw from reshaped data
            all_data = np.rollaxis(np.concatenate((data_m, data_s), axis=0), 1)

            all_data_shape = all_data.shape
            raw_all = RawArray(all_data.reshape(all_data.shape[0], -1),
                               info, verbose=False)

            # Transform training and testing data
            all_data_trans = ica._transform_raw(raw_all, None, None)

            # Un-reshape training and testing data to (n_comp  x  n_trials  x  n_times)
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
            scores[comp_i, ti, :, :, :] = \
                cross_val_ica(data_dict, params['kernel_list'],
                              params['C_range'], params['g_range'])

    return scores


def csp_routine(data_m, data_s, params):
    """Loop over all classification parameters using CSP"""

    scores = np.empty((len(params['reg_list']),
                       len(params['n_components_list']),
                       params['cvs'], len(params['kernel_list']),
                       len(params['C_range']), len(params['g_range'])))

    # Reshape into n_trials x n_features
    temp_m = data_m.reshape((data_m.shape[0], -1))
    temp_s = data_s.reshape((data_s.shape[0], -1))

    # Get data and trial labels
    data = np.concatenate((temp_m, temp_s), axis=0)
    target = np.concatenate((np.ones(data_m.shape[0]) * -1,
                            np.ones(data_s.shape[0])), axis=0)

    # Iterate over classifier params
    for reg_i, reg in enumerate(params['reg_list']):
        for comp_i, n_components in enumerate(params['n_components_list']):
            csp = CSP(n_components=n_components, reg=reg, log=False)

            # Prepare cross-validation
            cv = StratifiedKFold(target, n_folds=params['cvs'])

            for ti, (train_idx, test_idx) in enumerate(cv):
                # Create training and test sets
                data_dict = dict(X_train=data[train_idx],
                                 X_test=data[test_idx],
                                 y_train=target[train_idx],
                                 y_test=target[test_idx])

                # Compute scores over given SVM parameters
                scores[reg_i, comp_i, ti, :, :, :] = \
                    cross_val_csp(data_dict, csp, ['rbf'], params['C_range'],
                                  params['g_range'])

    return scores


def gen_x_raw(n_trials, raw_template, stc_sim, s_dict):
    """Helper to simulate multiple trials of raw data"""

    scaled_cov = deepcopy(s_dict['inv']['noise_cov'])
    scaled_cov['data'] = scaled_cov['data'] * s_dict['noise_scale']

    # XXX: blink rate: 9 to 21.5 blinks/min (higher than found experimentally)
    return simulate_raw(raw_template, stc_sim, s_dict['inv']['mri_head_t'],
                        src=s_dict['inv']['src'], bem=s_dict['bem_fname'],
                        cov=scaled_cov, blink=True, n_jobs=n_jobs,
                        verbose=False)


def do_all_classification(data_s, data_m, params):
    """Helper to run classification using all dim. redux methods"""
    scores = []

    for dim_redux_meth in decomp_list:
        print '\tRunning classification with %s' % dim_redux_meth
        # Dynamically get classification function and associated parameters
        func = globals()[dim_redux_meth + '_routine']
        algo_specific_params = deepcopy(globals()[dim_redux_meth.upper() +
                                                  '_PARAMS'])
        algo_specific_params.update(params)
        scores.append(func(data_m, data_s, algo_specific_params))
        print '\t... Done'

    return scores

# Loop over each subject to avoid having to store all data in memory
for si, di in enumerate(subj_inds):
    s_dict = subj_d[di]
    src_fname = op.join(struct_dir, s_dict['Struct'], 'bem', s_dict['Struct'] +
                        '-oct-6-src.fif')

    # Set vertices used to those in the label
    #verts_used = subj_d[di]['lab'].get_vertices_used(
    #   s_dict['inv']['src'][1]['vertno'])
    verts_used = fsaverage['lab'].get_vertices_used()

    # Generate souce estimate
    stc_activation = sim_amplitude * np.ones((1, trial_len_ms * n_trials)) / \
        len(s_dict['lab'].vertices)
    # Generate a template raw object
    raw_template = mne.io.RawArray(np.zeros((len(s_dict['info']['chs']),
                                             stc_activation.shape[1])),
                                   info=s_dict['info'])

    # Get scalar for noise covariance to achieve desired SNR
    noiseless_act = np.ones((1, 3)) * sim_amplitude
    raw_template_noiseless = mne.io.RawArray(np.zeros((len(s_dict['info']['chs']),
                                                       noiseless_act.shape[1])),
                                             info=s_dict['info'])
    stc_noiseless = simulate_stc(subj_d[di]['inv']['src'], [subj_d[di]['lab']],
                                 stc_data=noiseless_act, tmin=0, tstep=0.001)
    eeg_noiseless = simulate_raw(raw_template_noiseless, stc_noiseless,
                                 s_dict['inv']['mri_head_t'],
                                 src=s_dict['inv']['src'],
                                 bem=s_dict['bem_fname'], cov=None,
                                 blink=True, n_jobs=n_jobs, verbose=False)

    s_dict['noise_scale'] = get_noise_scale(eeg_noiseless, s_dict)
    print 'Noise covariance scalar calculated: %s' % s_dict['noise_scale']

    # Simulate cortical activations.
    #    Maintain: Noise
    #    Switch:   Noise + [unit current / label area] in RTPJ
    raw_sim = []
    stc_est = []
    stc_est_sph = []
    for act_scale, trial_type in zip([0, 1], ['switch', 'maintain']):
        # Generate simulated stc activation
        stc_sim = simulate_stc(subj_d[di]['inv']['src'], [subj_d[di]['lab']],
                               stc_data=act_scale * stc_activation, tmin=0,
                               tstep=0.001)

        # Generate simulated raw data
        raw_temp = gen_x_raw(n_trials, raw_template, stc_sim, s_dict)

        # Calculate source estimates
        stc_est_sph.append(apply_inverse_raw(raw_temp, s_dict['inv_sph'],
                                             lambda2, 'MNE',
                                             label=fsaverage['lab']))
        stc_est_temp = apply_inverse_raw(raw_temp, s_dict['inv'], lambda2, 'MNE')
                                         #label=s_dict['lab'])

        # Need to morph to fsaverage to match spherical and morphed data
        stc_est.append(mne.morph_data_precomputed(
            subject_from=s_dict['Struct'], subject_to='fsaverage',
            stc_from=stc_est_temp, vertices_to=fsaverage['vertices'],
            morph_mat=s_dict['fs_morph']).in_label(fsaverage['lab']))

        # Resample and store
        # Use copy in raw_sim because of strange error with in-place resample
        raw_sim.append(raw_temp.resample(1. / bin_width, copy=True,
                                         n_jobs=n_jobs, verbose=False))
        stc_est[-1].resample(1. / bin_width, n_jobs=n_jobs, verbose=False)
        stc_est_sph[-1].resample(1. / bin_width, n_jobs=n_jobs, verbose=False)

    print '\t%s: Data simulated, source estimates computed' % s_dict['Subj#']

    ###########################################################################
    # Classify
    ###########################################################################

    # Construct reshaped np arrays from resampled data
    target_sens_shape = (len(s_dict['info']['chs']), n_trials,
                         int(trial_len_ms / bin_width / 1000.))
    target_src_shape = (len(verts_used), n_trials,
                        int(trial_len_ms / bin_width / 1000.))

    data_sens = [np.rollaxis(raw_sim[0][:][0].reshape(target_sens_shape), 1),
                 np.rollaxis(raw_sim[1][:][0].reshape(target_sens_shape), 1)]
    data_src_sph = [np.rollaxis(stc_est_sph[0].data.reshape(target_src_shape), 1),
                    np.rollaxis(stc_est_sph[1].data.reshape(target_src_shape), 1)]
    data_src = [np.rollaxis(stc_est[0].data.reshape(target_src_shape), 1),
                np.rollaxis(stc_est[1].data.reshape(target_src_shape), 1)]

    print '\t%s: Data reshaped, starting classification' % s_dict['Subj#']

    #Data should be n-trials x n_features
    sens_scores = do_all_classification(data_sens[0], data_sens[1],
                                        CLSF_PARAMS)
    print '\t%s: Sens classification complete, starting Src_sph classification' % \
        s_dict['Subj#']

    src_sph_scores = do_all_classification(data_src_sph[0], data_src_sph[1],
                                           CLSF_PARAMS)

    print '\t%s: Src_sph classification complete, starting src classification' % \
        s_dict['Subj#']
    src_scores = do_all_classification(data_src[0], data_src[1], CLSF_PARAMS)
    print '\t%s: Src classification complete' % s_dict['Subj#']

    score_dicts.append(dict(sens_scores=sens_scores,
                            src_sph_scores=src_sph_scores,
                            src_scores=src_scores, s_dict=s_dict))

    p_bar.update(si)

    # Save pickled data
    if save_data:
        pkl_file = open(op.join(data_save_dir, 'sim_scores_%s.pkl' %
                                s_dict['Struct']), 'wb')
        cPickle.dump(score_dicts[-1], pkl_file)
        pkl_file.close()

    print '\nData saved.'

if save_data:
    sim_params = dict(decomp_list=decomp_list, n_trials=n_trials,
                      activation_mag=sim_amplitude)
    pkl_file = open(op.join(data_save_dir, 'sim_scores_params.pkl'), 'wb')
    cPickle.dump(sim_params, pkl_file)
    pkl_file.close()

print '\nFinished simulation processing @ ' + strftime('%D %H:%M:%S')
