"""
reformulate_sim_scores.py

@Author: wronk

Reorganize scores from simulation so they can be easily plotted using
(prewritten) plotting functions.
"""

from os import environ, path as op
from time import strftime

import numpy as np
import cPickle
from config import (DATA_PARAMS, CLSF_PARAMS, CSP_PARAMS, ICA_PARAMS,
                    PCA_PARAMS)

save_data = True
subj_info_dir = op.join(environ['CODE_ROOT'], 'switchBCIFiles')
data_save_dir = op.join(subj_info_dir, 'PickledData', 'SimPred')

subj_list = ['AKCLEE_107', 'AKCLEE_102', 'AKCLEE_103', 'AKCLEE_104',
             'AKCLEE_105', 'AKCLEE_106', 'AKCLEE_108', 'AKCLEE_110',
             'AKCLEE_113', 'AKCLEE_114']

with open(op.join(data_save_dir, 'sim_scores_params.pkl'), 'rb') as pkl_file:
    sim_params = cPickle.load(pkl_file)

score_dicts = []
for si, subj in enumerate(subj_list):
    print 'Loading %s' % subj
    with open(op.join(data_save_dir, 'sim_scores_%s.pkl' % subj), 'rb') as pkl_file:
        score_dicts.append(cPickle.load(pkl_file))
    print '\t... Done.'

###########################################################################
# Reshape structure of scores so that the first level is the decomp method.
# This makes data amenable to the prewritten score processing functions
###########################################################################
print '\nReformulating scores.'
decomp_score_dict = dict(subj_list=subj_list, clsf_params=CLSF_PARAMS,
                         finish_time=strftime('%D %H:%M:%S'))
decomp_score_dict.update(sim_params)

# Do this seperately for each dim. reduc. method
for di, decomp_method in enumerate(sim_params['decomp_list']):
    if decomp_method == 'pca':
        # Initialize array for this case
        scores_src = np.empty((1, len(PCA_PARAMS['n_components_list']),
                               len(score_dicts), CLSF_PARAMS['cvs'],
                               len(CLSF_PARAMS['kernel_list']),
                               len(CLSF_PARAMS['C_range']),
                               len(CLSF_PARAMS['g_range'])))
        # Initialize src_sph and sens arrays too
        scores_src_sph = np.empty_like(scores_src)
        scores_sens = np.empty_like(scores_src)

        for si, subj_dict in enumerate(score_dicts):
            scores_sens[0, :, si, :, :, :, :] = subj_dict['sens_scores'][di]
            scores_src[0, :, si, :, :, :, :] = subj_dict['src_scores'][di]
            scores_src_sph[0, :, si, :, :, :, :] = subj_dict['src_sph_scores'][di]
        decomp_score_dict[decomp_method] = \
            dict(sens=scores_sens, src=scores_src, src_sph=scores_src_sph,
                 params=PCA_PARAMS)

    elif decomp_method == 'ica':
        scores_src = np.empty((1, len(ICA_PARAMS['n_components_list']),
                               len(score_dicts), CLSF_PARAMS['cvs'],
                               len(CLSF_PARAMS['kernel_list']),
                               len(CLSF_PARAMS['C_range']),
                               len(CLSF_PARAMS['g_range'])))
        scores_src_sph = np.empty_like(scores_src)
        scores_sens = np.empty_like(scores_src)

        for si, subj_dict in enumerate(score_dicts):
            scores_sens[0, :, si, :, :, :, :] = subj_dict['sens_scores'][di]
            scores_src[0, :, si, :, :, :, :] = subj_dict['src_scores'][di]
            scores_src_sph[0, :, si, :, :, :, :] = subj_dict['src_sph_scores'][di]
        decomp_score_dict[decomp_method] = \
            dict(sens=scores_sens, src=scores_src, src_sph=scores_src_sph,
                 params=ICA_PARAMS)

    elif decomp_method == 'csp':
        scores_src = np.empty((1, len(CSP_PARAMS['reg_list']),
                               len(CSP_PARAMS['n_components_list']),
                               len(score_dicts), CLSF_PARAMS['cvs'],
                               len(CLSF_PARAMS['kernel_list']),
                               len(CLSF_PARAMS['C_range']),
                               len(CLSF_PARAMS['g_range'])))
        scores_src_sph = np.empty_like(scores_src)
        scores_sens = np.empty_like(scores_src)

        for si, subj_dict in enumerate(score_dicts):
            scores_sens[0, :, :, si, :, :, :, :] = subj_dict['sens_scores'][di]
            scores_src[0, :, :, si, :, :, :, :] = subj_dict['src_scores'][di]
            scores_src_sph[0, :, :, si, :, :, :, :] = subj_dict['src_sph_scores'][di]
        decomp_score_dict[decomp_method] = \
            dict(sens=scores_sens, src=scores_src, src_sph=scores_src_sph,
                 params=CSP_PARAMS)

    else:
        raise RuntimeError('No suitable decomposition method matched')

# Save pickled data
if save_data:
    pkl_file = open(op.join(data_save_dir, 'sim_scores.pkl'), 'wb')
    cPickle.dump(decomp_score_dict, pkl_file)
    pkl_file.close()

print '\nData saved.'
