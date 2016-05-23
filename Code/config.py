"""
config.py

Parameters for classification of attentional switching data

@author: wronk
"""

import numpy as np

MEAN_PARAMS = dict(score_mat_order=['Time window', 'Subject index',
                                    'Cross-validation index', 'Kernel index',
                                    'C-val index', 'Gamma val index'])

CSP_PARAMS = dict(reg_list=['ledoit_wolf', 'oas', .05, .1, .15, .2, .25],
                  n_components_list=range(1, 6),
                  score_mat_order=['Time window', 'Regularization',
                                   'Number of components', 'Subject index',
                                   'Cross-validation index', 'Kernel index',
                                   'C-val index', 'Gamma val index'])

PCA_PARAMS = dict(n_components_list=range(1, 6))

ICA_PARAMS = dict(n_components_list=[.85, .9, .95, .99],
                  ica_init_params=dict(method='extended-infomax',
                                       max_iter=50000))

CLSF_PARAMS = dict(cvs=10,  # 5 or 10-fold classification recommended (Bashashati et al., 2007)
                   C_range=10. ** np.arange(-4., 5.),
                   g_range=10. ** np.arange(-7., 3.),
                   cache_size=4096,
                   kernel_list=['rbf'],
                   t_wind_list=[[2.5, 3.2], [2.9, 3.5], [2.9, 3.2], [2.5, 3.5]])

DATA_PARAMS = dict(roi='RTPJAnatomical-rh.label',
                   bin_width=0.05,  # 50 ms
                   lambda2=1. / 9.,  # Standard regularization
                   eq='EQ',  # 'DQ' or 'EQ' Just disqualified trials or equalized trials
                   loaded_bin_width=None,
                   useAbsInMorph=False,
                   patterns=['space_maintain', 'space_switch'],  # Maintain should be first!
                   hemi_idx=1)  # 0 = left hemisphere; 1 = right hemisphere
