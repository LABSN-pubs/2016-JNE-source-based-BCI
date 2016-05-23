# Author: Mark Wronkiewicz

import os.path as op
import numpy as np
from numpy.testing import assert_equal

from mne.utils import run_tests_if_main
from switchPredFun import process_ica, process_csp, process_pca, _process_aux

t_winds = 3
n_regularizations = 3
n_components = 4
n_subjects = 10
n_CVs = 10
n_kernels = 1
n_Cs = 5
n_gs = 6


def test_scoring_functions():
    """Test that code used to process and average score arrays"""
    t_wind = 1

    pca_ica_arr = np.zeros((t_winds, n_components, n_subjects, n_CVs,
                            n_kernels, n_Cs, n_gs))
    csp_arr = np.zeros((t_winds, n_regularizations, n_components, n_subjects,
                        n_CVs, n_kernels, n_Cs, n_gs))

    pca_ica_arr[t_wind, :, :, :, 0, 0, 0] = 1.
    csp_arr[t_wind, 0, 0, :, :, 0, 0, 0] = 1.

    # Test auxiliary processing
    scores_aux = _process_aux(pca_ica_arr, mean_dim_1=3,
                              max_dims=(-3, -2, -1), switch_period_dim=t_wind)
    assert_equal(np.ones_like(scores_aux), scores_aux)

    # Test main processing of pca/ica and csp
    scores_temp = process_pca(pca_ica_arr)
    assert_equal(np.ones(n_subjects), scores_temp)

    scores_temp = process_ica(pca_ica_arr)
    assert_equal(np.ones(n_subjects), scores_temp)

    scores_temp = process_csp(csp_arr)
    assert_equal(np.ones(n_subjects), scores_temp)

run_tests_if_main()
