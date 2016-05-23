"""
switchPredFun.py

@Author: wronk

Script with helper functions for ML prediction of SOP data.
"""

import numpy as np
from sklearn.svm import SVC

cache_size = 4096


def cross_val_mean(data_dict, kernel_list, C_range, g_range):
    """
    Helper function to loop over SVC params using the spatial mean

    Parameters
    ----------
    data_dict: dict
        Dictionary with keys (X_test, X_train, y_test, y_train) containing the
        appropriate data matrices for each key.
    kernel_list: list
        Kernels to use in the SVM classifier.
    C_range: list
        Values of C to iterate over in SVM classifier.
    g_range: list
        Values of gamma to iterate over in SVM classifier.

    Returns
    -------
    score_array: ndarray
        Array containing classification scores over each dimension.
    """

    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']

    score_array = np.empty((len(kernel_list), len(C_range), len(g_range)))
    score_array[:] = np.nan

    # Loop over SVM parameters
    for ki, kernel in enumerate(kernel_list):
        for Ci, C in enumerate(C_range):
            for gi, g in enumerate(g_range):
                clf = SVC(C=C, kernel=kernel, gamma=g, cache_size=cache_size)

                # Train and test classifier
                clf.fit(X_train, y_train)
                score_array[ki, Ci, gi] = clf.score(X_test, y_test)

    return score_array


def cross_val_csp(data_dict, csp_filter, kernel_list, C_range, g_range):
    """
    Helper function to loop over all SVM parameters for a given data set.

    Parameters
    ----------
    data_dict: dict
        Dictionary with keys (X_test, X_train, y_test, y_train) containing the
        appropriate data matrices for each key.
    kernel_list: list
        Kernels to use in the SVM classifier.
    C_range: list
        Values of C to iterate over in SVM classifier.
    g_range: list
        Values of gamma to iterate over in SVM classifier.

    Returns
    -------
    score_array: ndarray
        Array containing classification scores over each dimension.
    """

    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']

    score_array = np.empty((len(kernel_list), len(C_range), len(g_range)))
    score_array[:] = np.nan

    # Make sure X_train and X_test are 3D (n_trials, n_chan, n_times)
    if len(X_train.shape) == 2:
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)

    # Compute CSP features
    X_train_csp = csp_filter.fit_transform(X_train, y_train)
    X_test_csp = csp_filter.transform(X_test)

    # Standard scaler not necessary since CSP manipulates variance

    # Loop over SVM parameters
    for ki, kernel in enumerate(kernel_list):
        for Ci, C in enumerate(C_range):
            for gi, g in enumerate(g_range):
                clf = SVC(C=C, kernel=kernel, gamma=g, cache_size=cache_size)

                # Train and test classifier
                clf.fit(X_train_csp, y_train)
                score_array[ki, Ci, gi] = clf.score(X_test_csp, y_test)

    return score_array


def cross_val_pca(data_dict, decomp_filter, kernel_list, C_range, g_range):
    """
    Helper function to loop over SVC params using PCA decomposition.

    Parameters
    ----------
    data_dict: dict
        Dictionary with keys (X_test, X_train, y_test, y_train) containing the
        appropriate data matrices for each key.
    decomp_filter: object
        PCA decomposition object.
    kernel_list: list
        Kernels to use in the SVM classifier.
    C_range: list
        Values of C to iterate over in SVM classifier.
    g_range: list
        Values of gamma to iterate over in SVM classifier.

    Returns
    -------
    score_array: ndarray
        Array containing classification scores over each dimension.
    """

    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']

    score_array = np.empty((len(kernel_list), len(C_range), len(g_range)))
    score_array[:] = np.nan

    # Make sure X_train and X_test are 3D
    #if len(X_train.shape) == 2:
    #    X_train = np.expand_dims(X_train, axis=2)
    #    X_test = np.expand_dims(X_test, axis=2)

    # Compute PCA features
    X_train_tran = decomp_filter.fit_transform(X_train, y_train)
    X_test_tran = decomp_filter.transform(X_test)

    # Loop over SVM parameters
    for ki, kernel in enumerate(kernel_list):
        for Ci, C in enumerate(C_range):
            for gi, g in enumerate(g_range):
                clf = SVC(C=C, kernel=kernel, gamma=g, cache_size=cache_size)

                # Train and test classifier
                clf.fit(X_train_tran, y_train)
                score_array[ki, Ci, gi] = clf.score(X_test_tran, y_test)

    return score_array


def cross_val_ica(data_dict, kernel_list, C_range, g_range):
    """
    Helper function to loop over SVC params using ICA decomposition.

    Parameters
    ----------
    data_dict: dict
        Dictionary with keys (X_test, X_train, y_test, y_train) containing the
        appropriate data matrices for each key.
    kernel_list: list
        Kernels to use in the SVM classifier.
    C_range: list
        Values of C to iterate over in SVM classifier.
    g_range: list
        Values of gamma to iterate over in SVM classifier.

    Returns
    -------
    score_array: ndarray
        Array containing classification scores over each dimension.
    """

    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']

    score_array = np.empty((len(kernel_list), len(C_range), len(g_range)))
    score_array[:] = np.nan

    # Loop over SVM parameters
    for ki, kernel in enumerate(kernel_list):
        for Ci, C in enumerate(C_range):
            for gi, g in enumerate(g_range):
                clf = SVC(C=C, kernel=kernel, gamma=g, cache_size=cache_size)

                # Train and test classifier
                clf.fit(X_train, y_train)
                score_array[ki, Ci, gi] = clf.score(X_test, y_test)

    return score_array


def apply_unmixing(data_3d, unmix_mat):
    """Helper function to apply unmixing matrix to 3D data

    Note: ``data_3d`` should be (n_trials x n_channels x n_times)"""

    data_rolled = np.rollaxis(data_3d, axis=1)
    rolled_shape = data_rolled.shape
    rolled_reshaped = data_rolled.reshape(rolled_shape[0], -1)

    unmixed = np.dot(unmix_mat, rolled_reshaped)
    unmix_reshaped = unmixed.reshape(rolled_shape)
    unmix_reshaped = np.rollaxis(unmix_reshaped, axis=1)

    return unmix_reshaped


def process_mean(scores):
    """Helper function to process scores calculated with a mean"""

    assert len(scores.shape) == 6, 'Incorrect number of dims'

    # Get scores meaned over CVs, max svm params, and time period of interest
    scores_temp = scores.mean(axis=2)  # Mean across cross-vals
    scores_temp = scores_temp.max(axis=(-3, -2, -1))  # Take max SVM params
    scores_temp = scores_temp[1, :]  # Pick switching period

    scores_proc = np.copy(scores_temp)  # Copy initial processed score array
    #scores_temp = scores_temp.mean(1)  # Mean across subjects
    #max_comp_ind = np.argmax(scores_temp)  # Find best num components

    # Take number of components that gave the best score
    scores_proc = scores_proc
    return scores_proc


def process_pca(scores):
    """Helper function to process PCA scores"""

    assert len(scores.shape) == 7, 'Incorrect number of dims'

    # Get scores meaned over CVs, max svm params, and time period of interest
    scores_temp = _process_aux(scores, 3, (-3, -2, -1), 1)

    scores_proc = np.copy(scores_temp)  # Copy initial processed score array
    scores_temp = scores_temp.mean(1)  # Mean across subjects
    max_comp_ind = np.argmax(scores_temp)  # Find best num components
    print 'Max PCA n_component index: ' + str(max_comp_ind)

    # Take number of components that gave the best score
    scores_proc = scores_proc[max_comp_ind, :]
    return scores_proc


def process_csp(scores):
    """Helper function to process CSP scores"""

    assert len(scores.shape) == 8, 'Incorrect number of dims'

    # Get scores meaned over CVs, max svm params, and time period of interest
    scores_temp = _process_aux(scores, 4, (-3, -2, -1), 1)

    scores_proc = np.copy(scores_temp)  # Copy initial processed score array
    scores_temp = scores_temp.mean(2)  # Mean across subjects

    # Get best regularization and number of components across subjects
    max_inds = np.where(scores_temp == np.max(scores_temp))
    best_inds = [ind[0] for ind in max_inds]
    print 'Max CSP regularization index: ' + str(best_inds[0])
    print 'Max CSP n_components index: ' + str(best_inds[1])

    scores_proc = scores_proc[best_inds[0], best_inds[1], :]

    return scores_proc


def process_ica(scores):
    """Helper function to process ICA scores"""

    assert len(scores.shape) == 7, 'Incorrect number of dims'

    # Get scores meaned over CVs, max svm params, and time period of interest
    scores_temp = _process_aux(scores, 3, (-3, -2, -1), 1)

    scores_proc = np.copy(scores_temp)  # Copy initial processed score array
    scores_temp = scores_temp.mean(1)  # Mean across subjects
    max_comp_ind = np.argmax(scores_temp)  # Find best num components
    print 'Max ICA n_component index: ' + str(max_comp_ind)

    # Take number of components that gave the best score
    scores_proc = scores_proc[max_comp_ind, :]
    return scores_proc


def _process_aux(scores, mean_dim_1, max_dims, switch_period_dim):
    """Auxiliary function.

    Average across CVs, get max SVMs parameters (C, gamma, kernel), and pick
    time period"""
    scores_temp = scores.mean(mean_dim_1)  # Mean across cross-vals
    scores_temp = scores_temp.max(axis=max_dims)  # Take max SVM params
    scores_temp = scores_temp[switch_period_dim, :, :]  # Pick switching period

    return scores_temp
