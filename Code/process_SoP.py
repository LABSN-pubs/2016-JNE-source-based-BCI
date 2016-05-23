"""
processSoP.py

@Author: wronk

Generate MNE-Python Epoch objects from minimally processed SoP data
"""


import mne
import numpy as np
from copy import deepcopy

import glob
from os import mkdir, environ, path as op


'''
def generateEpochs(raw_dir, list_dir, eq):
    """
    Generate an epoch object from raw data and events file

    Parameters:
    -----------
    raw_dir: str
        File path to folder containing raw data objects
    list_dir: str
        File path to events list
    eq: str
        equalize or just disqualified trials

    Returns:
    --------
    epochs: mne.Epochs
        Generated epochs object
    """

    raw_fileNames = glob.glob(op.join(raw_dir,
                              'Eric_SoP_*_allclean_fil55_raw_sss.fif'))
    list_fileNames = glob.glob(op.join(list_dir,
                                       'All_55-SSS_' + eq + '_Eric_SoP_*.lst'))

    # Read in list of events
    events = [mne.read_events(l_dir) for l_dir in list_fileNames]

    # Read in raw files
    raw = mne.io.RawFIFF(raw_fileNames, add_eeg_ref=False)

    tmin = -0.2
    tmax = 4.75


    # generate epochs object from events list and raw file
    return mne.Epochs(raw, events, event_id=None, tmin=tmin, tmax=tmax,
                      proj=False)
'''
###############################################################################
subj_info_dir = op.join(environ['CODE_ROOT'], 'switchBCIFiles')
subjects = ['Eric_SoP_001', 'Eric_SoP_002', 'Eric_SoP_003', 'Eric_SoP_004',
            'Eric_SoP_005', 'Eric_SoP_006', 'Eric_SoP_008', 'Eric_SoP_009',
            'Eric_SoP_010', 'Eric_SoP_013', 'Eric_SoP_014', 'Eric_SoP_015']

eq = 'EQ'  # 'EQ' or 'DQ' for equalized or non-equalized trial sets

subj_epochs = []
print 'Generating epochs for subject list...\n',
for subj_file in subjects:

    raw_dir = op.join(subj_info_dir, subj_file, 'SSS_PCA_FIF')
    list_dir = op.join(subj_info_dir, subj_file, 'LST')

    ###################################
    # Get lists of raw and LST files
    raw_fileNames = glob.glob(op.join(raw_dir,
                                      'Eric_SoP_*_allclean_fil55_raw_sss.fif'))
    # Use events corresponding to (space/pitch) X (maintain/switch)
    list_fileNames = glob.glob(op.join(list_dir,
                                       'ULxSR_55-SSS_' + eq + '_' + subj_file +
                                       '_??.lst'))
    # Sort lists so they're in the same trial order
    raw_fileNames.sort()
    list_fileNames.sort()

    ###################################
    # Read in raw files
    raw_list = []
    projOverwrite = {}
    sampTimes = np.zeros((len(raw_fileNames), 2))
    # Overwrite projection information
    for i, raw_fname in enumerate(raw_fileNames):
        raw_list.append(mne.io.RawFIFF(raw_fname, add_eeg_ref=True,
                                       preload=True, verbose=False))
        if i == 0:
            projOverwrite = deepcopy(raw_list[0].info['projs'])

        ### QUICK HACK ###
        raw_list[-1].apply_proj()
        #raw_list[-1].proj = False
        #raw_list[-1]._projector = None
        ############

        raw_list[-1].info['projs'] = projOverwrite

    #concatenate raw
    raw = mne.concatenate_raws(raw_list)

    #raw = mne.io.RawFIFF(raw_fileNames, add_eeg_ref=True)
    # Read in list of events
    events = mne.concatenate_events([mne.read_events(l_dir)
                                     for l_dir in list_fileNames],
                                    raw._first_samps, raw._last_samps)

    ###################################
    # generate epochs object from events list and raw file
    #tempEpo = generateEpochs(raw_dir, list_dir))
    eventDict = {'pitch_maintain': 1, 'space_maintain': 2,
                 'pitch_switch': 3, 'space_switch': 4}

    tmin = -0.2
    tmax = 4.75
    tempEpo = mne.Epochs(raw, events, event_id=eventDict, tmin=tmin,
                         tmax=tmax, preload=True, proj=True, verbose=False)
    #tempEpo.drop_bad_epochs()  #Done if preload=True above
    assert all(len(x) == 0 or 'IGNORED' in x for x in tempEpo.drop_log)

    # Check if epochs directory exists
    save_dir = op.join(subj_info_dir, subj_file, 'epochs')
    if not op.exists(save_dir):
        mkdir(save_dir)

    ###################################
    # Save epoch object
    tempEpo.save(op.join(save_dir, 'All_55_sss_' + eq + '_' + subj_file +
                         '-epo.fif'))

    #subj_epochs.append(mne.Epochs(raw, events, event_id=None, tmin=tmin,
    #                              tmax=tmax, proj=True, verbose=False)
    #subj_epochs.append(generateEpochs(raw_dir, list_dir))

    print '\t... ' + subj_file.split('/')[-1] + ' Done'
