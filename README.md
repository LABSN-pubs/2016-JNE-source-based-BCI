# A Source-based brain-computer interface
This repository contains code for the prediction of auditory attention switching with a source-based BCI.

## Reference
- Wronkiewicz, M., Larson, E., and Lee, A. KC (2016). Incorporating modern neuroscience findings to improve brain-computer interfaces: tracking auditory attention. _Journal of Neural Engineering_ 

## Code
The code makes use of at least these specialized libraries:
MNE-Python v0.11
Freesurfer
Pysurfer
Statsmodels

Raw data is processed with `process_SoP.py` and `process_createStcs.py`

Figures 1 and 2 were created using meshes obtained via MRI scans and [Blender](www.blender.org)
The simulation in Figure 4 is created with `plot_topoDifference.py`.

The script for computing spherical inverse models is `makeSphModels.py`
Code for sensor and source based classification are in `switchPredSensLoop_all.py` and `switchPredSrcLoop_all.py`, respectively.
The script used for the statistics and plotting of Figure 5 is `plot_switchPredLoop.py`.
General functions and parameters are in `switchPredFun.py` and `config.py`, respectively.

Unfortunately, the data is not included because 1) it contains (HIPAA-violating) identifying information and 2) the raw data are tens of gigabytes.