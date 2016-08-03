# Incorporating modern neuroscience findings to improve brain-computer interfaces: tracking auditory attention.
This repository contains code for a study comparing sensor- and source-based BCIs. 
Specifically, it shows the theoretical and quantitative advantages (using both simulated and real data) associated with using the source space instead of the sensor space in a BCI context.
We demonstrate this by classifying when subjects switched spatial auditory attention with data from a [previous task](http://www.ncbi.nlm.nih.gov/pubmed/24096028).

## Reference
- Wronkiewicz, M., Larson, E., and Lee, A. KC (2016). Incorporating modern neuroscience findings to improve brain-computer interfaces: tracking auditory attention. _Journal of Neural Engineering_ 

## Code
The code makes use of at least these specialized libraries:
  * MNE-Python v0.11
  * Freesurfer
  * Pysurfer
  * Statsmodels

Raw data is processed with `process_SoP.py` and `process_createStcs.py`

Figures 1 and 2 were created using meshes obtained via MRI scans and [Blender](www.blender.org).
The activation simulation in Figure 4 is created with `plot_topoDifference.py`.
The synthetic data for Figure 5 was created using `switchPredSim.py`, reorganized to link into previously written plotting code with `reformulate_sim_scores.py`, and plotted with `plot_switchPredSim.py`.

The script for computing spherical inverse models is `makeSphModels.py`.
Code for sensor and source based classification are in `switchPredSensLoop_all.py` and `switchPredSrcLoop_all.py`, respectively.
The script used for the statistics and plotting of Figure 6 is `plot_switchPredLoop.py`.
General functions and parameters are in `switchPredFun.py` and `config.py`, respectively.

Unfortunately, the raw data are not included because:
  1. it contains (HIPAA-violating) identifying information and 
  2. the raw data are many tens of gigabytes