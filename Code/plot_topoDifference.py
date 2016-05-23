"""
plot_topoDifference.py

@Author: wronk

Code for Figure 4
Plot unit activation of a several labels as they manifests in sensor space.
Plot mean topology and 50 percent of max equipotential line.
Plot cardinal direction indicators
"""

from os import environ, path as op
from copy import deepcopy
import csv
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.simulation import simulate_stc
from mne.minimum_norm import read_inverse_operator as read_inv

save_plot = True
hemi = ['lh', 'rh'][1]
model_subj = 'fsaverage'  # Brain to plot labels on
contour_cutoff = 0.5  # Proportion of max value used for equipotential lines

# Find/construct useful folder global vars
subj_info_dir = op.join(environ['CODE_ROOT'], 'switchBCIFiles')
subjListFName = op.join(subj_info_dir, 'SoP_MEG.txt')
struct_dir = op.join(environ['SUBJECTS_DIR'])
fwd_dir = op.join(environ['CODE_ROOT'], 'switchBCIFiles')

########################################
# Load subject information
########################################
subj_list = ['AKCLEE_107', 'AKCLEE_102', 'AKCLEE_103', 'AKCLEE_104',
             'AKCLEE_105', 'AKCLEE_106', 'AKCLEE_108', 'AKCLEE_110',
             'AKCLEE_113', 'AKCLEE_114']
#subj_list = ['AKCLEE_107', 'AKCLEE_114']

# Open/read subject info file into a list of dicts
with open(subjListFName, 'rb') as f:
    subj_d = list(csv.DictReader(f, dialect='excel-tab', delimiter='\t'))

# Get subject indices in the dict list
subj_inds = [i for i, d in enumerate(subj_d) if d['Struct'] in subj_list]

foldNames = [d['Subj#'] for d in [subj_d[i] for i in subj_inds]]

###############
# Load ROI info
###############
# Define labels to use
rois = ['RTPJAnatomical-rh.label',
        'G_precentral_handMotor_radius_10mm-rh.label',
        'G_temp_sup-G_T_transv-rh.label']

aud_labels = None
if 'G_temp_sup-G_T_transv-rh.label' in rois:
    print 'Aud label found'

    aud_labels = []
    aud_label_fsaverage = mne.read_labels_from_annot(
        model_subj, parc='aparc.a2009s', hemi=hemi,
        regexp='G_temp_sup-G_T_transv', verbose=False)[0]
    for si in subj_inds:
        labels = mne.read_labels_from_annot(
            subj_d[si]['Struct'], parc='aparc.a2009s', hemi=hemi,
            regexp='G_temp_sup-G_T_transv', verbose=False)

        aud_labels.append(labels[0])
        #aud_labels.append(labels[0] + labels[1])

########################################
# Compute effect of labels on sensors
########################################

sens_data_list = []
p_bar = mne.utils.ProgressBar(max_value=len(subj_list) - 1, spinner=True,
                              mesg='Creating Subject Evokeds')

#Load all forward, inverse, and epochs
print 'Loading info for:'
for si, (di, s_name) in enumerate(zip(subj_inds, foldNames)):

    struct_name = subj_d[di]['Struct']
    fwd_fname = op.join(fwd_dir, s_name, 'forward', s_name + '-SSS-fwd.fif')
    inv_fname = op.join(fwd_dir, s_name, 'inverse', s_name +
                        '-55-SSS-eeg-Fixed-inv.fif')
    src_fname = op.join(struct_dir, struct_name, 'bem', struct_name +
                        '-7-src.fif')

    ###################################
    # Load src space, forawrd solution, inverse solution, and info
    subj_d[di]['src'] = mne.read_source_spaces(src_fname, verbose=False)

    fwd = mne.read_forward_solution(fwd_fname, force_fixed=True,
                                    surf_ori=False, verbose=False)
    fwd = mne.pick_types_forward(fwd, meg=False, eeg=True,
                                 ref_meg=False, exclude=[])
    subj_d[di]['vertices'] = [s['vertno'] for s in fwd['src']]
    inv = read_inv(inv_fname, verbose=False)
    cov = inv['noise_cov']
    subj_d[di]['info'] = deepcopy(fwd['info'])
    subj_d[di]['info']['sfreq'] = 1000
    subj_d[di]['info']['comps'] = []

    n_dipoles = sum([len(v) for v in subj_d[di]['vertices']])

    ########################################
    # Create stc object for label of interest

    subj_d[di]['rois'] = []
    subj_d[di]['evo'] = []
    sens_data_list_rois = []
    for ri, roi_name in enumerate(rois):
        # Deal with labels that potentially had to be loaded from parc
        if roi_name == 'G_temp_sup-G_T_transv-rh.label':
            subj_d[di]['rois'].append(aud_labels[si])
        else:
            label_fname = op.join(struct_dir, subj_d[di]['Struct'], 'label', roi_name)
            subj_d[di]['rois'].append(mne.read_label(label_fname,
                                                     subject=subj_d[di]['Struct']))

        # Construct stc with active vertices belonging to the correct label/ROI
        n_verts = len(subj_d[di]['rois'][-1].vertices)
        data_arr = np.array([(1. / n_verts)])[np.newaxis, :]

        temp_stc = simulate_stc(src=subj_d[di]['src'],
                                labels=[subj_d[di]['rois'][ri]],
                                stc_data=data_arr, tmin=0, tstep=1)

        # Create evoked object to plot on scalp map
        fake_evo = mne.simulation.simulate_evoked(fwd, temp_stc,
                                                  subj_d[di]['info'], cov,
                                                  snr=np.inf, verbose=False)
        normed_data = fake_evo.data / np.max(np.abs(fake_evo.data))
        sens_data_list_rois.append(np.squeeze(normed_data))

        subj_d[di]['evo'].append(mne.EvokedArray(normed_data,
                                                 subj_d[di]['info'], 0,
                                                 verbose=False))
    sens_data_list.append(sens_data_list_rois)
    p_bar.update(si)  # update progress bar

###########################################################################
# Average evoked objects
###########################################################################
epo_model = 0  # Subject index whose head should be used for the sensor plots

# sens_data_list.shape = (n_subj x n_rois x n_channels)
# sens_data_arr.shape = (n_channels x n_rois)  (exploiting times dim.)
sens_data = np.array(sens_data_list)
sens_data_arr_mean = sens_data.mean(axis=0).T
epo_avg = mne.EvokedArray(sens_data_arr_mean, subj_d[epo_model]['info'],
                          tmin=0)

#sens_data_arr_max = sens_data.T.squeeze().copy()
#evo_all = mne.EvokedArray(sens_data_arr_max, subj_d[epo_model]['info'],
#                          tmin=0)

##############################################################################
# Helper functions to create 3D coordinate frames next to brain
##############################################################################
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    '''Custom arrow design'''
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def draw_lateral_coord_frame(ax):
    '''Helper to draw lateral coordinate frame indicator'''

    vecs_s = [[0, -0.125, 0],
              [0, 0, -0.125]]
    vecs_e = [[0, 1, 0],
              [0, 0, 1]]
    dir_label_pos = [[1.1, -0.6, -0.5],
                     [0, 1.1, -0.15],
                     [0.0, -0.3, 1.1]]
    dir_labels = ['L', 'A', 'D']

    # Plot arrow vectors
    for vec_s, vec_e in zip(vecs_s, vecs_e):
        a = Arrow3D([vec_s[0], vec_e[0]], [vec_s[1], vec_e[1]], [vec_s[2], vec_e[2]],
                    mutation_scale=10, lw=1, arrowstyle="-|>")
        ax.add_artist(a)

        # Label arrow tips
    for l_pos, label in zip(dir_label_pos, dir_labels):
        ax.text(l_pos[0], l_pos[1], l_pos[2], label, None, fontsize=12)

    # Set to lateral view
    ax.view_init(0, 0)

    ax.scatter3D(0, 0, zs=0, s=40, marker='o', c='w')
    ax.scatter3D(0, 0, zs=0, s=7, marker='.', c='k')
    ax.dist = 100  # required when adding the scatter points

    _cleanup_ax(ax)


def draw_oblique_coord_frame(ax, view_angles):
    '''Helper to coordinate frame indicator from any angle'''
    vecs_s = [[-0.125, 0, 0],
              [0, -0.125, 0],
              [0, 0, -0.125]]
    vecs_e = [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]
    dir_label_pos = [[1.3, -0.15, -0.2],
                     [0, 1.1, -0.2],
                     [0.0, -0.35, 1.1]]
    dir_labels = ['L', 'A', 'D']

    # Plot arrow vectors
    for vec_s, vec_e, l_pos, label in zip(vecs_s, vecs_e, dir_label_pos, dir_labels):
        a = Arrow3D([vec_s[0], vec_e[0]], [vec_s[1], vec_e[1]], [vec_s[2], vec_e[2]],
                    mutation_scale=10, lw=1, arrowstyle="-|>")
        ax.add_artist(a)

        # Label arrow tips
        ax.text(l_pos[0], l_pos[1], l_pos[2], label, None, fontsize=12)

    # Set to oblique view
    ax.view_init(view_angles[0], view_angles[1])

    _cleanup_ax(ax)


def _cleanup_ax(ax):
    '''Clear ticks, spines, lines'''

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    white_rgb = (1, 1, 1, 0)
    ax.w_xaxis.set_pane_color(white_rgb)
    ax.w_yaxis.set_pane_color(white_rgb)
    ax.w_zaxis.set_pane_color(white_rgb)

    ax.w_xaxis.line.set_color(white_rgb)
    ax.w_yaxis.line.set_color(white_rgb)
    ax.w_zaxis.line.set_color(white_rgb)

###########################################################################
# Plotting
###########################################################################
from surfer import Brain
plt.close('all')
plt.ion()

fig, axes = plt.subplots(3, len(rois), figsize=(7.5, 6), facecolor='white')
if len(axes.shape) == 1:
    axes = axes[:, np.newaxis]
#fig.set_facecolor('white')

clims = dict(kind='percent', lims=[90, 99, 100])
mask = None  # mask for epo maps
res = 128  # Resolution for topo map

# Specify View Dict for brain point of view
v_rh_aud_angled = {'azimuth': 30., 'elevation': 70., 'distance': 100.}

view_dict = {}
view_dict['G_temp_sup-G_T_transv-rh.label'] = [v_rh_aud_angled]
view_dict['G_precentral_handMotor_radius_10mm-rh.label'] = ['lat']
view_dict['RTPJAnatomical-rh.label'] = ['lat']
label_color = '#DD0000'

# Few variables for small coordinate frame indicators next to brains
sq_size = 0.085
coord_ax_dims = [[0.28, 0.79, sq_size, sq_size],
                 [0.60, 0.79, sq_size, sq_size],
                 [0.91, 0.79, sq_size, sq_size]]

print '\nPlotting...\n'
for ri, roi_name in enumerate(rois):
    ###################################
    # Plot source space with roi
    ###################################
    width = 500 if ri == 2 else 1000
    height = 400 if ri == 2 else 800
    brain = Brain(subject_id='fsaverage', hemi=hemi,
                  surf='inflated_pre', size=(height, width),
                  offscreen=True, background='white')

    # Load corresponding label from fsaverage
    if roi_name == 'G_temp_sup-G_T_transv-rh.label':
        brain.add_label(aud_label_fsaverage, color=label_color, alpha=0.75)
    else:
        fname_load_label = op.join(struct_dir, model_subj, 'label', roi_name)
        temp_roi = mne.read_label(fname_load_label, subject=model_subj)
        brain.add_label(temp_roi, color=label_color, alpha=0.75)

    # Save montage as an image
    montage = brain.save_montage(None, order=view_dict[roi_name],
                                 orientation='v', border_size=15,
                                 colorbar=None)
    axes[0, ri].imshow(montage, interpolation='nearest', origin='upper',
                       aspect='equal')
    brain.close()

    ax_coord = plt.axes(coord_ax_dims[ri], axisbg='none', projection='3d',
                        aspect='equal')
    # Turn off appropriate ticks and spines and plot coordinate frame indicator
    if ri == 0:
        # This roundabout way preserves ability to set ylabel
        axes[0, 0].xaxis.set_visible(False)
        axes[0, 0].xaxis.set_ticklabels([])
        axes[0, 0].yaxis.set_ticks([])
        for spine in axes[0, 0].spines.values():
            spine.set_visible(False)
        for spine in axes[0, 0].spines.values():
            spine.set_visible(False)

        # Draw coordinate frame indicator
        draw_lateral_coord_frame(ax_coord)
    elif ri == 1:
        axes[0, ri].set_axis_off()
        draw_lateral_coord_frame(ax_coord)
    else:
        axes[0, ri].set_axis_off()
        draw_oblique_coord_frame(ax_coord, (20, 45))

    ######################################
    # Plot sensor space electrode topology
    ######################################
    # Middle row (topomaps)
    cbar = False
    epo_avg.plot_topomap(times=epo_avg.times[ri], ch_type='eeg', size=2,
                         vmin=-1, vmax=1, scale=1, cmap='RdBu_r',
                         time_format="", unit='Activation (AU)', mask=mask,
                         contours=5, colorbar=cbar, res=res,
                         axes=axes[1, ri], sensors=False)

    # Bottom row (equipotential lines)
    sens_data_arr_max = sens_data[:, ri, :].T.copy()
    for si, di in enumerate(subj_inds):
        evo_temp = mne.EvokedArray(sens_data_arr_max[:, si][:, np.newaxis],
                                   subj_d[epo_model]['info'], tmin=0)
        evo_temp.plot_topomap(times=0, ch_type='eeg', size=2, vmin=-100000,
                              vmax=100000, scale=1, cmap='RdBu_r',
                              time_format="", unit='Units', mask=mask,
                              contours=[contour_cutoff], colorbar=False,
                              res=res, axes=axes[2, ri], sensors=False)

    print 'Epo Max: ' + str(np.max(epo_avg.data[:, ri]))
    print 'Epo Min: ' + str(np.min(epo_avg.data[:, ri]))

# Position relative to axes for column labels
xpos, ypos = 0.5, 2.55
fs = 15  # fontsize
pad = 9.

# Add text to label the columns/rows
axes[1, 0].annotate('RTPJ', xy=(xpos, ypos), xycoords=axes[1, 0].transAxes,
                    ha='center', fontsize=fs, va='top')
axes[1, 1].annotate('Right hand-motor\nregion', xy=(xpos, ypos),
                    xycoords=axes[1, 1].transAxes, ha='center', fontsize=fs,
                    va='top')
axes[1, 2].annotate('Right ant. trans.\ntemporal gyrus', xy=(xpos, ypos),
                    xycoords=axes[1, 2].transAxes, ha='center', fontsize=fs,
                    va='top')

axes[0, 0].set_ylabel('Source space', fontsize=fs)
axes[1, 0].set_ylabel('Sensor space\n(mean influence)', fontsize=fs,
                      va='bottom', labelpad=pad)
axes[2, 0].set_ylabel('Sensor space\n(%02d%% of max)' %
                      (100 * contour_cutoff), fontsize=fs, va='bottom',
                      labelpad=pad)

fig.subplots_adjust(left=0.05, right=.975, bottom=0.04, top=0.875,
                    hspace=0.175, wspace=.1)

plt.show()
if save_plot:
    fname_save = op.join(subj_info_dir, 'figures_epoTopo_simulatedActivity',
                         'roi_avg_max%02d' % (100 * contour_cutoff))
    fig.savefig(fname_save + '.png')
    fig.savefig(fname_save + '.pdf', dpi=300)

print '\n...Complete'
