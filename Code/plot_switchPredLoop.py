"""
plot_switchPredLoop.py

@Author: wronk

Plot results from attentional switching ML prediction (3 groups) + mean
Used for source (including AND excluding MRI) and sensor space in Figure 5
"""

from os import environ, path as op

import numpy as np
import pickle as pkl
import pandas as pd
from scipy.stats import sem
import patsy
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM

from expyfun.analyze import barplot
from switchPredFun import process_pca, process_ica, process_csp, process_mean

n_subj = 10
bin_width = '50'
savePlot = False
decomp_methods = ['pca', 'ica', 'csp']
ica_method = 'extended-infomax'
inv_methods = ['sens', 'src_sph', 'src']
anatomical_label = True  # Whether or not to use anatomical RTPJ definition
spherical_inv = True  # Whether or not to include spherical inverse in plots

stats_interaction = True

#decomp_methods = ['pca', 'csp']
n_dm = len(decomp_methods)

# Programatically get list of functions needed
proc_functions = [globals()['process_' + method] for method in decomp_methods]

###############################################################################
# LOAD DATA

scores = dict(sens={}, src_sph={}, src={})
data_fold_header = op.join(environ['CODE_ROOT'], 'switchBCIFiles',
                           'PickledData')
# Directory to save plots/results to
save_dir = op.join(data_fold_header, 'scores_srcBCI_manuscript')

for decomp_method in decomp_methods:
    ica_add_on = '_' + ica_method if decomp_method == 'ica' else ''
    anatom_add_on = '_AnatomRTPJ' if anatomical_label else ''

    # Build filenames to source and sensor data
    src_fname = op.join(data_fold_header, 'SrcPred_ParamLoop', 'Space',
                        decomp_method, 'scoreDict_spaceMorphed_' +
                        decomp_method + anatom_add_on + ica_add_on + '_' +
                        bin_width + 'ms.pkl')
    src_sph_fname = op.join(data_fold_header, 'SrcPred_ParamLoop', 'Space',
                            decomp_method, 'scoreDict_spaceMorphed_' +
                            decomp_method + anatom_add_on + '_sph' +
                            ica_add_on + '_' + bin_width + 'ms.pkl')
    sens_fname = op.join(data_fold_header, 'SensPred_ParamLoop', 'Space',
                         decomp_method, 'scoreDict_spaceSens_' +
                         decomp_method + ica_add_on + '_' + bin_width + 'ms.pkl')

    # Load and store score information
    for key, fname in zip(inv_methods,
                          [sens_fname, src_sph_fname, src_fname]):
        with open(fname, 'rb') as pkl_file:
            scores[key][decomp_method] = pkl.load(pkl_file)
            print (decomp_method + '  ' + key + '\tdata loaded from: ' +
                   scores[key][decomp_method]['finish_time'])

# Add processing that used spatial mean (instead of CSP/ICA/PCA)
src_mean_fname = op.join(data_fold_header, 'SrcPred_ParamLoop', 'Space',
                         'mean', 'scoreDict_spaceMorphed_' + 'mean' +
                         anatom_add_on + '_' + bin_width + 'ms.pkl')
with open(src_mean_fname, 'rb') as pkl_file:
    scores['src']['mean'] = pkl.load(pkl_file)
    print ('mean' + '  ' + 'src' + '\tdata loaded from: ' +
           scores['src']['mean']['finish_time'])

###############################################################################
# PROCESS DATA

# Process each decomposition method appropriately
data_list = []
subj_num_list = []
for space in inv_methods:
    print '\nRunning ' + space + '\n============'
    for decomp_method, proc_func in zip(decomp_methods, proc_functions):
        scores[space][decomp_method]['scores_proc'] = \
            proc_func(scores[space][decomp_method]['scores'])

        data_list.append(scores[space][decomp_method]['scores_proc'])
        subj_num_list.append(range(len(data_list[-1])))

# Separately compute mean scores as there is no sensor equivalent
src_bem_mean = process_mean(scores['src']['mean']['scores'])

# Convert data to array of percentages and save
data = np.array(data_list).T
assert data.shape == (n_subj, len(decomp_methods) * 3), 'Data shape incorrect.'
data *= 100  # Convert to percentages

subj_num = np.array(subj_num_list).T.reshape(-1)

np.savetxt(op.join(save_dir, 'bci_classification_data.csv'), data,
           delimiter=',')

# Compute mean and SEM for each classification strategy
data_means = np.mean(data, 0)
sems = sem(data, axis=0)

###############################################################################
# STATISTICS (with or without interaction)

# Create dummy variables
dummy_inv_list = []
for inv_method in inv_methods:
    dummy_inv_list.extend([inv_method] * len(decomp_methods))
dummy_inv_list = dummy_inv_list * n_subj
dummy_decomp_list = decomp_methods * n_subj * len(inv_methods)

# Create pandas dataframe for OLS calculation
d_tot_pd = pd.DataFrame(dict(Accuracy=data.reshape(-1), Inverse=dummy_inv_list,
                             Decomposition=dummy_decomp_list,
                             Subject=subj_num))

l_inverse = [inv_methods[2], inv_methods[0], inv_methods[1]]
l_decomp = decomp_methods

# Create and fit linear regression model
inter = '*' if stats_interaction else '+'
call_str = ('Accuracy ~ C(Inverse, levels=l_inverse) ' + inter +
            ' C(Decomposition, levels=l_decomp)')

y, X = patsy.dmatrices(call_str, d_tot_pd, return_type='dataframe')
model_results = MixedLM(y, X, groups=subj_num).fit()

print model_results.summary()

###############################################################################
# PLOT DATA

import matplotlib.pyplot as plt
#plt.rcParams['pdf.fonttype'] = 42  # Seems to mess up rendering of text
#plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.default'] = 'regular'  # Set math font to normal

# Close all previous plots
plt.close('all')
plt.ion()

label_fontsize = 14
tickLabel_fontsize = 12
fig_width_1col = 3.25
all_groups = [range(n_dm), range(n_dm, n_dm * 2), range(n_dm * 2, n_dm * 3)]
groups = all_groups[:len(decomp_methods)]
brackets = [(range(n_dm, n_dm * 2), range(n_dm * 2, n_dm * 3)),
            (range(n_dm), range(n_dm * 2, n_dm * 3))]
# Show p-val corresponding to diff between src and src_sph (index 2) and
#      p-val corresponding to diff between src and sens (index 1)
p_val_text_auto = ['p=%0.2e' % model_results.pvalues[ind] for ind in [2, 1]]

# Manually reformat p-vals to use  10^x format (no automatic way to do this)
p_val_text = [r'p=$3.66 \times 10^{-3}$', r'p=$1.49 \times 10^{-6}$']
print '\nWARNING: double check calculated p-vals:' + str(p_val_text_auto) + \
    ' match hard-coded text' + str(p_val_text)

bar_labels = [method.upper() for method in decomp_methods] * 3
bw = 0.4

# Properties for `barplot` call
fig_kw = {}
bar_kw = {'linewidth': 2, 'color': '1'}
line_kw = {'color': 'k', 'marker': 'o', 'fillstyle': 'full', 'markersize': 5,
           'alpha': 0.3, 'lw': 0}
error_kw = {'ecolor': 'k', 'linewidth': 2, 'capsize': 5, 'capthick': 1.25}
bracket_kw = {'color': 'k', 'clip_on': False}
pval_kwargs = {'fontsize': 10.5}

#######################################
# Create figure and plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width_1col * 1.5, 4.5),
                       sharey=True, facecolor='white')

fig.subplots_adjust(top=0.75, bottom=0.2, left=0.14, right=0.975)
p, b = barplot(data, axis=0, ylim=[45, 70], err_bars='se', lines=False,
               groups=groups, gap_size=0.4, brackets=brackets,
               bracket_text=p_val_text, pval_kwargs=pval_kwargs,
               bracket_group_lines=True, bar_names=bar_labels,
               bar_kwargs=bar_kw, err_kwargs=error_kw, line_kwargs=line_kw,
               bracket_kwargs=bracket_kw, ax=ax, smart_defaults=False)

ax.axhline(50., c='r', ls='dashed', label='Chance', lw=1.25)
ax.set_ylabel('Accuracy (%)', fontsize=label_fontsize + 2)

# Add text showing actual scores
for mi, mean in enumerate(data_means):
    bbox = b[mi].get_bbox()
    xpos = (bbox.x0 + bbox.x1) / 2.
    ax.annotate('%0.1f' % (mean), xy=(xpos, 47), xycoords=ax.transData,
                ha='center', va='center', fontsize=9, color='0.35')

# Fix ticks and margins
ax.locator_params(axis='y', nbins=6)
ax.tick_params(axis='both', labelsize=tickLabel_fontsize)
ax.margins(0.5)

#######################################
# Add x-axis labels
xpos = (b[1].get_bbox().x0 + b[1].get_bbox().x1) / 2.
ypos = ax.get_ylim()[0] - 4
ypos_low = ax.get_ylim()[0] - 7
lw = 1
ax.annotate('Sensor', xy=(xpos, ypos), xycoords=ax.transData,
            va='center', ha='center', fontsize=16)

xpos = (b[5].get_bbox().x1 + b[6].get_bbox().x0) / 2.
ax.annotate('Source', xy=(xpos, ypos), xycoords=ax.transData,
            va='center', ha='center', fontsize=16)

xpos = (b[4].get_bbox().x0 + b[4].get_bbox().x1) / 2.
ax.annotate('(generic)', xy=(xpos, ypos_low), xycoords=ax.transData,
            va='center', ha='center', fontsize=12)
# add two lines connecting labels
ax.annotate('', xy=(xpos, ypos_low + 1.25), xycoords=ax.transData,
            xytext=(xpos, ypos + 0.25), ha='center', fontsize=16,
            arrowprops=dict(arrowstyle='-', lw=lw))
ax.annotate('', xy=(xpos - 0.03, ypos), xycoords=ax.transData,
            xytext=(xpos + 0.45, ypos), va='center', ha='center', fontsize=16,
            arrowprops=dict(arrowstyle='-', lw=lw))

xpos = (b[7].get_bbox().x0 + b[7].get_bbox().x1) / 2.
ax.annotate('(individualized)', xy=(xpos, ypos_low), xycoords=ax.transData,
            va='center', ha='center', fontsize=12)
# add two lines connecting labels
ax.annotate('', xy=(xpos, ypos_low + 1.25), xycoords=ax.transData,
            xytext=(xpos, ypos + 0.25), ha='center', fontsize=16,
            arrowprops=dict(arrowstyle='-', lw=lw))
ax.annotate('', xy=(xpos + 0.03, ypos), xycoords=ax.transData,
            xytext=(xpos - 0.45, ypos), va='center', ha='center', fontsize=16,
            arrowprops=dict(arrowstyle='-', lw=lw))

###############################################################################
# SAVE PLOT

if savePlot:
    interaction = '_interaction' if stats_interaction else '_noInteraction'
    plot_fname = op.join(save_dir, 'AccDif_' + '_'.join(decomp_methods) +
                         '_bin' + bin_width + 'ms' + interaction)
    fig.savefig(plot_fname + '.pdf')
    fig.savefig(plot_fname + '.png')
