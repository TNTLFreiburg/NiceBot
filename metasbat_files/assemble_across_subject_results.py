# %% Imports
import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from utils import svg_stack as ss
from utils.df_utils import fill_results_in_configurations_at_index
from visualizations.plots import plot_transfer_matrix

mpl.rcParams['figure.dpi'] = 300

# %%
configurations_file = '/mnt/meta-cluster/home/fiederer/nicebot/metasbat_files/configs_no_valid.csv'
# configurations_file = '/home/lukas/nicebot/metasbat_files/configs_all_models.csv'
configurations = pd.read_csv(configurations_file)  # Load existing configs

# Rename fields to make plots easier to understand
assert all(configurations['data'].unique() == ['onlyRobotData', 'onlyAux', 'onlyEEGData', 'RobotEEGAux', 'RobotEEG'])
for old_data_type, new_data_type in zip(configurations['data'].unique(), ['Robot', 'Aux', 'EEG',
                                                                          'Robot+Aux+EEG', 'Robot+EEG']):
    configurations.replace(old_data_type, new_data_type, inplace=True)
configurations = configurations[configurations['data'] != 'Robot EEG']

assert all(
    configurations['model_name'].unique() == ['eegnet', 'deep4', 'resnet', 'lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg'])
for old_model_name, new_model_name in zip(configurations['model_name'].unique(), ['EEGNetv4', 'Deep4Net',
                                                                                  'EEGResNet-29',
                                                                                  'Linear', 'Linear SV',
                                                                                  'Radial basis SV',
                                                                                  'Random forest']):
    configurations.replace(old_model_name, new_model_name, inplace=True)

# Initialize storage. One needs to create a new list for each, else these are just linked copies of each other and
# values get updated everywhere
configurations['config_has_run'] = False
configurations['mse_train'] = np.tile(np.zeros(9) * np.nan, (len(configurations),1)).tolist()
configurations['mse_valid'] = np.tile(np.zeros(9) * np.nan, (len(configurations),1)).tolist()
configurations['mse_test'] = np.tile(np.zeros(9) * np.nan, (len(configurations),1)).tolist()
configurations['corr_train'] = np.tile(np.zeros(9) * np.nan, (len(configurations),1)).tolist()
configurations['corr_valid'] = np.tile(np.zeros(9) * np.nan, (len(configurations),1)).tolist()
configurations['corr_test'] = np.tile(np.zeros(9) * np.nan, (len(configurations),1)).tolist()
configurations['corr_p_train'] = np.tile(np.zeros(9) * np.nan, (len(configurations),1)).tolist()
configurations['corr_p_valid'] = np.tile(np.zeros(9) * np.nan, (len(configurations),1)).tolist()
configurations['corr_p_test'] = np.tile(np.zeros(9) * np.nan, (len(configurations),1)).tolist()
configurations['r^2_train'] = np.tile(np.zeros(9) * np.nan, (len(configurations), 1)).tolist()
configurations['r^2_valid'] = np.tile(np.zeros(9) * np.nan, (len(configurations), 1)).tolist()
configurations['r^2_test'] = np.tile(np.zeros(9) * np.nan, (len(configurations), 1)).tolist()

for i_config in configurations.index:
    result_folder = configurations.loc[i_config, 'result_folder']
    unique_id = configurations.loc[i_config, 'unique_id']
    matching_results = np.sort(glob.glob('/mnt/meta-cluster' + result_folder + '/*' + unique_id + '*Exp.csv'))
    # TODO: Check that we get only one result per subject, but not sure what to do if not. Check that results are
    #  identical?

    if any(matching_results):
        configurations.loc[i_config, 'config_has_run'] = True
        i_subject = -1
        for subject in matching_results:
            i_subject += 1
            df = pd.read_csv(subject)
            configurations = fill_results_in_configurations_at_index(configurations, i_config, df, i_subject)

# configurations.to_csv(configurations_file[:-4] + '_with_results.csv')
# Somehow converts my lists into strings!!!


#%% Load only missing results only instead of having to rerun the above every time
for i_config in configurations.index:
    if np.isnan(configurations.loc[i_config, 'mse_train']).any():
        result_folder = configurations.loc[i_config, 'result_folder']
        unique_id = configurations.loc[i_config, 'unique_id']
        matching_results = np.sort(glob.glob('/mnt/meta-cluster/' + result_folder + '/*' + unique_id + '*.csv'))
        # TODO: Check that we get only one result per subject, but not sure what to do if not. Check that results are
        #  identical?

        if any(matching_results):
            configurations.loc[i_config, 'config_has_run'] = True
            i_subject = -1
            for subject in matching_results:
                i_subject += 1
                df = pd.read_csv(subject)
                configurations = fill_results_in_configurations_at_index(configurations, i_config, df, i_subject)

# %% Plot transfer matrices
data_conditions = configurations['data'].unique()
metric_conditions = ['mse_test', 'corr_test']
condition_values = ['S1 (no exp.)', 'S2 (mod. exp.)', 'S3 (subst. exp.)']
# subject_original_order = ['mod2mod', 'mod2no', 'mod2sub', 'no2mod', 'no2no', 'no2sub', 'sub2mod', 'sub2no',
#                           'sub2sub']  # Alphabetical order
subject_new_order_indices = [4, 3, 5, 1, 0, 2, 7, 6, 8]
# subject_new_order = np.array(subject_original_order)[subject_new_order_indices]
# subject_matrix = subject_new_order.reshape((3, 3))
# fig, ax = plt.subplots()
# ax.set(xticks=np.arange(subject_matrix.shape[1]),
#        yticks=np.arange(subject_matrix.shape[0]),
#        # ... and label them with the respective list entries
#        xticklabels=condition_values, yticklabels=condition_values,
#        title='Reference matrix',
#        ylabel='Subject trained on',
#        xlabel='Subject tested on')
# plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
#          rotation_mode="anchor")
# plt.setp(ax.get_yticklabels(), rotation=90, ha="center",
#          rotation_mode="anchor")
# for i in range(subject_matrix.shape[0]):
#     for j in range(subject_matrix.shape[1]):
#         ax.text(j, i, subject_matrix[i, j],
#                 ha='center', va='center',
#                 color='black')
# plt.savefig('reference_transfer_matrix.svg', dpi=300)

for data in data_conditions:
    for model in configurations['model_name'].unique():
        for metric in metric_conditions:
            transfer_matrix = configurations[(configurations['model_name'] == model) & (configurations['data'] ==
                                                                                        data)][metric].values[0]
            transfer_matrix = np.array(transfer_matrix)[subject_new_order_indices].reshape((3, 3))
            if metric == 'mse_test':
                cmap = LinearSegmentedColormap.from_list('white2colorblind',
                                                         [[1, 1, 1], sns.color_palette(
                                                             'colorblind')[0]], N=256).reversed()
                transfer_matrix = np.sqrt(transfer_matrix)
                clabel = 'Root mean square error'
            elif metric == 'corr_test':
                cmap = LinearSegmentedColormap.from_list('white2colorblind',
                                                         [[1, 1, 1], sns.color_palette(
                                                             'colorblind')[1]], N=256)
                clabel = "Pearson's rho"
            else:
                cmap = 'Blues'
                clabel = ''
            plot_transfer_matrix(transfer_matrix, condition_values=condition_values,
                                 xlabel='Subject tested on', ylabel='Subject trained on', clabel=clabel,
                                 title=model + ' regressor, ' + data + ' data', cmap=cmap)
            plt.savefig(model.replace(' ', '_') + '_' + data.replace(' ', '_') + '_' + metric.replace(' ',
                                                                                                      '_') + '_transfer_matrix.svg',
                        dpi=300)
            plt.close()
            # plt.show()

# %% Stack svgs to single figures
model = configurations['model_name'].unique()
for data in data_conditions:
    for metric in metric_conditions:
        doc = ss.Document()
        layout0 = ss.VBoxLayout()
        layout0.addSVG(model[0].replace(' ', '_') + '_' + data.replace(' ', '_') + '_' + metric.replace(' ',
                                                                                                        '_') + '_transfer_matrix.svg',
                       alignment=ss.AlignHCenter)
        layout1 = ss.HBoxLayout()
        layout1.addSVG(model[1].replace(' ', '_') + '_' + data.replace(' ', '_') + '_' + metric.replace(' ',
                                                                                                        '_') + '_transfer_matrix.svg',
                       alignment=ss.AlignVCenter)
        layout1.addSVG(model[2].replace(' ', '_') + '_' + data.replace(' ', '_') + '_' + metric.replace(' ',
                                                                                                        '_') + '_transfer_matrix.svg',
                       alignment=ss.AlignVCenter)

        layout2 = ss.HBoxLayout()
        layout2.addSVG(model[3].replace(' ', '_') + '_' + data.replace(' ', '_') + '_' + metric.replace(' ',
                                                                                                        '_') + '_transfer_matrix.svg',
                       alignment=ss.AlignVCenter)
        layout2.addSVG(model[4].replace(' ', '_') + '_' + data.replace(' ', '_') + '_' + metric.replace(' ',
                                                                                                        '_') + '_transfer_matrix.svg',
                       alignment=ss.AlignVCenter)

        layout3 = ss.HBoxLayout()
        layout3.addSVG(model[5].replace(' ', '_') + '_' + data.replace(' ', '_') + '_' + metric.replace(' ',
                                                                                                        '_') + '_transfer_matrix.svg',
                       alignment=ss.AlignVCenter)
        layout3.addSVG(model[6].replace(' ', '_') + '_' + data.replace(' ', '_') + '_' + metric.replace(' ',
                                                                                                        '_') + '_transfer_matrix.svg',
                       alignment=ss.AlignVCenter)

        layout0.addLayout(layout1)
        layout0.addLayout(layout2)
        layout0.addLayout(layout3)

        doc.setLayout(layout0)
        doc.save('transfer_matrix_' + data.replace(' ', '_') + '_' + metric.replace(' ', '_') + '.svg')

# %%
###################
# Work in progress#
###################
for metric in metric_conditions:
    for data in data_conditions:
        doc = ss.Document()
        layout0 = ss.VBoxLayout()
        layout0.addSVG(configurations['model_name'].unique()[0] + '_' + data + '_transfer_matrix.svg',
                       alignment=ss.AlignHCenter)
        for model_index, model in enumerate(configurations['model_name'].unique()[1:]):
            if np.mod(model_index, 2) != 0:
                layout1 = ss.HBoxLayout()
                layout1.addSVG(model + '_' + data + '_transfer_matrix.svg', alignment=ss.AlignVCenter)

            layout2 = ss.HBoxLayout()
            layout2.addSVG('subject_performance_matrix.svg', alignment=ss.AlignVCenter)
            layout2.addSVG('frequency_performance_matrix.svg', alignment=ss.AlignVCenter | ss.AlignRight)

            layout0.addLayout(layout1)
            layout0.addLayout(layout2)

        doc.setLayout(layout0)
        doc.save('performance_matrix_plot.svg')
