# %% Imports
import glob
import re

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

import utils.svg_stack as ss
from utils.df_utils import initialize_empty_columns_at_index, prefill_columns_configuration_at_index, \
    fill_results_in_configurations_at_index, compute_performance_matrix
from utils.statistics import fdr_corrected_pvals
from visualizations.plots import split_combi_plot, add_significance_bars_above, add_significance_bars_below, \
    plot_diff_matrix

mpl.rcParams['figure.dpi'] = 300

# %% Load all existing single subject results
configurations_file = '/mnt/meta-cluster/home/fiederer/nicebot/metasbat_files/configs_no_valid.csv'
# configurations_file = '/home/lukas/nicebot/metasbat_files/configs_all_models.csv'
configurations = pd.read_csv(configurations_file)  # Load existing configs

# Rename fields to make plots easier to understand
assert all(configurations['data'].unique() == ['onlyRobotData', 'onlyAux', 'onlyEEGData', 'RobotEEGAux', 'RobotEEG'])
configurations = configurations[configurations['data'] != 'RobotEEG']
for old_data_type, new_data_type in zip(configurations['data'].unique(), ['Robot', 'Periphery', 'EEG',
                                                                          'Combined']):
    configurations.replace(old_data_type, new_data_type, inplace=True)

assert all(
    configurations['model_name'].unique() == ['eegnet', 'deep4', 'resnet', 'lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg'])
for old_model_name, new_model_name in zip(configurations['model_name'].unique(), ['EEGNetv4', 'Deep4Net',
                                                                                  'EEGResNet-29',
                                                                                  'Linear', 'Linear SV',
                                                                                  'Radial basis SV',
                                                                                  'Random forest']):
    configurations.replace(old_model_name, new_model_name, inplace=True)

for i_config in configurations.index:
    result_folder = configurations.loc[i_config, 'result_folder']  # '/data/schirrmr/fiederer/nicebot/results'  #
    unique_id = configurations.loc[i_config, 'unique_id']
    matching_results = np.sort(glob.glob('/mnt/meta-cluster/' + result_folder + '/*' + unique_id + '*Exp.csv'))
    # matching_results = np.sort(glob.glob(result_folder + '/*' + unique_id[:7] + '.csv'))
    # Check that we get only one result per subject, but not sure what to do if not. Check that results are identical?

    configurations = initialize_empty_columns_at_index(configurations, i_config)

    if any(matching_results):

        configurations = prefill_columns_configuration_at_index(configurations, i_config, has_run=True)
        i_subject = -1
        for subject in matching_results:
            # Check if this subject was applied on himself.
            word_list = re.split('[_/.]', subject)  # Split path to .csv file into single words delimited by _ /
            # and .
            if word_list.count(word_list[-2]) == 2:  # If the second to
                # last word (either subject model was applied to or unique ID of the experiment) is present twice
                # then the model was trained on the same subject
                i_subject += 1
                df = pd.read_csv(subject)  # , names=['epoch',
                # 'train_loss','valid_loss','test_loss',
                # 'train_corr','valid_corr','test_corr', 'runtime'])
                configurations = fill_results_in_configurations_at_index(configurations, i_config, df, i_subject)
    else:
        configurations = prefill_columns_configuration_at_index(configurations, i_config, has_run=False)

# configurations.to_csv(configurations_file[:-4] + '_with_results.csv')
# Somehow converts my lists into strings!!!

# %% Load only missing results only instead of having to rerun the above every time
for i_config in configurations.index:
    if np.isnan(configurations.loc[i_config, 'mse_train']).any():
        print('Missing results in config {}'.format(i_config))
        result_folder = configurations.loc[
            i_config, 'result_folder']  # '/data/schirrmr/fiederer/nicebot/results_without_bp_bug'  #
        unique_id = configurations.loc[i_config, 'unique_id']
        matching_results = np.sort(glob.glob('/mnt/meta-cluster/' + result_folder + '/*' + unique_id + '*Exp.csv'))
        # matching_results = np.sort(glob.glob(result_folder + '/*' + unique_id[:7] + '.csv'))
        # Check that we get only one result per subject, but not sure what to do if not. Check that results are identical?

        configurations = initialize_empty_columns_at_index(configurations, i_config)

        if any(matching_results):
            configurations = prefill_columns_configuration_at_index(configurations, i_config, True)
            i_subject = -1
            for subject in matching_results:  # Check if this subject was applied on himself.
                word_list = re.split('[_/.]', subject)  # Split path to .csv file into single words delimited by _ /
                # and .
                if word_list.count(word_list[-2]) == 2:  # If the second to
                    # last word (either subject model was applied to or unique ID of the experiment) is present twice
                    # then the model was trained on the same subject
                    i_subject += 1
                    df = pd.read_csv(subject)  # , names=['epoch',
                    # 'train_loss','valid_loss','test_loss',
                    # 'train_corr','valid_corr','test_corr', 'runtime'])
                    configurations = fill_results_in_configurations_at_index(configurations, i_config, df, i_subject)

# %% Plot fancy results
# configurations_file = '/mnt/meta-cluster/home/fiederer/nicebot/metasbat_files/configs_all_models_with_results.csv'
# configurations = pd.read_csv(configurations_file)  # Load existing configs

f = plt.figure(figsize=[15, 10])
# plt.autoscale(False)
x_positions = np.array([2, 4, 1, 3, 5, 2, 4])
y_positions = np.array([1, 1, 2, 2, 2, 3, 3])
scaling_factor = 10.0
subplot_index = 1
x_shift = 0
x_shift_increment = 6
y_shift = 0
y_shift_increment = 4
data_values = configurations['data'].unique()
bandpass_values = configurations['band_pass'].unique()
electrodes_values = configurations['electrodes'].unique()
subject_values = np.array(['S2\n(mod. exp.)', 'S1\n(no exp.)', 'S3\n(subst. exp)'])
model_order = [5, 6, 0, 1, 2, 3, 4]
data_split = 'test'

for i_no_eeg_data in np.arange(2):
    for i_subject in np.arange(3):
        plt.subplot(5, 3, subplot_index)
        # plt.title('Subject %g'.format(i_subject+1))
        df = configurations[configurations['data'] == data_values[i_no_eeg_data]]
        assert (df['model_name'].size == 7)
        assert (
            all(df['model_name'].values == ['eegnet', 'deep4', 'resnet', 'lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg']))
        df = df.iloc[model_order]

        # tmp = np.full(7, np.nan)
        metrics = []
        # metrics = [a[i_subject]/b[i_subject] for a, b in zip(df['corr_test'].values, df['mse_test'].values)]
        for i_model in range(df['model_name'].size):
            # if not any(np.isnan(metric)):
            try:
                metrics.append(df['corr_' + data_split].values[i_model][i_subject] /
                               df['mse_' + data_split].values[i_model][i_subject])
                # metrics.append(1 / np.sqrt(df['mse_' + data_split].values[i_model][i_subject]))
                # metrics.append(1 / df['mse_' + data_split].values[i_model][i_subject])
                # metrics.append(df['corr_' + data_split].values[i_model][i_subject])
            # else:
            except:
                metrics.append(0)

        plt.scatter(x_positions, y_positions, s=np.abs(np.array(metrics)) * scaling_factor,
                    c=('b', 'g', 'r', 'c', 'm', 'y', 'k'))
        plt.xlim(np.array([np.mean(x_positions) - (x_shift_increment * bandpass_values.size), np.mean(x_positions) + (
                x_shift_increment * bandpass_values.size)]) / 2)
        plt.ylim(np.array([np.mean(y_positions) - (y_shift_increment * electrodes_values.size), np.mean(y_positions) + (
                y_shift_increment * electrodes_values.size)]) / 2)
        plt.xticks([], ())
        # plt.yticks(np.unique(y_positions), ('linear', 'neural net', 'non-linear'), rotation=45)
        plt.yticks([], ())

        subplot_index += 1

x_shift = 0
y_shift = 0
for i_eeg_data in np.arange(3):
    for i_subject in np.arange(3):
        plt.subplot(5, 3, subplot_index)
        for i_bandpass in np.arange(7):
            for i_electrode in np.arange(3):
                df = configurations[(configurations['data'] == data_values[i_eeg_data + 2]) &
                                    (configurations['band_pass'] == bandpass_values[i_bandpass]) &
                                    (configurations['electrodes'] == electrodes_values[i_electrode])]
                # df = df[df['band_pass'] == ]
                assert (df['model_name'].size == 7)
                assert (all(df['model_name'].values == ['EEGNetv4', 'Deep4Net', 'EEGResNet-29', 'Linear',
                                                        'Linear SV', 'Radial basis SV', 'Random forest']))
                df = df.iloc[model_order]

                # tmp = np.full(7, np.nan)
                metrics = []
                # metrics = [a[i_subject]/b[i_subject] for a, b in zip(df['corr_test'].values, df['mse_test'].values)]
                for i_model in range(df['model_name'].size):
                    # if not any(np.isnan(metric)):
                    try:
                        metrics.append(df['corr_' + data_split].values[i_model][i_subject] /
                                       df['mse_' + data_split].values[i_model][i_subject])
                        # metrics.append(1 / np.sqrt(df['mse_test'].values[i_model][i_subject]))
                        # metrics.append(1 / df['mse_test'].values[i_model][i_subject])
                        # metrics.append(df['corr_test'].values[i_model][i_subject])
                    # else:
                    except:
                        metrics.append(0)

                plt.scatter(x_positions + x_shift, y_positions + y_shift, s=np.abs(np.array(metrics)) * scaling_factor,
                            c=('b', 'g', 'r', 'c', 'm', 'y', 'k'))
                plt.xlim([0, x_shift_increment * bandpass_values.size])
                plt.ylim([0, y_shift_increment * electrodes_values.size])
                y_shift += y_shift_increment
            x_shift += x_shift_increment
            y_shift = 0
        x_shift = 0
        subplot_index += 1
        plt.xticks(x_positions.mean() + np.arange(x_shift, x_shift_increment * (i_bandpass) + 1, x_shift_increment),
                   ('None', '[0,4]', '[4,8]', '[8,14]', '[14,20]',
                    '[20,30]', '[30,40]'))
        plt.yticks(y_positions.mean() + np.arange(y_shift, y_shift_increment * (i_electrode) + 1, y_shift_increment),
                   ('*', '*z', '*C*'))
        # plt.axis('tight')
# plt.ylabel('Electrode selection')
# plt.xlabel('Band-pass (Hz)')

for i_data in range(data_values.size):
    plt.text(0.05, 0.166 * (i_data + 1), data_values[::-1][i_data], horizontalalignment='center',
             verticalalignment='center',
             transform=f.transFigure, rotation=90)

for i_subject in range(subject_values.size):
    plt.text(0.25 * (i_subject + 1), 0.9, subject_values[i_subject], horizontalalignment='center',
             verticalalignment='center',
             transform=f.transFigure)

plt.text(0.5, 0.95, 'Data split: ' + data_split + ' set', horizontalalignment='center',
         verticalalignment='center',
         transform=f.transFigure)
# plt.figlegend(['lin_reg', 'lin_svr', 'eegNet', 'deep4', 'resNet', 'rbf_svr', 'rf_reg'], loc='center right')#, bbox_to_anchor=(1, 0.5))
plt.show()

# %% Overall best performing method
# data_split_values = ['train', 'test']
data_split_values = ['test']
model_values = configurations['model_name'].unique()
assert (all(model_values == ['EEGNetv4', 'Deep4Net', 'EEGResNet-29', 'Linear', 'Linear SV', 'Radial basis SV',
                             'Random forest']))

# f.axes[0].set_xscale("log", nonposx='clip')
# f.axes[0].set_xscale("log")
metric_df = pd.DataFrame()
for data_split in data_split_values:
    # subplot_index += 1
    for model_name in model_values:
        df_mse = configurations[(configurations['model_name'] == model_name)]['mse_' + data_split]
        df_corr = configurations[(configurations['model_name'] == model_name)]['corr_' + data_split]
        # x = np.abs([np.array([i if i is not None else np.nan for i in b]) /
        #             np.array([i if i is not None else np.nan for i in a])
        #             for a, b in zip(df_mse, df_corr)])
        x = np.array([i if i is not None else np.nan for i in df_mse])
        y = np.array([i if i is not None else np.nan for i in df_corr])
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        if any(x):
            metric_df = metric_df.append(pd.DataFrame({'Models': model_name,
                                                       'data_split': data_split,
                                                       'metric_value': np.sqrt(x),
                                                       'Metrics': 'Root mean square error',
                                                       'data': 'Robot, Aux, EEG, Robot+Aux+EEG'}))
            metric_df = metric_df.append(pd.DataFrame({'Models': model_name,
                                                       'data_split': data_split,
                                                       'metric_value': y,
                                                       'Metrics': "Pearson's rho",
                                                       'data': 'Robot, Aux, EEG, Robot+Aux+EEG'}))

f = plt.figure(figsize=[15, 10])
split_combi_plot(f, metric_df, x_col='Models', y_col='metric_value', hue_col='Metrics', palette='colorblind')
add_significance_bars_above(metric_df, condition_col='Models', metric_name_col='Metrics',
                            metric_name='Root mean square error',
                            metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
add_significance_bars_below(metric_df, condition_col='Models', metric_name_col='Metrics', metric_name="Pearson's rho",
                            metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
plt.title('Regressors')
handles, labels = f.axes[0].get_legend_handles_labels()
f.axes[0].legend(handles=handles[0:2], labels=labels[0:2], title='Metrics', loc=2)
plt.savefig('model_violins.svg', bbox_inches='tight', dpi=300)
plt.close()

# f = plt.figure(figsize=[15, 10])
# split_combi_plot(f, metric_df, x_col='model', y_col='metric_value', hue_col='Metrics', palette='colorblind')
# add_significance_bars_below(metric_df, condition_col='model', metric_name_col='Metrics', metric_name="Pearson's rho",
#                             metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
# plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
# plt.show()

plot_diff_matrix(metric_df, condition_col='Models', metric_name_col='Metrics', metric_name=['Root mean square error',
                                                                                            "Pearson's rho"],
                 metric_value_col='metric_value',
                 title='Regressors',
                 cmap='gray')
plt.savefig('model_diff_matrix.pdf', bbox_inches='tight', dpi=300)
plt.close()

# %% Best performing method without selection
# data_split_values = ['train', 'test']
data_split_values = ['test']
model_values = configurations['model_name'].unique()
assert (all(model_values == ['EEGNetv4', 'Deep4Net', 'EEGResNet-29', 'Linear', 'Linear SV', 'Radial basis SV',
                             'Random forest']))

# f.axes[0].set_xscale("log", nonposx='clip')
# f.axes[0].set_xscale("log")
metric_df = pd.DataFrame()
for data_split in data_split_values:
    # subplot_index += 1
    for model_name in model_values:
        df_mse = configurations[(configurations['model_name'] == model_name) &
                                (configurations['band_pass'] == "'[None, None]'") &
                                (configurations['electrodes'] == "'*'")]['mse_' + data_split]
        df_corr = configurations[(configurations['model_name'] == model_name) &
                                 (configurations['band_pass'] == "'[None, None]'") &
                                 (configurations['electrodes'] == "'*'")]['corr_' + data_split]
        # x = np.abs([np.array([i if i is not None else np.nan for i in b]) /
        #             np.array([i if i is not None else np.nan for i in a])
        #             for a, b in zip(df_mse, df_corr)])
        x = np.array([i if i is not None else np.nan for i in df_mse])
        y = np.array([i if i is not None else np.nan for i in df_corr])
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        if any(x):
            metric_df = metric_df.append(pd.DataFrame({'Models': model_name,
                                                       'data_split': data_split,
                                                       'metric_value': np.sqrt(x),
                                                       'Metrics': 'Root mean square error',
                                                       'data': 'Robot, Aux, EEG, Robot+Aux+EEG'}))
            metric_df = metric_df.append(pd.DataFrame({'Models': model_name,
                                                       'data_split': data_split,
                                                       'metric_value': y,
                                                       'Metrics': "Pearson's rho",
                                                       'data': 'Robot, Aux, EEG, Robot+Aux+EEG'}))

f = plt.figure(figsize=[15, 10])
split_combi_plot(f, metric_df, x_col='Models', y_col='metric_value', hue_col='Metrics', palette='colorblind')
add_significance_bars_above(metric_df, condition_col='Models', metric_name_col='Metrics',
                            metric_name='Root mean square error',
                            metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
add_significance_bars_below(metric_df, condition_col='Models', metric_name_col='Metrics', metric_name="Pearson's rho",
                            metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
plt.title('Regressors')
handles, labels = f.axes[0].get_legend_handles_labels()
f.axes[0].legend(handles=handles[0:2], labels=labels[0:2], title='Metrics', loc=2)
plt.savefig('model_no_selection_violins.svg', bbox_inches='tight', dpi=300)
plt.close()

# f = plt.figure(figsize=[15, 10])
# split_combi_plot(f, metric_df, x_col='model', y_col='metric_value', hue_col='Metrics', palette='colorblind')
# add_significance_bars_below(metric_df, condition_col='model', metric_name_col='Metrics', metric_name="Pearson's rho",
#                             metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
# plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
# plt.show()

plot_diff_matrix(metric_df, condition_col='Models', metric_name_col='Metrics', metric_name=['Root mean square error',
                                                                                            "Pearson's rho"],
                 metric_value_col='metric_value',
                 title='Regressors',
                 cmap='gray')
plt.savefig('model_no_selection_diff_matrix.pdf', bbox_inches='tight', dpi=300)
plt.close()

# %% Best performing method for each data type

data_values = configurations['data'].unique()
data_split_values = ['test']
model_values = configurations['model_name'].unique()
assert (all(model_values == ['EEGNetv4', 'Deep4Net', 'EEGResNet-29', 'Linear', 'Linear SV', 'Radial basis SV',
                             'Random forest']))

for data in data_values:
    df = configurations[configurations['data'] == data]

    metric_df = pd.DataFrame()
    for data_split in data_split_values:
        for model_name in model_values:
            df_mse = df[(df['model_name'] == model_name)]['mse_' + data_split]
            df_corr = df[(df['model_name'] == model_name)]['corr_' + data_split]
            # x = np.abs([np.array([i if i is not None else np.nan for i in b]) /
            #             np.array([i if i is not None else np.nan for i in a])
            #             for a, b in zip(df_mse, df_corr)])
            x = np.array([i if i is not None else np.nan for i in df_mse])
            y = np.array([i if i is not None else np.nan for i in df_corr])
            x = x[~np.isnan(x)]
            y = y[~np.isnan(y)]
            if any(x):
                metric_df = metric_df.append(pd.DataFrame({'Models': model_name,
                                                           'data_split': data_split,
                                                           'metric_value': np.sqrt(x),
                                                           'Metrics': 'Root mean square error',
                                                           'data': data}))
                metric_df = metric_df.append(pd.DataFrame({'Models': model_name,
                                                           'data_split': data_split,
                                                           'metric_value': y,
                                                           'Metrics': "Pearson's rho",
                                                           'data': data}))

    f = plt.figure(figsize=[15, 10])
    split_combi_plot(f, metric_df, x_col='Models', y_col='metric_value', hue_col='Metrics', palette='colorblind')
    add_significance_bars_above(metric_df, condition_col='Models', metric_name_col='Metrics',
                                metric_name='Root mean square error',
                                metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
    add_significance_bars_below(metric_df, condition_col='Models', metric_name_col='Metrics', metric_name="Pearson's "
                                                                                                          "rho",
                                metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
    plt.title(data)
    handles, labels = f.axes[0].get_legend_handles_labels()
    f.axes[0].legend(handles=handles[0:2], labels=labels[0:2], title='Metrics', loc=2)
    plt.savefig('model_' + data.replace(' ', '_') + '_violins.svg', bbox_inches='tight', dpi=300)
    plt.close()

    # f = plt.figure(figsize=[15, 10])
    # split_combi_plot(f, metric_df, x_col='Models', y_col='metric_value', hue_col='Metrics', palette='colorblind')
    # add_significance_bars_above(metric_df, condition_col='Models', metric_name_col='Metrics', metric_name="Pearson's rho",
    #                             metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
    # plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
    # plt.show()

    plot_diff_matrix(metric_df, condition_col='Models', metric_name_col='Metrics',
                     metric_name=['Root mean square error',
                                  "Pearson's rho"],
                     metric_value_col='metric_value',
                     title=data,
                     cmap='gray')
    plt.savefig('model_' + data.replace(' ', '_') + '_diff_matrix.pdf', bbox_inches='tight', dpi=300)
    plt.close()

# %% Best electrode set EEG only
# data_split_values = ['train', 'test']
data_split_values = ['test']
electrode_sets = configurations['electrodes'].unique()
assert (all(electrode_sets == ["'*'", "'*z'", "'*C*'"]))

metric_df = pd.DataFrame()
for electrode_set_name in electrode_sets:
    for data_split in data_split_values:
        df_mse = configurations[(configurations['electrodes'] == electrode_set_name)
                                & (configurations['data'] == 'EEG')]['mse_' + data_split]
        df_corr = configurations[(configurations['electrodes'] == electrode_set_name)
                                 & (configurations['data'] == 'EEG')]['corr_' + data_split]
        x = np.array([i if i is not None else np.nan for i in df_mse])
        y = np.array([i if i is not None else np.nan for i in df_corr])
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        if any(x):
            metric_df = metric_df.append(pd.DataFrame({'Electrode selection': electrode_set_name[1:-1],
                                                       'data_split': data_split,
                                                       'metric_value': np.sqrt(x),
                                                       'Metrics': 'Root mean square error',
                                                       'data': 'EEG'}))
            metric_df = metric_df.append(pd.DataFrame({'Electrode selection': electrode_set_name[1:-1],
                                                       'data_split': data_split,
                                                       'metric_value': y,
                                                       'Metrics': "Pearson's rho",
                                                       'data': 'EEG'}))

f = plt.figure(figsize=[15, 10])
split_combi_plot(f, metric_df, x_col='Electrode selection', y_col='metric_value', hue_col='Metrics',
                 palette='colorblind')
add_significance_bars_above(metric_df, condition_col='Electrode selection', metric_name_col='Metrics',
                            metric_name='Root mean square error',
                            metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
add_significance_bars_below(metric_df, condition_col='Electrode selection', metric_name_col='Metrics',
                            metric_name="Pearson's rho",
                            metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
plt.title('Electrodes (EEG data)')
handles, labels = f.axes[0].get_legend_handles_labels()
f.axes[0].legend(handles=handles[0:2], labels=labels[0:2], title='Metrics', loc=2)
plt.savefig('electrode_violins.svg', bbox_inches='tight', dpi=300)
plt.close()

# f = plt.figure(figsize=[15, 10])
# split_combi_plot(f, metric_df, x_col='Electrode selection', y_col='metric_value', hue_col='Metrics', palette='colorblind')
# add_significance_bars_above(metric_df, condition_col='Electrode selection', metric_name_col='Metrics', metric_name="Pearson's rho",
#                             metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
# plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
# plt.show()

plot_diff_matrix(metric_df, condition_col='Electrode selection', metric_name_col='Metrics',
                 metric_name=['Root mean square error',
                              "Pearson's rho"],
                 metric_value_col='metric_value',
                 title='Electrode selections (EEG data)',
                 cmap='gray')
plt.savefig('electrode_diff_matrix.pdf', bbox_inches='tight', dpi=300)
plt.close()

# %% Best data combination
# data_split_values = ['train', 'test']
data_split_values = ['test']
data_sets = configurations['data'].unique()

metric_df = pd.DataFrame()
for data_split in data_split_values:
    for data in data_sets:
        df_mse = configurations[(configurations['data'] == data)]['mse_' + data_split]
        df_corr = configurations[(configurations['data'] == data)]['corr_' + data_split]
        # x = np.abs([np.array([i if i is not None else np.nan for i in b]) /
        #             np.array([i if i is not None else np.nan for i in a])
        #             for a, b in zip(df_mse, df_corr)])
        x = np.array([i if i is not None else np.nan for i in df_mse])
        y = np.array([i if i is not None else np.nan for i in df_corr])
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        if any(x):
            metric_df = metric_df.append(pd.DataFrame({'data_split': data_split,
                                                       'metric_value': np.sqrt(x),
                                                       'Metrics': 'Root mean square error',
                                                       'Data': data}))
            metric_df = metric_df.append(pd.DataFrame({'data_split': data_split,
                                                       'metric_value': y,
                                                       'Metrics': "Pearson's rho",
                                                       'Data': data}))

f = plt.figure(figsize=[15, 10])
split_combi_plot(f, metric_df, x_col='Data', y_col='metric_value', hue_col='Metrics', data_type_col='Data',
                 palette='colorblind')
add_significance_bars_above(metric_df, condition_col='Data', metric_name_col='Metrics',
                            metric_name='Root mean square error',
                            metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
add_significance_bars_below(metric_df, condition_col='Data', metric_name_col='Metrics', metric_name="Pearson's rho",
                            metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
plt.title('Data types')
handles, labels = f.axes[0].get_legend_handles_labels()
f.axes[0].legend(handles=handles[0:2], labels=labels[0:2], title='Metrics', loc=2)
plt.savefig('data_violins.svg', bbox_inches='tight', dpi=300)
plt.close()

# f = plt.figure(figsize=[15, 10])
# split_combi_plot(f, metric_df, x_col='Data', y_col='metric_value', hue_col='Metrics', data_type_col='Data', palette='colorblind')
# add_significance_bars_above(metric_df, condition_col='Data', metric_name_col='Metrics', metric_name="Pearson's rho",
#                             metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
# plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
# plt.show()

plot_diff_matrix(metric_df, condition_col='Data', metric_name_col='Metrics', metric_name=['Root mean square error',
                                                                                          "Pearson's rho"],
                 metric_value_col='metric_value',
                 title='Data types',
                 cmap='gray')
plt.savefig('data_diff_matrix.pdf', bbox_inches='tight', dpi=300)
plt.close()

# %% Best frequency band EEG only
# data_split_values = ['train', 'test']
data_split_values = ['test']
bandpass_sets = configurations['band_pass'].unique()

metric_df = pd.DataFrame()
for bandpass in bandpass_sets:
    for data_split in data_split_values:
        df_mse = configurations[(configurations['band_pass'] == bandpass)
                                & (configurations['data'] == 'EEG')]['mse_' + data_split]
        df_corr = configurations[(configurations['band_pass'] == bandpass)
                                 & (configurations['data'] == 'EEG')]['corr_' + data_split]
        x = np.array([i if i is not None else np.nan for i in df_mse])
        y = np.array([i if i is not None else np.nan for i in df_corr])
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        if any(x):
            metric_df = metric_df.append(pd.DataFrame({'Band-pass frequencies': bandpass[1:-1],
                                                       'data_split': data_split,
                                                       'metric_value': np.sqrt(x),
                                                       'Metrics': 'Root mean square error',
                                                       'data': 'EEG'}))
            metric_df = metric_df.append(pd.DataFrame({'Band-pass frequencies': bandpass[1:-1],
                                                       'data_split': data_split,
                                                       'metric_value': y,
                                                       'Metrics': "Pearson's rho",
                                                       'data': 'EEG'}))

f = plt.figure(figsize=[15, 10])
split_combi_plot(f, metric_df, x_col='Band-pass frequencies', y_col='metric_value', hue_col='Metrics',
                 palette='colorblind')
add_significance_bars_above(metric_df, condition_col='Band-pass frequencies', metric_name_col='Metrics',
                            metric_name='Root mean square error',
                            metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
add_significance_bars_below(metric_df, condition_col='Band-pass frequencies', metric_name_col='Metrics',
                            metric_name="Pearson's rho",
                            metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
plt.title('Frequency bands (EEG data)')
handles, labels = f.axes[0].get_legend_handles_labels()
f.axes[0].legend(handles=handles[0:2], labels=labels[0:2], title='Metrics', loc=2)
plt.savefig('frequency_violins.svg', bbox_inches='tight', dpi=300)
plt.close()

# f = plt.figure(figsize=[15, 10])
# split_combi_plot(f, metric_df, x_col='Band-pass frequencies', y_col='metric_value', hue_col='Metrics', palette='colorblind')
# add_significance_bars_above(metric_df, condition_col='Band-pass frequencies', metric_name_col='Metrics', metric_name="Pearson's rho",
#                             metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
# plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
# plt.show()

plot_diff_matrix(metric_df, condition_col='Band-pass frequencies', metric_name_col='Metrics',
                 metric_name=['Root mean square error',
                              "Pearson's rho"],
                 metric_value_col='metric_value',
                 title='Frequency bands (EEG data)',
                 cmap='gray')
plt.savefig('frequency_diff_matrix.pdf', bbox_inches='tight', dpi=300)
plt.close()

# %% Overall best subject
# data_split_values = ['train', 'test']
data_split_values = ['test']
subject_values = np.array(['S2\n(mod. exp.)', 'S1\n(no exp.)', 'S3\n(subst. exp)'])

metric_df = pd.DataFrame()
for data_split in data_split_values:
    for i_subject in range(len(subject_values)):
        df_mse = configurations['mse_' + data_split]
        df_corr = configurations['corr_' + data_split]
        x = np.array([i[i_subject] if i is not None else np.nan for i in df_mse])
        y = np.array([i[i_subject] if i is not None else np.nan for i in df_corr])
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        if any(x):
            metric_df = metric_df.append(pd.DataFrame({'Subjects': subject_values[i_subject],
                                                       'data_split': data_split,
                                                       'metric_value': np.sqrt(x),
                                                       'Metrics': 'Root mean square error',
                                                       'data': 'Robot, Aux, EEG, Robot+Aux+EEG'}))
            metric_df = metric_df.append(pd.DataFrame({'Subjects': subject_values[i_subject],
                                                       'data_split': data_split,
                                                       'metric_value': y,
                                                       'Metrics': "Pearson's rho",
                                                       'data': 'Robot, Aux, EEG, Robot+Aux+EEG'}))

metric_df.sort_values(by='Subjects', ascending=True, inplace=True)
f = plt.figure(figsize=[15, 10])
split_combi_plot(f, metric_df, x_col='Subjects', y_col='metric_value', hue_col='Metrics', palette='colorblind')
add_significance_bars_above(metric_df, condition_col='Subjects', metric_name_col='Metrics',
                            metric_name='Root mean square error',
                            metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
add_significance_bars_below(metric_df, condition_col='Subjects', metric_name_col='Metrics', metric_name="Pearson's rho",
                            metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
plt.title('Subjects')
handles, labels = f.axes[0].get_legend_handles_labels()
f.axes[0].legend(handles=handles[0:2], labels=labels[0:2], title='Metrics', loc=2)
plt.savefig('subject_violins.svg', bbox_inches='tight', dpi=300)
plt.close()

# f = plt.figure(figsize=[15, 10])
# split_combi_plot(f, metric_df, x_col='Subjects', y_col='metric_value', hue_col='Metrics', palette='colorblind')
# add_significance_bars_above(metric_df, condition_col='Subjects', metric_name_col='Metrics', metric_name="Pearson's rho",
#                             metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
# plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
# plt.show()

f = plt.figure(figsize=[15, 10])
plot_diff_matrix(metric_df, condition_col='Subjects', metric_name_col='Metrics', metric_name=['Root mean square error',
                                                                                              "Pearson's rho"],
                 metric_value_col='metric_value',
                 title='Subjects',
                 cmap='gray')
plt.savefig('subject_diff_matrix.pdf', bbox_inches='tight', dpi=300)
plt.close()

# %% Combine SVGs

doc_pm = ss.Document()

layout0 = ss.VBoxLayout()
layout0.addSVG('model_diff_matrix.svg', alignment=ss.AlignHCenter)

layout1 = ss.HBoxLayout()
layout1.addSVG('data_diff_matrix.svg', alignment=ss.AlignVCenter)
layout1.addSVG('electrode_diff_matrix.svg', alignment=ss.AlignVCenter)

layout2 = ss.HBoxLayout()
layout2.addSVG('subject_diff_matrix.svg', alignment=ss.AlignVCenter)
layout2.addSVG('frequency_diff_matrix.svg', alignment=ss.AlignVCenter | ss.AlignRight)

layout0.addLayout(layout1)
layout0.addLayout(layout2)

doc_pm.setLayout(layout0)
doc_pm.save('diff_matrix_plot.svg')

doc_v = ss.Document()
layout0 = ss.VBoxLayout()
layout0.addSVG('model_violins.svg', alignment=ss.AlignHCenter)
layout1 = ss.HBoxLayout()
layout1.addSVG('data_violins.svg', alignment=ss.AlignVCenter)
layout1.addSVG('electrode_violins.svg', alignment=ss.AlignVCenter)
layout2 = ss.HBoxLayout()
layout2.addSVG('subject_violins.svg', alignment=ss.AlignVCenter)
layout2.addSVG('frequency_violins.svg', alignment=ss.AlignVCenter)
layout0.addLayout(layout1)
layout0.addLayout(layout2)
doc_v.setLayout(layout0)
doc_v.save('violins_plot.svg')

doc_v_d = ss.Document()
layout0 = ss.VBoxLayout()
layout1 = ss.HBoxLayout()
layout1.addSVG('model_Robot_violins.svg', alignment=ss.AlignVCenter)
layout1.addSVG('model_Aux_violins.svg', alignment=ss.AlignVCenter)
layout2 = ss.HBoxLayout()
layout2.addSVG('model_EEG_violins.svg', alignment=ss.AlignVCenter)
layout2.addSVG('model_Robot+Aux+EEG_violins.svg', alignment=ss.AlignVCenter)
layout0.addLayout(layout1)
layout0.addLayout(layout2)
doc_v_d.setLayout(layout0)
doc_v_d.save('violins_models4data_plot.svg')

doc_pm_d = ss.Document()
layout0 = ss.VBoxLayout()
layout1 = ss.HBoxLayout()
layout1.addSVG('model_Robot_diff_matrix.svg', alignment=ss.AlignVCenter)
layout1.addSVG('model_Aux_diff_matrix.svg', alignment=ss.AlignVCenter)
layout2 = ss.HBoxLayout()
layout2.addSVG('model_EEG_diff_matrix.svg', alignment=ss.AlignVCenter)
layout2.addSVG('model_Robot+Aux+EEG_diff_matrix.svg', alignment=ss.AlignVCenter)
layout0.addLayout(layout1)
layout0.addLayout(layout2)
doc_v_d.setLayout(layout0)
doc_v_d.save('diff_matrix_models4data_plot.svg')

# %% Plot per model data type overview
data_values = configurations['data'].unique()
data_split_values = ['test']
model_values = configurations['model_name'].unique()
assert (all(model_values == ['EEGNetv4', 'Deep4Net', 'EEGResNet-29', 'Linear', 'Linear SV', 'Radial basis SV',
                             'Random forest']))

for model_name in model_values:
    df = configurations[configurations['model_name'] == model_name]

    metric_df = pd.DataFrame()
    for data_split in data_split_values:
        for data in data_values:
            df_mse = df[(df['data'] == data)]['mse_' + data_split]
            df_corr = df[(df['data'] == data)]['corr_' + data_split]
            # x = np.abs([np.array([i if i is not None else np.nan for i in b]) /
            #             np.array([i if i is not None else np.nan for i in a])
            #             for a, b in zip(df_mse, df_corr)])
            x = np.array([i if i is not None else np.nan for i in df_mse])
            y = np.array([i if i is not None else np.nan for i in df_corr])
            x = x[~np.isnan(x)]
            y = y[~np.isnan(y)]
            if any(x):
                metric_df = metric_df.append(pd.DataFrame({'Models': model_name,
                                                           'data_split': data_split,
                                                           'metric_value': np.sqrt(x),
                                                           'Metrics': 'Root mean square error',
                                                           'Data': data}))
                metric_df = metric_df.append(pd.DataFrame({'Models': model_name,
                                                           'data_split': data_split,
                                                           'metric_value': y,
                                                           'Metrics': "Pearson's rho",
                                                           'Data': data}))

    plot_diff_matrix(metric_df, condition_col='Data', metric_name_col='Metrics', metric_name=['Root mean square '
                                                                                              'error',
                                                                                              "Pearson's rho"],
                     metric_value_col='metric_value',
                     title=model_name,
                     cmap='gray')
    plt.savefig(model_name.replace(' ', '_') + '_diff_matrix.svg', bbox_inches='tight', dpi=300)
    plt.close()

# %% Randperm baseline
n_seconds_test_set = configurations['n_seconds_test_set'].unique()
assert n_seconds_test_set.size == 1
sampling_rate = configurations['sampling_rate'].unique()
assert sampling_rate.size == 1
seed = 0
metric_functions = [stats.pearsonr, mean_squared_error, r2_score]
n_permutes = int(1e6)
score_files = glob.glob("./data/BBCIformat/*_score.mat")
randperm_scores = []  # np.nan * np.ndarray(len(score_files))
to_test = []
to_permute = []
for idx, score_filename in enumerate(score_files):
    score_tmp = sp.io.loadmat(score_filename)
    score = score_tmp['score_resample']
    to_permute.append(score[0, :-1])
    cut_ind_test = int(np.size(to_permute[idx]) - n_seconds_test_set[0] * sampling_rate[0])
    to_test.append(to_permute[idx][cut_ind_test:])
    to_permute[idx] = to_permute[idx][:cut_ind_test]
    # randperm_scores.append(random_permutation(to_permute,
    #                                           to_test,
    #                                           n_permutes=n_permutes,
    #                                           metric_functions=metric_functions,
    #                                           seed=seed))

np.random.seed(seed=seed)
permutation_metrics = np.nan * np.zeros((len(to_permute), n_permutes, len(metric_functions)))
next_power = 0
for i_permute in range(n_permutes):
    for i_to_permute in range(len(to_permute)):
        permuted = np.random.permutation(to_permute[i_to_permute])
        to_test_len = len(to_test[i_to_permute])
        for i_metric in range(len(metric_functions)):
            permutation_metrics[i_to_permute, i_permute, i_metric] = \
                np.atleast_1d(metric_functions[i_metric](to_test[i_to_permute],
                                                         permuted[:to_test_len]))[0]

    if np.mod(i_permute + 1, 10 ** next_power) == 0:
        joblib.dump(permutation_metrics, 'permutation_metrics.pkl.z')
        print("Stats at {:g} iterations:".format(10 ** next_power))
        next_power += 1
        print('min')
        print(np.nanmin(permutation_metrics, axis=1))
        print("max")
        print(np.nanmax(permutation_metrics, axis=1))
        print("mean")
        print(np.nanmean(permutation_metrics, axis=1))
        print("median")
        print(np.nanmedian(permutation_metrics, axis=1))

# %% Plot per model data type without selections overview
data_values = configurations['data'].unique()
data_split_values = ['test']
model_values = configurations['model_name'].unique()
reorder_modalities = [0, 2, 1, 3]
assert (all(model_values == ['EEGNetv4', 'Deep4Net', 'EEGResNet-29', 'Linear', 'Linear SV', 'Radial basis SV',
                             'Random forest']))
metrics = ["Pearson's rho", 'Root mean square error']  # The order is very important here as it is hardcoded in the
# permutation metric matrix
assert (metric_functions[0] == stats.pearsonr) & (metric_functions[1] == mean_squared_error)
permutation_metrics = joblib.load("permutation_metrics.pkl.z")
p_values = np.nan * np.ndarray([len(subject_values), len(data_values), len(model_values), len(metrics)])
q_values = np.nan * np.ndarray(p_values.shape)

for idz, model_name in enumerate(model_values):
    df = configurations[(configurations['model_name'] == model_name) &
                        (configurations['band_pass'] == "'[None, None]'") &
                        (configurations['electrodes'] == "'*'")]

    metric_df = pd.DataFrame()
    for data_split in data_split_values:
        for data in data_values:
            for i_subject in [1, 0, 2]:
                df_mse = df[(df['data'] == data)]['mse_' + data_split]
                df_corr = df[(df['data'] == data)]['corr_' + data_split]
                # x = np.abs([np.array([i if i is not None else np.nan for i in b]) /
                #             np.array([i if i is not None else np.nan for i in a])
                #             for a, b in zip(df_mse, df_corr)])
                x = np.array([i[i_subject] if i is not None else np.nan for i in df_mse])
                y = np.array([i[i_subject] if i is not None else np.nan for i in df_corr])
                x = x[~np.isnan(x)]
                y = y[~np.isnan(y)]
                if any(x):
                    metric_df = metric_df.append(pd.DataFrame({'Subjects': subject_values[i_subject],
                                                               'Models': model_name,
                                                               'data_split': data_split,
                                                               'metric_value': np.sqrt(x),
                                                               'Metrics': 'Root mean square error',
                                                               'Data': data}))
                    metric_df = metric_df.append(pd.DataFrame({'Subjects': subject_values[i_subject],
                                                               'Models': model_name,
                                                               'data_split': data_split,
                                                               'metric_value': y,
                                                               'Metrics': "Pearson's rho",
                                                               'Data': data}))

    # f = plt.figure(figsize=[15, 10])
    # split_combi_plot(f, metric_df, x_col='Data', y_col='metric_value', hue_col='Metrics', palette='colorblind',
    #                  data_type_col='Data')
    # add_significance_bars_above(metric_df, condition_col='Data', metric_name_col='Metrics',
    #                             metric_name='Root mean square error',
    #                             metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
    # add_significance_bars_below(metric_df, condition_col='Data', metric_name_col='Metrics', metric_name="Pearson's "
    #                                                                                                       "rho",
    #                             metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
    # plt.title(model_name)
    # handles, labels = f.axes[0].get_legend_handles_labels()
    # f.axes[0].legend(handles=handles[0:2], labels=labels[0:2], title='Metrics', loc=2)
    # plt.savefig(model_name.replace(' ', '_') + '_no_selection_violins.svg', bbox_inches='tight', dpi=300)
    # plt.close()
    #
    # plot_diff_matrix(metric_df, condition_col='Data', metric_name_col='Metrics', metric_name=['Root mean square '
    #                                                                                               'error',
    #                                                                                                       "Pearson's rho"],
    #                  metric_value_col='metric_value',
    #                  title=model_name,
    #                  cmap='gray',
    #                  averaging_func=np.mean)
    # plt.savefig(model_name.replace(' ', '_') + '_no_selection_diff_matrix.svg', bbox_inches='tight', dpi=300)
    # plt.close()

    for index, metric_name in enumerate(metrics):
        clist = [[1, 1, 1], sns.color_palette('colorblind')[index]]
        if index == 1:
            cmap = LinearSegmentedColormap.from_list('white2colorblind', clist, N=256)
        else:
            cmap = LinearSegmentedColormap.from_list('white2colorblind', clist[::-1], N=256)
        # plot_performance_matrix(metric_df,
        #                         x_col='Subjects',
        #                         y_col='Data',
        #                         metric_name_col='Metrics',
        #                         metric_name=metric_name,
        #                         metric_value_col='metric_value',
        #                         title=model_name,
        #                         cmap=cmap,
        #                         averaging_func=np.mean)
        # plt.savefig(
        #     model_name.replace(' ', '_') + '_no_selection_performance_matrix_' + metric_name.replace(' ', '_') + '.svg',
        #     bbox_inches='tight', dpi = 300)
        # plt.close()
        performance_matrix, x_labels, y_labels = compute_performance_matrix(metric_df,
                                                                            x_col='Subjects',
                                                                            y_col='Data',
                                                                            metric_name_col='Metrics',
                                                                            metric_name=metric_name,
                                                                            metric_value_col='metric_value',
                                                                            averaging_func=np.mean)
        print(model_name + ' ' + metric_name)
        print(y_labels[reorder_modalities])
        print(" \\\\\n".join(
            [" & ".join(map('{0:.3f}'.format, line)) for line in performance_matrix[:, reorder_modalities]]))
        for idx, metric_value_per_subject in enumerate(performance_matrix[:, reorder_modalities]):
            performed_randperms = np.sum(~np.isnan(permutation_metrics[idx, :, 0]))
            if performed_randperms != permutation_metrics.shape[1]:
                print("Warning, p-values based on {:g} permutations instead of {:g}!".format(performed_randperms,
                                                                                             permutation_metrics.shape[
                                                                                                 1]))
            for idy, metric_value_per_data in enumerate(metric_value_per_subject):
                if metric_name == 'Root mean square error':
                    metric_index = 1
                    p_values[idx, idy, idz, metric_index] = (np.sum(permutation_metrics[idx, :, metric_index] <=
                                                                    metric_value_per_data) + 1) / \
                                                            (performed_randperms + 1)
                elif metric_name == "Pearson's rho":
                    metric_index = 0
                    p_values[idx, idy, idz, metric_index] = (np.sum(permutation_metrics[idx, :, metric_index] >=
                                                                    metric_value_per_data) + 1) / \
                                                            (performed_randperms + 1)
                else:
                    assert 0 == 1, "Unknown metric!"
        print("p-values")
        print(" \\\\\n".join(
            [" & ".join(map('{0:.3f}'.format, line)) for line in p_values[:, :, idz, metric_index]]))

for i_subject in range(len(subject_values)):
    for i_metric in range(len(metrics)):
        q_values[i_subject, :, :, i_metric] = fdr_corrected_pvals(p_values[i_subject, :, :, i_metric])

for idz, model_name in enumerate(model_values):
    for index, metric_name in enumerate(metrics):
        print(model_name + ' ' + metric_name + ' q-values')
        print(y_labels[reorder_modalities])
        print(
            " \\\\\n".join([" & ".join(map('{0:.3f}'.format, line)) for line in q_values[:, :, idz, index]]))


################################
######## work in progress ######
################################
# %% Plot comparison matrix
mat_padding = ((10, 10), (10, 10))
subject_values = np.array(['S2 moderate experience', 'S1 no experience', 'S3 substantial experience'])
data_split_values = ['train', 'valid', 'test']

f = plt.figure(figsize=[15, 15])
subplot_index = 1

model_vector = []
for name in configurations['model_name']:
    if name == 'lin_reg':
        model_vector.append(1)
    elif name == 'lin_svr':
        model_vector.append(2)
    elif name == 'eegnet':
        model_vector.append(3)
    elif name == 'deep4':
        model_vector.append(4)
    elif name == 'resnet':
        model_vector.append(5)
    elif name == 'rbf_svr':
        model_vector.append(6)
    elif name == 'rf_reg':
        model_vector.append(7)
    else:
        model_vector.append(np.nan)  # do something

electrode_vector = []
for electrodes in configurations['electrodes']:
    if electrodes == '*':
        electrode_vector.append(1)
    elif electrodes == '*z':
        electrode_vector.append(2)
    elif electrodes == '*C*':
        electrode_vector.append(3)
    else:
        electrode_vector.append(np.nan)  # do something

bandpass_vector = []
for band_pass in configurations['band_pass']:
    if band_pass == "'[None, None]'":
        bandpass_vector.append(1)
    elif band_pass == "'[0, 4]'":
        bandpass_vector.append(2)
    elif band_pass == "'[4, 8]'":
        bandpass_vector.append(3)
    elif band_pass == "'[8, 14]'":
        bandpass_vector.append(4)
    elif band_pass == "'[14, 20]'":
        bandpass_vector.append(5)
    elif band_pass == "'[20, 30]'":
        bandpass_vector.append(6)
    elif band_pass == "'[30, 40]'":
        bandpass_vector.append(7)
    else:
        bandpass_vector.append(np.nan)  # do something

for i_subject in range(len(subject_values)):
    for data_split in data_split_values:
        df = configurations['mse_' + data_split]
        df = df.apply(lambda x: [np.sqrt(a) if a is not None else np.nan for a in x])
        rmse = np.vstack(df.values).flatten()[i_subject::3]
        diff_mat = np.reshape([a - b for a in rmse for b in rmse], [rmse.size, rmse.size], order='C')
        diff_mat = np.pad(diff_mat, mat_padding, 'constant', constant_values=np.nan)
        ax = plt.subplot(len(subject_values), np.size(data_split_values), subplot_index)
        subplot_index += 1
        im = ax.imshow(diff_mat, cmap=plt.get_cmap('RdBu'))
        f.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # ax.imshow(np.tile(model_vector,(10, 1)).T, cmap=plt.get_cmap('Set1'))
        # ax.plot()

        ax.bar()
        ax.set_title(data_split + ' set, ' + subject_values[i_subject])
        plt.show()

# %% Circle plot in one axes

f = plt.figure()
x_positions = np.array([7, 19, 1, 13, 25, 7, 19])
y_positions = np.array([1, 1, 1.5, 1.5, 1.5, 2, 2])
subplot_index = 1
x_shift = 18
y_shift = 11 * 4
for i_no_eeg_data in np.arange(2):
    for i_subject in np.arange(3):
        # plt.subplot(5, 3, subplot_index)
        plt.scatter(x_positions + x_shift, y_positions + y_shift, s=np.random.rand(7) * 10,
                    c=('b', 'g', 'r', 'c', 'm', 'y', 'k'))
        x_shift += 6 * 21
        # subplot_index += 1
        x_shift = 18
        y_shift -= 2

x_shift = 0
y_shift = 10 * 4
for i_eeg_data in np.arange(3):
    for i_subject in np.arange(3):
        # plt.subplot(5, 3,subplot_index)
        for i_bandpass in np.arange(7):
            for i_electrode in np.arange(3):
                plt.scatter(x_positions + x_shift, y_positions + y_shift, s=np.random.rand(7) * 10,
                            c=('b', 'g', 'r', 'c', 'm', 'y', 'k'))
                y_shift -= 2
                x_shift += 50
                y_shift = 10 * 4
                # subplot_index += 1

plt.show()

# for i_config in configurations.keys():
#     result_folder = configurations[i_config]['result_folder']
#     unique_id = configurations[i_config]['unique_id']
#     matching_results = np.sort(glob.glob(result_folder + '/*' + unique_id[:7] + '.csv'))
#     # Check that we get only one result per subject, but not sure what to do if not. Check that results are identical?
#
#     if any(matching_results):
#         configurations[i_config]['config_has_run'] = True
#         configurations[i_config]['mse_train'] = [None, None, None]
#         configurations[i_config]['mse_valid'] = [None, None, None]
#         configurations[i_config]['mse_test'] = [None, None, None]
#         configurations[i_config]['corr_train'] = [None, None, None]
#         configurations[i_config]['corr_valid'] = [None, None, None]
#         configurations[i_config]['corr_test'] = [None, None, None]
#         configurations[i_config]['corr_p_train'] = [None, None, None]
#         configurations[i_config]['corr_p_valid'] = [None, None, None]
#         configurations[i_config]['corr_p_test'] = [None, None, None]
#         i_subject = -1
#         for subject in matching_results:
#             i_subject += 1
#             df = pd.read_csv(subject)#, names=['epoch',
#                                              # 'train_loss','valid_loss','test_loss',
#                                              # 'train_corr','valid_corr','test_corr', 'runtime'])
#             configurations[i_config]['mse_train'][i_subject] = df.tail(1).values[0, 1]
#             configurations[i_config]['mse_valid'][i_subject] = df.tail(1).values[0, 2]
#             configurations[i_config]['mse_test'][i_subject] = df.tail(1).values[0, 3]
#             configurations[i_config]['corr_train'][i_subject] = df.tail(1).values[0, 4]
#             configurations[i_config]['corr_valid'][i_subject] = df.tail(1).values[0, 5]
#             configurations[i_config]['corr_test'][i_subject] = df.tail(1).values[0, 6]
#             # configurations[i_config]['corr_p_train'][i_subject] = df.tail(1).values[0, None]
#             # configurations[i_config]['corr_p_valid'][i_subject] = df.tail(1).values[0, None]
#             # configurations[i_config]['corr_p_test'][i_subject] = df.tail(1).values[0, None]
#     else:
#         configurations[i_config]['config_has_run'] = False
#         configurations[i_config]['mse_train'] = [None, None, None]
#         configurations[i_config]['mse_valid'] = [None, None, None]
#         configurations[i_config]['mse_test'] = [None, None, None]
#         configurations[i_config]['corr_train'] = [None, None, None]
#         configurations[i_config]['corr_valid'] = [None, None, None]
#         configurations[i_config]['corr_test'] = [None, None, None]
#         configurations[i_config]['corr_p_train'] = [None, None, None]
#         configurations[i_config]['corr_p_valid'] = [None, None, None]
#         configurations[i_config]['corr_p_test'] = [None, None, None]
#
# configurations.T.to_csv(configurations_file)
