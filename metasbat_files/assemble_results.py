# %% Imports
import glob
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.df_utils import initialize_empty_columns_at_index, prefill_columns_configuration_at_index, \
    fill_results_in_configurations_at_index
from visualizations.plots import split_combi_plot, add_significance_bars, plot_performance_matrix

mpl.rcParams['figure.dpi'] = 300

# %% Load all existing single subject results
configurations_file = '/mnt/meta-cluster/home/fiederer/nicebot/metasbat_files/configs_no_valid.csv'
# configurations_file = '/home/lukas/nicebot/metasbat_files/configs_all_models.csv'
configurations = pd.read_csv(configurations_file)  # Load existing configs

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
subject_values = np.array(['moderate experience (S2)', 'no experience (S1)', 'substantial experience (S3)'])
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
                assert (all(df['model_name'].values == ['eegnet', 'deep4', 'resnet', 'lin_reg', 'lin_svr', 'rbf_svr',
                                                        'rf_reg']))
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
assert (all(model_values == ['eegnet', 'deep4', 'resnet', 'lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg']))

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
            metric_df = metric_df.append(pd.DataFrame({'model': model_name,
                                                       'data_split': data_split,
                                                       'metric_value': np.sqrt(x),
                                                       'metric_name': 'rmse',
                                                       'data': 'All data'}))
            metric_df = metric_df.append(pd.DataFrame({'model': model_name,
                                                       'data_split': data_split,
                                                       'metric_value': 1 - np.abs(y),
                                                       'metric_name': '1-abs(corr)',
                                                       'data': 'All data'}))

f = plt.figure(figsize=[15, 10])
split_combi_plot(f, metric_df, x_col='model', y_col='metric_value', hue_col='metric_name', palette='colorblind')
add_significance_bars(metric_df, condition_col='model', metric_name_col='metric_name', metric_name='rmse',
                      metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
plt.show()

f = plt.figure(figsize=[15, 10])
split_combi_plot(f, metric_df, x_col='model', y_col='metric_value', hue_col='metric_name', palette='colorblind')
add_significance_bars(metric_df, condition_col='model', metric_name_col='metric_name', metric_name='1-abs(corr)',
                      metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
plt.show()

plot_performance_matrix(metric_df, condition_col='model', metric_name_col='metric_name', metric_name=['rmse',
                                                                                                      '1-abs(corr)'],
                        metric_value_col='metric_value',
                        title=None,
                        cmap=plt.cm.Blues_r)
plt.show()

# %% Best performing method for each data type

data_values = configurations['data'].unique()
data_split_values = ['test']
model_values = configurations['model_name'].unique()
assert (all(model_values == ['eegnet', 'deep4', 'resnet', 'lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg']))

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
                metric_df = metric_df.append(pd.DataFrame({'model': model_name,
                                                           'data_split': data_split,
                                                           'metric_value': np.sqrt(x),
                                                           'metric_name': 'rmse',
                                                           'data': data}))
                metric_df = metric_df.append(pd.DataFrame({'model': model_name,
                                                           'data_split': data_split,
                                                           'metric_value': 1 - np.abs(y),
                                                           'metric_name': '1-abs(corr)',
                                                           'data': data}))

    f = plt.figure(figsize=[15, 10])
    split_combi_plot(f, metric_df, x_col='model', y_col='metric_value', hue_col='metric_name', palette='colorblind')
    add_significance_bars(metric_df, condition_col='model', metric_name_col='metric_name', metric_name='rmse',
                          metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
    plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    f = plt.figure(figsize=[15, 10])
    split_combi_plot(f, metric_df, x_col='model', y_col='metric_value', hue_col='metric_name', palette='colorblind')
    add_significance_bars(metric_df, condition_col='model', metric_name_col='metric_name', metric_name='1-abs(corr)',
                          metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
    plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    plot_performance_matrix(metric_df, condition_col='model', metric_name_col='metric_name', metric_name=['rmse',
                                                                                                          '1-abs(corr)'],
                            metric_value_col='metric_value',
                            title=data,
                            cmap=plt.cm.Blues_r)
    plt.show()

# %% Best electrode set EEG only
# data_split_values = ['train', 'test']
data_split_values = ['test']
electrode_sets = configurations['electrodes'].unique()
assert (all(electrode_sets == ["'*'", "'*z'", "'*C*'"]))

metric_df = pd.DataFrame()
for electrode_set_name in electrode_sets:
    for data_split in data_split_values:
        df_mse = configurations[(configurations['electrodes'] == electrode_set_name)
                                & (configurations['data'] == 'onlyEEGData')]['mse_' + data_split]
        df_corr = configurations[(configurations['electrodes'] == electrode_set_name)
                                 & (configurations['data'] == 'onlyEEGData')]['corr_' + data_split]
        x = np.array([i if i is not None else np.nan for i in df_mse])
        y = np.array([i if i is not None else np.nan for i in df_corr])
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        if any(x):
            metric_df = metric_df.append(pd.DataFrame({'electrodes': electrode_set_name[1:-1],
                                                       'data_split': data_split,
                                                       'metric_value': np.sqrt(x),
                                                       'metric_name': 'rmse',
                                                       'data': 'onlyEEGData'}))
            metric_df = metric_df.append(pd.DataFrame({'electrodes': electrode_set_name[1:-1],
                                                       'data_split': data_split,
                                                       'metric_value': 1 - np.abs(y),
                                                       'metric_name': '1-abs(corr)',
                                                       'data': 'onlyEEGData'}))

f = plt.figure(figsize=[15, 10])
split_combi_plot(f, metric_df, x_col='electrodes', y_col='metric_value', hue_col='metric_name', palette='colorblind')
add_significance_bars(metric_df, condition_col='electrodes', metric_name_col='metric_name', metric_name='rmse',
                      metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
plt.show()

f = plt.figure(figsize=[15, 10])
split_combi_plot(f, metric_df, x_col='electrodes', y_col='metric_value', hue_col='metric_name', palette='colorblind')
add_significance_bars(metric_df, condition_col='electrodes', metric_name_col='metric_name', metric_name='1-abs(corr)',
                      metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
plt.show()

plot_performance_matrix(metric_df, condition_col='electrodes', metric_name_col='metric_name', metric_name=['rmse',
                                                                                                           '1-abs(corr)'],
                        metric_value_col='metric_value',
                        title=None,
                        cmap=plt.cm.Blues_r)
plt.show()

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
                                                       'metric_name': 'rmse',
                                                       'data': data}))
            metric_df = metric_df.append(pd.DataFrame({'data_split': data_split,
                                                       'metric_value': 1 - np.abs(y),
                                                       'metric_name': '1-abs(corr)',
                                                       'data': data}))

f = plt.figure(figsize=[15, 10])
split_combi_plot(f, metric_df, x_col='data', y_col='metric_value', hue_col='metric_name', palette='colorblind')
add_significance_bars(metric_df, condition_col='data', metric_name_col='metric_name', metric_name='rmse',
                      metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
plt.show()

f = plt.figure(figsize=[15, 10])
split_combi_plot(f, metric_df, x_col='data', y_col='metric_value', hue_col='metric_name', palette='colorblind')
add_significance_bars(metric_df, condition_col='data', metric_name_col='metric_name', metric_name='1-abs(corr)',
                      metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
plt.show()

plot_performance_matrix(metric_df, condition_col='data', metric_name_col='metric_name', metric_name=['rmse',
                                                                                                     '1-abs(corr)'],
                        metric_value_col='metric_value',
                        title=None,
                        cmap=plt.cm.Blues_r)
plt.show()

# %% Best frequency band EEG only
# data_split_values = ['train', 'test']
data_split_values = ['test']
bandpass_sets = configurations['band_pass'].unique()

metric_df = pd.DataFrame()
for bandpass in bandpass_sets:
    for data_split in data_split_values:
        df_mse = configurations[(configurations['band_pass'] == bandpass)
                                & (configurations['data'] == 'onlyEEGData')]['mse_' + data_split]
        df_corr = configurations[(configurations['band_pass'] == bandpass)
                                 & (configurations['data'] == 'onlyEEGData')]['corr_' + data_split]
        x = np.array([i if i is not None else np.nan for i in df_mse])
        y = np.array([i if i is not None else np.nan for i in df_corr])
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        if any(x):
            metric_df = metric_df.append(pd.DataFrame({'band_pass': bandpass[1:-1],
                                                       'data_split': data_split,
                                                       'metric_value': np.sqrt(x),
                                                       'metric_name': 'rmse',
                                                       'data': 'onlyEEGData'}))
            metric_df = metric_df.append(pd.DataFrame({'band_pass': bandpass[1:-1],
                                                       'data_split': data_split,
                                                       'metric_value': 1 - np.abs(y),
                                                       'metric_name': '1-abs(corr)',
                                                       'data': 'onlyEEGData'}))

f = plt.figure(figsize=[15, 10])
split_combi_plot(f, metric_df, x_col='band_pass', y_col='metric_value', hue_col='metric_name', palette='colorblind')
add_significance_bars(metric_df, condition_col='band_pass', metric_name_col='metric_name', metric_name='rmse',
                      metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
plt.show()

f = plt.figure(figsize=[15, 10])
split_combi_plot(f, metric_df, x_col='band_pass', y_col='metric_value', hue_col='metric_name', palette='colorblind')
add_significance_bars(metric_df, condition_col='band_pass', metric_name_col='metric_name', metric_name='1-abs(corr)',
                      metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
plt.show()

plot_performance_matrix(metric_df, condition_col='band_pass', metric_name_col='metric_name', metric_name=['rmse',
                                                                                                          '1-abs(corr)'],
                        metric_value_col='metric_value',
                        title=None,
                        cmap=plt.cm.Blues_r)
plt.show()

# %% Overall best subject
# data_split_values = ['train', 'test']
data_split_values = ['test']

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
            metric_df = metric_df.append(pd.DataFrame({'subject': subject_values[i_subject],
                                                       'data_split': data_split,
                                                       'metric_value': np.sqrt(x),
                                                       'metric_name': 'rmse',
                                                       'data': 'All data'}))
            metric_df = metric_df.append(pd.DataFrame({'subject': subject_values[i_subject],
                                                       'data_split': data_split,
                                                       'metric_value': 1 - np.abs(y),
                                                       'metric_name': '1-abs(corr)',
                                                       'data': 'All data'}))

f = plt.figure(figsize=[15, 10])
split_combi_plot(f, metric_df, x_col='subject', y_col='metric_value', hue_col='metric_name', palette='colorblind')
add_significance_bars(metric_df, condition_col='subject', metric_name_col='metric_name', metric_name='rmse',
                      metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
plt.show()

f = plt.figure(figsize=[15, 10])
split_combi_plot(f, metric_df, x_col='subject', y_col='metric_value', hue_col='metric_name', palette='colorblind')
add_significance_bars(metric_df, condition_col='subject', metric_name_col='metric_name', metric_name='1-abs(corr)',
                      metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1])
plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
plt.show()

plot_performance_matrix(metric_df, condition_col='subject', metric_name_col='metric_name', metric_name=['rmse',
                                                                                                        '1-abs(corr)'],
                        metric_value_col='metric_value',
                        title=None,
                        cmap=plt.cm.Blues_r)
plt.show()


################################
######## work in progress ######
################################
# %% Plot comparison matrix
mat_padding = ((10, 10), (10, 10))
subject_values = np.array(['moderate experience (S2)', 'no experience (S1)', 'substantial experience (S3)'])
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
