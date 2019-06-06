import pandas as pd
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

configurations_file = '/mnt/meta-cluster/home/fiederer/nicebot/metasbat_files/configs_no_valid.csv'
# configurations_file = '/home/lukas/nicebot/metasbat_files/configs_all_models.csv'
configurations = pd.read_csv(configurations_file)  # Load existing configs

# Initialize storage. One needs to create a new list for each, else these are just linked copyies of each other and
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
reapplication_error = []
original_mse = []
is_model = []
is_config = []
corr_coef_error = []

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
            # Check if this subject was applied on himself. If yes make sure that the results are identical to
            # the original ones
            word_list = re.split('[_/.]', subject)  # Split path to .csv file into single words delimited by _ /
            # and .
            if word_list.count(word_list[-2]) == 2 and bool(re.match('.*Net.*', word_list[-4])):  # If the second to
                # last word (either subject model was applied to or unique ID of the experiment) is present twice
                # then the model was trained on the same subject
                df_original = pd.read_csv(subject.replace('_' + word_list[-2], '_epochs'))  # load result of original
                # predictions
                df_same = pd.read_csv(subject)# load result of across subjects predictions on same subject
                if len(df_original) == configurations.loc[i_config, 'max_epochs']+1:  # then these are the results
                    # stored in an epochs dataframe
                    tmp = df_original.tail(1)['test_loss'].values[0]
                elif len(df_original) == 12:
                    assert df_original.values[8, 0] == 'Test mse', 'Value at hardcoded index is not Test mse!'
                    tmp = df_original.values[8, 1]
                else:
                    print('Unexpected situation! Please check the code!')
                    exit()

                assert df_same.values[8, 0] == 'Test mse', 'Value at hardcoded index is not Test mse!'
                if tmp == df_same.values[8, 1]:  # Check if test mse is equal
                    continue  # if equal skip as we already have written the results to the dataframe
                else:
                    is_config.append(i_config)
                    original_mse.append(tmp)
                    reapplication_error.append(tmp - df_same.values[8, 1])
                    corr_coef_error.append(df_original.tail(1)['test_corr'].values[0] - df_same.values[9, 1])

                    if word_list[12] == 'EEGNetv4':
                        is_model.append(0)
                    elif word_list[12] == 'Deep4Net':
                        is_model.append(1)
                    elif word_list[12] == 'ResNet':
                        is_model.append(2)
                    elif word_list[12] == 'lin':
                        if word_list[13] == 'reg':
                            is_model.append(3)
                        elif word_list[13] == 'svr':
                            is_model.append(4)
                        else:
                            print(word_list[13] + 'Unexpected situation! Please check the code!')
                            exit()# if not equal throw error
                    elif word_list[12] == 'rbf':
                        is_model.append(5)
                    elif word_list[12] == 'rf':
                        is_model.append(6)
                    else:
                        print(word_list[12] + 'Unexpected situation! Please check the code!')
                        exit()# if not equal throw error
                    # if len(df_original) == 12:
                    #     is_model.append(False)
                    # else:
                    #     is_model.append(True)
                    continue
                    # print('Unexpected situation! Please check the code!')
                    # exit()# if not equal throw error

            elif word_list.count(word_list[-2]) > 2:
                print('Unexpected situation! Please check the code!')
                exit()
# #%%
# plt.figure
# plt.yscale("log"),plt.xscale("log")
# plt.scatter(original_mse, np.abs(reapplication_error)), plt.xlabel('original mse'), plt.ylabel('absolute '
#                                                                                        'difference to '
#                                                                                        'reapplied')
# plt.show()
#
# #%%
# plt.figure
# plt.yscale("log"),plt.xscale("log")
# plt.scatter(original_mse, np.abs(corr_coef_error)), plt.xlabel('original corr coef'), plt.ylabel('absolute '
#                                                                                        'difference to '
#                                                                                        'reapplied')
# plt.show()
# #%%
# plt.figure
# # plt.yscale("log"),plt.xscale("log")
# plt.scatter(original_mse, [a+b for a, b in zip(original_mse, reapplication_error)])
# plt.xlabel('original mse'),
# plt.ylabel('new mse')
# plt.show()
# #%%
            i_subject += 1
            df = pd.read_csv(subject)
            # if configurations.loc[i_config, 'model_name'] in ['deep4', 'eegnet', 'resnet']:
            if len(df) == configurations.loc[i_config, 'max_epochs']+1:  # then these are the results of the training
                # of a cnn model
                for column_name in df.columns:
                    if column_name == 'train_loss':
                        configurations.loc[i_config, 'mse_train'][i_subject] = df.tail(1)[column_name].values[0]#, 1]
                    if column_name == 'valid_loss':
                        configurations.loc[i_config, 'mse_valid'][i_subject] = df.tail(1)[column_name].values[0]#, 2]
                    if column_name == 'test_loss':
                        configurations.loc[i_config, 'mse_test'][i_subject] = df.tail(1)[column_name].values[0]#, 3]
                    if column_name == 'train_corr':
                        configurations.loc[i_config, 'corr_train'][i_subject] = df.tail(1)[column_name].values[0]#, 4]
                    if column_name == 'valid_corr':
                        configurations.loc[i_config, 'corr_valid'][i_subject] = df.tail(1)[column_name].values[0]#, 5]
                    if column_name == 'test_corr':
                        configurations.loc[i_config, 'corr_test'][i_subject] = df.tail(1)[column_name].values[0]#, 6]
            elif len(df) == 12:  # these are the results of the training of a non-cnn model
            # elif configurations.loc[i_config, 'model_name'] in ['lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg']:
                configurations.loc[i_config, 'mse_train'][i_subject] = df.values[0, 1]
                configurations.loc[i_config, 'mse_valid'][i_subject] = df.values[4, 1]
                configurations.loc[i_config, 'mse_test'][i_subject] = df.values[8, 1]
                configurations.loc[i_config, 'corr_train'][i_subject] = df.values[1, 1]
                configurations.loc[i_config, 'corr_valid'][i_subject] = df.values[5, 1]
                configurations.loc[i_config, 'corr_test'][i_subject] = df.values[9, 1]
                configurations.loc[i_config, 'corr_p_train'][i_subject] = df.values[2, 1]
                configurations.loc[i_config, 'corr_p_valid'][i_subject] = df.values[6, 1]
                configurations.loc[i_config, 'corr_p_test'][i_subject] = df.values[10, 1]
            elif len(df) == 4:  # these are the results of a transfer
                configurations.loc[i_config, 'mse_test'][i_subject] = df.values[0, 1]
                configurations.loc[i_config, 'corr_test'][i_subject] = df.values[1, 1]
                configurations.loc[i_config, 'corr_p_test'][i_subject] = df.values[2, 1]

            else:
                # print('Unknown model name: {}'.format(configurations.loc[i_config, 'model_name']))
                print('Unknown csv layout with len {:d}'.format(len(df)))
                break

# configurations.to_csv(configurations_file[:-4] + '_with_results.csv')
# Somehow converts my lists into strings!!!

#%% Load only missing results only instead of having to rerun the above every time
for i_config in configurations.index:
    if configurations.loc[i_config, 'config_has_run'] == False:
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
            # if configurations.loc[i_config, 'model_name'] in ['deep4', 'eegnet', 'resnet']:
            if len(df) == configurations.loc[i_config, 'max_epochs']+1:  # then these are the results of the training
                # of a cnn model
                for column_name in df.columns:
                    if column_name == 'train_loss':
                        configurations.loc[i_config, 'mse_train'][i_subject] = df.tail(1)[column_name].values[0]#, 1]
                    if column_name == 'valid_loss':
                        configurations.loc[i_config, 'mse_valid'][i_subject] = df.tail(1)[column_name].values[0]#, 2]
                    if column_name == 'test_loss':
                        configurations.loc[i_config, 'mse_test'][i_subject] = df.tail(1)[column_name].values[0]#, 3]
                    if column_name == 'train_corr':
                        configurations.loc[i_config, 'corr_train'][i_subject] = df.tail(1)[column_name].values[0]#, 4]
                    if column_name == 'valid_corr':
                        configurations.loc[i_config, 'corr_valid'][i_subject] = df.tail(1)[column_name].values[0]#, 5]
                    if column_name == 'test_corr':
                        configurations.loc[i_config, 'corr_test'][i_subject] = df.tail(1)[column_name].values[0]#, 6]
            elif len(df) == 12:  # these are the results of the training of a non-cnn model
            # elif configurations.loc[i_config, 'model_name'] in ['lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg']:
                configurations.loc[i_config, 'mse_train'][i_subject] = df.values[0, 1]
                configurations.loc[i_config, 'mse_valid'][i_subject] = df.values[4, 1]
                configurations.loc[i_config, 'mse_test'][i_subject] = df.values[8, 1]
                configurations.loc[i_config, 'corr_train'][i_subject] = df.values[1, 1]
                configurations.loc[i_config, 'corr_valid'][i_subject] = df.values[5, 1]
                configurations.loc[i_config, 'corr_test'][i_subject] = df.values[9, 1]
                configurations.loc[i_config, 'corr_p_train'][i_subject] = df.values[2, 1]
                configurations.loc[i_config, 'corr_p_valid'][i_subject] = df.values[6, 1]
                configurations.loc[i_config, 'corr_p_test'][i_subject] = df.values[10, 1]
            elif len(df) == 4:  # these are the results of a transfer
                configurations.loc[i_config, 'mse_test'][i_subject] = df.values[0, 1]
                configurations.loc[i_config, 'corr_test'][i_subject] = df.values[1, 1]
                configurations.loc[i_config, 'corr_p_test'][i_subject] = df.values[2, 1]

            else:
                # print('Unknown model name: {}'.format(configurations.loc[i_config, 'model_name']))
                print('Unknown csv layout with len {:d}'.format(len(df)))
                break

#%% Plot fancy results
# configurations_file = '/mnt/meta-cluster/home/fiederer/nicebot/metasbat_files/configs_all_models_with_results.csv'
# configurations = pd.read_csv(configurations_file)  # Load existing configs
# TODO: Visualize the across subject results as second ring using the same fill colors but different edge colors. That
#  way we have the direct comparison of within and across results
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
        #plt.title('Subject %g'.format(i_subject+1))
        df = configurations[configurations['data'] == data_values[i_no_eeg_data]]
        assert(df['model_name'].size == 7)
        assert(all(df['model_name'].values == ['eegnet', 'deep4', 'resnet', 'lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg']))
        df = df.iloc[model_order]

        # tmp = np.full(7, np.nan)
        metrics = []
        # metrics = [a[i_subject]/b[i_subject] for a, b in zip(df['corr_test'].values, df['mse_test'].values)]
        for i_model in range(df['model_name'].size):
            #if not any(np.isnan(metric)):
            try:
                metrics.append(df['corr_' + data_split].values[i_model][i_subject] /
                               df['mse_' + data_split].values[i_model][i_subject])
                # metrics.append(1 / np.sqrt(df['mse_' + data_split].values[i_model][i_subject]))
                # metrics.append(1 / df['mse_' + data_split].values[i_model][i_subject])
                # metrics.append(df['corr_' + data_split].values[i_model][i_subject])
            #else:
            except:
                metrics.append(0)

        plt.scatter(x_positions, y_positions, s=np.abs(np.array(metrics)) * scaling_factor,
                    c=('b', 'g', 'r', 'c', 'm', 'y', 'k'))
        plt.xlim(np.array([np.mean(x_positions)-(x_shift_increment*bandpass_values.size), np.mean(x_positions)+(
                x_shift_increment*bandpass_values.size)]) / 2)
        plt.ylim(np.array([np.mean(y_positions)-(y_shift_increment*electrodes_values.size), np.mean(y_positions)+(
                y_shift_increment*electrodes_values.size)]) / 2)
        plt.xticks([], ())
        # plt.yticks(np.unique(y_positions), ('linear', 'neural net', 'non-linear'), rotation=45)
        plt.yticks([], ())

        subplot_index += 1


x_shift = 0
y_shift = 0
for i_eeg_data in np.arange(3):
    for i_subject in np.arange(3):
        plt.subplot(5, 3,subplot_index)
        for i_bandpass in np.arange(7):
            for i_electrode in np.arange(3):
                df = configurations[(configurations['data'] == data_values[i_eeg_data+2]) &
                                    (configurations['band_pass'] == bandpass_values[i_bandpass]) &
                                     (configurations['electrodes'] == electrodes_values[i_electrode])]
                #df = df[df['band_pass'] == ]
                assert(df['model_name'].size == 7)
                assert(all(df['model_name'].values == ['eegnet', 'deep4', 'resnet', 'lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg']))
                df = df.iloc[model_order]

                # tmp = np.full(7, np.nan)
                metrics = []
                # metrics = [a[i_subject]/b[i_subject] for a, b in zip(df['corr_test'].values, df['mse_test'].values)]
                for i_model in range(df['model_name'].size):
                    #if not any(np.isnan(metric)):
                    try:
                        metrics.append(df['corr_' + data_split].values[i_model][i_subject] /
                                       df['mse_' + data_split].values[i_model][i_subject])
                        # metrics.append(1 / np.sqrt(df['mse_test'].values[i_model][i_subject]))
                        # metrics.append(1 / df['mse_test'].values[i_model][i_subject])
                        # metrics.append(df['corr_test'].values[i_model][i_subject])
                    #else:
                    except:
                        metrics.append(0)

                plt.scatter(x_positions+x_shift, y_positions+y_shift, s=np.abs(np.array(metrics))*scaling_factor,
                            c=('b', 'g', 'r', 'c', 'm', 'y', 'k'))
                plt.xlim([0, x_shift_increment*bandpass_values.size])
                plt.ylim([0, y_shift_increment*electrodes_values.size])
                y_shift += y_shift_increment
            x_shift += x_shift_increment
            y_shift = 0
        x_shift = 0
        subplot_index += 1
        plt.xticks(x_positions.mean()+np.arange(x_shift, x_shift_increment*(i_bandpass)+1, x_shift_increment),
        ('None', '[0,4]', '[4,8]', '[8,14]', '[14,20]',
                                    '[20,30]', '[30,40]'))
        plt.yticks(y_positions.mean()+np.arange(y_shift, y_shift_increment*(i_electrode)+1, y_shift_increment),
        ('*', '*z', '*C*'))
        # plt.axis('tight')
# plt.ylabel('Electrode selection')
# plt.xlabel('Band-pass (Hz)')

for i_data in range(data_values.size):
    plt.text(0.05, 0.166*(i_data+1), data_values[::-1][i_data], horizontalalignment='center', verticalalignment='center',
         transform=f.transFigure, rotation=90)

for i_subject in range(subject_values.size):
    plt.text(0.25*(i_subject+1), 0.9, subject_values[i_subject], horizontalalignment='center',
             verticalalignment='center',
         transform=f.transFigure)

plt.text(0.5, 0.95, 'Data split: ' + data_split + ' set', horizontalalignment='center',
         verticalalignment='center',
         transform=f.transFigure)
# plt.figlegend(['lin_reg', 'lin_svr', 'eegNet', 'deep4', 'resNet', 'rbf_svr', 'rf_reg'], loc='center right')#, bbox_to_anchor=(1, 0.5))
plt.show()

#%% Overall best performing method
# data_split_values = ['train', 'test']
data_split_values = ['test']
model_values = configurations['model_name'].unique()
assert(all(model_values == ['eegnet', 'deep4', 'resnet', 'lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg']))

f = plt.figure(figsize=[15, 10])
# plt.axes()
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
        x = np.array([np.sqrt(i) if i is not None else np.nan for i in df_mse])
        y = np.array([1-np.abs(i) if i is not None else np.nan for i in df_corr])
#         x = x[~np.isnan(x)]
#         y = y[~np.isnan(y)]
#         if any(x):
#             metric_df = metric_df.append(pd.DataFrame({'model': model_name,
#                                                        'data_split': data_split,
#                                                        'metric_value': x,
#                                                        'metric_name': 'mse'}))
#             metric_df = metric_df.append(pd.DataFrame({'model': model_name,
#                                                        'data_split': data_split,
#                                                        'metric_value': y,
#                                                        'metric_name': 'corr'}))
# # Plot the orbital period with horizontal boxes
# sns.boxplot(x='metric_value', y='model', hue='metric_name', data=metric_df, whis="range", palette="vlag")
#
# # Add in points to show each observation
# # sns.swarmplot(x='mse', y='model', hue='data_split', data=metric_df, size=2, color=".3", linewidth=0)
# sns.stripplot(x='metric_value', y='model', hue='metric_name', data=metric_df, dodge=True, jitter=True, alpha=.25, zorder=1)

        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        if any(x):
            metric_df = metric_df.append(pd.DataFrame({'model': model_name, 'data_split': data_split, 'metric_value':
                np.concatenate((x, y)), 'metric_name': np.concatenate((np.repeat('rmse', len(x)), np.repeat('1-abs('
                                                                                                            'corr)',
                                                                                                           len(y))))}))
# Plot the orbital period with horizontal boxes
sns.violinplot(x='metric_value', y='model', hue='metric_name', data=metric_df, palette="colorblind",
               color='1', split=True, inner='quartiles', scale='area', scale_hue=True, cut=2, bw=0.15)
# sns.boxplot(x='metric_value', y='model', hue='metric_name', data=metric_df, whis="range", width=0.1)

# Add in points to show each observation
# sns.swarmplot(x='metric_value', y='model', hue='metric_name', data=metric_df, size=2, color=".3", linewidth=0,
#               dodge=True)
sns.stripplot(x='metric_value', y='model', hue='metric_name', data=metric_df, dodge=True, jitter=True, alpha=.25,
              zorder=1, palette="colorblind", color='1', edgecolor='black', linewidth=1)
# colors = sns.color_palette('vlag',n_colors=7)
# # colors = sp.repeat(colors,2,axis=0)
#
# for artist, collection, color in zip(f.axes[0].artists, f.axes[0].collections, colors):
#     artist.set_facecolor(color)
#     collection.set_facecolor(color)


# Tweak the visual presentation
# f.axes[0].get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
# f.axes[0].get_xaxis().set_major_locator(mpl.ticker.MultipleLocator(0.1))
# f.axes[0].get_xaxis().set_major_locator(mpl.ticker.LogLocator())
# ax.set_xticks(np.arange(0,8)-0.5, minor=True)
# f.axes[0].xaxis.grid(b=True, which='minor', linewidth=0.5)
# f.axes[0].xaxis.grid(b=True, which='major', linewidth=1)
f.axes[0].set_axisbelow(True)
f.axes[0].set(ylabel="")
# sns.despine(trim=True, left=True)
xlims = plt.xlim()
plt.xlim(0.01, xlims[1])

plt.title('All models, all data')
# f.tight_layout()
plt.show()

# # User test to determine if anything significantly different
# stats.wilcoxon()