import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy import stats

configurations_file = '/mnt/meta-cluster/home/fiederer/nicebot/metasbat_files/configs_no_valid.csv'
# configurations_file = '/home/lukas/nicebot/metasbat_files/configs_all_models.csv'
configurations = pd.read_csv(configurations_file)  # Load existing configs

for i_config in configurations.index:
    result_folder = configurations.loc[i_config, 'result_folder']
    unique_id = configurations.loc[i_config, 'unique_id']
    matching_results = np.sort(glob.glob('/mnt/meta-cluster/' + result_folder + '/*' + unique_id + '.csv'))
    # matching_results = np.sort(glob.glob(result_folder + '/*' + unique_id[:7] + '.csv'))
    # Check that we get only one result per subject, but not sure what to do if not. Check that results are identical?

    # I have to do this ugly thing here because initializing with 3 nans is somehow not possible because pandas sets
    # the dtype of the cell as float instead ob object
    configurations.loc[i_config, 'config_has_run'] = False
    configurations.loc[i_config, 'mse_train'] = False
    configurations.loc[i_config, 'mse_valid'] = False
    configurations.loc[i_config, 'mse_test'] = False
    configurations.loc[i_config, 'corr_train'] = False
    configurations.loc[i_config, 'corr_valid'] = False
    configurations.loc[i_config, 'corr_test'] = False
    configurations.loc[i_config, 'corr_p_train'] = False
    configurations.loc[i_config, 'corr_p_valid'] = False
    configurations.loc[i_config, 'corr_p_test'] = False

    if any(matching_results):
        configurations.loc[i_config, 'config_has_run'] = True
        configurations.loc[i_config, 'mse_train'] = [[np.nan], [np.nan], [np.nan]]
        configurations.loc[i_config, 'mse_valid'] = [[np.nan], [np.nan], [np.nan]]
        configurations.loc[i_config, 'mse_test'] = [[np.nan], [np.nan], [np.nan]]
        configurations.loc[i_config, 'corr_train'] = [[np.nan], [np.nan], [np.nan]]
        configurations.loc[i_config, 'corr_valid'] = [[np.nan], [np.nan], [np.nan]]
        configurations.loc[i_config, 'corr_test'] = [[np.nan], [np.nan], [np.nan]]
        configurations.loc[i_config, 'corr_p_train'] = [[np.nan], [np.nan], [np.nan]]
        configurations.loc[i_config, 'corr_p_valid'] = [[np.nan], [np.nan], [np.nan]]
        configurations.loc[i_config, 'corr_p_test'] = [[np.nan], [np.nan], [np.nan]]
        i_subject = -1
        for subject in matching_results:
            i_subject += 1
            df = pd.read_csv(subject)#, names=['epoch',
                                             # 'train_loss','valid_loss','test_loss',
                                             # 'train_corr','valid_corr','test_corr', 'runtime'])
            if configurations.loc[i_config, 'model_name'] in ['deep4', 'eegnet', 'resnet']:
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
            elif configurations.loc[i_config, 'model_name'] in ['lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg']:
                configurations.loc[i_config, 'mse_train'][i_subject] = df.values[0, 1]
                configurations.loc[i_config, 'mse_valid'][i_subject] = df.values[4, 1]
                configurations.loc[i_config, 'mse_test'][i_subject] = df.values[8, 1]
                configurations.loc[i_config, 'corr_train'][i_subject] = df.values[1, 1]
                configurations.loc[i_config, 'corr_valid'][i_subject] = df.values[5, 1]
                configurations.loc[i_config, 'corr_test'][i_subject] = df.values[9, 1]
                configurations.loc[i_config, 'corr_p_train'][i_subject] = df.values[2, 1]
                configurations.loc[i_config, 'corr_p_valid'][i_subject] = df.values[6, 1]
                configurations.loc[i_config, 'corr_p_test'][i_subject] = df.values[10, 1]
            else:
                print('Unknown model name: {}'.format(configurations.loc[i_config, 'model_name']))
                break
    else:
        configurations.loc[i_config, 'config_has_run'] = False
        configurations.loc[i_config, 'mse_train'] = [[np.nan], [np.nan], [np.nan]]
        configurations.loc[i_config, 'mse_valid'] = [[np.nan], [np.nan], [np.nan]]
        configurations.loc[i_config, 'mse_test'] = [[np.nan], [np.nan], [np.nan]]
        configurations.loc[i_config, 'corr_train'] = [[np.nan], [np.nan], [np.nan]]
        configurations.loc[i_config, 'corr_valid'] = [[np.nan], [np.nan], [np.nan]]
        configurations.loc[i_config, 'corr_test'] = [[np.nan], [np.nan], [np.nan]]
        configurations.loc[i_config, 'corr_p_train'] = [[np.nan], [np.nan], [np.nan]]
        configurations.loc[i_config, 'corr_p_valid'] = [[np.nan], [np.nan], [np.nan]]
        configurations.loc[i_config, 'corr_p_test'] = [[np.nan], [np.nan], [np.nan]]

# configurations.to_csv(configurations_file[:-4] + '_with_results.csv')
# Somehow converts my lists into strings!!!

#%% Load only missing results only instead of having to rerun the above every time
for i_config in configurations.index:
    if configurations.loc[i_config, 'config_has_run'] == False:
        result_folder = configurations.loc[i_config, 'result_folder']
        unique_id = configurations.loc[i_config, 'unique_id']
        matching_results = np.sort(glob.glob('/mnt/meta-cluster/' + result_folder + '/*' + unique_id + '.csv'))
        # matching_results = np.sort(glob.glob(result_folder + '/*' + unique_id[:7] + '.csv'))
        # Check that we get only one result per subject, but not sure what to do if not. Check that results are identical?

        # I have to do this ugly thing here because initializing with 3 nans is somehow not possible because pandas sets
        # the dtype of the cell as float instead ob object
        configurations.loc[i_config, 'config_has_run'] = False
        configurations.loc[i_config, 'mse_train'] = False
        configurations.loc[i_config, 'mse_valid'] = False
        configurations.loc[i_config, 'mse_test'] = False
        configurations.loc[i_config, 'corr_train'] = False
        configurations.loc[i_config, 'corr_valid'] = False
        configurations.loc[i_config, 'corr_test'] = False
        configurations.loc[i_config, 'corr_p_train'] = False
        configurations.loc[i_config, 'corr_p_valid'] = False
        configurations.loc[i_config, 'corr_p_test'] = False

        if any(matching_results):
            configurations.loc[i_config, 'config_has_run'] = True
            configurations.loc[i_config, 'mse_train'] = [[np.nan], [np.nan], [np.nan]]
            configurations.loc[i_config, 'mse_valid'] = [[np.nan], [np.nan], [np.nan]]
            configurations.loc[i_config, 'mse_test'] = [[np.nan], [np.nan], [np.nan]]
            configurations.loc[i_config, 'corr_train'] = [[np.nan], [np.nan], [np.nan]]
            configurations.loc[i_config, 'corr_valid'] = [[np.nan], [np.nan], [np.nan]]
            configurations.loc[i_config, 'corr_test'] = [[np.nan], [np.nan], [np.nan]]
            configurations.loc[i_config, 'corr_p_train'] = [[np.nan], [np.nan], [np.nan]]
            configurations.loc[i_config, 'corr_p_valid'] = [[np.nan], [np.nan], [np.nan]]
            configurations.loc[i_config, 'corr_p_test'] = [[np.nan], [np.nan], [np.nan]]
            i_subject = -1
            for subject in matching_results:
                i_subject += 1
                df = pd.read_csv(subject)#, names=['epoch',
                                                 # 'train_loss','valid_loss','test_loss',
                                                 # 'train_corr','valid_corr','test_corr', 'runtime'])
                if configurations.loc[i_config, 'model_name'] in ['deep4', 'eegnet', 'resnet']:
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
                elif configurations.loc[i_config, 'model_name'] in ['lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg']:
                    configurations.loc[i_config, 'mse_train'][i_subject] = df.values[0, 1]
                    configurations.loc[i_config, 'mse_valid'][i_subject] = df.values[4, 1]
                    configurations.loc[i_config, 'mse_test'][i_subject] = df.values[8, 1]
                    configurations.loc[i_config, 'corr_train'][i_subject] = df.values[1, 1]
                    configurations.loc[i_config, 'corr_valid'][i_subject] = df.values[5, 1]
                    configurations.loc[i_config, 'corr_test'][i_subject] = df.values[9, 1]
                    configurations.loc[i_config, 'corr_p_train'][i_subject] = df.values[2, 1]
                    configurations.loc[i_config, 'corr_p_valid'][i_subject] = df.values[6, 1]
                    configurations.loc[i_config, 'corr_p_test'][i_subject] = df.values[10, 1]
                else:
                    print('Unknown model name: {}'.format(configurations.loc[i_config, 'model_name']))
                    break

#%% Plot comparison matrix
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
        model_vector.append(np.nan)# do something

electrode_vector = []
for electrodes in configurations['electrodes']:
    if electrodes == '*':
        electrode_vector.append(1)
    elif electrodes == '*z':
        electrode_vector.append(2)
    elif electrodes == '*C*':
        electrode_vector.append(3)
    else:
        electrode_vector.append(np.nan)# do something

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
        bandpass_vector.append(np.nan)# do something

for i_subject in range(len(subject_values)):
    for data_split in data_split_values:
        df = configurations['mse_' + data_split]
        df = df.apply(lambda x: [np.sqrt(a) if a is not None else np.nan for a in x])
        rmse = np.vstack(df.values).flatten()[i_subject::3]
        diff_mat = np.reshape([a-b for a in rmse for b in rmse], [rmse.size, rmse.size], order='C')
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


#%% Plot fancy results
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

#%% Some metrics
i_subject = 0
model_name = 'lin_svr'
# electrodes =
df_mse = configurations[(configurations['model_name'] == model_name) & (configurations['electrodes'] == "'*C*'")][
    'mse_test']
df_corr = configurations[(configurations['model_name'] == model_name) & (configurations['electrodes'] == "'*C*'")][
    'corr_test']

print(np.median([b[i_subject]/a[i_subject] for a, b in zip(df_mse, df_corr)]))

df_mse = configurations[(configurations['model_name'] == model_name) & (configurations['electrodes'] == "'*z'")][
    'mse_test']
df_corr = configurations[(configurations['model_name'] == model_name) & (configurations['electrodes'] == "'*z'")][
    'corr_test']
print(np.median([b[i_subject]/a[i_subject] for a, b in zip(df_mse, df_corr)]))

df_mse = configurations[(configurations['model_name'] == model_name) & (configurations['electrodes'] == "'*'")][
    'mse_test']
df_corr = configurations[(configurations['model_name'] == model_name) & (configurations['electrodes'] == "'*'")][
    'corr_test']
print(np.median([b[i_subject]/a[i_subject] for a, b in zip(df_mse, df_corr)]))

#%% Overall best performing method

data_split_values = ['train', 'valid', 'test']
model_values = configurations['model_name'].unique()
assert(all(model_values == ['eegnet', 'deep4', 'resnet', 'lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg']))

f = plt.figure(figsize=[15, 10])
for i_split in range(len(data_split_values)):
    ax = plt.subplot(np.size(model_values) + 1, np.size(data_split_values), i_split+1)
    plt.text(0.5, 0.5, 'All data types: ' + data_split_values[i_split], size=16, horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes)
    ax.axis('off')

# ax = plt.subplot(np.size(model_values)+1, np.size(data_split_values), 1)
# ax.text(data_split_values[0])
# ax = plt.subplot(np.size(model_values)+1, np.size(data_split_values), 2)
# ax.set_title(data_split_values[1])
# ax = plt.subplot(np.size(model_values)+1, np.size(data_split_values), 3)
# ax.set_title(data_split_values[2])

subplot_index = 4
for model_name in model_values:
    for data_split in data_split_values:
        ax = plt.subplot(np.size(model_values)+1, np.size(data_split_values), subplot_index)
        subplot_index += 1
        df_mse = configurations[(configurations['model_name'] == model_name)]['mse_' + data_split]
        df_corr = configurations[(configurations['model_name'] == model_name)]['corr_' + data_split]
        # x = np.abs([np.array([i if i is not None else np.nan for i in b]) /
        #             np.array([i if i is not None else np.nan for i in a])
        #             for a, b in zip(df_mse, df_corr)])
        x = np.array([i if i is not None else np.nan for i in df_mse])
        # x = np.array([i if i is not None else np.nan for i in df_corr])

        ax.hist(x[~np.isnan(x)], bins='auto')
        ax.axis('tight')
        x_median = np.nanmedian(x.flatten())
        ax.axvline(x_median, color='r')
        ax.text(x_median, 0, '{:.3f}'.format(x_median), color='k')
        ax.set_title(model_name)
# for i_split in range(len(data_split_values)):
#     plt.text(0.25*(i_split+1), 0.95, data_split_values[i_split], horizontalalignment='center',
#              verticalalignment='center',
#          transform=f.transFigure)

f.tight_layout()
plt.show()

# User test to determine if anything significantly different
stats.wilcoxon()

#%% Best performing method for each data type

data_values = configurations['data'].unique()
data_split_values = ['train', 'valid', 'test']
model_values = configurations['model_name'].unique()
assert(all(model_values == ['eegnet', 'deep4', 'resnet', 'lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg']))

for data in data_values:
    df = configurations[configurations['data'] == data]

    f = plt.figure(figsize=[15, 10])
    for i_split in range(len(data_split_values)):
        ax = plt.subplot(np.size(model_values) + 1, np.size(data_split_values), i_split+1)
        plt.text(0.5, 0.5, data + ': ' + data_split_values[i_split], size=16, horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax.transAxes)
        ax.axis('off')

    # ax = plt.subplot(np.size(model_values)+1, np.size(data_split_values), 1)
    # ax.text(data_split_values[0])
    # ax = plt.subplot(np.size(model_values)+1, np.size(data_split_values), 2)
    # ax.set_title(data_split_values[1])
    # ax = plt.subplot(np.size(model_values)+1, np.size(data_split_values), 3)
    # ax.set_title(data_split_values[2])

    subplot_index = 4
    for model_name in model_values:
        for data_split in data_split_values:
            ax = plt.subplot(np.size(model_values)+1, np.size(data_split_values), subplot_index)
            subplot_index += 1
            df_mse = df[(df['model_name'] == model_name)]['mse_' + data_split]
            df_corr = df[(df['model_name'] == model_name)]['corr_' + data_split]
            # x = np.abs([np.array([i if i is not None else np.nan for i in b]) /
            #             np.array([i if i is not None else np.nan for i in a])
            #             for a, b in zip(df_mse, df_corr)])
            x = np.abs([np.array([i if i is not None else np.nan for i in df_mse])])
            # x = np.abs([np.array([i if i is not None else np.nan for i in df_corr])])

            ax.hist(x[~np.isnan(x)].flatten(), bins='auto')
            ax.axis('tight')
            x_median = np.nanmedian(x.flatten())
            ax.axvline(x_median, color='r')
            ax.text(x_median, 0, '{:.3f}'.format(x_median), color='k')
            ax.set_title(model_name)
    # for i_split in range(len(data_split_values)):
    #     plt.text(0.25*(i_split+1), 0.95, data_split_values[i_split], horizontalalignment='center',
    #              verticalalignment='center',
    #          transform=f.transFigure)

    f.tight_layout()
    plt.show()

#%% Overall best electrode set
data_split_values = ['train', 'valid', 'test']
electrode_sets = configurations['electrodes'].unique()
assert(all(electrode_sets == ["'*'", "'*z'", "'*C*'"]))

f = plt.figure(figsize=[15, 10])
for i_split in range(len(data_split_values)):
    ax = plt.subplot(np.size(electrode_sets) + 1, np.size(data_split_values), i_split+1)
    plt.text(0.5, 0.5, 'All data types: ' + data_split_values[i_split], size=16, horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes)
    ax.axis('off')

# ax = plt.subplot(np.size(model_values)+1, np.size(data_split_values), 1)
# ax.text(data_split_values[0])
# ax = plt.subplot(np.size(model_values)+1, np.size(data_split_values), 2)
# ax.set_title(data_split_values[1])
# ax = plt.subplot(np.size(model_values)+1, np.size(data_split_values), 3)
# ax.set_title(data_split_values[2])

subplot_index = 4
for electrode_set_name in electrode_sets:
    for data_split in data_split_values:
        ax = plt.subplot(np.size(electrode_sets)+1, np.size(data_split_values), subplot_index)
        subplot_index += 1
        df_mse = configurations[(configurations['electrodes'] == electrode_set_name)]['mse_' + data_split]
        df_corr = configurations[(configurations['electrodes'] == electrode_set_name)]['corr_' + data_split]
        # x = np.abs([np.array([i if i is not None else np.nan for i in b]) /
        #             np.array([i if i is not None else np.nan for i in a])
        #             for a, b in zip(df_mse, df_corr)])
        x = np.array([i if i is not None else np.nan for i in df_mse])
        # x = np.array([i if i is not None else np.nan for i in df_corr])

        ax.hist(x[~np.isnan(x)], bins='auto')
        ax.axis('tight')
        x_median = np.nanmedian(x.flatten())
        ax.axvline(x_median, color='r')
        ax.text(x_median, 0, '{:.3f}'.format(x_median), color='k')
        ax.set_title(electrode_set_name)
# for i_split in range(len(data_split_values)):
#     plt.text(0.25*(i_split+1), 0.95, data_split_values[i_split], horizontalalignment='center',
#              verticalalignment='center',
#          transform=f.transFigure)

f.tight_layout()
plt.show()


#%% Overall best frequency band
data_split_values = ['train', 'valid', 'test']
bandpass_sets = configurations['band_pass'].unique()
# assert(all(bandpass_sets == ["'*'", "'*z'", "'*C*'"]))

f = plt.figure(figsize=[15, 10])
for i_split in range(len(data_split_values)):
    ax = plt.subplot(np.size(bandpass_sets) + 1, np.size(data_split_values), i_split+1)
    plt.text(0.5, 0.5, 'All data types: ' + data_split_values[i_split], size=16, horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes)
    ax.axis('off')

# ax = plt.subplot(np.size(model_values)+1, np.size(data_split_values), 1)
# ax.text(data_split_values[0])
# ax = plt.subplot(np.size(model_values)+1, np.size(data_split_values), 2)
# ax.set_title(data_split_values[1])
# ax = plt.subplot(np.size(model_values)+1, np.size(data_split_values), 3)
# ax.set_title(data_split_values[2])

subplot_index = 4
for bandpass in bandpass_sets:
    for data_split in data_split_values:
        ax = plt.subplot(np.size(bandpass_sets)+1, np.size(data_split_values), subplot_index)
        subplot_index += 1
        df_mse = configurations[(configurations['band_pass'] == bandpass)]['mse_' + data_split]
        df_corr = configurations[(configurations['band_pass'] == bandpass)]['corr_' + data_split]
        # x = np.abs([np.array([i if i is not None else np.nan for i in b]) /
        #             np.array([i if i is not None else np.nan for i in a])
        #             for a, b in zip(df_mse, df_corr)])
        x = np.array([i if i is not None else np.nan for i in df_mse])
        # x = np.array([i if i is not None else np.nan for i in df_corr])

        ax.hist(x[~np.isnan(x)], bins='auto')
        ax.axis('tight')
        x_median = np.nanmedian(x.flatten())
        ax.axvline(x_median, color='r')
        ax.text(x_median, 0, '{:.3f}'.format(x_median), color='k')
        ax.set_title(bandpass)
# for i_split in range(len(data_split_values)):
#     plt.text(0.25*(i_split+1), 0.95, data_split_values[i_split], horizontalalignment='center',
#              verticalalignment='center',
#          transform=f.transFigure)

f.tight_layout()
plt.show()

#%% Circle plot in one axes

f = plt.figure()
x_positions = np.array([7, 19, 1, 13, 25, 7, 19])
y_positions = np.array([1, 1, 1.5, 1.5, 1.5, 2, 2])
subplot_index = 1
x_shift = 18
y_shift = 11*4
for i_no_eeg_data in np.arange(2):
    for i_subject in np.arange(3):
        # plt.subplot(5, 3, subplot_index)
        plt.scatter(x_positions+x_shift, y_positions+y_shift, s=np.random.rand(7) * 10,
                    c=('b', 'g', 'r', 'c', 'm', 'y', 'k'))
        x_shift += 6*21
        # subplot_index += 1
    x_shift = 18
    y_shift -= 2


x_shift = 0
y_shift = 10*4
for i_eeg_data in np.arange(3):
    for i_subject in np.arange(3):
        # plt.subplot(5, 3,subplot_index)
        for i_bandpass in np.arange(7):
            for i_electrode in np.arange(3):
                plt.scatter(x_positions+x_shift, y_positions+y_shift, s=np.random.rand(7)*10,
                            c=('b', 'g', 'r', 'c', 'm', 'y', 'k'))
                y_shift -= 2
            x_shift += 50
            y_shift = 10*4
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
