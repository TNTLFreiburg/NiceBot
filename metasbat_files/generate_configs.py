import pandas as pd
import numpy as np
import itertools as it
import shortuuid

# data_folder = ['/data/schirrmr/fiederer/nicebot/data']
# batch_size = [64]
# max_epochs = [200]
# cuda = ['True']
# result_folder = ['/data/schirrmr/fiederer/nicebot/results']
# model_name = ['eegnet', 'deep4', 'resnet'] # lin_reg, lin_svr, rbf_svr, rf_reg
# init_lr = [0.001]
# weight_decay =[0]
# band_pass = np.array([['None', 'None'], [0, 4], [4, 8], [8, 14], [14, 20], [20, 30], [30, 40]])
# band_pass_low = band_pass[:, 0]
# band_pass_high = band_pass[:, 1]
# electrodes = ['*', '*C*']
# sampling_rate = [256]
# n_seconds_valid_set = [180]
# n_seconds_test_set = [180]
# data = ['onlyRobotData', 'onlyEEGData', 'onlyAux', 'RobotEEGAux', 'RobotEEG']

# Execute this section only if you need to create the initial config file.
# To append new configs use the section at the end of the file
# config_space_EEG = {
#     'unique_id': '0',
#     'data_folder': ['/data/schirrmr/fiederer/nicebot/data'],
#     'batch_size': [64],
#     'max_epochs': [200],
#     'cuda': ['True'],
#     'result_folder': ['/data/schirrmr/fiederer/nicebot/results'],
#     'model_name': ['eegnet', 'deep4', 'resnet', 'lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg'],
#     'init_lr': [0.001],
#     'weight_decay': [0],
#     # 'band_pass_low': band_pass[:, 0],
#     # 'band_pass_high': band_pass[:, 1],
#     'band_pass': np.array(["'[None, None]'", "'[0, 4]'", "'[4, 8]'", "'[8, 14]'", "'[14, 20]'", "'[20, 30]'", "'[30, 40]'"]),
#     'electrodes': ["'*'", "'*z'", "'*C*'"],
#     'sampling_rate': [256],
#     'n_seconds_valid_set': [180],
#     'n_seconds_test_set': [180],
#     'data': ['onlyEEGData', 'RobotEEGAux', 'RobotEEG']
# }
# rows = it.product(*config_space_EEG.values())
# configs_EEG = pd.DataFrame.from_records(rows, columns=config_space_EEG.keys())
#
# config_space_no_EEG = {
#     'unique_id': '0',
#     'data_folder': ['/data/schirrmr/fiederer/nicebot/data'],
#     'batch_size': [64],
#     'max_epochs': [200],
#     'cuda': ['True'],
#     'result_folder': ['/data/schirrmr/fiederer/nicebot/results'],
#     'model_name': ['eegnet', 'deep4', 'resnet', 'lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg'],
#     'init_lr': [0.001],
#     'weight_decay': [0],
#     # 'band_pass_low': ['None'],
#     # 'band_pass_high': ['None'],
#     'band_pass': np.array(["'[None, None]'"]),
#     'electrodes': ["'*'"],
#     'sampling_rate': [256],
#     'n_seconds_valid_set': [180],
#     'n_seconds_test_set': [180],
#     'data': ['onlyRobotData', 'onlyAux']
# }
# rows = it.product(*config_space_no_EEG.values())
# configs_no_EEG = pd.DataFrame.from_records(rows, columns=config_space_no_EEG.keys())
#
# configs = pd.concat([configs_no_EEG, configs_EEG], ignore_index=True)
#
# for index in configs.index:
#     configs['unique_id'][index] = shortuuid.uuid()
#
# configs.to_csv('/home/fiederer/nicebot/metasbat_files/configs.csv')

# To add new configurations to an existing configuration file without overwriting identical
# existing configurations (and their uuid) use this section.
existing_configs = pd.DataFrame.from_csv('/home/fiederer/nicebot/metasbat_files/configs.csv') # Load existing configs

config_space_EEG = {
    'unique_id': '0',
    'data_folder': ['/data/schirrmr/fiederer/nicebot/data'],
    'batch_size': [64],
    'max_epochs': [200],
    'cuda': ['True'],
    'result_folder': ['/data/schirrmr/fiederer/nicebot/results'],
    'model_name': ['eegnet', 'deep4', 'resnet', 'lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg'],
    'init_lr': [0.001],
    'weight_decay': [0],
    'band_pass': np.array(["'[None, None]'", "'[0, 4]'", "'[4, 8]'", "'[8, 14]'", "'[14, 20]'", "'[20, 30]'", "'[30, 40]'"]),
    'electrodes': ["'*'", "'*z'", "'*C*'"],
    'sampling_rate': [256],
    'n_seconds_valid_set': [180],
    'n_seconds_test_set': [180],
    'data': ['onlyEEGData', 'RobotEEGAux', 'RobotEEG']
}
rows = it.product(*config_space_EEG.values())
configs_EEG = pd.DataFrame.from_records(rows, columns=config_space_EEG.keys())

config_space_no_EEG = {
    'unique_id': '0',
    'data_folder': ['/data/schirrmr/fiederer/nicebot/data'],
    'batch_size': [64],
    'max_epochs': [200],
    'cuda': ['True'],
    'result_folder': ['/data/schirrmr/fiederer/nicebot/results'],
    'model_name': ['eegnet', 'deep4', 'resnet', 'lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg'],
    'init_lr': [0.001],
    'weight_decay': [0],
    'band_pass': np.array(["'[None, None]'"]),
    'electrodes': ["'*'"],
    'sampling_rate': [256],
    'n_seconds_valid_set': [180],
    'n_seconds_test_set': [180],
    'data': ['onlyRobotData', 'onlyAux']
}
rows = it.product(*config_space_no_EEG.values())
configs_no_EEG = pd.DataFrame.from_records(rows, columns=config_space_no_EEG.keys())

configs = pd.concat([configs_no_EEG, configs_EEG], ignore_index=True)

pd.concat([existing_configs, configs]).drop_duplicates(subset=existing_configs.keys()[1:]).reset_index(drop=True)