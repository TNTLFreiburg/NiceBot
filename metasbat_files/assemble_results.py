#%% Imports
import pandas as pd
import numpy as np
import scipy as sp
import glob
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import re
import statsmodels.stats.multitest
import itertools as it

#%% Definitions
def initialize_empty_columns_at_index(dataframe, row_index):
    # I have to do this ugly thing here because initializing with 3 nans is somehow not possible because pandas sets
    # the dtype of the cell as float instead ob object
    dataframe.loc[row_index, 'config_has_run'] = False
    dataframe.loc[row_index, 'mse_train'] = False
    dataframe.loc[row_index, 'mse_valid'] = False
    dataframe.loc[row_index, 'mse_test'] = False
    dataframe.loc[row_index, 'corr_train'] = False
    dataframe.loc[row_index, 'corr_valid'] = False
    dataframe.loc[row_index, 'corr_test'] = False
    dataframe.loc[row_index, 'corr_p_train'] = False
    dataframe.loc[row_index, 'corr_p_valid'] = False
    dataframe.loc[row_index, 'corr_p_test'] = False
    dataframe.loc[row_index, 'r^2_train'] = False
    dataframe.loc[row_index, 'r^2_valid'] = False
    dataframe.loc[row_index, 'r^2_test'] = False
    return dataframe


def prefill_columns_configuration_at_index(dataframe, row_index, has_run):
    dataframe.loc[row_index, 'config_has_run'] = has_run
    dataframe.loc[row_index, 'mse_train'] = [[np.nan], [np.nan], [np.nan]]
    dataframe.loc[row_index, 'mse_valid'] = [[np.nan], [np.nan], [np.nan]]
    dataframe.loc[row_index, 'mse_test'] = [[np.nan], [np.nan], [np.nan]]
    dataframe.loc[row_index, 'corr_train'] = [[np.nan], [np.nan], [np.nan]]
    dataframe.loc[row_index, 'corr_valid'] = [[np.nan], [np.nan], [np.nan]]
    dataframe.loc[row_index, 'corr_test'] = [[np.nan], [np.nan], [np.nan]]
    dataframe.loc[row_index, 'corr_p_train'] = [[np.nan], [np.nan], [np.nan]]
    dataframe.loc[row_index, 'corr_p_valid'] = [[np.nan], [np.nan], [np.nan]]
    dataframe.loc[row_index, 'corr_p_test'] = [[np.nan], [np.nan], [np.nan]]
    dataframe.loc[row_index, 'r^2_train'] = [[np.nan], [np.nan], [np.nan]]
    dataframe.loc[row_index, 'r^2_valid'] = [[np.nan], [np.nan], [np.nan]]
    dataframe.loc[row_index, 'r^2_test'] = [[np.nan], [np.nan], [np.nan]]
    return dataframe


def fill_results_in_configurations_at_index(dataframe, row_index, df):
    if len(df) == 12:
        dataframe.loc[row_index, 'mse_train'][i_subject] = df.values[0, 1]
        dataframe.loc[row_index, 'mse_valid'][i_subject] = df.values[4, 1]
        dataframe.loc[row_index, 'mse_test'][i_subject] = df.values[8, 1]
        dataframe.loc[row_index, 'corr_train'][i_subject] = df.values[1, 1]
        dataframe.loc[row_index, 'corr_valid'][i_subject] = df.values[5, 1]
        dataframe.loc[row_index, 'corr_test'][i_subject] = df.values[9, 1]
        dataframe.loc[row_index, 'corr_p_train'][i_subject] = df.values[2, 1]
        dataframe.loc[row_index, 'corr_p_valid'][i_subject] = df.values[6, 1]
        dataframe.loc[row_index, 'corr_p_test'][i_subject] = df.values[10, 1]
        dataframe.loc[row_index, 'r^2_train'][i_subject] = df.values[3, 1]
        dataframe.loc[row_index, 'r^2_valid'][i_subject] = df.values[7, 1]
        dataframe.loc[row_index, 'r^2_test'][i_subject] = df.values[11, 1]
    elif len(df) == 9:
        print('WARNING! Residual result df missing r^2 (len = 9)!')
        dataframe.loc[row_index, 'mse_train'][i_subject] = df.values[0, 1]
        dataframe.loc[row_index, 'mse_valid'][i_subject] = df.values[3, 1]
        dataframe.loc[row_index, 'mse_test'][i_subject] = df.values[6, 1]
        dataframe.loc[row_index, 'corr_train'][i_subject] = df.values[1, 1]
        dataframe.loc[row_index, 'corr_valid'][i_subject] = df.values[4, 1]
        dataframe.loc[row_index, 'corr_test'][i_subject] = df.values[7, 1]
        dataframe.loc[row_index, 'corr_p_train'][i_subject] = df.values[2, 1]
        dataframe.loc[row_index, 'corr_p_valid'][i_subject] = df.values[5, 1]
        dataframe.loc[row_index, 'corr_p_test'][i_subject] = df.values[8, 1]
    else:
        print('Unknown result df format of len: {}'.format(len(df)))
        exit()
    # if configurations.loc[i_config, 'model_name'] in ['deep4', 'eegnet', 'resnet']:
    #     for column_name in df.columns:
    #         if column_name == 'train_loss':
    #             configurations.loc[i_config, 'mse_train'][i_subject] = df.tail(1)[column_name].values[0]#, 1]
    #         if column_name == 'valid_loss':
    #             configurations.loc[i_config, 'mse_valid'][i_subject] = df.tail(1)[column_name].values[0]#, 2]
    #         if column_name == 'test_loss':
    #             configurations.loc[i_config, 'mse_test'][i_subject] = df.tail(1)[column_name].values[0]#, 3]
    #         if column_name == 'train_corr':
    #             configurations.loc[i_config, 'corr_train'][i_subject] = df.tail(1)[column_name].values[0]#, 4]
    #         if column_name == 'valid_corr':
    #             configurations.loc[i_config, 'corr_valid'][i_subject] = df.tail(1)[column_name].values[0]#, 5]
    #         if column_name == 'test_corr':
    #             configurations.loc[i_config, 'corr_test'][i_subject] = df.tail(1)[column_name].values[0]#, 6]
    # elif configurations.loc[i_config, 'model_name'] in ['lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg']:
    #     dataframe.loc[row_index, 'mse_train'][i_subject] = df.values[0, 1]
    #     dataframe.loc[row_index, 'mse_valid'][i_subject] = df.values[4, 1]
    #     dataframe.loc[row_index, 'mse_test'][i_subject] = df.values[8, 1]
    #     dataframe.loc[row_index, 'corr_train'][i_subject] = df.values[1, 1]
    #     dataframe.loc[row_index, 'corr_valid'][i_subject] = df.values[5, 1]
    #     dataframe.loc[row_index, 'corr_test'][i_subject] = df.values[9, 1]
    #     dataframe.loc[row_index, 'corr_p_train'][i_subject] = df.values[2, 1]
    #     dataframe.loc[row_index, 'corr_p_valid'][i_subject] = df.values[6, 1]
    #     dataframe.loc[row_index, 'corr_p_test'][i_subject] = df.values[10, 1]
    # else:
    #     print('Unknown model name: {}'.format(configurations.loc[i_config, 'model_name']))
    #     break
    return dataframe


def split_combi_plot(dataframe, x_col='metric_value', y_col='model_name', hue_col='metric_name', palette='colorblind'):

    sns.violinplot(x=x_col, y=y_col, hue=hue_col, data=dataframe, palette=palette,
                   color='1', split=True, inner='quartiles', scale='count', scale_hue=True, cut=2, bw=0.15)
    add_data_and_format_to_plot(dataframe, x_col, y_col, hue_col, palette)


def combi_plot(dataframe, x_col='metric_value', y_col='model_name', hue_col='metric_name', palette='colorblind'):

    sns.violinplot(x=x_col, y=y_col, hue=hue_col, data=dataframe, palette=palette,
                   color='1', split=False, inner='quartiles', scale='count', scale_hue=True, cut=2, bw=0.15)
    add_data_and_format_to_plot(dataframe, x_col, y_col, hue_col, palette)


def add_data_and_format_to_plot(dataframe, x_col='metric_value', y_col='model_name', hue_col='metric_name', palette='colorblind'):
    # sns.boxplot(x=x_col, y=y_col, hue=hue_col, data=dataframe, whis="range", width=0.1)

    # Add in points to show each observation
    # sns.swarmplot(x=x_col, y=y_col, hue=hue_col, data=dataframe, size=2, color=".3", linewidth=0,
    #               dodge=True)
    sns.stripplot(x=x_col, y=y_col, hue=hue_col, data=dataframe, dodge=True, jitter=True, alpha=.25,
                  zorder=1, palette=palette, color='1', edgecolor='black', linewidth=1)
    # colors = sns.color_palette('vlag',n_colors=7)
    # # colors = sp.repeat(colors,2,axis=0)
    #
    # for artist, collection, color in zip(f.axes[0].artists, f.axes[0].collections, colors):
    #     artist.set_facecolor(color)
    #     collection.set_facecolor(color)

    # Tweak the visual presentation
    f.axes[0].get_yaxis().set_minor_locator(mpl.ticker.MultipleLocator(0.02))  # AutoMinorLocator())
    f.axes[0].get_yaxis().set_major_locator(mpl.ticker.MultipleLocator(0.1))
    # ax.set_xticks(np.arange(0,8)-0.5, minor=True)
    f.axes[0].yaxis.grid(b=True, which='minor', linewidth=0.5)
    f.axes[0].yaxis.grid(b=True, which='major', linewidth=1.5)
    f.axes[0].set_axisbelow(True)
    f.axes[0].set(ylabel="")
    # sns.despine(trim=True, left=True)
    # xlims = plt.xlim()
    # plt.xlim(0, xlims[1])
    data_types = np.unique(dataframe['data'].values)
    data_types = ', '.join(data_types)
    plt.title('All models, ' + data_types)
    # f.tight_layout()


def sign_test(a, b, p=0.5, alternative='two-sided'):
    # Should be same as https://onlinecourses.science.psu.edu/stat464/node/49
    # Link is dead now...

    a = np.array(a)
    b = np.array(b)
    assert len(a) == len(b)
    n_samples = len(a)
    diffs = a - b
    n_positive = np.sum(diffs > 0)
    n_equal = np.sum(diffs == 0)
    # adding half of equal to positive (so implicitly
    # other half is added to negative total)
    n_total = n_positive + (n_equal / 2)
    # rounding conservatively
    if n_total < (n_samples / 2):
        n_total = int(np.ceil(n_total))
    else:
        n_total = int(np.floor(n_total))

    return sp.stats.binom_test(n_total, n_samples, p, alternative)


def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, barc='black', fs=None,
                              fw=None, maxasterix=None):
    """
    Modified from https://stackoverflow.com/a/52333561/6786600
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param barc: bar color in matplotlib color code
    :param fs: font size
    :param fw: font weigth
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, rx]  # [lx, lx, rx, rx]
    bary = [y+barh, y+barh]  # [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c=barc)

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs
    if fw is not None:
        kwargs['fontweight'] = fw

    plt.text(*mid, text, **kwargs)


def fdr_corrected_pvals(pvals):
    return statsmodels.stats.multitest.multipletests(pvals, alpha=0.05, method='fdr_by',
                                                                       is_sorted=False,
                                     returnsorted=False)[1]


def add_significance_bars(dataframe, condition_col, metric_name_col='metric_name', metric_name='rmse',
                          metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0]):
    plt.axis('tight')
    ylim = plt.ylim()
    plt.ylim(ylim[0], ylim[1] * 2)
    condition_values, condition_indices = np.unique(dataframe[condition_col], return_index=True)
    condition_values = condition_values[np.argsort(condition_indices)]
    condition_indices = range(len(condition_values))
    barh = 0
    pvals = []
    pair_index = 0
    for pair in it.combinations(range(len(condition_indices)), 2):
        pvals.append(dataframe_significance_test(dataframe, condition_col, condition_values[pair[0]],
                                                 condition_values[pair[1]],
                                                 metric_name_col, metric_name, metric_value_col))
    pvals_corrected = fdr_corrected_pvals(pvals)
    q2fw = lambda x: 1000 if x < 0.05 else 0
    for pair in it.combinations(range(len(condition_indices)), 2):
        barplot_annotate_brackets(pair[0],
                                  pair[1],
                                  'q = {:.4g}'.format(pvals_corrected[pair_index]),
                                  condition_indices,
                                  # np.repeat(0, len(condition_indices)),
                                  np.repeat(np.max(dataframe[metric_value_col]) - 0.4, len(condition_indices)),
                                  # max_of_each_entry(metric_df, 'model', 'metric_value'),
                                  yerr=None,
                                  dh=.05,
                                  barh=barh,
                                  barc=barc,
                                  fs=8,
                                  fw=q2fw(pvals_corrected[pair_index]),
                                  maxasterix=None)
        barh += .01
        pair_index += 1
    plt.axis('tight')


def reverse_colormap(cmap, name='my_cmap_r'):
    """
    In:
    cmap, name
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1-t[0],t[2],t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r


def plot_performance_matrix(dataframe, condition_col, metric_name_col='metric_name', metric_name=['rmse',
                                                                                                  '1-abs(corr)'],
                          metric_value_col='metric_value',
                          title=None,
                          cmap=reverse_colormap(sns.light_palette(sns.color_palette('colorblind')[2],
                                                                  as_cmap=True))):
    # Compute performance matrix
    if type(metric_name) != str:
        performance_matrix_a, significance_matrix_a, condition_values_a = compute_performance_matrix(dataframe,
                                                                                               condition_col,
                                                                                               metric_name_col,
                                                                                               metric_name[0],
                                                                                               metric_value_col)
        performance_matrix_b, significance_matrix_b, condition_values_b = compute_performance_matrix(dataframe,
                                                                                               condition_col,
                                                                                               metric_name_col,
                                                                                               metric_name[1],
                                                                                               metric_value_col)
        assert all(condition_values_a == condition_values_b), 'Extracted conditions do not match!'
        condition_values = condition_values_a
        performance_matrix = np.tril(performance_matrix_a)
        performance_matrix[np.triu_indices_from(performance_matrix)] = performance_matrix_b.T[np.triu_indices_from(
            performance_matrix)]
        significance_matrix = np.tril(significance_matrix_a)
        significance_matrix[np.triu_indices_from(significance_matrix)] = significance_matrix_b.T[np.triu_indices_from(
            significance_matrix)]
    else:
        performance_matrix, significance_matrix, condition_values = compute_performance_matrix(dataframe,
                                                                                               condition_col,
                                                                                               metric_name_col,
                                                                                               metric_name,
                                                                                               metric_value_col)
    vmin = 0
    vmax = 0.05
    tick_step = (vmax-vmin)/5
    ticks = np.arange(vmin, vmax+tick_step, tick_step)
    fig, ax = plt.subplots()
    im = ax.imshow(significance_matrix, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    # im = ax.pcolormesh(significance_matrix, edgecolors='w', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = ax.figure.colorbar(im, ax=ax, label='q-values', ticks=ticks)
    q2fw = lambda x : 1000 if x < vmax else 0  #1000*np.exp(-25*x)
    # for cbar_label_idx, cbar_label in enumerate(cbar.ax.get_yticklabels()):
    #     cbar_label.set_fontweight(q2fw(ticks[cbar_label_idx]))
    # We want to show all ticks...
    ax.set(xticks=np.arange(performance_matrix.shape[1]),
           yticks=np.arange(performance_matrix.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=condition_values, yticklabels=condition_values,
           title=title,
           ylabel='',
           xlabel='')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    def _remove_leading_zero(value, string):
        if 1 > value > -1:
            if string[0] == '0':
                string = string.replace('0', '', 1)
        return string

    class MyFloat(float):
        def __str__(self):
            string = super().__str__()
            return _remove_leading_zero(self, string)

        def __format__(self, format_string):
            string = super().__format__(format_string)
            return _remove_leading_zero(self, string)

    fmt = '.2g'
    thresh = vmax/2 # significance_matrix.max() / 2.
    for i in range(performance_matrix.shape[0]):
        for j in range(performance_matrix.shape[1]):
            ax.text(j, i, format(MyFloat(performance_matrix[i, j]), fmt),
                    ha="center", va="center",
                    color="black" if significance_matrix[i, j] > thresh else "white",
                    weight=q2fw(significance_matrix[i, j]) if significance_matrix[i, j] < vmax else 0)  #
            # Font weight seems to behave non-linearly, at least to my perception
    xtick_positions = ax.get_xticks()
    ytick_positions = ax.get_yticks()
    ax.set_xticks(xtick_positions + 0.5, minor=True)  # xaxis.set_minor_locator(xtick_positions)
    ax.set_yticks(ytick_positions + 0.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', length=0)
    if type(metric_name) != str:
        tril_x = np.array([0., 0., performance_matrix.shape[0]])-0.5
        tril_y = np.array([0., performance_matrix.shape[1], performance_matrix.shape[1]])-0.5
        ax.fill(tril_x, tril_y, fill=False, hatch='/', edgecolor=sns.color_palette('colorblind')[0])
        triu_x = np.array([0, performance_matrix.shape[0], performance_matrix.shape[0]])-0.5
        triu_y = np.array([0, performance_matrix.shape[1], 0])-0.5
        ax.fill(triu_x, triu_y, fill=False, hatch='\\', edgecolor=sns.color_palette('colorblind')[1])
    fig.tight_layout()
    return ax


def compute_performance_matrix(dataframe, condition_col, metric_name_col='metric_name', metric_name='rmse',
                          metric_value_col='metric_value'):
    condition_values, condition_indices = np.unique(dataframe[condition_col], return_index=True)
    condition_values = condition_values[np.argsort(condition_indices)]
    condition_indices = range(len(condition_values))
    performance_matrix = np.ndarray((len(condition_indices), len(condition_indices)))
    significance_matrix = np.ndarray(performance_matrix.shape)
    for x, y in it.product(range(len(condition_indices)), range(len(condition_indices))):
        a = get_dataframe_values_matching_two_criteria_in_two_columns(
                                                                        dataframe,
                                                                        condition_col,
                                                                        condition_values[x],
                                                                        metric_name_col,
                                                                        metric_name,
                                                                        metric_value_col)
        b = get_dataframe_values_matching_two_criteria_in_two_columns(
                                                                        dataframe,
                                                                        condition_col,
                                                                        condition_values[y],
                                                                        metric_name_col,
                                                                        metric_name,
                                                                        metric_value_col)
        performance_matrix[x, y] = np.median(a) - np.median(b)
        significance_matrix[x, y] = significance_test(a, b, alpha=0.5, alternative='two-sided', use_continuity=True)
    tril_indices = np.tril_indices_from(significance_matrix, k=-1)
    significance_matrix[tril_indices] = fdr_corrected_pvals(significance_matrix[tril_indices])
    return performance_matrix, significance_matrix, condition_values

def significance_test(a, b, alpha=0.5, alternative='two-sided', use_continuity=True):
        if len(a) == len(b):
            return sign_test(a, b, alpha, alternative)
        else:
            return stats.mannwhitneyu(a, b, use_continuity=use_continuity, alternative=alternative)[1]
    # return sp.stats.wilcoxon(a, b, zero_method='zsplit', alternative='two-sided')
    # return sm.stats.descriptivestats.sign_test(a-b, mu0=0)


def max_of_each_entry(df, label_key, value_key):

    max_value = []
    for unique_label in np.unique(df[label_key]):
        max_value.append(np.max(df[df[label_key] == unique_label][value_key]))

    return max_value

def get_dataframe_values_matching_two_criteria_in_two_columns(dataframe, condition_col, condition_name,
                                                              metric_name_col, metric_name, metric_value_col):

    return dataframe[(dataframe[condition_col] == condition_name)
                     & (dataframe[metric_name_col] == metric_name)][metric_value_col].values


def dataframe_significance_test(dataframe, condition_col, condition_name_a, condition_name_b, metric_name_col,
                                metric_name, metric_value_col):

    a = get_dataframe_values_matching_two_criteria_in_two_columns(dataframe, condition_col, condition_name_a,
                                                                  metric_name_col, metric_name, metric_value_col)
    b = get_dataframe_values_matching_two_criteria_in_two_columns(dataframe, condition_col, condition_name_b,
                                                                  metric_name_col, metric_name, metric_value_col)

    return significance_test(a, b, alpha=0.5, alternative='two-sided', use_continuity=True)

def single_value_row_configurations(configurations):
    # Reformat configuration dataframe to have one row per metric value
    single_value_configurations = pd.DataFrame()
    for i_config in range(len(configurations)):
        for metric in ['mse', 'corr', 'corr_p', 'r^2']:
            for data_split in ['train', 'valid', 'test']:
                for i_subject in range(len(subject_values)):
                    single_value_configurations = single_value_configurations.append(
                        pd.DataFrame({'subject_train': subject_values[i_subject],
                                      'subject_apply': subject_values[i_subject],
                                      'data_split': data_split,
                                      'metric_value': configurations[metric + '_' + data_split][i_config][i_subject],
                                      'metric_name': metric,
                                      'data': configurations.loc[i_config, 'data'],
                                      'electrodes': configurations.loc[i_config, 'electrodes'],
                                      'band_pass': configurations.loc[i_config, 'band_pass'],
                                      'model_name': configurations.loc[i_config, 'model_name'],
                                      'unique_id': configurations.loc[i_config, 'unique_id'],
                                      'data_folder': configurations.loc[i_config, 'data_folder'],
                                      'batch_size': configurations.loc[i_config, 'batch_size'],
                                      'max_epochs': configurations.loc[i_config, 'max_epochs'],
                                      'cuda': configurations.loc[i_config, 'cuda'],
                                      'result_folder': configurations.loc[i_config, 'result_folder'],
                                      'init_lr': configurations.loc[i_config, 'init_lr'],
                                      'weight_decay': configurations.loc[i_config, 'weight_decay'],
                                      'sampling_rate': configurations.loc[i_config, 'sampling_rate'],
                                      'n_seconds_valid_set': configurations.loc[i_config, 'n_seconds_valid_set'],
                                      'n_seconds_test_set': configurations.loc[i_config, 'n_seconds_test_set'],
                                      'config_has_run': configurations.loc[i_config, 'config_has_run']}, index=[0]),
                        ignore_index=True)
    return single_value_configurations


#%% Load all existing single subject results
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
                df = pd.read_csv(subject)#, names=['epoch',
                                                 # 'train_loss','valid_loss','test_loss',
                                                 # 'train_corr','valid_corr','test_corr', 'runtime'])
                configurations = fill_results_in_configurations_at_index(configurations, i_config, df)
    else:
        configurations = prefill_columns_configuration_at_index(configurations, i_config, has_run=False)

# configurations.to_csv(configurations_file[:-4] + '_with_results.csv')
# Somehow converts my lists into strings!!!

#%% Load only missing results only instead of having to rerun the above every time
for i_config in configurations.index:
    if np.isnan(configurations.loc[i_config, 'mse_train']).any():
        print('Missing results in config {}'.format(i_config))
        result_folder = configurations.loc[i_config, 'result_folder']  # '/data/schirrmr/fiederer/nicebot/results_without_bp_bug'  #
        unique_id = configurations.loc[i_config, 'unique_id']
        matching_results = np.sort(glob.glob('/mnt/meta-cluster/' + result_folder + '/*' + unique_id + '*Exp.csv'))
        # matching_results = np.sort(glob.glob(result_folder + '/*' + unique_id[:7] + '.csv'))
        # Check that we get only one result per subject, but not sure what to do if not. Check that results are identical?

        configurations = initialize_empty_columns_at_index(configurations, i_config)

        if any(matching_results):
            configurations = prefill_columns_configuration_at_index(configurations, i_config, True)
            i_subject = -1
            for subject in matching_results:# Check if this subject was applied on himself.
                word_list = re.split('[_/.]', subject)  # Split path to .csv file into single words delimited by _ /
                # and .
                if word_list.count(word_list[-2]) == 2:  # If the second to
                # last word (either subject model was applied to or unique ID of the experiment) is present twice
                # then the model was trained on the same subject
                    i_subject += 1
                    df = pd.read_csv(subject)#, names=['epoch',
                                                     # 'train_loss','valid_loss','test_loss',
                                                     # 'train_corr','valid_corr','test_corr', 'runtime'])
                    configurations = fill_results_in_configurations_at_index(configurations, i_config, df)


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


#%% Overall best performing method
# data_split_values = ['train', 'test']
data_split_values = ['test']
model_values = configurations['model_name'].unique()
assert(all(model_values == ['eegnet', 'deep4', 'resnet', 'lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg']))


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
                                                       'metric_value': 1-np.abs(y),
                                                       'metric_name': '1-abs(corr)',
                                                       'data': 'All data'}))

f = plt.figure(figsize=[15, 10])
split_combi_plot(metric_df, x_col='model', y_col='metric_value', hue_col='metric_name', palette='colorblind')
add_significance_bars(metric_df, condition_col='model', metric_name_col='metric_name', metric_name='rmse',
                      metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
plt.show()

f = plt.figure(figsize=[15, 10])
split_combi_plot(metric_df, x_col='model', y_col='metric_value', hue_col='metric_name', palette='colorblind')
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

#%% Best performing method for each data type

data_values = configurations['data'].unique()
data_split_values = ['test']
model_values = configurations['model_name'].unique()
assert(all(model_values == ['eegnet', 'deep4', 'resnet', 'lin_reg', 'lin_svr', 'rbf_svr', 'rf_reg']))

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
    split_combi_plot(metric_df, x_col='model', y_col='metric_value', hue_col='metric_name', palette='colorblind')
    add_significance_bars(metric_df, condition_col='model', metric_name_col='metric_name', metric_name='rmse',
                          metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
    plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    f = plt.figure(figsize=[15, 10])
    split_combi_plot(metric_df, x_col='model', y_col='metric_value', hue_col='metric_name', palette='colorblind')
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

#%% Best electrode set EEG only
# data_split_values = ['train', 'test']
data_split_values = ['test']
electrode_sets = configurations['electrodes'].unique()
assert(all(electrode_sets == ["'*'", "'*z'", "'*C*'"]))

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
                                                       'metric_value': 1-np.abs(y),
                                                       'metric_name': '1-abs(corr)',
                                                       'data': 'onlyEEGData'}))

f = plt.figure(figsize=[15, 10])
split_combi_plot(metric_df, x_col='electrodes', y_col='metric_value', hue_col='metric_name', palette='colorblind')
add_significance_bars(metric_df, condition_col='electrodes', metric_name_col='metric_name', metric_name='rmse',
                      metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
plt.show()

f = plt.figure(figsize=[15, 10])
split_combi_plot(metric_df, x_col='electrodes', y_col='metric_value', hue_col='metric_name', palette='colorblind')
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



#%% Best data combination
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
split_combi_plot(metric_df, x_col='data', y_col='metric_value', hue_col='metric_name', palette='colorblind')
add_significance_bars(metric_df, condition_col='data', metric_name_col='metric_name', metric_name='rmse',
                      metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
plt.show()

f = plt.figure(figsize=[15, 10])
split_combi_plot(metric_df, x_col='data', y_col='metric_value', hue_col='metric_name', palette='colorblind')
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


#%% Best frequency band EEG only
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
                                                       'metric_value': 1-np.abs(y),
                                                       'metric_name': '1-abs(corr)',
                                                       'data': 'onlyEEGData'}))

f = plt.figure(figsize=[15, 10])
split_combi_plot(metric_df, x_col='band_pass', y_col='metric_value', hue_col='metric_name', palette='colorblind')
add_significance_bars(metric_df, condition_col='band_pass', metric_name_col='metric_name', metric_name='rmse',
                      metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
plt.show()

f = plt.figure(figsize=[15, 10])
split_combi_plot(metric_df, x_col='band_pass', y_col='metric_value', hue_col='metric_name', palette='colorblind')
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



#%% Overall best subject
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
                                                       'metric_value': 1-np.abs(y),
                                                       'metric_name': '1-abs(corr)',
                                                       'data': 'All data'}))

f = plt.figure(figsize=[15, 10])
split_combi_plot(metric_df, x_col='subject', y_col='metric_value', hue_col='metric_name', palette='colorblind')
add_significance_bars(metric_df, condition_col='subject', metric_name_col='metric_name', metric_name='rmse',
                      metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0])
plt.savefig('test.pdf', bbox_inches='tight', dpi=300)
plt.show()

f = plt.figure(figsize=[15, 10])
split_combi_plot(metric_df, x_col='subject', y_col='metric_value', hue_col='metric_name', palette='colorblind')
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
