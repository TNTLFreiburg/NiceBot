import itertools as it

import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from utils.statistics import fdr_corrected_pvals, significance_test
from utils.df_utils import get_dataframe_values_matching_two_criteria_in_two_columns, dataframe_significance_test


def split_combi_plot(f, dataframe, x_col='metric_value', y_col='model_name', hue_col='metric_name',
                     palette='colorblind'):
    sns.violinplot(x=x_col, y=y_col, hue=hue_col, data=dataframe, palette=palette,
                   color='1', split=True, inner='quartiles', scale='count', scale_hue=True, cut=2, bw=0.15)
    add_data_and_format_to_plot(f, dataframe, x_col, y_col, hue_col, palette)


def combi_plot(f, dataframe, x_col='metric_value', y_col='model_name', hue_col='metric_name', palette='colorblind'):
    sns.violinplot(x=x_col, y=y_col, hue=hue_col, data=dataframe, palette=palette,
                   color='1', split=False, inner='quartiles', scale='count', scale_hue=True, cut=2, bw=0.15)
    add_data_and_format_to_plot(f, dataframe, x_col, y_col, hue_col, palette)


def add_data_and_format_to_plot(f, dataframe, x_col='metric_value', y_col='model_name', hue_col='metric_name',
                                palette='colorblind'):
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
    bary = [y + barh, y + barh]  # [y, y+barh, y+barh, y]
    mid = ((lx + rx) / 2, y + barh)

    plt.plot(barx, bary, c=barc)

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs
    if fw is not None:
        kwargs['fontweight'] = fw

    plt.text(*mid, text, **kwargs)


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
            data.append((1 - t[0], t[2], t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k, reverse))
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
    tick_step = (vmax - vmin) / 5
    ticks = np.arange(vmin, vmax + tick_step, tick_step)
    fig, ax = plt.subplots()
    im = ax.imshow(significance_matrix, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    # im = ax.pcolormesh(significance_matrix, edgecolors='w', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = ax.figure.colorbar(im, ax=ax, label='q-values', ticks=ticks)
    q2fw = lambda x: 1000 if x < vmax else 0  # 1000*np.exp(-25*x)
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
    thresh = vmax / 2  # significance_matrix.max() / 2.
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
        tril_x = np.array([0., 0., performance_matrix.shape[0]]) - 0.5
        tril_y = np.array([0., performance_matrix.shape[1], performance_matrix.shape[1]]) - 0.5
        ax.fill(tril_x, tril_y, fill=False, hatch='/', edgecolor=sns.color_palette('colorblind')[0])
        triu_x = np.array([0, performance_matrix.shape[0], performance_matrix.shape[0]]) - 0.5
        triu_y = np.array([0, performance_matrix.shape[1], 0]) - 0.5
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
