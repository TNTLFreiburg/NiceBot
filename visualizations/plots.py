import itertools as it

import matplotlib as mpl
import matplotlib.transforms as transforms
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils.df_utils import dataframe_significance_test, compute_diff_matrix, compute_performance_matrix
from utils.statistics import fdr_corrected_pvals


def _remove_leading_zero(value, string):
    if 1 > value > -1:
        if string[0] == '0':
            string = string.replace('0', '', 1)
        elif (string[0] == '-') & (string[1] == '0'):
            string = string.replace('0', '', 1)
    return string


class MyFloat(float):
    def __str__(self):
        string = super().__str__()
        return _remove_leading_zero(self, string)

    def __format__(self, format_string):
        string = super().__format__(format_string)
        return _remove_leading_zero(self, string)


def devel_plot():
    tmp = sconfs[(sconfs['data_split'] == 'test') & ((sconfs['metric_name'] == 'mse') | (sconfs['metric_name'] ==
                                                                                         'corr'))]
    tmp['metric_value'] = np.sqrt(tmp['metric_value'])
    f = plt.figure(figsize=np.array([15, 10]) * 2)
    f, ax = plt.subplots()
    offset = lambda p: transforms.ScaledTranslation(p / 72., 0, plt.gcf().dpi_scale_trans)
    trans = ax.transData
    sns.violinplot(x='data', y='metric_value', hue='metric_name', data=tmp,
                   palette='colorblind',
                   color='1', split=False, inner='quartiles', scale='count', scale_hue=True, cut=0, bw=0.15)
    add_significance_bars(tmp, condition_col='data', metric_name_col='metric_name', metric_name='mse',
                          metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0], above=True)
    shift = [-10, 2.5, 15]
    for idx, subject in enumerate(tmp['subject_train'].unique()):
        tmp = sconfs[(sconfs['data_split'] == 'test') & ((sconfs['metric_name'] == 'mse') | (sconfs['metric_name'] ==
                                                                                             'corr'))]
        tmp['metric_value'] = np.sqrt(tmp['metric_value'])
        tmp = tmp[
            (tmp['model_name'] == 'eegnet') &
            (tmp['subject_train'] == subject) &
            (tmp['subject_apply'] == subject)]
        # STRIPPLOT GEHT NICHT WEIL DER OFFSET IN DIESELBE RINTUNG PRO SPLIT/DODGE GEHT... MUSS ALLES EINZELN GEPLOTTET
        # WERDEN!!

        sns.stripplot(x="data", y="metric_value", hue="metric_name", data=tmp, dodge=True,
                      jitter=0.05, alpha=.25,
                      zorder=1, palette='colorblind', color='1', edgecolor='black', linewidth=1,
                      transform=trans + offset(shift[idx]))

        # tmp = sconfs[(sconfs['data_split'] == 'test') & ((sconfs['metric_name'] == 'mse') | (sconfs['metric_name'] ==
        #                                                                                     'corr'))]
        # tmp = tmp[
        #     (tmp['model_name'] == 'eegnet') &
        #     (tmp['band_pass'] == "'[None, None]'") &
        #     (tmp['electrodes'] == "'*'") &
        #     (tmp['subject_train'] == 'moderate experience (S2)') &
        #     (tmp['subject_apply'] == 'moderate experience (S2)')]
        # sns.stripplot(x="data", y="metric_value", hue="metric_name", data=tmp, dodge=True,
        #               jitter=False, alpha=.25,
        #               zorder=1, palette='colorblind', color='1', edgecolor='black', linewidth=1, transform=trans+offset(5))
        #
        tmp = sconfs[(sconfs['data_split'] == 'test') & ((sconfs['metric_name'] == 'mse') | (sconfs['metric_name'] ==
                                                                                             'corr'))]
        tmp['metric_value'] = np.sqrt(tmp['metric_value'])
        tmp = tmp[
            (tmp['model_name'] == 'deep4') &
            (tmp['subject_train'] == subject) &
            (tmp['subject_apply'] == subject)]
        sns.stripplot(x="data", y="metric_value", hue="metric_name", data=tmp, dodge=True,
                      jitter=0.05, alpha=.25,
                      zorder=1, palette='colorblind', color='1', edgecolor='black', linewidth=1,
                      transform=trans + offset(shift[idx]), marker="D")
    # ax.set_ylim([-.5, 2])
    ax.legend_.remove()
    plt.show()


def split_combi_plot(f, dataframe, x_col='metric_value', y_col='model_name', hue_col='metric_name',
                     data_type_col='data', palette='colorblind'):
    sns.violinplot(x=x_col, y=y_col, hue=hue_col, data=dataframe, palette=palette,
                   color='1', split=True, inner='quartiles', scale='count', scale_hue=True, cut=2, bw=0.15)
    add_data_and_format_to_plot(f, dataframe, x_col, y_col, hue_col, data_type_col, palette)


def combi_plot(f, dataframe, x_col='metric_value', y_col='model_name', hue_col='metric_name', data_type_col='data',
               palette='colorblind'):
    sns.violinplot(x=x_col, y=y_col, hue=hue_col, data=dataframe, palette=palette,
                   color='1', split=False, inner='quartiles', scale='count', scale_hue=True, cut=2, bw=0.15)
    add_data_and_format_to_plot(f, dataframe, x_col, y_col, hue_col, data_type_col, palette)


def add_data_and_format_to_plot(f, dataframe, x_col='metric_value', y_col='model_name', hue_col='metric_name',
                                data_type_col='data', palette='colorblind'):
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
    f.axes[0].yaxis.grid(b=False)  # True, which='minor', linewidth=0.5)
    f.axes[0].yaxis.grid(b=True, which='major', linewidth=0.5)
    f.axes[0].set_axisbelow(True)
    f.axes[0].set(ylabel=' & '.join(np.unique(dataframe[hue_col])))
    # sns.despine(trim=True, left=True)
    # xlims = plt.xlim()
    # plt.xlim(0, xlims[1])
    data_types = np.unique(dataframe[data_type_col].values)
    data_types = ', '.join(data_types)
    plt.title(data_types)
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


def add_significance_bars_above(dataframe, condition_col, metric_name_col='metric_name', metric_name='rmse',
                                metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0]):
    add_significance_bars(dataframe, condition_col, metric_name_col=metric_name_col, metric_name=metric_name,
                          metric_value_col=metric_value_col, barc=barc, above=True)


def add_significance_bars_below(dataframe, condition_col, metric_name_col='metric_name', metric_name='corr',
                                metric_value_col='metric_value', barc=sns.color_palette('colorblind')[1]):
    add_significance_bars(dataframe, condition_col, metric_name_col=metric_name_col, metric_name=metric_name,
                          metric_value_col=metric_value_col, barc=barc, above=False)


def add_significance_bars(dataframe, condition_col, metric_name_col='metric_name', metric_name='rmse',
                          metric_value_col='metric_value', barc=sns.color_palette('colorblind')[0], above=True):
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
                                  height=np.repeat(np.max(dataframe[metric_value_col]), len(condition_indices)) if
                                  above else np.repeat(np.min(dataframe[metric_value_col]), len(condition_indices)),
                                  # max_of_each_entry(metric_df, 'model', 'metric_value'),
                                  yerr=None,
                                  dh=.05 if above else -.05,
                                  barh=barh,
                                  barc=barc,
                                  fs=6,
                                  fw=q2fw(pvals_corrected[pair_index]),
                                  maxasterix=None)
        barh = barh + .01 if above else barh - 0.01
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


def plot_performance_matrix(dataframe,
                            x_col='Data',
                            y_col='Subject',
                            metric_name_col='Metrics',
                            metric_name='Root mean square error',
                            metric_value_col='metric_value',
                            title='',
                            cmap=LinearSegmentedColormap.from_list('white2colorblind', [[1, 1, 1], sns.color_palette(
                                'colorblind')[0]], N=256),
                            averaging_func=np.median,
                            transformation_func=lambda x: x,
                            vmin=0,
                            vmax=1):
    # Compute diff matrix
    if type(metric_name) != str:
        print('Unsupported!')
        return
    else:
        performance_matrix, x_values, y_values = compute_performance_matrix(dataframe,
                                                                            x_col,
                                                                            y_col,
                                                                            metric_name_col,
                                                                            metric_name,
                                                                            metric_value_col,
                                                                            averaging_func,
                                                                            transformation_func)
    tick_step = (vmax - vmin) / 5
    ticks = np.arange(vmin, vmax + tick_step, tick_step)
    fig, ax = plt.subplots()
    im = ax.imshow(performance_matrix, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = ax.figure.colorbar(im, ax=ax, label=metric_name, ticks=ticks)
    ax.set(xticks=np.arange(performance_matrix.shape[1]),
           yticks=np.arange(performance_matrix.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=y_values, yticklabels=x_values,
           title=title,
           ylabel='',
           xlabel='')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=90, ha="center",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f'
    thresh = (performance_matrix.max() + performance_matrix.min()) / 2. / vmax
    cmap = im.get_cmap()
    for i in range(performance_matrix.shape[0]):
        for j in range(performance_matrix.shape[1]):
            ax.text(j, i, format(performance_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="black" if np.sum(cmap(performance_matrix[i, j] / vmax)) >= np.sum(cmap(
                        thresh)) else "white")

    # Center ticks
    xtick_positions = ax.get_xticks()
    ytick_positions = ax.get_yticks()
    ax.set_xticks(xtick_positions + 0.5, minor=True)  # xaxis.set_minor_locator(xtick_positions)
    ax.set_yticks(ytick_positions + 0.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', length=0)
    fig.tight_layout()
    return ax


def plot_diff_matrix(dataframe,
                     condition_col,
                     metric_name_col='metric_name',
                     metric_name=['rmse', '1-abs(corr)'],
                     metric_value_col='metric_value',
                     title=None,
                     cmap=reverse_colormap(
                         LinearSegmentedColormap.from_list('white2colorblind', [[1, 1, 1], sns.color_palette(
                             'colorblind')[2]], N=256), ),
                     averaging_func=np.median):
    # Compute diff matrix
    if type(metric_name) != str:
        diff_matrix_a, significance_matrix_a, condition_values_a = compute_diff_matrix(dataframe,
                                                                                       condition_col,
                                                                                       metric_name_col,
                                                                                       metric_name[0],
                                                                                       metric_value_col,
                                                                                       averaging_func)
        diff_matrix_b, significance_matrix_b, condition_values_b = compute_diff_matrix(dataframe,
                                                                                       condition_col,
                                                                                       metric_name_col,
                                                                                       metric_name[1],
                                                                                       metric_value_col,
                                                                                       averaging_func)
        assert all(condition_values_a == condition_values_b), 'Extracted conditions do not match!'
        condition_values = condition_values_a
        diff_matrix = np.tril(diff_matrix_a)
        diff_matrix[np.triu_indices_from(diff_matrix)] = diff_matrix_b.T[np.triu_indices_from(
            diff_matrix)]
        significance_matrix = np.tril(significance_matrix_a)
        significance_matrix[np.triu_indices_from(significance_matrix)] = significance_matrix_b.T[np.triu_indices_from(
            significance_matrix)]
    else:
        diff_matrix, significance_matrix, condition_values = compute_diff_matrix(dataframe,
                                                                                 condition_col,
                                                                                 metric_name_col,
                                                                                 metric_name,
                                                                                 metric_value_col,
                                                                                 averaging_func)
    vmin = 0
    vmax = 0.05
    tick_step = (vmax - vmin) / 5
    ticks = np.arange(vmin, vmax + tick_step, tick_step)
    fig, ax = plt.subplots()
    im = ax.imshow(significance_matrix, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    # im = ax.pcolormesh(significance_matrix, edgecolors='w', cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = ax.figure.colorbar(im, ax=ax, label='q-value', ticks=ticks)
    q2fw = lambda x: 1000 if x < vmax else 0  # 1000*np.exp(-25*x)
    # for cbar_label_idx, cbar_label in enumerate(cbar.ax.get_yticklabels()):
    #     cbar_label.set_fontweight(q2fw(ticks[cbar_label_idx]))
    # We want to show all ticks...
    ax.set(xticks=np.arange(diff_matrix.shape[1]),
           yticks=np.arange(diff_matrix.shape[0]),
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
    fmt = '.1g'
    # thresh = vmax / 2  # significance_matrix.max() / 2.
    tril_indices = np.column_stack(np.tril_indices_from(diff_matrix))
    triu_indices = np.column_stack(np.triu_indices_from(diff_matrix))
    for i in range(diff_matrix.shape[0]):
        for j in range(diff_matrix.shape[1]):
            if type(metric_name) == str:
                color = 'black'
            elif any(np.sum((i, j) == tril_indices, axis=1) == 2):
                color = sns.color_palette('colorblind')[0]
            elif any(np.sum((i, j) == triu_indices, axis=1) == 2):
                color = sns.color_palette('colorblind')[1]
            ax.text(j, i + .25, format(MyFloat(diff_matrix[i, j]), fmt),
                    ha="center", va="center",
                    color=color,  # "black" if significance_matrix[i, j] > thresh else "white",
                    weight=q2fw(significance_matrix[i, j]) if significance_matrix[i, j] < vmax else 0)  #
            # Font weight seems to behave non-linearly, at least to my perception
            if i != j:
                if diff_matrix[i, j] > 0:
                    ax.arrow(j, i - .25, 0, +.25, color=color, head_width=.125, length_includes_head=True)
                else:
                    ax.arrow(j + .125, i - .25, -.25, 0, color=color, head_width=.125, length_includes_head=True)

    # Center ticks
    xtick_positions = ax.get_xticks()
    ytick_positions = ax.get_yticks()
    ax.set_xticks(xtick_positions + 0.5, minor=True)  # xaxis.set_minor_locator(xtick_positions)
    ax.set_yticks(ytick_positions + 0.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', length=0)

    # Add transparent triangles to further separate lower and upper triangles
    if type(metric_name) != str:
        tril_x = np.array([0., 0., diff_matrix.shape[0]]) - 0.5
        tril_y = np.array([0., diff_matrix.shape[1], diff_matrix.shape[1]]) - 0.5
        ax.fill(tril_x, tril_y, fill=True, alpha=0.1, hatch=None, edgecolor=sns.color_palette('colorblind')[0])
        triu_x = np.array([0, diff_matrix.shape[0], diff_matrix.shape[0]]) - 0.5
        triu_y = np.array([0, diff_matrix.shape[1], 0]) - 0.5
        ax.fill(triu_x, triu_y, fill=True, alpha=0.1, hatch=None, edgecolor=sns.color_palette('colorblind')[1])
    fig.tight_layout()
    return ax


def plot_transfer_matrix(transfer_matrix, condition_values=['S1 (no exp.)', 'S2 (mod. exp.)', 'S3 (subst. exp.)'],
                         xlabel='subject tested on', ylabel='subject trained on', clabel='Root mean square error',
                         title='',
                         cmap=LinearSegmentedColormap.from_list('white2colorblind', [[1, 1, 1], sns.color_palette(
                             'colorblind')[0]], N=256)):
    fig, ax = plt.subplots()
    # vmax = transfer_matrix.max() if transfer_matrix.max() < 1 else 1
    # im = ax.imshow(transfer_matrix, interpolation='nearest', cmap=cmap, vmin=transfer_matrix.min(),
    #                vmax=vmax)
    vmin = 0
    vmax = 1
    im = ax.imshow(transfer_matrix, interpolation='nearest', cmap=cmap, vmin=vmin,
                   vmax=vmax)
    cbar = ax.figure.colorbar(im, ax=ax, label=clabel)
    ax.set(xticks=np.arange(transfer_matrix.shape[1]),
           yticks=np.arange(transfer_matrix.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=condition_values, yticklabels=condition_values,
           title=title,
           ylabel=ylabel,
           xlabel=xlabel)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=90, ha="center",
             rotation_mode="anchor")
    fmt = '.3f'
    # thresh = (transfer_matrix.max() + transfer_matrix.min()) / 2.
    thresh = (vmax - vmin) / 2.
    cmap = im.get_cmap()
    for i in range(transfer_matrix.shape[0]):
        for j in range(transfer_matrix.shape[1]):
            ax.text(j, i, format(transfer_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="black" if np.sum(cmap(transfer_matrix[i, j] / vmax)) >= np.sum(cmap(
                        thresh / vmax)) else "white")
