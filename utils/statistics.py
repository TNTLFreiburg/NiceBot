import numpy as np
import scipy as sp
import statsmodels.stats.multitest
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score


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


def fdr_corrected_pvals(pvals):
    pvals_shape = np.shape(pvals)
    qvals = statsmodels.stats.multitest.multipletests(np.ravel(pvals), alpha=0.05, method='fdr_by',
                                                      is_sorted=False,
                                                      returnsorted=False)[1]
    return qvals.reshape(pvals_shape)


def significance_test(a, b, alpha=0.5, alternative='two-sided', use_continuity=True):
    if len(a) == len(b):
        return sign_test(a, b, alpha, alternative)
        # return sp.stats.wilcoxon(a, b, zero_method='zsplit', alternative='two-sided')
        # return sm.stats.descriptivestats.sign_test(a-b, mu0=0)
    else:
        return stats.mannwhitneyu(a, b, use_continuity=use_continuity, alternative=alternative)[1]


def random_permutation(to_permute, to_test, n_permutes=int(10e6), metric_functions=[stats.pearsonr, mean_squared_error,
                                                                                    r2_score], seed=0):
    np.random.seed(seed=seed)
    permutation_metrics = np.nan * np.zeros((n_permutes, len(metric_functions)))
    to_test_len = len(to_test)
    for i_permute in range(n_permutes):
        permuted = np.random.permutation(to_permute)
        for i_metric in range(len(metric_functions)):
            permutation_metrics[i_permute, i_metric] = np.atleast_1d(metric_functions[i_metric](to_test,
                                                                                                permuted[
                                                                                                :to_test_len]))[0]

    return permutation_metrics
