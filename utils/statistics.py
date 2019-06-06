import numpy as np
import scipy as sp
import statsmodels.stats
from scipy import stats


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
    return statsmodels.stats.multitest.multipletests(pvals, alpha=0.05, method='fdr_by',
                                                     is_sorted=False,
                                                     returnsorted=False)[1]


def significance_test(a, b, alpha=0.5, alternative='two-sided', use_continuity=True):
    if len(a) == len(b):
        return sign_test(a, b, alpha, alternative)
        # return sp.stats.wilcoxon(a, b, zero_method='zsplit', alternative='two-sided')
        # return sm.stats.descriptivestats.sign_test(a-b, mu0=0)
    else:
        return stats.mannwhitneyu(a, b, use_continuity=use_continuity, alternative=alternative)[1]
