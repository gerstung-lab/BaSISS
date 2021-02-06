import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FixedLocator, FuncFormatter
from scipy import stats

from basiss.utits.pval_bayes import pval_bayes


def squeezed_barplot(data_dict, sample_names, E_pannel_from, title=None):
    dat = [data_dict[k][:, :, 0] / data_dict[k][:, :, 1] for k in sample_names]
    dat = [np.log10(dat[i]) for i, k in enumerate(sample_names)]
    p_vals = [pval_bayes(data_dict[k][:, :, 0], data_dict[k][:, :, 1], 1.5) for k in sample_names]
    gene_filtered = np.where(np.logical_or(p_vals[0] < 2, p_vals[1] < 2))[0]
    dat = [dat[i][:, gene_filtered] for i, k in enumerate(sample_names)]

    ###### Locators for Y-axis
    # set tickmarks at multiples of 1.

    percentiles_dat = [[np.percentile(dat[s], pct, axis=0) for pct in [2.5, 50, 97.5]] for s in range(2)]

    majorLocator = MultipleLocator(1.)
    # create custom minor ticklabels at logarithmic positions
    ra = np.array(
        [[n + (1. - np.log10(i))] for n in range(-3, 3) for i in [2, 3, 4, 5, 6, 7, 8, 9][::-1]]).flatten() * -1.
    minorLocator = FixedLocator(ra)
    ###### Formatter for Y-axis (chose any of the following two)
    # show labels as powers of 10 (looks ugly)
    # majorFormatter= FuncFormatter(lambda x,p: "{:.1e}".format(10**x) )
    # or using MathText (looks nice, but not conform to the rest of the layout)
    majorFormatter = FuncFormatter(lambda x, p: r"$10^{" + "{x:d}".format(x=int(x)) + r"}$")
    subset = np.where(E_pannel_from[gene_filtered] == ['imm', 'exp'][pannel_id])[0]
    sub_percentiles_dat = [[percentiles_dat[s][i][subset] for i in range(3)] for s in range(2)]

    log_sum_pval = []
    for i in range(len(p_vals[0][gene_filtered][subset])):
        if np.sign(sub_percentiles_dat[0][1][i]) == np.sign(sub_percentiles_dat[1][1][i]):
            val = -2 * (np.log(p_vals[0][gene_filtered][subset][i]) + np.log(p_vals[1][gene_filtered][subset][i]))
        else:
            p1 = p_vals[0][gene_filtered][subset][i]
            p2 = p_vals[1][gene_filtered][subset][i]

            p1, p2 = np.minimum(p1, p2), np.maximum(p1, p2)
            val = -2 * (np.log(p1) + np.log(1))
        log_sum_pval.append(val)

    # p_vals_chi2 = np.minimum(p_vals[0][gene_filtered][subset], p_vals[1][gene_filtered][subset])
    p_vals_chi2 = np.round(stats.chi2.sf(log_sum_pval, 4), 5)

    ranks = -np.round(1 / p_vals_chi2 * np.sign(sub_percentiles_dat[0][1] + sub_percentiles_dat[1][1]), 2)
    order = []
    for r in np.unique(ranks):
        ranked_order = np.where(ranks == r)[0]
        magnitude_order = list(ranked_order[np.argsort(
            (sub_percentiles_dat[0][1] + sub_percentiles_dat[1][1])[np.where(ranks == r)[0]])[::-1]])
        order += magnitude_order

    boundaries = [0] + list(np.cumsum([len(np.where(ranks == r)[0]) for r in np.unique(ranks)]))
    boundaries_pval = np.array([0] + list([p_vals_chi2[np.where(ranks == r)[0]][0] for r in np.unique(ranks)]))

    p_val_th = 0.05

    gene_names = E_names[gene_filtered][subset][order]
    if pannel_id == 1:
        color = ['darkgrey', 'lightgrey']
    else:
        color = ['darkgrey', 'lightgrey']
    bpv = np.array(boundaries)[np.where(np.diff(boundaries_pval < p_val_th))[0]]
    xposdiff = np.ones(len(order))
    squeezed = 0.2
    xposdiff[bpv[0] + 1:bpv[1]] = squeezed
    xpos = np.cumsum(xposdiff) * 2
    widths = xposdiff.copy()
    widths[bpv[0]] = squeezed
    for s in range(2):
        for i in range(3):
            sub_percentiles_dat[s][i] = sub_percentiles_dat[s][i][order]

    plt.figure(figsize=(xpos[-1] * 0.3, 3))
    plt.bar(xpos - 0.4 * widths, sub_percentiles_dat[0][1],
            yerr=[np.abs(sub_percentiles_dat[0][0] - sub_percentiles_dat[0][1]),
                  np.abs(sub_percentiles_dat[0][2] - sub_percentiles_dat[0][1])], width=0.7 * widths,
            color=color[0], error_kw={'linewidth': widths * 1, 'alpha': 0.8})
    plt.bar(xpos + 0.4 * widths, sub_percentiles_dat[1][1],
            yerr=[np.abs(sub_percentiles_dat[1][0] - sub_percentiles_dat[1][1]),
                  np.abs(sub_percentiles_dat[1][2] - sub_percentiles_dat[1][1])], width=0.7 * widths,
            color=color[1], error_kw={'linewidth': widths * 1, 'alpha': 0.8})

    plt.gca().set_xticks(xpos)
    gene_names[widths == squeezed] = ''
    plt.gca().set_xticklabels(gene_names, rotation=90)
    plt.gca().set_title(title)
    plt.xlim(xpos[0] - 2, xpos[-1] + 2)
    plt.ylim(-1.3, 1.3)
    plt.gca().yaxis.set_major_locator(majorLocator)
    plt.gca().yaxis.set_minor_locator(minorLocator)
    plt.gca().yaxis.set_major_formatter(majorFormatter)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.show()
