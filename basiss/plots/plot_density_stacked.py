import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from scipy.interpolate import make_interp_spline

from basiss.pile_of_code import mut_sample_list, sample_dims, samples_hierarchical_errosion, cells_counts, cmaps
from basiss.utits.sample import Sample


def plot_density_stacked(sampleID, site, save=False, ax=None, flipped=False, rescale_y=1):
    names = ['grey', 'green', 'purple', 'blue', 'red', 'orange', 'wt']

    SampleID = sampleID
    site = site
    if flipped:
        site = -site
    grid_mm2 = (Sample.get_img_size(mut_sample_list[SampleID].image)[0] / sample_dims[SampleID][0]) ** 2 / 1e6
    data = samples_hierarchical_errosion['F_{}'.format(sampleID)][:, :, :].reshape(300, *sample_dims[sampleID], 9)[:,
           ::, -site, :] * \
           cells_counts[sampleID].reshape(sample_dims[sampleID])[None, ::, -site, None] / grid_mm2
    data = np.concatenate([data[:, :, [0, 1, 2, 3, 4, 5]], data[:, :, [6, 7]].sum(axis=2)[:, :, None]], axis=2)

    CI = (2.5, 97.5)
    color_list = [get_cmap(cmaps[name])(150) for name in names]
    line_id_list = [0, 1, 2, 3, 4, 5, 6]
    # 1e3 / 0.325 / 15
    xold = np.linspace(0, sample_dims[SampleID][0], sample_dims[SampleID][0])
    xnew = np.linspace(0, (sample_dims[SampleID][0]), 5000)

    lines_smooth = []
    for i, idx in enumerate(line_id_list):
        line = data.mean(axis=0)[:, idx].T
        line_low = np.percentile(data, CI[0], axis=0)[:, idx].T
        line_up = np.percentile(data, CI[1], axis=0)[:, idx].T

        spl_line = make_interp_spline(xold, line, k=2)
        line_smooth = spl_line(xnew)
        line_smooth[line_smooth < 0] = 0
        lines_smooth.append(line_smooth)

    cum = np.zeros(lines_smooth[0].shape)
    if ax is not None:
        for i in range(len(line_id_list) - 1):
            ax.fill_between(xnew * rescale_y, cum, cum + lines_smooth[i], color=color_list[i], alpha=1)
            cum += lines_smooth[i]
        ax.plot(xnew * rescale_y, cum + lines_smooth[i + 1], color='black', alpha=1)

        ax.set_xlim(-1)
        ax.set_ylim(0, 2000)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_visible(False)
    else:
        # plt.figure(figsize=(int(Sample.get_img_size(mut_sample_list[SampleID].image)[0]/2000),2))
        for i in range(len(line_id_list) - 1):
            plt.fill_between(xnew * rescale_y, cum, cum + lines_smooth[i], color=color_list[i], alpha=1)
            cum += lines_smooth[i]
        plt.plot(xnew * rescale_y, cum + lines_smooth[i + 1], color='black', alpha=1)
        # cum += lines_smooth[i]

        plt.xlim(-1)
        plt.ylim(0, 2000)

        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().get_xaxis().set_visible(False)
