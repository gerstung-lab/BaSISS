import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import cv2 as cv
from scipy import stats

cmaps_global = {'grey':"Greys", 'green':"Greens", 'purple':"Purples",'magenta':"RdPu",'blue':"Blues",'red':"Reds",'orange':"YlOrBr",'wt':"Greys", 'residuals':"Greys"}

def display_raw_fields(params_samples, field_names, sample_names=None, figsize=None):
    fields_loc = [k.startswith('F') for k in params_samples.keys()]
    n_s = np.sum(fields_loc)
    n_f = params_samples[np.array(list(params_samples.keys()))[np.where(fields_loc)[0][0]]].shape[-1]
    
    c = 0
    if figsize is not None:
        plt.figure(figsize=figsize)
    for i, ide in enumerate(np.arange(n_f)):
        for s in range(n_s):
            dims = np.array(params_samples['lm_n_{}'.format(s)].shape)[[1,2]]
            plt.subplot(n_f,n_s,c+1)
            #plt.imshow(resized_img_list[s])
            plt.imshow(cv.resize((np.percentile(params_samples['F_{}'.format(s)][:,:,:,ide] * params_samples['lm_n_{}'.format(s)], 50, axis=0) ).T[::-1, :], tuple(np.array(dims*4))),
                       cmap=plt.get_cmap(cmaps_global[field_names[ide]]), vmin=0, vmax=50)
            
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().axes.get_xaxis().set_visible(False)

            plt.gca().spines['left'].set_visible(False)
            plt.gca().set_yticklabels([])
            plt.gca().set_yticks([])

            
            if s == 0:
                plt.ylabel(field_names[ide])
            if sample_names is not None:
                if i == 0:
                    plt.title(sample_names[i])
            c += 1
        plt.colorbar()
        
def diagnostic_residual(extended_model_params_samples, iss_data, mask, tree, sample_dims, samples=[0], figsize=(9,12), subplots=(9,6), plot_type='hist'):
    assert plot_type in ['hist', 'map'], 'plot_type should be either "hist" or "map"'
    
    n_genes = tree.shape[1]
    for s in samples:
        mu = np.array(extended_model_params_samples.posterior[f'lm_er_{s}'])[0].mean(axis=0)
        k = iss_data[s]
        u = stats.uniform.rvs(size=mu.shape[0]*mu.shape[1]).reshape(mu.shape)
        ppois = u * stats.poisson.cdf(k, mu) + (1-u) * stats.poisson.cdf(k-1, mu)
        qnorm = stats.norm.ppf(ppois) * mask[s].reshape(-1,1)
        plt.figure(figsize=figsize)
        for g in range(n_genes):
            plt.subplot(subplots[0],subplots[1],g+1)
            if plot_type == 'hist':
                q = qnorm[:,g][mask[s]]
                q = q[~np.isinf(q)]
                plt.hist(q, bins=100)
                plt.axvline(-1, color='red')
                plt.axvline(1, color='red')
                plt.title(tree.columns[g])
            elif plot_type == 'map':
                plt.subplot(subplots[0],subplots[1],g+1)
                plt.imshow(qnorm[:,g].reshape(*sample_dims[s]).T[::-1, :], vmin=-3, vmax=3)
                plt.title(tree.columns[g])

        plt.subplot(subplots[0],subplots[1],n_genes+1)
        plt.imshow(qnorm.mean(axis=1).reshape(*sample_dims[s]).T[::-1, :], vmin=-3, vmax=3)
        plt.title('mean logp')
        plt.colorbar()
        plt.rcParams['figure.facecolor'] = 'w'
        plt.tight_layout()
        plt.show()
        
def gaussian_prior_check(n_factors, n_aug, t, ax=None):
    fs = np.random.randn(1000,n_factors)
    fs = np.concatenate([fs, np.ones((1000,1))*(-1.7)], axis=1)
    fs = (np.exp(fs * t) / np.exp(fs * t).sum(1)[:,None])
    fractions = fs[:,:n_factors] / fs[:,:n_factors].sum(1)[:,None]
    if ax is None:
        for i in range(n_factors):
            plt.hist(fractions[:,i], range=(0,1), bins=100, alpha=0.5, density=True)
        plt.show()
    else:
        for i in range(n_factors):
            ax.hist(fractions[:,i], range=(0,1), bins=100, alpha=0.5, density=True)