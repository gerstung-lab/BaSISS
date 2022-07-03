import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import cv2 as cv
from scipy import stats
from matplotlib.cm import get_cmap
from skimage import exposure

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
                    plt.title(sample_names[s])
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
            
            
def plot_field_comparison(mut_sample_list1, mut_sample_list2,
                    model_params_samples1, model_params_samples2,
                    mask1, mask2, sample_names,
                    n_samples=3,
                    names = ['grey','green', 'purple', 'blue', 'red', 'orange', 'wt'],
                    set_name1 = 'R0', set_name2 = 'R1'):


    c = [get_cmap(cmaps_global[n])(150) for n in names]

    FR0 = [model_params_samples1[f'F_{i}'] for i in range(n_samples)]
    FR1 = [model_params_samples2[f'F_{i}'] for i in range(n_samples)]
    for i in range(3):
        FR0[i][:,:,:,-3] += FR0[i][:,:,:,-2]
        FR1[i][:,:,:,-3] += FR1[i][:,:,:,-2]
    plt.figure(figsize=(1.5*6, 1.5*8))

    n_factors = FR0[0].shape[-1] - 2
    counter = 0
    for i in range(n_samples):
        plt.subplot(n_factors+1,n_samples*2,counter+1)
        plt.imshow(mask1[i].reshape(FR0[i].shape[1:3]).T[::-1], cmap='Greys')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().yaxis.set_ticks([])
        plt.gca().yaxis.set_ticklabels([])
        plt.gca().get_xaxis().set_visible(False)
        plt.title(f'{set_name1} {sample_names[i]}')
        if i == 0:
            plt.ylabel(f'valid tiles')
        counter +=1
        plt.subplot(n_factors+1,n_samples*2,counter+1)
        plt.imshow(mask2[i].reshape(FR1[i].shape[1:3]).T[::-1], cmap='Greys')
        plt.title(f'{set_name2} {sample_names[i]}')
        plt.gca().axis('off')
        counter +=1

    for factor in range(n_factors):
        for i in range(n_samples):
            imgR0 = mut_sample_list1[i]._scaffold_image
            imgR1 = mut_sample_list2[i]._scaffold_image #tifffile.imread(mut_sample_list[i].image)
            sR0 = imgR0.shape
            sR0 = tuple([int(x) for x in list(sR0)[::-1]])
            sR1 = imgR1.shape
            sR1 = tuple([int(x) for x in list(sR1)[::-1]])

            p35, p90 = np.percentile(imgR0, (35, 90))
            processed_img = exposure.rescale_intensity(imgR0, in_range=(p35, p90))
            b0 = cv.resize(processed_img, sR0)[::-1,:]/255.

            p35, p90 = np.percentile(imgR1, (35, 90))
            processed_img = exposure.rescale_intensity(imgR1, in_range=(p35, p90))
            b1 = cv.resize(processed_img, sR1)[::-1,:]/255.

            for percentile in [50]:
                if factor == n_factors-1:
                    vmax = 0.7
                else:
                    vmax = 0.20
                b0 = np.maximum(np.minimum(b0,1),0)
                Fc = np.minimum(np.percentile(FR0[i][:,:,:,factor], percentile, axis=0), vmax) / vmax
                Fc = (Fc.T[::-1,:][:,:,None] * (1-np.array(c[factor])))[:,:,:3]
                plt.subplot(n_factors+1,n_samples*2,counter+1)
                plt.imshow(1-cv.resize(Fc, sR0) * b0.reshape(*b0.shape,1))
                if i == 0:
                    plt.gca().spines['right'].set_visible(False)
                    plt.gca().spines['top'].set_visible(False)
                    plt.gca().spines['bottom'].set_visible(False)
                    plt.gca().spines['left'].set_visible(False)
                    plt.gca().yaxis.set_ticks([])
                    plt.gca().yaxis.set_ticklabels([])
                    plt.gca().get_xaxis().set_visible(False)
                    plt.ylabel(names[factor])
                else:
                    plt.gca().axis('off')

                if factor == 6:
                    plt.plot([b0.shape[1]*0.1, b0.shape[1]*0.1 + 2.5e3 / 0.325 / 15 ], [b0.shape[0] * 1.05, b0.shape[0] * 1.05], color='black', lw=3)
                counter += 1


                plt.subplot(n_factors+1,n_samples*2,counter+1)
                b1 = np.maximum(np.minimum(b1,1),0)
                Fc = np.minimum(np.percentile(FR1[i][:,:,:,factor], percentile, axis=0), vmax) / vmax
                Fc = (Fc.T[::-1,:][:,:,None] * (1-np.array(c[factor])))[:,:,:3]
                plt.imshow(1-cv.resize(Fc, sR1) * b1.reshape(*b1.shape,1))
                if factor == 6:
                    plt.plot([b0.shape[1]*0.1, b0.shape[1]*0.1 + 2.5e3 / 0.325 / 15 ], [b0.shape[0] * 1.05, b0.shape[0] * 1.05], color='black', lw=3)
                plt.gca().axis('off')
                counter += 1