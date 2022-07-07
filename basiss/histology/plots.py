from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib import colors
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from basiss.utils import categ_df2color


def subcl_comp2color(mat, colornames = np.array(['grey','green', 'mediumorchid', 'dodgerblue', 'orangered','darkorange']), colorscale=1):
    ns, nr = mat.shape
    rgba = np.array([[colors.to_rgba(c)] * nr  for c in colornames])
    rgba[:,:,-1] = np.minimum(mat*colorscale, 1)
    return rgba


def plot_by_epith_extra(hga_selected, sbcl_names, epith_names, hist_columns=None, title=None, plot_labels=False, reorder=None, colorscale=1, th=1):


    if sbcl_names == ['blue', 'green', 'orange', 'purple']: 
        colornames = ['dodgerblue', 'green', 'darkorange', 'purple']
    elif sbcl_names ==  ['grey', 'green', 'purple', 'blue', 'red', 'orange']: 
        colornames = ['grey','green', 'mediumorchid', 'dodgerblue', 'orangered','darkorange']

    
    if reorder is None:
        reorder = np.arange(len(sbcl_names))
    
    sbcl_names = np.array(sbcl_names)[reorder]
    
    # Genomics
    dat_matrix = []
    labels = []
    n_type = []
    
    hist_dat = []
    
    if hist_columns is not None:
        if hga_selected.histology_df[hist_columns[0]].dtype.name == 'category':
            flag_categorical = True

            cmap = mpl.cm.get_cmap('YlOrBr')
            cat_colours = np.array([np.array(cmap(0))[:3],
                                     np.array(cmap(0.33))[:3],
                                     np.array(cmap(0.66))[:3],
                                     np.array(cmap(1.0))[:3]]) * 255
        else:
            flag_categorical = False

    for i in range(len(epith_names)):
        hga_selected.subset_ids(hist_condition = np.isin(hga_selected.histology_df['Epithelial cells'], epith_names[i]), comp_condition = hga_selected.exist_comp_condition())
        hga_selected.sort_ids(th=th)
        proportions = hga_selected.comp_matrix.T / hga_selected.comp_matrix.T.sum(axis=0)[None,:]
        dat_matrix.append(proportions)
        n_type.append(len(hga_selected))
        labels = labels + hga_selected.common_ids
        
        if hist_columns is not None:
            if flag_categorical:
                hist_dat.append(hga_selected.hist_matrix.loc[:, hist_columns].apply(lambda x: x.cat.codes).values.T)
            else:
                hist_dat.append(hga_selected.hist_matrix[hist_columns].values.T)
            
    #plotting
    
    plt.rcParams['figure.facecolor'] = 'w'
    
    if hist_columns is None:
        fig, axs = plt.subplots(2,2, sharex = True, figsize=((len(labels) + 5)/1.5/6*1.5,5/2*1.5),
                                gridspec_kw={'wspace':0.1,
                                             'hspace': 0.6,
                                             'width_ratios': [len(labels),5],
                                             'height_ratios': [len(sbcl_names),1]}, dpi=100)
    else:
        fig, axs = plt.subplots(3,2, sharex = True, figsize=((len(labels) + 5)/1.5/6*1.5,6/2*1.5),
                                gridspec_kw={'wspace':0.1,
                                             'hspace': 0.6,
                                             'width_ratios': [len(labels),5],
                                             'height_ratios': [len(sbcl_names),1,len(hist_columns)]}, dpi=100)

    axs[0,0].imshow(subcl_comp2color(np.concatenate(dat_matrix, axis=1)[:-1], colornames=colornames, colorscale=colorscale)[reorder,:], vmin=0, vmax=1)
    axs[0,0].set_yticks(np.arange(len(sbcl_names)))
    axs[0,0].set_yticklabels(sbcl_names)
    axs[0,0].set_yticks(np.arange(len(sbcl_names)+1) - .5, minor=True)
    axs[0,0].tick_params(which='minor', length=0)  
    axs[0,0].set_xticks(np.arange(len(labels)+1) - .5, minor=True)
    axs[0,0].tick_params(which='minor', length=0)
    axs[0,0].grid(which='minor', color='w', linestyle='-', linewidth=2)

    
    axs[1,0].imshow(np.concatenate(dat_matrix, axis=1)[-1,None,:], vmin=0, vmax=1, cmap='Greys')
    axs[1,0].set_yticks([0])
    axs[1,0].set_yticklabels(['wt'])
    axs[1,0].set_yticks(np.arange(2) - .5, minor=True)
    axs[1,0].tick_params(which='minor', length=0)  
    axs[1,0].set_xticks(np.arange(len(labels)+1) - .5, minor=True)
    axs[1,0].tick_params(which='minor', length=0)
    axs[1,0].grid(which='minor', color='w', linestyle='-', linewidth=2)
    
    if hist_columns is not None:

        if flag_categorical:
            axs[2,0].imshow(categ_df2color(np.concatenate(hist_dat, axis=1).astype(int), colours=cat_colours), vmin=0, vmax=1)
        else:
            axs[2,0].imshow(np.concatenate(hist_dat, axis=1), vmin=0, vmax=100, cmap='YlOrBr')
        axs[2,0].set_yticks(np.arange(len(hist_columns)))
        axs[2,0].set_yticklabels(hist_columns);
        axs[2,0].set_yticks(np.arange(len(hist_columns)+1) - .5, minor=True)
        axs[2,0].tick_params(which='minor', length=0)  
        axs[2,0].set_xticks(np.arange(len(labels)+1) - .5, minor=True)
        axs[2,0].tick_params(which='minor', length=0)
        axs[2,0].grid(which='minor', color='w', linestyle='-', linewidth=2)


        if plot_labels:
            axs[2,0].set_xticks(np.arange(len(labels)))
            axs[2,0].set_xticklabels(labels, rotation=90);
        else:
            axs[2,0].set_xticks([])
            axs[2,0].set_xticklabels([])
    else:
        if plot_labels:
            axs[1,0].set_xticks(np.arange(len(labels)))
            axs[1,0].set_xticklabels(labels, rotation=90);
        else:
            axs[1,0].set_xticks([])
            axs[1,0].set_xticklabels([])

    cumusm = 0
    for i in range(len(n_type)):
        if n_type[i] != 0:
            if n_type[i] < 5:
                rotation = 45
            else:
                rotation = 0
            axs[0,0].text(cumusm + n_type[i]/2 - 1,-0.75, epith_names[i], rotation=rotation)
        if i < len(n_type) - 1:
            axs[0,0].axvline(cumusm + n_type[i] - 0.5, color='black', lw=1)
        cumusm += n_type[i]
        
    axs[0,1].axis('off')
    lsp = np.linspace(0, 1, 11)
    colors = subcl_comp2color(np.ones((len(sbcl_names),1)), colornames=colornames, colorscale=colorscale)[:, 0, :]
    for i in range(len(sbcl_names)):
        inset_ax = inset_axes(axs[0,1],
                              height="100%", # set height
                              width="100%", # and width
                              loc=10,
                              bbox_to_anchor=(0.1,1-1/len(sbcl_names)*(i+1)+0.01,1,1/len(sbcl_names)),
                              bbox_transform=axs[0,1].transAxes) # center, you can check the different codes in plt.legend?

        inset_ax.imshow(np.array([list(colors[i,:3]) + [x] for x in lsp]).reshape(1,len(lsp),4), vmin=0, vmax=1)
        inset_ax.tick_params(labelleft=False)   
        inset_ax.set_xticks(np.arange(len(lsp))[::2])
        inset_ax.set_xticklabels(np.round(lsp,1)[::2], rotation=90);
        inset_ax.set_title(sbcl_names[i])
        if i != len(sbcl_names) - 1 :
            inset_ax.tick_params(labelbottom=False)
            
            
    axs[1,1].axis('off')
    
    lsp = np.linspace(0, 1, 11)

    inset_ax = inset_axes(axs[1,1],
                          height="100%", # set height
                          width="100%", # and width
                          loc=10,
                          bbox_to_anchor=(0.1,0.2,1,0.4),
                          bbox_transform=axs[1,1].transAxes) # center, you can check the different codes in plt.legend?

    inset_ax.imshow(lsp.reshape(1,len(lsp)), vmin=0, vmax=1, cmap='Greys')
    inset_ax.tick_params(labelleft=False)   
    inset_ax.set_xticks(np.arange(len(lsp))[::2])
    inset_ax.set_xticklabels(np.round(lsp,1)[::2], rotation=90);
    inset_ax.set_title('Normal genotype')

    if hist_columns is not None:

        axs[2,1].axis('off')

        if flag_categorical:
            selected_matrix = hga_selected.hist_matrix.loc[:,hist_columns]
            for i in range(len(hist_columns)):
                categories = selected_matrix.iloc[:,i].cat.categories
                for j in range(len(categories)):
                    inset_ax = inset_axes(axs[2,1],
                                              height="40%", # set height
                                              width="40%", # and width
                                              loc=10,
                                              bbox_to_anchor=(0.1 + 1/len(categories)*j,1-1/len(hist_columns)*(i+1)-0.1,1/len(categories),1/len(hist_columns)),
                                              bbox_transform=axs[2,1].transAxes) # center, you can check the different codes in plt.legend?

                    inset_ax.scatter(0, 0, marker='s', s=100, color=categ_df2color(np.array([[j]]), colours=cat_colours))
                    inset_ax.axis('off')
                    inset_ax.set_title(categories[j])
        else:
            lsp = np.linspace(0, 100, 11)

            inset_ax = inset_axes(axs[2,1],
                                  height="100%", # set height
                                  width="100%", # and width
                                  loc=10,
                                  bbox_to_anchor=(0.1,0.2,1,0.4),
                                  bbox_transform=axs[2,1].transAxes) # center, you can check the different codes in plt.legend?

            inset_ax.imshow(lsp.reshape(1,len(lsp)), vmin=0, vmax=100, cmap='YlOrBr')
            inset_ax.tick_params(labelleft=False)   
            inset_ax.set_xticks(np.arange(len(lsp))[::2])
            inset_ax.set_xticklabels(np.round(lsp,1)[::2], rotation=90);
            inset_ax.set_title('Estimated area %')
            axs[2,0].set_aspect('auto')
        
        
    axs[0,0].set_aspect('auto')
    axs[1,0].set_aspect('auto')

    if title is not None:
        fig.suptitle(title)
