from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib import colors
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from basiss.utils import categ_df2color
from scipy.stats import chi2
from adjustText import adjust_text

from basiss.utils import boxplot_annotate_brackets
from basiss.utils.stats import fdr

import itertools

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
        
def plot_dcis_histogenomics(dcis_data, th = 0.4, plotnames = ['DCIS PD9694d', 'DCIS PD9694l'], sbcl_names = ['grey', 'green', 'purple', 'blue', 'red', 'orange']):
    hga_selected = dcis_data
    plt.rcParams['figure.facecolor'] = 'w'
    plotnames = ['DCIS PD9694d', 'DCIS PD9694l']
    sbcl_names = ['grey', 'green', 'purple', 'blue', 'red', 'orange']
    n_rows = 3
    n_cols = len(hga_selected)


    fig, axs = plt.subplots(n_rows,n_cols + 1, figsize=(40/1.3,3*n_rows/1.3*1.5),
                            gridspec_kw={'wspace':0.1,
                                         'hspace': 0.1,
                                         'width_ratios': [len(hga_selected[0]), len(hga_selected[1]), 15 ],
                                         'height_ratios': [6*1,3,4]}, dpi=200)



    reorder = [2,3,1,0,5,4,6]

    # Genomics
    for i, sample_name in enumerate(plotnames):
        hga_selected[i].sort_ids(th=th)
        proportions = hga_selected[i].comp_matrix.T / hga_selected[i].comp_matrix.T.sum(axis=0)[None,:]

        axs[0,i].imshow(subcl_comp2color(proportions, colornames=np.array(['grey', 'green', 'mediumorchid', 'dodgerblue', 'orangered',
           'darkorange', 'black']))[reorder,:], vmin=0, vmax=1)
        print('total', proportions.shape[1])

        sweep = proportions > 0.95

        dominant = proportions > 0.8

    
        axs[0,i].set_aspect('auto')

        axs[0,i].set_title(sample_name)

        for j in range(n_rows):
            axs[j,i].set_xticks(np.arange(len(hga_selected[i])))
            axs[j,i].set_xticks(np.arange(len(hga_selected[i])+1) - .5, minor=True)
            axs[j,i].tick_params(which='minor', length=0)    
            axs[j,i].set_xticklabels(hga_selected[i].common_ids, rotation=90);

            axs[j,i].spines['right'].set_visible(False)
            axs[j,i].spines['top'].set_visible(False)
            axs[j,i].tick_params(labelleft=False)   
            axs[j,i].grid(which='minor', color='w', linestyle='-', linewidth=2)
            if j < n_rows-1:
                axs[j,i].tick_params(labelbottom=False)    

        #axs[1,i].imshow(np.log10(hga_selected[i].comp_matrix.T.sum(axis=0)[None,:]), vmin=1, vmax=5, cmap='YlOrBr')

        #histology
        hist_df = hga_selected[i].hist_matrix

        ##anatomy
        selected_columns = ['Anatomical position', 'single_unit_in_mutISS']
        selected_matrix = hga_selected[i].hist_matrix.loc[:,selected_columns].apply(lambda x: x.cat.codes).values.T

        ### colourmap for n cells
        mapable = mpl.cm.ScalarMappable(cmap='YlOrBr')
        mapable.set_clim(vmin=1, vmax=5)
        n_cell_colour = mapable.to_rgba(np.log10(hga_selected[i].comp_matrix.T.sum(axis=0)[None,:]))

        axs[1,i].imshow(np.concatenate([n_cell_colour, categ_df2color(selected_matrix)], axis=0))
        ##Grades
        selected_columns = ['Grade', 'Nuclear pleomorphism', 'Vacuoli','Necrosis', 'Growth pattern']
        selected_matrix = hga_selected[i].hist_matrix.loc[:,selected_columns].apply(lambda x: x.cat.codes).values.T
        axs[2,i].imshow(categ_df2color(selected_matrix))


        axs[1,i].set_aspect('auto')
        axs[2,i].set_aspect('auto')


    axs[0,0].tick_params(labelleft=True)    
    axs[0,0].set_yticks(np.arange(6+1))
    axs[0,0].set_yticklabels(np.array(sbcl_names + ['wt'])[reorder]);
    axs[0,0].set_ylabel('Clone distribution', weight = 'bold')
    for i, sample_name in enumerate(plotnames):
        axs[0,i].set_yticks(np.arange(len(sbcl_names)+1) - .5, minor=True)
        axs[0,i].tick_params(which='minor', length=0)      

    selected_columns = ['N cells', 'Position', 'Is single']
    axs[1,0].tick_params(labelleft=True)    
    axs[1,0].set_yticks(np.arange(len(selected_columns)))
    axs[1,0].set_yticklabels(selected_columns);
    axs[1,0].set_ylabel('Anatomy', weight = 'bold')


    for i, sample_name in enumerate(plotnames):
        axs[1,i].set_yticks(np.arange(len(selected_columns)+1) - .5, minor=True)
        axs[1,i].tick_params(which='minor', length=0)      

    selected_columns = ['Grade', 'Nuclear pleomorphism', 'Vacuoli','Necrosis', 'Growth pattern']
    axs[2,0].tick_params(labelleft=True)    
    axs[2,0].set_yticks(np.arange(len(selected_columns)))
    axs[2,0].set_yticklabels(selected_columns);
    axs[2,0].set_ylabel('Tumour cell', weight = 'bold')  

    for i, sample_name in enumerate(plotnames):
        axs[2,i].set_yticks(np.arange(len(selected_columns)+1) - .5, minor=True)
        axs[2,i].tick_params(which='minor', length=0)      




    ###legends###
    axs[0,n_cols].axis('off')
    lsp = np.linspace(0, 1, 11)
    colors = subcl_comp2color(np.ones((6,1)))[:, 0, :]
    for i in range(6):
        inset_ax = inset_axes(axs[0,2],
                              height="100%", # set height
                              width="100%", # and width
                              loc=10,
                              bbox_to_anchor=(0.1,1-1/6*(i+1)+0.01,1,1/6),
                              bbox_transform=axs[0,2].transAxes) # center, you can check the different codes in plt.legend?

        inset_ax.imshow(np.array([list(colors[i,:3]) + [x] for x in lsp]).reshape(1,len(lsp),4), vmin=0, vmax=1)
        inset_ax.tick_params(labelleft=False)   
        inset_ax.set_xticks(np.arange(len(lsp))[::2])
        inset_ax.set_xticklabels(np.round(lsp,1)[::2], rotation=90);
        inset_ax.set_title(sbcl_names[i])
        if i != 5:
            inset_ax.tick_params(labelbottom=False)  

    axs[1,n_cols].axis('off')
    mapable = mpl.cm.ScalarMappable(cmap='YlOrBr')
    mapable.set_clim(vmin=1, vmax=5)
    lsp = np.linspace(0, 5, 11)
    inset_ax = inset_axes(axs[1,2],
                              height="80%", # set height
                              width="100%", # and width
                              loc=10,
                              bbox_to_anchor=(0.1,1-1/3*(0+1),1,1/3),
                              bbox_transform=axs[1,2].transAxes) # center, you can check the different codes in plt.legend?

    n_cell_colour = mapable.to_rgba(lsp.reshape(1,-1))
    inset_ax.imshow(n_cell_colour)
    inset_ax.tick_params(labelleft=False)   
    inset_ax.set_xticks(np.arange(len(lsp))[::2])
    inset_ax.set_xticklabels(np.round(lsp,0)[::2], rotation=90);
    inset_ax.set_title('Log10 n cells')

    selected_columns = ['Anatomical position', 'single_unit_in_mutISS']
    selected_matrix = hga_selected[0].hist_matrix.loc[:,selected_columns]
    for i in range(2):
        categories = selected_matrix.iloc[:,i].cat.categories
        for j in range(len(categories)):
            inset_ax = inset_axes(axs[1,2],
                                      height="40%", # set height
                                      width="40%", # and width
                                      loc=10,
                                      bbox_to_anchor=(0.1 + 1/len(categories)*j,1-1/4*(i+1+2)-0.1,1/len(categories),1/4),
                                      bbox_transform=axs[1,2].transAxes) # center, you can check the different codes in plt.legend?

            inset_ax.scatter(0, 0, marker='s', s=100, color=categ_df2color(np.array([[j]])))
            inset_ax.axis('off')
            inset_ax.set_title(categories[j])

    axs[2,n_cols].axis('off')
    selected_columns = ['Grade', 'Nuclear pleomorphism', 'Vacuoli','Necrosis central', 'Growth pattern']
    selected_matrix = hga_selected[0].hist_matrix.loc[:,selected_columns]
    for i in range(len(selected_columns)):
        categories = selected_matrix.iloc[:,i].cat.categories
        for j in range(len(categories)):
            inset_ax = inset_axes(axs[2,2],
                                      height="40%", # set height
                                      width="40%", # and width
                                      loc=10,
                                      bbox_to_anchor=(0.1 + 1/len(categories)*j,1-1/len(selected_columns)*(i+1)-0.1,1/len(categories),1/len(selected_columns)),
                                      bbox_transform=axs[2,2].transAxes) # center, you can check the different codes in plt.legend?

            inset_ax.scatter(0, 0, marker='s', s=100, color=categ_df2color(np.array([[j]])))
            inset_ax.axis('off')
            inset_ax.set_title(categories[j])


        
def vulcano_plot_glmm(diff_data, signal_data, colors,
                      fold_change_cutoff = 1.5,
                      fdr_val_cutoff = 0.1,
                      pre_test_cut = 1,
                      adjust=False):
    
    genes = np.array(signal_data.index.to_frame()['gene'].astype('category').cat.categories.to_list())
    
    mask = (signal_data['value'].sum(level=0)/signal_data['n_nucl'].sum(level=0) * signal_data['n_nucl'].mean(level=0)) > pre_test_cut
        
    color1 = colors[0]
    color2 = colors[1]

    delta = diff_data[:,0,:] - diff_data[:,1,:]
    z = delta.mean(0)/delta.std(0)

    p = chi2.sf(z**2,1)
    
    xs = []
    ys = []
    cs = []
    txts = []
    
    fdr_val = p.copy()
    fdr_val[mask] = fdr(p[mask])
    fdr_val[~mask] = 1.0
    for xyzf in zip(delta.mean(0), p, genes, fdr_val):
        xs.append(np.exp(xyzf[0]))
        ys.append(xyzf[1])
        if xyzf[3] < fdr_val_cutoff and not (np.exp(xyzf[0]) < fold_change_cutoff and  np.exp(xyzf[0]) > 1/fold_change_cutoff):
            txts.append(plt.text(x=np.exp(xyzf[0]), y=xyzf[1], s=xyzf[2]))
            if np.exp(xyzf[0]) > 1:
                cs.append(color1)
            else:
                cs.append(color2)
        else:
            cs.append('lightgrey')

    plt.axvline(1, c='k', alpha=0.4)
    plt.scatter(xs,ys,c=cs)
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.gca().invert_yaxis()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xlabel('$\mathregular{Log}_2$ fold change')
    plt.ylabel('$\mathregular{-Log}_{10}$ p-value')
    ymin = (np.minimum(p.min(), 10**-4))
    plt.ylim(1,ymin / 10)
    xminmax = np.maximum(np.max([np.max(xs), 1/np.min(xs)]), 2**(5))
    plt.xlim(1/xminmax/2, xminmax*2)
    if adjust:
        adjust_text(txts, arrowprops=dict(arrowstyle='-', color='black'), lim=100)
        
        
def plot_cell_composition(data, composition_estimates, ids2compare, cancer_type, colors, ylims=(0.12,0.5)):

    data.index = data.index.rename(['celltype', 'region', 'panel'])
    f = data.index.to_frame()
    df_pivoted = pd.pivot(pd.concat([data, f], 1), index=['panel', 'celltype'], columns='region', values='value')
    panels = f['panel'].unique()

    celltypes = df_pivoted.index.to_frame()['celltype'].values
    regions = df_pivoted.columns.values
    r2c = np.array([int(data.droplevel(0).droplevel(1).loc[r].iloc[0].clone_id) for r in regions])


    celltype2position = {k:v for k,v in zip(df_pivoted.index.to_frame()['celltype'], np.arange(df_pivoted.shape[0]))}

    probs = np.concatenate([np.exp(np.array(beta)) / np.exp(np.array(beta)).sum(2)[:,:,None] for beta in composition_estimates], axis=2)
    data_probs = np.concatenate([(df_pivoted.loc[p] / df_pivoted.loc[p].sum(0)).values for p in panels]).T



    celltypes_of_interest = ['B-cells', 'T-cells', 'Myeloid']
    subset = [celltype2position[x] for x in celltypes_of_interest]

    #fdr value
    n_entities = len(ids2compare)
    
    pairwise_comparisons = list(itertools.combinations(np.arange(n_entities), 2))

    p_values = []
    for pc in pairwise_comparisons:
        delta = probs[:,pc[0],subset] - probs[:,pc[1],subset]
        z = delta.mean(0)/delta.std(0)
        p = chi2.sf(z**2,1)
        p_values.append(fdr(p))
    fdr_values = np.stack(p_values)


    plt.subplot(1,2,1)
    
    if n_entities == 3:
        move = [-0.27, 0, 0.27]
    elif n_entities == 2:
        move = [-0.2, 0.2]
    else:
        raise ValueError('Only 2 and 3 group comparison is implemented, I was too tired to think how to nicely spread violin groups')
        
    for i in range(n_entities):
        ps = probs[:,i,subset]
        vp = plt.violinplot([ps[:,sub_gene][(ps[:,sub_gene] < np.percentile(ps[:,sub_gene], 99)) & (ps[:,sub_gene] > np.percentile(ps[:,sub_gene], 1))]
                             for sub_gene in range(len(subset))], positions=np.arange(len(subset)) + move[i], showextrema=False, widths=0.3)
        for j in range(len(subset)):
            jit = np.random.randn(data_probs[r2c == i].shape[0])
            plt.scatter(np.array([j] * data_probs[r2c == i].shape[0]) + move[i] + jit/50, data_probs[r2c == i][:,subset[j]], color='black', s=5)
        plt.xticks(np.arange(len(subset)), celltypes_of_interest, rotation=-45)

        for body in vp['bodies']:
            body.set_alpha(1)
            body.set_facecolor(colors[ids2compare[i]])
            if cancer_type[i] != 'TC1':
                body.set_edgecolor('black')
                body.set_linewidth(1)


    for j in range(len(subset)):
        dh_extra = 0
        for i, ps in enumerate(pairwise_comparisons):
            num1, num2 = ps
            if fdr_values[i,j] < 0.1:
                boxplot_annotate_brackets(num1, num2, j + np.array(move), [np.max(data_probs[r2c == x][:,subset[j]]) for x in range(n_entities)],  str(np.round(fdr_values[i,j],5)), dh=0.01 + dh_extra)
                dh_extra += 0.1


    plt.ylim(0,ylims[0])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.title('Immune')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)


    celltypes_of_interest = ['B-cells', 'T-cells', 'Myeloid']
    subset = [celltype2position[x] for x in celltypes_of_interest]

    probs_combined = np.concatenate([probs[:,:,[celltype2position[x] for x in ['B-cells', 'T-cells', 'Myeloid', 'Immune broad']]].sum(2)[:,:,None],
                                     probs[:,:,celltype2position['Fibroblasts + PVL']][:,:,None],
                                     probs[:,:,celltype2position['Epithelial broad']][:,:,None]], axis=2)

    data_probs_combined = np.concatenate([data_probs[:,[celltype2position[x] for x in ['B-cells', 'T-cells', 'Myeloid', 'Immune broad']]].sum(1)[:,None],
                                          data_probs[:,celltype2position['Fibroblasts + PVL']][:,None],
                                          data_probs[:,celltype2position['Epithelial broad']][:,None]], axis=1)

    p_values = []
    for pc in pairwise_comparisons:
        delta = probs_combined[:,pc[0],:] - probs_combined[:,pc[1],:]
        z = delta.mean(0)/delta.std(0)
        p = chi2.sf(z**2,1)
        p_values.append(fdr(p))
    fdr_values = np.stack(p_values)

    plt.subplot(1,2,2)

    for i in range(n_entities):
        ps = probs_combined[:,i,:]
        vp = plt.violinplot([ps[:,sub_gene][(ps[:,sub_gene] < np.percentile(ps[:,sub_gene], 99)) & (ps[:,sub_gene] > np.percentile(ps[:,sub_gene], 1))] for sub_gene in range(3)],
                            positions=np.arange(3) + move[i], showextrema=False, widths=0.3)
        for j in range(3):
            jit = np.random.randn(data_probs_combined[r2c == i].shape[0])
            plt.scatter(np.array([j] * data_probs_combined[r2c == i].shape[0]) + move[i] + jit/50, data_probs_combined[r2c == i][:,j], color='black', s=5)
        plt.xticks(np.arange(ps.shape[-1]), ['Immune', 'Fibrobalsts', 'Epithelial'], rotation=-45)

        for body in vp['bodies']:
            body.set_alpha(1)
            body.set_facecolor(colors[ids2compare[i]])
            if cancer_type[i] != 'TC1':
                body.set_edgecolor('black')
                body.set_linewidth(1)

    for j in range(len(subset)):
        dh_extra = 0
        for i, ps in enumerate(pairwise_comparisons):
            num1, num2 = ps
            if fdr_values[i,j] < 0.1:
                boxplot_annotate_brackets(num1, num2, j + np.array(move), [np.max(data_probs_combined[r2c == x][:,j]) for x in range(n_entities)], str(np.round(fdr_values[i,j],5)), dh=0.01 + dh_extra)
                dh_extra += 0.1

    plt.ylim(0,ylims[1])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.title('Broad')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)