import numpy as np
import matplotlib.pyplot as plt

def mask_infisble(mut_sample_list, scale, probability=0.6, plot=False):
    """Create a mask of infisible areas according to a set of emperical rules

    Parameters
    ----------
    mut_sample_list : list
        List with mutation layers for each sample
    scale : int
        Scaeling parameter for the grid size
    probability : float
        Threshold probability to filter out non-confident iss signal, default is 0.6 
    plot : bool
        If True plot resulting mask array 

    Returns
    -------
    list
        List of bool matrices where False correspond to infisible tiles.
    """
    mask = []
    for i in range(len(mut_sample_list)):
        mut_sample_list[i].data_to_grid(scale_factor=scale, probability=0.6)
        t = np.array([s for s in mut_sample_list[i].gene_grid.values()])[:-3].sum(0)
        mask_infisiable = mut_sample_list[i].gene_grid['infeasible']/t < 0.1
        mask_infisiable *= mut_sample_list[i].cell_grid > 5
        
        if plot:
            plt.figure(figsize=(8,4))
            plt.imshow(mask_infisiable.T[::-1,:])
            
        mask.append(mask_infisiable.flatten())
            
    return mask

def generate_data4model(samples_list, genes, M, n_aug=1, unified_names=None):
    """Output data in a suitable format for bassis inference

    Parameters
    ----------
    samples_list : list
        List with mutation layers for each sample
    genes : list
        List of gene names which are used for inference 
    M : matrix
        Genotype matrix (clones x genes) with copy number alterations for each clone-locus pare
    n_aug : int
        Number of extra pseudoclones (flexibility of inhomogeneous noise capture), default 1
    unified_names : list
        Map of sample names -> starndard names for each sample
    
    Returns
    -------
    dict
        Dictionary of bassis inference related input data and parameters
    """
    n_samples = len(samples_list)
    n_genes = len(genes)
    if not unified_names is None:
        iss_data = [np.transpose(np.array([samples_list[i].gene_grid[unified_name1[i][k]] for k in genes]), [1,2,0]).reshape(-1, n_genes) for i in range(n_samples)]
    else:
        iss_data = [np.transpose(np.array([samples_list[i].gene_grid[k] for k in genes]), [1,2,0]).reshape(-1, n_genes) for i in range(n_samples)]

    tiles_axes = [samples_list[i].tile_axis for i in range(n_samples)]

    cells_counts = [samples_list[i].cell_grid.flatten() for i in range(n_samples)]
    sample_dims = [(int(tiles_axes[i][0][-1]+1), int(tiles_axes[i][1][-1]+1)) for i in range(n_samples)]
    n_factors = M.shape[0]
    n_aug=n_aug
    
    return {'iss_data': iss_data, 'tiles_axes': tiles_axes, 'cells_counts': cells_counts, 'sample_dims': sample_dims,
            'n_factors': n_factors, 'n_aug': n_aug, 'tree_matrix':M, 'n_samples': n_samples, 'n_genes': n_genes, 'genes': genes}


def to_grid(x_raw, y_raw, grid_params, img_size):
    """Data binning to a predifined grid

    Parameters
    ----------
    x_raw : np.array
        X positions of point data
    y_raw : np.array
        Y position of point data 
    grid_params : tuple
        number of grid cells along x and y dimensions 
    img_size : list
        Image dimensions in pixels
    
    Returns
    -------
    np.array
        2D array of binned data
    """
    arr = np.zeros((grid_params[0], grid_params[1]))
    x_step = img_size[0] / (grid_params[0]-1)
    y_step = img_size[1] / (grid_params[1]-1)
    tiles = np.array([(x_raw//x_step).astype(int),
                        (y_raw//y_step).astype(int)]).T
    k_id, v = np.unique(tiles, return_counts=True, axis=0)
    for i in range(len(v)):
        arr[tuple(k_id[i,:])] = v[i]
        
    return arr