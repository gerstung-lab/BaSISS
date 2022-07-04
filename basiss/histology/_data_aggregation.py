import shapely.geometry as shp
import numpy as np
import pandas as pd
import matplotlib.path as mpltPath
from collections import Counter


def shoelace(x_y):
    x_y = np.array(x_y)
    x_y = x_y.reshape(-1,2)

    x = x_y[:,0]
    y = x_y[:,1]

    S1 = np.sum(x*np.roll(y,-1))
    S2 = np.sum(y*np.roll(x,-1))

    area = .5*np.absolute(S1 - S2)

    return area

def length(x_y):
    x_y = np.array(x_y)
    x_y = x_y.reshape(-1,2)

    x = x_y[:,0]
    y = x_y[:,1]

    l = np.sum(((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)**0.5)
    
    return l

def expand_poly(x_y, distance):
    ar_sh = shp.Polygon(x_y)
    ar_exp_sh = ar_sh.buffer(distance) 
    try:
        ar_exp = np.array(ar_exp_sh.exterior.coords)
        return tuple([ar_exp])
    except AttributeError:
        return (np.array(x.exterior.coords) for x in ar_exp_sh.geoms)

    
def output_ducts_composition_raw(mut_sample_list, model_params_samples, indx, names=['grey','green', 'purple', 'blue', 'red','orange', 'wt']):
    mut_sample_list[indx].data_to_grid(3, probability=0.6)
    sample_d_data = mut_sample_list[indx]
    

    gfields_d = model_params_samples[f'F_{indx}']
    n_subcl_dim = gfields_d.shape[-1]#.reshape(300, *sample_d_data.grid_params, 9)
    gfields_d = np.concatenate([gfields_d[:,:,:,:(n_subcl_dim-3)], gfields_d[:,:,:,(n_subcl_dim-3):(n_subcl_dim-1)].sum(axis=3)[:,:,:,None]], axis=3) #collapse wt, remove residuals
    cell_densities_d = model_params_samples[f'lm_n_{indx}']#.reshape(300, *sample_d_data.grid_params)
    
    rescale_xy = sample_d_data.grid_params / np.array(sample_d_data.spatial_dims)
    paths = sample_d_data.ducts['paths']
    #print(sample_d_data.ducts.keys())
    paths_id = [string[1:] if string.startswith('_') else string for string in sample_d_data.ducts['id']]
    
    paths_matplot = [mpltPath.Path(paths[i]* rescale_xy) for i in range(len(paths))]
    
    xg1, yg1 = np.meshgrid(np.arange(sample_d_data.grid_params[0]), np.arange(sample_d_data.grid_params[1]))
    xg1, yg1 = xg1.flatten(), yg1.flatten()
    paths_matplot = [mpltPath.Path(paths[i]* rescale_xy) for i in range(len(paths))]
    dots_within_paths = {}
    for i in range(len(paths)):
        dots_within = paths_matplot[i].contains_points(np.array([xg1, yg1]).T, radius=1)
        dots_within_paths[paths_id[i]] = dots_within
        
    subclone_proportions = []
    subclone_cell_densities = {}
    for i in range(len(paths)):
        x = xg1[dots_within_paths[paths_id[i]]]
        y = yg1[dots_within_paths[paths_id[i]]]

        cell_numbers = (gfields_d[:, x, y, :(n_subcl_dim-2)] * cell_densities_d[:, x, y, None]).sum(axis=1)
        subclone_cell_densities[paths_id[i]] = cell_numbers
       
    return subclone_cell_densities



'''
Be careful, with the names 
'''

def output_cell_composition_raw(sample_panel_list: list, cell_type_df_list: list, expression_type_df_list: list,
                                cell_groups = ['Epithelial broad', 'Immune broad', 'Stromal broad',
                                               'Fibroblasts', 'Endothelial', 'Myeloid',
                                               'T-cells', 'B-cells', 'None', 'Immune total'],
                                expression_types = ['Epithelial broad', 'Fibroblasts', 'Stromal broad',
                                                    'Immune broad', 'Immune total', 'Myeloid', 'T-cells'],
                                mode='inner',
                                distance_inside=-10,
                                distance_outside=80,
                                pixel2um=0.325):
    
    paths_id = list(set.intersection(*map(set,[[string[1:] if string.startswith('_') else string for string in sample_panel.ducts['id']] for sample_panel in sample_panel_list])))
    cell_type_numbers = {gr: {k:np.zeros(len(paths_id)) for k in cell_groups} for gr in range(len(sample_panel_list))}
    
    expression_numbers = {gr: {k:[] for k in expression_types} for gr in range(len(sample_panel_list))}
    for sample_panel_i, sample_panel in enumerate(sample_panel_list):
        paths = np.array(sample_panel.ducts['paths'])[[np.where(np.array(sample_panel.ducts['id']) == k)[0][0]
                                                       for k in paths_id]]
        
        cell_type_df = cell_type_df_list[sample_panel_i]
        xg1, yg1 = cell_type_df.x , cell_type_df.y

        dots_within_paths = {}
        poly_area = []
        if mode == 'inner':
            for i in range(len(paths)):
                poly_area.append(np.round(np.sum([shoelace(x) for x in expand_poly(paths[i] * pixel2um, distance=distance_inside)])))
                paths_altered = expand_poly(paths[i] * pixel2um, distance=distance_inside)
                
                dots_within = None
                for path_alt in paths_altered:
                    path_matplot = mpltPath.Path(path_alt)
                    if dots_within is not None:
                        dots_within |= path_matplot.contains_points(np.array([xg1, yg1]).T)
                    else:
                        dots_within = path_matplot.contains_points(np.array([xg1, yg1]).T)
                
                dots_within_paths[paths_id[i]] = dots_within
                
                for expression_type in expression_types:                        
                    cell_by_inclusion = cell_type_df[dots_within_paths[paths_id[i]]]
                    
                    if expression_type == 'Immune total':
                        cell_by_type = cell_by_inclusion[np.isin(cell_by_inclusion.assignment, ['Immune broad',
                                                                                                'Myeloid',
                                                                                                'T-cells', 'B-cells'])]
                    else:
                        cell_by_type = cell_by_inclusion[cell_by_inclusion.assignment == expression_type]

                    sum_exp = expression_type_df_list[sample_panel_i].loc[cell_by_type.nucl_id,:].sum(axis=0)
                    expression_numbers[sample_panel_i][expression_type].append(sum_exp)
                    
                entries = Counter(cell_type_df.assignment[dots_within_paths[paths_id[i]]])
                for k in cell_groups:
                    cell_type_numbers[sample_panel_i][k][i] += entries[k]
        elif mode == 'outer':
            for i in range(len(paths)):
                poly_area.append(np.round(np.sum([shoelace(x) for x in expand_poly(paths[i] * pixel2um, distance=distance_outside)]) -
                                 np.sum([shoelace(x) for x in expand_poly(paths[i] * pixel2um, distance=distance_inside)])))
                #inside
                
                paths_altered = expand_poly(paths[i] * pixel2um, distance=distance_inside)
                
                dots_within = None
                for path_alt in paths_altered:
                    path_matplot = mpltPath.Path(path_alt)
                    if dots_within is not None:
                        dots_within |= path_matplot.contains_points(np.array([xg1, yg1]).T)
                    else:
                        dots_within = path_matplot.contains_points(np.array([xg1, yg1]).T)
                
                dots_within_paths[paths_id[i]] = dots_within

                entries = Counter(cell_type_df.assignment[dots_within_paths[paths_id[i]]])
                for k in cell_groups:
                    cell_type_numbers[sample_panel_i][k][i] -= entries[k]
                                    
                for expression_type in expression_types:
                    
                    cell_by_inclusion = cell_type_df[dots_within_paths[paths_id[i]]]
                    if expression_type == 'Immune total':
                        cell_by_type = cell_by_inclusion[np.isin(cell_by_inclusion.assignment, ['Immune broad',
                                                                                                'Myeloid',
                                                                                                'T-cells', 'B-cells'])]
                    else:
                        cell_by_type = cell_by_inclusion[cell_by_inclusion.assignment == expression_type]
                    
                    sum_exp = expression_type_df_list[sample_panel_i].loc[cell_by_type.nucl_id,:].sum(axis=0)
                    expression_numbers[sample_panel_i][expression_type].append(-sum_exp)
                
                #outside
                paths_altered = expand_poly(paths[i] * pixel2um, distance=distance_outside)
                
                dots_within = None
                for path_alt in paths_altered:
                    path_matplot = mpltPath.Path(path_alt)
                    if dots_within is not None:
                        dots_within |= path_matplot.contains_points(np.array([xg1, yg1]).T)
                    else:
                        dots_within = path_matplot.contains_points(np.array([xg1, yg1]).T)

                dots_within_paths[paths_id[i]] = dots_within
                entries = Counter(cell_type_df.assignment[dots_within_paths[paths_id[i]]])
                for k in cell_groups:
                    cell_type_numbers[sample_panel_i][k][i] += entries[k]
                    
                for expression_type in expression_types:
                    
                    cell_by_inclusion = cell_type_df[dots_within_paths[paths_id[i]]]
                    if expression_type == 'Immune total':
                        cell_by_type = cell_by_inclusion[np.isin(cell_by_inclusion.assignment, ['Immune broad',
                                                                                                'Myeloid',
                                                                                                'T-cells', 'B-cells'])]
                    else:
                        cell_by_type = cell_by_inclusion[cell_by_inclusion.assignment == expression_type]
                    
                    sum_exp = expression_type_df_list[sample_panel_i].loc[cell_by_type.nucl_id,:].sum(axis=0)
                    expression_numbers[sample_panel_i][expression_type][i] += sum_exp
                    
        for expression_type in expression_types:            
            expression_numbers[sample_panel_i][expression_type] = pd.DataFrame(np.stack(expression_numbers[sample_panel_i][expression_type]),
                                                                               index=paths_id, 
                                                                               columns=expression_type_df_list[sample_panel_i].columns)
        cell_type_numbers[sample_panel_i]['area'] = poly_area
    
    df = {k: pd.DataFrame(v, index=paths_id) for k,v in cell_type_numbers.items()}
    return df, expression_numbers
