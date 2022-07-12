import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from basiss.utils import inverse_dict


def cell_type_decision(expression_table, markers, condition=None, previous_annots=None):
    """Cell type decision based on the assigned marker genes

    Parameters
    ----------
    expression_table : pd.DataFrame
        Nuclei x genes count table
    markers : dict
        Marker gene dictionary {'cell type' : ['gene1', ...], ...}
    condition : None or dict
        Conditional structure of cell type assignment based on the hierarchy,
         {'old type': ['new type 1', 'new type 2', ...]}
    previous_annots : np.array
        Array of the previous annotations

    Returns
    -------
    np.array
        Cell type names
    """
    decisions_made = np.zeros(expression_table.shape[0])
    if (condition is not None) and (previous_annots is not None):
        decision = previous_annots.copy()
        for ct, gs in markers.items():
            try:
                cond = condition[ct]
                # print(cond)
                gs_upd = np.array(gs)[np.isin(gs, list(expression_table.columns))]
                decisions_made[(expression_table.loc[:, gs_upd].sum(axis=1) > 0) & (decision == cond)] += 1
                decision[(expression_table.loc[:, gs_upd].sum(axis=1) > 0) & (decision == cond)] = ct
            except KeyError:
                # print(gs_upd)
                pass
        decision[decisions_made != 1] = previous_annots[decisions_made != 1]
    else:
        decision = np.array([None] * expression_table.shape[0])
        for ct, gs in markers.items():
            try:
                gs_upd = np.array(gs)[np.isin(gs, list(expression_table.columns))]
                decisions_made[expression_table.loc[:, gs_upd].sum(axis=1) > 0] += 1
                decision[expression_table.loc[:, gs_upd].sum(axis=1) > 0] = ct
            except KeyError:
                # print(gs_upd)
                pass
        decision[decisions_made != 1] = None
    return decision.astype('<U50')


def iss_annotation(sample, broad_markers, narrow_markers, condition, th_dist=5, pix2um=0.325):
    """Annotation of cell types based on the ISS data and definitive cell type markers

    Parameters
    ----------
    sample : basiss.preprocessing.Sample
        Sample with ISS data
    broad_markers : dict
        Broad cell type marker gene dictionary {'cell type' : ['gene1', ...], ...}
    narrow_markers : dict
        Narrow cell type marker gene dictionary {'cell type' : ['gene1', ...], ...}
    condition : dict
        Conditional structure of cell type assignment based on the hierarchy,
         {'old type': ['new type 1', 'new type 2', ...]}
    th_dist : float
        Distance of the ISS signal from the nucleus centre that is considered
    pix2um : float
        pixel to um conversion

    Returns
    -------
    dict
        Dictionary of 2 DataFrames - expression_per_nucleus and nuclei_types
    """
    nucl = sample.cellpos
    iss_spots = pd.DataFrame(sample.data)[sample.iss_probability > 0.6]
    iss_spots = iss_spots[~np.isin(iss_spots.Gene, ['infeasible', 'background'])]
    nucl_pos = np.array([nucl[:, 0], nucl[:, 1]]).T * pix2um
    iss_pos = np.array([iss_spots.PosX, iss_spots.PosY]).T * pix2um

    tree = KDTree(nucl_pos)
    distance, indx = tree.query(iss_pos, k=1, return_distance=True)

    iss_spots['assigned_nucl'] = -1
    iss_spots.loc[iss_spots.index[distance.flatten() < th_dist], 'assigned_nucl'] = indx[distance < th_dist].astype(int)
    expression_table = iss_spots.groupby(['Gene', 'assigned_nucl']).size().unstack('assigned_nucl',
                                                                                   fill_value=0).T.iloc[1:, :]

    res_low = cell_type_decision(expression_table, broad_markers)
    res_high = cell_type_decision(expression_table, narrow_markers, condition=inverse_dict(condition),
                                  previous_annots=res_low)

    answer_df = pd.DataFrame({'nucl_id': np.array(expression_table.index),
                              'assignment': res_high})
    answer_df['x'] = nucl_pos[answer_df.nucl_id, 0]
    answer_df['y'] = nucl_pos[answer_df.nucl_id, 1]

    ids = np.arange(nucl_pos.shape[0])
    ids = ids[~np.isin(ids, answer_df.nucl_id)]

    t = pd.DataFrame({'nucl_id': ids, 'assignment': 'None', 'x': nucl_pos[ids, 0], 'y': nucl_pos[ids, 1]})

    answer_df = pd.concat([answer_df, t]).reset_index(drop=True)
    return {'expression_per_nucleus': expression_table, 'nuclei_types': answer_df}
