import numpy as np
import pandas as pd

def construct_data_for_DE(hga,
                          gene_expression_panel = 'epithelial',
                          iss_panel = 'inner_onco',
                          cell_type = 'Epithelial',
                          cancer_type = ['DCIS', 'DCIS'],
                          ids2compare = [[1], [5]],
                          th = [0.7, 0.7],
                          histo_type = 'Epithelial cells',
                          drop=['POU5F1', 'PTPRC_trans5']):
    
    full_matrix = []

    for i in range(2):
        hga.subset_ids()

        hga.subset_ids(hist_condition = np.isin(hga.histology_df[histo_type], cancer_type[i]),
                                  comp_condition = [(v[:,:-1]/v.sum(1)[:,None]).mean(0).argmax() in ids2compare[i] and 
                                                    (v[:,(v[:,:-1]/v.sum(1)[:,None]).mean(0).argmax()]/v.sum(1)).mean() > th[i]
                                                    for k,v in hga.composition_dict.items()])
        data_matrix = hga.extra_matrix(gene_expression_panel)
        data_matrix['clone'] = hga.comp_matrix.argmax(1)
        data_matrix['clone_id'] = i
        print(data_matrix.shape)
        data_matrix.index = data_matrix.index.rename('region')
        cell_df = hga.extra_matrix(iss_panel)
        cell_df['total'] = hga.extra_matrix(iss_panel).iloc[:,:-1].sum(1)
        data_matrix['n_nucl'] = cell_df[cell_type]
        full_matrix.append(data_matrix)
        #data_matrix['histology'] = hga.hist_matrix['Epithelial cells']

    full_matrix = pd.concat(full_matrix)
    data = full_matrix.reset_index().melt(id_vars=['region', 'clone', 'n_nucl','clone_id'], value_vars=data_matrix.columns[:-3], var_name='gene') 
    data = data[~np.isin(data.gene, drop)]
    return data

def construct_data_for_regression(hga,
                          gene_expression_panel = 'epithelial',
                          iss_panel = 'inner_onco',
                          cell_type = 'Epithelial',
                          cancer_type = ['DCIS', 'DCIS'],
                          ids2compare = [1,5],
                          th = [0.7, 0.7],
                          histo_type = 'Epithelial cells',
                          drop=['POU5F1', 'PTPRC_trans5']):
    
    full_matrix = []

    for i in range(len(ids2compare)):
        hga.subset_ids()
        hga.subset_ids(hist_condition = np.isin(hga.histology_df[histo_type], cancer_type[i]),
                                  comp_condition = [(v[:,:-1]/v.sum(1)[:,None]).mean(0).argmax() == ids2compare[i] and 
                                                    (v[:,(v[:,:-1]/v.sum(1)[:,None]).mean(0).argmax()]/v.sum(1)).mean() > th[i]
                                                    for k,v in hga.composition_dict.items()])
        data_matrix = hga.extra_matrix(gene_expression_panel)
        data_matrix['clone'] = hga.comp_matrix.argmax(1)
        data_matrix['clone_id'] = i
        print(data_matrix.shape)
        data_matrix.index = data_matrix.index.rename('region')
        cell_df = hga.extra_matrix(iss_panel)
        cell_df['total'] = hga.extra_matrix(iss_panel).iloc[:,:-1].sum(1)
        data_matrix['n_nucl'] = cell_df[cell_type]
        full_matrix.append(data_matrix)
        #data_matrix['histology'] = hga.hist_matrix['Epithelial cells']

    full_matrix = pd.concat(full_matrix)
    data = full_matrix.reset_index().melt(id_vars=['region', 'clone', 'n_nucl','clone_id'], value_vars=data_matrix.columns[:-3], var_name='gene') 
    data = data[~np.isin(data.gene, drop)]
    return data


def prepare_cell_composition_data(hga, sample_id, ids2compare, cancer_type, panel='inner_immune', cells_to_include = ['Immune broad', 'B-cells', 'Myeloid', 'T-cells'],
                                  histology_type='Epithelial cells', th=-1):
    
   # if panel == 'inner_immune':
   #     cells_to_include = ['Immune broad', 'B-cells', 'Myeloid', 'T-cells']
   # elif panel == 'inner_onco':
   #     cells_to_include = ['Epithelial broad', 'Fibroblasts broad']
    
    full_matrix = []

    for i in range(len(ids2compare)):
        hgas.subset_ids()
        hgas.subset_ids(hist_condition = np.isin(hgas.histology_df[histology_type], cancer_type[i]),
                                  comp_condition = [(v[:,:-1]/v.sum(1)[:,None]).mean(0).argmax() == ids2compare[i] and 
                                                    (v[:,(v[:,:-1]/v.sum(1)[:,None]).mean(0).argmax()]/v.sum(1)).mean() > th[i]
                                                    for k,v in hgas.composition_dict.items()])
        data_matrix = pd.concat([hgas.extra_matrix(panel)[cells_to_include],
                   hgas.extra_matrix(panel).iloc[:, ~np.isin(hgas.extra_matrix(panel).columns, cells_to_include)].iloc[:,:-2].sum(1).rename('None')], axis=1)
        data_matrix['clone'] = hgas.comp_matrix.argmax(1)
        data_matrix['clone_id'] = i
        print(data_matrix.shape)
        data_matrix.index = data_matrix.index.rename('region')
        full_matrix.append(data_matrix)
        #data_matrix['histology'] = hgas.hist_matrix['Epithelial cells']

    full_matrix = pd.concat(full_matrix)
    data = full_matrix.reset_index().melt(id_vars=['region', 'clone','clone_id'], value_vars=data_matrix.columns[:-2], var_name='cell') 
    return data
    
