import numpy as np
import pandas as pd


def construct_data_for_DE(hga,
                          gene_expression_panel='epithelial',
                          iss_panel='inner_onco',
                          cell_type='Epithelial',
                          cancer_type=['DCIS', 'DCIS'],
                          ids2compare=[[1], [5]],
                          th=[0.7, 0.7],
                          histo_type='Epithelial cells',
                          drop=['PTPRC_trans5', 'nan']):
    """Construct data for differential expression glmm

    Parameters
    ----------
    hga : basiss.histology.Histogenomic_associations
        Histogenomic association object of a sample of interest
    gene_expression_panel : str
        Name of the cell type specific expression panel (as in hga)
    iss_panel : str
        Name of the expression panel
    cell_type : str
        Cell type of interest
    cancer_type : list
        List of region type
    ids2compare : list
        List of clonal ids to compare
    th :  list
        List of minimal clone cell fraction thresholds for the regions to be considered
    histo_type : str
        Name of the column in hga.histoloy_df where region type is stored
    drop : list
        List of genes to drop

    Returns
    -------
    pd.DataFrame
        Dataframe ready for glmm differential expression analysis
    """
    full_matrix = []

    for i in range(2):
        hga.subset_ids()

        hga.subset_ids(hist_condition=np.isin(hga.histology_df[histo_type], cancer_type[i]),
                       comp_condition=[(v[:, :-1] / v.sum(1)[:, None]).mean(0).argmax() in ids2compare[i] and
                                       (v[:, (v[:, :-1] / v.sum(1)[:, None]).mean(0).argmax()] / v.sum(1)).mean() > th[
                                           i]
                                       for k, v in hga.composition_dict.items()])
        data_matrix = hga.extra_matrix(gene_expression_panel)
        data_matrix['clone'] = hga.comp_matrix[:, :-1].argmax(1)
        data_matrix['clone_id'] = i
        print(data_matrix.shape)
        data_matrix.index = data_matrix.index.rename('region')
        cell_df = hga.extra_matrix(iss_panel)
        cell_df['total'] = hga.extra_matrix(iss_panel).iloc[:, :-1].sum(1)
        data_matrix['n_nucl'] = cell_df[cell_type]
        full_matrix.append(data_matrix)
        # data_matrix['histology'] = hga.hist_matrix['Epithelial cells']

    full_matrix = pd.concat(full_matrix)
    data = full_matrix.reset_index().melt(id_vars=['region', 'clone', 'n_nucl', 'clone_id'],
                                          value_vars=data_matrix.columns[:-3], var_name='gene')
    data = data[~np.isin(data.gene, drop)]
    return data


def construct_data_for_regression(hga,
                                  gene_expression_panel='epithelial',
                                  iss_panel='inner_onco',
                                  cell_type='Epithelial',
                                  cancer_type=['DCIS', 'DCIS'],
                                  ids2compare=[1, 5],
                                  th=[0.7, 0.7],
                                  histo_type='Epithelial cells',
                                  drop=['PTPRC_trans5', 'nan']):
    """Construct data for multiregional expression glmm

        Parameters
        ----------
        hga : basiss.histology.Histogenomic_associations
            Histogenomic association object of a sample of interest
        gene_expression_panel : str
            Name of the cell type specific expression panel (as in hga)
        iss_panel : str
            Name of the expression panel
        cell_type : str
            Cell type of interest
        cancer_type : list
            List of region type
        ids2compare : list
            List of clonal ids to compare
        th :  list
            List of minimal clone cell fraction thresholds for the regions to be considered
        histo_type : str
            Name of the column in hga.histoloy_df where region type is stored
        drop : list
            List of genes to drop

        Returns
        -------
        pd.DataFrame
            Dataframe ready for multiregional glmm differential expression analysis
        """

    full_matrix = []

    for i in range(len(ids2compare)):
        hga.subset_ids()
        hga.subset_ids(hist_condition=np.isin(hga.histology_df[histo_type], cancer_type[i]),
                       comp_condition=[(v[:, :-1] / v.sum(1)[:, None]).mean(0).argmax() == ids2compare[i] and
                                       (v[:, (v[:, :-1] / v.sum(1)[:, None]).mean(0).argmax()] / v.sum(1)).mean() > th[
                                           i]
                                       for k, v in hga.composition_dict.items()])
        data_matrix = hga.extra_matrix(gene_expression_panel)
        data_matrix['clone'] = hga.comp_matrix[:, :-1].argmax(1)
        data_matrix['clone_id'] = i
        print(data_matrix.shape)
        data_matrix.index = data_matrix.index.rename('region')
        cell_df = hga.extra_matrix(iss_panel)
        cell_df['total'] = hga.extra_matrix(iss_panel).iloc[:, :-1].sum(1)
        data_matrix['n_nucl'] = cell_df[cell_type]
        full_matrix.append(data_matrix)
        # data_matrix['histology'] = hga.hist_matrix['Epithelial cells']

    full_matrix = pd.concat(full_matrix)
    data = full_matrix.reset_index().melt(id_vars=['region', 'clone', 'n_nucl', 'clone_id'],
                                          value_vars=data_matrix.columns[:-3], var_name='gene')
    data = data[~np.isin(data.gene, drop)]
    return data


def prepare_cell_composition_data(hga,
                                  ids2compare,
                                  cancer_type,
                                  panel='inner_immune',
                                  cells_to_include=['Immune broad', 'B-cells', 'Myeloid', 'T-cells'],
                                  histology_type='Epithelial cells',
                                  th=-1):

    """Construct data for multiregional composition glmm

            Parameters
            ----------
            hga : basiss.histology.Histogenomic_associations
                Histogenomic association object of a sample of interest
            ids2compare : list
                List of clonal ids to compare
            cancer_type : list
                List of region type
            panel : str
                Name of the cell composition panel (as in hga)
            cells_to_include : list
                List of cell types to consider
            histology_type : str
                Name of the column in hga.histoloy_df where region type is stored
            th :  list
                List of minimal clone cell fraction thresholds for the regions to be considered

            Returns
            -------
            pd.DataFrame
                Dataframe ready for multiregional glmm differential composition analysis
            """

    full_matrix = []

    for i in range(len(ids2compare)):
        hga.subset_ids()
        hga.subset_ids(hist_condition=np.isin(hga.histology_df[histology_type], cancer_type[i]),
                       comp_condition=[(v[:, :-1] / v.sum(1)[:, None]).mean(0).argmax() == ids2compare[i] and
                                       (v[:, (v[:, :-1] / v.sum(1)[:, None]).mean(0).argmax()] / v.sum(1)).mean() > th[
                                           i]
                                       for k, v in hga.composition_dict.items()])
        data_matrix = pd.concat([hga.extra_matrix(panel)[cells_to_include],
                                 hga.extra_matrix(panel).iloc[:,
                                 ~np.isin(hga.extra_matrix(panel).columns, cells_to_include)].iloc[:, :-2].sum(
                                     1).rename('None')], axis=1)
        data_matrix['clone'] = hga.comp_matrix[:, :-1].argmax(1)
        data_matrix['clone_id'] = i
        print(data_matrix.shape)
        data_matrix.index = data_matrix.index.rename('region')
        full_matrix.append(data_matrix)

    full_matrix = pd.concat(full_matrix)
    data = full_matrix.reset_index().melt(id_vars=['region', 'clone', 'clone_id'], value_vars=data_matrix.columns[:-2],
                                          var_name='cell')
    return data
