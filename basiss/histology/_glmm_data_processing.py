import numpy as np
import pandas as pd
from basiss.utils.stats import fdr
from scipy.stats import chi2


def signif_comparisons(diff_data, signal_data, comparisons, fdr_th=0.1, pre_test_cut=0.4):
    signif_gene_pval = {}
    all_genes = np.array(signal_data.index.to_frame()['gene'].astype('category').cat.categories.to_list())
    for comp in comparisons:
        signal_data_sub = signal_data[signal_data['clone_id'].isin(comp)]
        diff_data_sub = diff_data[:, comp, :]
        mask = (signal_data_sub['value'].sum(level=0) / signal_data_sub['n_nucl'].sum(level=0) * signal_data_sub[
            'n_nucl'].mean(level=0)) > pre_test_cut
        delta = diff_data_sub[:, 0, :] - diff_data_sub[:, 1, :]
        z = delta.mean(0) / delta.std(0)
        p = chi2.sf(z ** 2, 1)

        fdr_val = p.copy()
        fdr_val[mask] = fdr(p[mask])
        fdr_val[~mask] = 1

        f = pd.DataFrame(diff_data_sub.mean(0).T, index=all_genes, columns=[0, 1])
        f = f[fdr_val < fdr_th]

        for singif_gene in list(np.where(fdr_val < fdr_th)[0]):
            if singif_gene not in signif_gene_pval.keys():
                signif_gene_pval[singif_gene] = {}
                signif_gene_pval[singif_gene][tuple(comp)] = fdr_val[singif_gene]
            else:
                signif_gene_pval[singif_gene][tuple(comp)] = fdr_val[singif_gene]
    return signif_gene_pval
