import numpy as np
import pymc3 as pm


class Model(object):

    def __init__(self, n_bio_samples=1, sample_separate_flag=True, exp_genes=None, imm_genes=None):
        self.model = pm.Model()
        self.n_bio_samples = n_bio_samples
        self.sample_separate_flag = sample_separate_flag
        if exp_genes is not None:
            self.exp_genes = exp_genes
            self.n_exp_genes = len(exp_genes)
        if imm_genes is not None:
            self.imm_genes = imm_genes
            self.n_imm_genes = len(imm_genes)

    def compile_model(self):
        with pm.Model() as model:
            pass
        self.model = model

    def fit(self, n_iter=10000, learning_rate=0.01):
        with self.model:
            advi = pm.ADVI()
            self.mean_field_approx = advi.fit(n=n_iter, obj_optimizer=pm.adam(learning_rate=learning_rate))

    def sample(self, n_samples=300):
        if not hasattr(self, 'mean_field_approx'):
            raise ValueError('Use .fit() method first to obtain the mean-field approximation.')

        return self.mean_field_approx.sample(n_samples)

    def process_raw_samples(self, posterior_samples):
        if self.sample_separate_flag:
            E_data_pre = np.concatenate(
                [np.stack([posterior_samples['E_exp_{s}'] for s in range(self.n_bio_samples)], axis=3),
                 np.stack([posterior_samples['E_imm_{s}'] for s in range(self.n_bio_samples)], axis=3)],
                axis=1)
            mu_data_pre = np.concatenate(
                [np.stack([posterior_samples['r_mu_exp_{s}'] for s in range(self.n_bio_samples)], axis=2),
                 np.stack([posterior_samples['r_mu_imm_{s}'] for s in range(self.n_bio_samples)], axis=2)],
                axis=1)
        else:
            E_data_pre = np.concatenate([posterior_samples['E_exp'], posterior_samples['E_imm']], axis=1)
            mu_data_pre = np.concatenate([posterior_samples['r_mu_exp'], posterior_samples['r_mu_imm']], axis=1)

        E_data = E_data_pre
        E_names = np.concatenate([self.exp_genes, self.imm_genes])
        E_panel_from = np.array(['exp'] * self.n_exp_genes + ['imm'] * self.n_imm_genes)
        E_panel_color = np.array(['orange'] * self.n_exp_genes + ['skyblue'] * self.n_imm_genes)

        # chagne gene names in plots
        gene_name2good = {'PTPRC': 'CD45', 'CD274': 'PD-L1', 'Ki-67': 'MKI67'}
        for k in gene_name2good.keys():
            E_names[np.where(E_names == k)] = gene_name2good[k]

        # remove PTPRC_trans5

        ptprc_trans5_loc = np.where(E_names == 'PTPRC_trans5')[0]

        E_data = np.delete(E_data, ptprc_trans5_loc, axis=1)
        E_names = np.delete(E_names, ptprc_trans5_loc)
        E_panel_from = np.delete(E_panel_from, ptprc_trans5_loc)
        E_panel_color = np.delete(E_panel_color, ptprc_trans5_loc)

        return E_data, E_names, E_panel_from, E_panel_color
