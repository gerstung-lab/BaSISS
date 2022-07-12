import pymc as pm
import numpy as np
from tqdm import tqdm
from ._distr import rho2sigma


def gp_params_real_space(mu, rho, cov_funcs, tiles_axes):
    k1, k2 = cov_funcs[0](tiles_axes[0]).eval(), cov_funcs[1](tiles_axes[1]).eval()
    k1 += np.eye(k1.shape[0]) * 1e-7
    k2 += np.eye(k2.shape[0]) * 1e-7
    k1_chol = np.linalg.cholesky(k1)
    k2_chol = np.linalg.cholesky(k2)

    k_chol = np.kron(k1_chol, k2_chol)
    mu_corr = k_chol @ mu
    sigma_corr = np.diagonal(k_chol @ np.diag(rho2sigma(rho)) @ k_chol.T)
    return np.stack([mu_corr, sigma_corr])


def store_essential_params(var_approx, n_samples, n_factors, n_aug, tiles_axes, scale=3, lsn=1):
    ss = np.arange(n_samples)
    fs = np.arange(n_factors - 1 + n_aug)

    cov_func_f = pm.gp.cov.ExpQuad(1, ls=lsn * np.sqrt(scale))

    varname2slice = var_approx.ordering
    flat_mus = var_approx.params[0].eval()
    flat_rhos = var_approx.params[1].eval()

    # mus = var_approx.bij.rmap(var_approx.params[0].eval())
    # rhos = var_approx.bij.rmap(var_approx.params[1].eval())

    fields_f = {}
    for s in ss:
        for f in tqdm(fs):
            name = f'f_f_{f}_{s}'
            fields_f[name] = gp_params_real_space(flat_mus[varname2slice[name + '_rotated_'][1]],
                                                  flat_rhos[varname2slice[name + '_rotated_'][1]],
                                                  (cov_func_f, cov_func_f), tiles_axes[s]).reshape(2, int(
                tiles_axes[s][0][-1]) + 1, int(tiles_axes[s][1][-1]) + 1)
        name = f'lm_n_{s}'
        fields_f[name] = np.stack([flat_mus[varname2slice[name + '_log__'][1]],
                                   rho2sigma(flat_rhos[varname2slice[name + '_log__'][1]])]).reshape(2, int(
            tiles_axes[s][0][-1]) + 1, int(tiles_axes[s][1][-1]) + 1)
    return fields_f


def sample_fields(params, sample, n_factors, n_aug=1, n_draws=500, t=2, seed=1234):
    rng = np.random.default_rng(seed)
    s = sample
    fs = np.arange(n_factors - 1 + n_aug)
    dims = params[f'f_f_0_{s}'][0].shape
    F_matrix = np.stack([rng.normal(params[f'f_f_{f}_{s}'][0][None, :, :], params[f'f_f_{f}_{s}'][1][None, :, :],
                                    size=(n_draws, dims[0], dims[1])) for f in fs] + [
                            np.ones((n_draws, dims[0], dims[1])) * (-2.3)], axis=1)
    F_matrix = np.transpose(F_matrix, axes=(0, 2, 3, 1))
    F_matrix = np.exp(F_matrix * t) / np.exp(F_matrix * t).sum(axis=-1)[:, :, :, None]
    return F_matrix


def sample_densities(params, sample, n_draws=500, seed=1234):
    rng = np.random.default_rng(seed)
    s = sample
    dims = params[f'lm_n_{s}'][0].shape
    lm_n = np.exp(rng.normal(params[f'lm_n_{s}'][0], params[f'lm_n_{s}'][1], size=(n_draws, dims[0], dims[1])))
    return lm_n


def sample_essential(params, n_factors, samples=[0], n_draws=500, seed=42, n_aug=1, t=2):
    output = {}
    for s in tqdm(samples):
        output[f'F_{s}'] = sample_fields(params, sample=s, n_factors=n_factors, n_aug=n_aug, n_draws=n_draws, t=t,
                                         seed=seed)
        output[f'lm_n_{s}'] = sample_densities(params, sample=s, n_draws=n_draws, seed=seed)
    return output
