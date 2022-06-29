import pymc3 as pm
import numpy as np

def gp_params_real_space(mu, rho, cov_funcs, tiles_axes):
    k1, k2 = cov_funcs[0](tiles_axes[0]).eval(), cov_funcs[1](tiles_axes[1]).eval()
    k = np.kron(k1,k2)
    chol = np.linalg.cholesky(k)
    mu_corr = np.linalg.cholesky(k) @ mu
    sigma_corr = np.diagonal(chol @ np.diag(rho2sigma(rho)) @ chol.T)
    return np.stack([mu_corr, sigma_corr])

def store_essential_params(var_approx, n_samples, n_factors, n_aug, scale):
    ss = np.arange(n_samples)
    fs = np.arange(n_factors - 1 + n_aug)

    cov_func1_f = [[pm.gp.cov.ExpQuad(1, ls=1*np.sqrt(scale)) for i in range(n_factors-1 + n_aug)] for s in range(n_samples)]
    cov_func2_f = [[pm.gp.cov.ExpQuad(1, ls=1*np.sqrt(scale)) for i in range(n_factors-1 + n_aug)] for s in range(n_samples)]
    
    mus = approx_hierarchical_errosion.bij.rmap(approx_hierarchical_errosion.params[0].eval())
    rhos = approx_hierarchical_errosion.bij.rmap(approx_hierarchical_errosion.params[1].eval())
    
    fields_f = {}
    for s in ss:
        for f in tqdm(fs):
            name = f'f_f_{f}_{s}'
            fields_f[name] = gp_params_real_space(mus[name + '_rotated_'], rhos[name + '_rotated_'], (cov_func2_f[s][0], cov_func2_f[s][1]), tiles_axes[s])
        name = f'lm_n_{s}'
        fields_f[name] = np.stack([mus[name + '_log__'], rho2sigma(rhos[name + '_log__'])])
    return fields_f

def sample_fields(params, sample, n_factors, n_aug=1, n_draws = 500, t=2, seed=1234):
    rng = np.random.default_rng(seed)
    s = sample
    fs = np.arange(n_factors - 1 + n_aug)
    F_matrix = np.stack([rng.normal(params[f'f_f_{f}_{s}'][0][None,:], params[f'f_f_{f}_{s}'][1][None,:],
                  size=(n_draws, len(params[f'f_f_0_{s}'][0]))) for f in fs] + [np.ones((n_draws, len(params[f'f_f_0_{s}'][0]))) * (-2.3)], axis=1)
    F_matrix = np.transpose(F_matrix, axes=(0,2,1))
    F_matrix = np.exp(F_matrix * t) / np.exp(F_matrix * t).sum(axis=-1)[:,:,None]
    return F_matrix

def sample_densities(params, sample, n_draws = 500, seed=1234):
    rng = np.random.default_rng(seed)
    s = sample
    lm_n = np.exp(rng.normal(params[f'lm_n_{s}'][0], params[f'lm_n_{s}'][1], size=(n_draws, len(params[f'lm_n_{s}'][0]))))
    return lm_n