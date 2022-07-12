import numpyro as npy
import numpyro.distributions as dist
import jax.numpy as jnp
import jax.nn


def model_jit_de(x, y, g, r, r2c):
    """GLMM model for differential expression

    Parameters
    ----------
    x : np.array
    y : np.array
    g : np.array
    r : np.array
    r2c : np.array

    Returns
    -------

    """
    with npy.plate("gene", 10):
        with npy.plate("clone", 2):
            beta_gene_clone = npy.sample("beta_gene", dist.Uniform(-10., 10.))
            sigma_gene_clone = npy.sample("sigma_gene_clone", dist.HalfNormal(0.05))
        with npy.plate("region", n_regions):
            alpha_gene_region = npy.sample("alpha_gene_region", dist.Normal(0., sigma_gene_clone[r2c]))
    # print(sigma_gene_clone.shape)
    rho = jnp.array([alpha_gene_region[i] for i in zip(r, g)])
    mu = jnp.array([beta_gene_clone[i] for i in zip(r2c[r], g)])  # alpha_gene[g] * (r2c[r]+0.0) + beta_gene[g] + rho
    with npy.plate("data", n_regions * 10):
        npy.sample("obs", dist.Poisson(rate=jnp.exp(mu + rho) * x), obs=y)


def model_jit_multiregion(x, y, g, r, r2c):
    """GLMM model for multiregional expression

    Parameters
    ----------
    x : np.array
    y : np.array
    g : np.array
    r : np.array
    r2c : np.array

    Returns
    -------
    """
    with npy.plate("gene", 10):
        with npy.plate("clone", n_clones):
            beta_gene_clone = npy.sample("beta_gene", dist.Uniform(-10., 10.))
            sigma_gene_clone = npy.sample("sigma_gene_clone", dist.HalfNormal(0.05))
        with npy.plate("region", n_regions):
            alpha_gene_region = npy.sample("alpha_gene_region", dist.Normal(0., sigma_gene_clone[r2c]))
    # print(sigma_gene_clone.shape)
    rho = jnp.array([alpha_gene_region[i] for i in zip(r, g)])
    mu = jnp.array([beta_gene_clone[i] for i in zip(r2c[r], g)])  # alpha_gene[g] * (r2c[r]+0.0) + beta_gene[g] + rho
    with npy.plate("data", n_regions * 10):
        npy.sample("obs", dist.Poisson(rate=jnp.exp(mu + rho) * x), obs=y)


def model_jit_composition():
    """GLMM model for regional cell composition

    Parameters
    ----------
    x : np.array
    y : np.array
    g : np.array
    r : np.array
    r2c : np.array

    Returns
    -------
    """
    with npy.plate("celltype", n_celltypes - 1):
        with npy.plate("clone", n_clones):
            beta_celltype_clone = npy.sample("beta_gene", dist.Uniform(-10., 10.))
            sigma_celltype_clone = npy.sample("sigma_celltype_clone", dist.HalfNormal(0.5))
        with npy.plate("region", n_regions):
            alpha_celltype_region = npy.sample("alpha_celltype_region", dist.Normal(0., sigma_celltype_clone[r2c]))
    rho = alpha_celltype_region
    mu = jnp.array([beta_celltype_clone[i] for i in r2c[r]])  # print(sigma_gene_clone.shape)

    lam = jnp.concatenate([mu + rho, jnp.ones((n_regions, 1))], axis=1)
    with npy.plate("data", n_regions):
        npy.sample("obs", dist.Multinomial(probs=jax.nn.softmax(lam)), obs=y)