import os

import numpy as np
from matplotlib import pyplot as plt


def mask_infeasible(mut_sample_list, scale, probability=0.6, critical_genes=False, plot=False):
    mask = []
    for i in range(len(mut_sample_list)):
        mut_sample_list[i].data_to_grid(scale_factor=scale, probability=0.6)
        t = np.array([s for s in mut_sample_list[i].gene_grid.values()])[:-3].sum(0)
        mask_infisiable = mut_sample_list[i].gene_grid["infeasible"] / t < 0.1
        mask_infisiable *= mut_sample_list[i].cell_grid > 5

        if critical_genes:
            if i == 0:
                mask_infisiable *= (
                        mut_sample_list[i].gene_grid["PTEN2mut"]
                        + mut_sample_list[i].gene_grid["LRP1Bmut"]
                        + mut_sample_list[i].gene_grid["NOB1wt"] <= 3
                )

        if plot:
            plt.figure(figsize=(8, 4))
            plt.imshow(mask_infisiable.T[::-1, :])

        mask.append(mask_infisiable.flatten())

    return mask


def generate_data(samples_list, genes, M, n_aug=1):
    n_samples = len(samples_list)
    n_genes = len(genes)

    iss_data = [
        np.transpose(np.array([samples_list[i].gene_grid[k] for k in genes]), [1, 2, 0]).reshape(-1, n_genes)
        for i in range(n_samples)
    ]

    tiles_axes = [samples_list[i].tile_axis for i in range(n_samples)]

    cells_counts = [samples_list[i].cell_grid.flatten() for i in range(n_samples)]
    sample_dims = [(int(tiles_axes[i][0][-1] + 1), int(tiles_axes[i][1][-1] + 1)) for i in range(n_samples)]
    n_factors = M.shape[0]
    n_aug = 1

    return {
        "iss_data": iss_data,
        "tiles_axes": tiles_axes,
        "cells_counts": cells_counts,
        "sample_dims": sample_dims,
        "n_factors": n_factors,
        "n_aug": n_aug,
        "tree_matrix": M,
        "n_samples": n_samples,
        "n_genes": n_genes,
        "genes": genes,
    }


def get_autobright_file(dirname):
    for file in os.listdir(dirname):
        if 'full_autobright.tif' in file:
            return file
    for file in os.listdir(dirname):
        if 'autobright.tif' in file or 'autbright.tif' in file:
            return file
    raise ValueError('No autobright file')
