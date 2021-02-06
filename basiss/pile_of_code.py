import copy
import os
import pickle as pkl
import re
import subprocess
from copy import copy, deepcopy
from itertools import chain

import cv2
import cv2 as cv
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy.interpolate as si
import theano
import tifffile
from scipy import stats
from skimage import exposure
from skimage import transform
from svgpathtools import svg2paths
from theano import tensor as tt
from tqdm import tqdm


class GridWarpAlignment:
    def __init__(self, warp_matrix_file, source_coords, resizing_params, small_img_source, small_img_target,
                 approx=100):
        self.warp_file = warp_matrix_file
        self.source_coords = source_coords
        self.resizing_params = resizing_params
        self.img_source = small_img_source
        self.img_target = small_img_target
        self.approx = approx

        self.img_source_dim = self.img_source.shape
        self.img_target_dim = self.img_target.shape

        self.n_spl = None
        self.trans_matrix = self._read_file(warp_matrix_file)

        self.norm_matrix = np.array(np.meshgrid(
            np.linspace(-(self.img_source_dim[1] / (self.n_spl - 3)),
                        self.img_source_dim[1] + (self.img_source_dim[1] / (self.n_spl - 3)), self.n_spl),
            np.linspace(-(self.img_source_dim[0] / (self.n_spl - 3)),
                        self.img_source_dim[0] + (self.img_source_dim[0] / (self.n_spl - 3)), self.n_spl)))

    def _to_float(self, x):
        if x == '':
            return ''
        else:
            return float(x)

    def _read_file(self, filename):
        with open(filename, "r") as file:
            raw_transform_file = file.readlines()
            raw_transform = {"X": [], "Y": []}

            for i in range(len(raw_transform_file)):
                if raw_transform_file[i] == "X Coeffs -----------------------------------\n":
                    c = i + 1
                    while raw_transform_file[c] != "\n":
                        raw_transform["X"].append(
                            list(map(lambda x: self._to_float(x), raw_transform_file[c].strip().split()))
                        )
                        c += 1
                elif raw_transform_file[i] == "Y Coeffs -----------------------------------\n":
                    c = i + 1
                    while c < len(raw_transform_file) and raw_transform_file[c] != "\n":
                        raw_transform["Y"].append(
                            list(map(lambda x: self._to_float(x), raw_transform_file[c].strip().split()))
                        )
                        c += 1

            raw_transform["X"] = list(chain(*(raw_transform["X"])))
            raw_transform["Y"] = list(chain(*(raw_transform["Y"])))

            raw_transform["X"] = list(filter(lambda x: x != "", raw_transform["X"]))
            raw_transform["Y"] = list(filter(lambda x: x != "", raw_transform["Y"]))

            self.n_spl = int(np.array(raw_transform["X"]).shape[0] ** 0.5)
        return np.array(
            [
                np.array(raw_transform["X"]).reshape(self.n_spl, self.n_spl),
                np.array(raw_transform["Y"]).reshape(self.n_spl, self.n_spl),
            ]
        )

    def _get_spline_coords_interpolation(self, approx=None):

        if type(approx) == type(None):
            approx = self.approx

        tx = np.clip(np.arange(self.n_spl + 3 + 1) - 3, 0, self.n_spl - 3)

        intact_x = si.bisplev(
            np.linspace(0, self.n_spl - 3, approx),
            np.linspace(0, self.n_spl - 3, approx),
            (tx, tx, self.norm_matrix[:, :, :].ravel(), 3, 3),
        )
        intact_y = si.bisplev(
            np.linspace(0, self.n_spl - 3, approx),
            np.linspace(0, self.n_spl - 3, approx),
            (tx, tx, np.transpose(self.norm_matrix[::-1, :, :], (0, 2, 1)).ravel(), 3, 3),
        )

        # print(intact_x)
        interp_intact_x = si.interp1d(
            intact_x[0], np.linspace(0, self.n_spl - 3, approx), kind="cubic", fill_value="extrapolate"
        )
        interp_intact_y = si.interp1d(
            intact_y[0], np.linspace(0, self.n_spl - 3, approx), kind="cubic", fill_value="extrapolate"
        )

        return interp_intact_x, interp_intact_y

    def resize_coords(self, source_coords=None, resize_from_to=[[1, 1], [1, 1]]):

        if type(source_coords) == type(None):
            source_coords = self.source_coords

        M = np.array([[resize_from_to[1][0] / resize_from_to[0][0], 0],
                      [0, resize_from_to[1][1] / resize_from_to[0][1]]])

        source_coords_rescaled = np.array(source_coords).T @ M

        return source_coords_rescaled

    def warp_coords(self, source_coords=None):

        if source_coords is None:
            source_coords = self.source_coords

        tx = np.clip(np.arange(self.n_spl + 3 + 1) - 3, 0, self.n_spl - 3)

        rescaled_coords = self.resize_coords(self.source_coords, resize_from_to=self.resizing_params['source']).T
        X, Y = rescaled_coords[0], rescaled_coords[1]
        # print(X, Y)
        interp_intact_x, interp_intact_y = self._get_spline_coords_interpolation()

        coord_spline_space_X = interp_intact_x(X)
        coord_spline_space_Y = interp_intact_y(Y)

        transformed_coord_X = list(map(lambda x: si.bisplev(coord_spline_space_Y[x],
                                                            coord_spline_space_X[x],
                                                            (tx, tx, self.trans_matrix[:, :, :].ravel(), 3, 3)),
                                       list(range(len(X)))))

        transformed_coord_Y = list(map(lambda x: si.bisplev(coord_spline_space_X[x],
                                                            coord_spline_space_Y[x],
                                                            (tx, tx, np.transpose(self.trans_matrix[::-1, :, :],
                                                                                  (0, 2, 1)).ravel(), 3, 3)),
                                       list(range(len(X)))))

        resized_transformed_coords = self.resize_coords([transformed_coord_X, transformed_coord_Y],
                                                        [self.resizing_params['target'][1],
                                                         self.resizing_params['target'][0]]).T

        return (resized_transformed_coords[0], resized_transformed_coords[1])

    def plot_transformation(self, approx=None, alpha=0.1, before=False):

        if type(approx) == type(None):
            approx = self.approx

        transformed_coord_X, transformed_coord_Y = self.warp_coords()
        transformed_coord_X, transformed_coord_Y = self.resize_coords([transformed_coord_X,
                                                                       transformed_coord_Y],
                                                                      resize_from_to=self.resizing_params['target']).T

        # plt.figure(figsize=(20, 15))
        tx = np.clip(np.arange(self.n_spl + 3 + 1) - 3, 0, self.n_spl - 3)
        plt.imshow(self.img_target)

        if before:
            rescaled_coords = self.resize_coords(self.source_coords,
                                                 resize_from_to=self.resizing_params['source']).T
            plt.scatter(rescaled_coords[0], rescaled_coords[1], alpha=alpha)

            plt.plot(np.linspace(self.norm_matrix[0].min(), self.norm_matrix[0].max(), approx),
                     si.bisplev(np.linspace(0, self.n_spl - 3, approx), np.linspace(0, self.n_spl - 3, approx),
                                (tx, tx, np.transpose(self.norm_matrix[::-1, :, :], (0, 2, 1)).ravel(), 3, 3)), 'k-',
                     alpha=0.3)
            plt.plot(si.bisplev(np.linspace(0, self.n_spl - 3, approx), np.linspace(0, self.n_spl - 3, approx),
                                (tx, tx, self.norm_matrix[::, :, :].ravel(), 3, 3)),
                     np.linspace(self.norm_matrix[1].min(), self.norm_matrix[1].max(), approx), 'k-', alpha=0.3)
        else:
            plt.scatter(transformed_coord_X, transformed_coord_Y, alpha=alpha)

            plt.plot(np.linspace(self.trans_matrix[0].min(), self.trans_matrix[0].max(), approx),
                     si.bisplev(np.linspace(0, self.n_spl - 3, approx), np.linspace(0, self.n_spl - 3, approx),
                                (tx, tx, np.transpose(self.trans_matrix[::-1, :, :], (0, 2, 1)).ravel(), 3, 3)), 'k-',
                     alpha=0.3)
            plt.plot(si.bisplev(np.linspace(0, self.n_spl - 3, approx), np.linspace(0, self.n_spl - 3, approx),
                                (tx, tx, self.trans_matrix[::, :, :].ravel(), 3, 3)),
                     np.linspace(self.trans_matrix[1].min(), self.trans_matrix[1].max(), approx), 'k-', alpha=0.3)
        plt.axvline(x=0, color='red')
        plt.axvline(x=self.img_target_dim[1], color='red')
        plt.axhline(y=0, color='red')
        plt.axhline(y=self.img_target_dim[0], color='red')
        plt.show()


class Sample:
    def __init__(self, iss_data, image, cell_data=None, masks_svg=None):
        self._iss_df_raw = pd.read_csv(iss_data)
        self.image = image
        self.spatial_dims = self.get_img_size(self.image)
        self.data = self.standardise_data()
        self.genes = np.unique(self.data["Gene"])
        self.cell_data = cell_data
        self.masks_svg = masks_svg

        self.gene_grid = None
        self.tile_axis = None
        self.grid_params = None
        self.cell_grid = None
        self.cell_types = None
        self.error_flag = False
        if self.cell_data is not None:
            self._cell_density_df_raw = pd.read_csv(self.cell_data)
            self.cellpos = np.array(self._cell_density_df_raw[["x", "y"]])
            if "cell_type" in self._cell_density_df_raw.columns:
                self.cell_types = np.array(self._cell_density_df_raw["cell_type"])
            else:
                self.cell_types = np.array(["core"] * self.cellpos.shape[0])

        else:
            self._cell_density_df_raw = None
            self.cellpos = None

        if masks_svg is not None:
            self.ducts = self._get_ducts(masks_svg)

    @staticmethod
    def get_img_size(filename):
        image_size = list(
            map(lambda x: int(x), re.findall("\ (\d+)x(\d+)\ ", subprocess.getoutput("identify " + filename))[-1])
        )
        return image_size

    def standardise_data(self):
        aliases = {
            "Gene": "Gene",
            "Name": "Gene",
            "gene": "Gene",
            "name": "Gene",
            "PosX": "PosX",
            "PosY": "PosY",
            "X": "PosX",
            "Y": "PosY",
            "global_X_pos": "PosX",
            "global_Y_pos": "PosY",
        }

        data = {}
        for col in self._iss_df_raw.columns:
            if col in aliases.keys():
                data[aliases[col]] = np.array(list(self._iss_df_raw[col]))

        if "Probability" in self._iss_df_raw.columns:
            self.iss_probability = np.array(self._iss_df_raw["Probability"])
        else:
            self.iss_probability = None

        return data

    def data_to_grid(self, scale_factor=4, gene_list="all", probability=None):

        # TODO creates parameters outside of the constructor, deal with it if necessery
        self.grid_params = (np.array(self.spatial_dims) / 1000 * scale_factor).astype(int)
        x_step = self.spatial_dims[0] / (self.grid_params[0] - 1)
        y_step = self.spatial_dims[1] / (self.grid_params[1] - 1)

        if gene_list == "all":
            gene_list = self.genes

        self.gene_grid = {}

        self.tile_axis = [np.arange(self.grid_params[0])[:, None], np.arange(self.grid_params[1])[:, None]]

        for gene in tqdm(gene_list):
            arr = np.zeros((self.grid_params[0], self.grid_params[1]))
            if (probability is not None) and (self.iss_probability is not None):
                cur_gene = np.where(np.logical_and(self.data["Gene"] == gene, self.iss_probability >= probability))
            else:
                cur_gene = np.where(self.data["Gene"] == gene)
            tiles = np.array(
                [(self.data["PosX"][cur_gene] // x_step).astype(int),
                 (self.data["PosY"][cur_gene] // y_step).astype(int)]).T
            k_id, v = np.unique(tiles, return_counts=True, axis=0)

            for i in range(len(v)):
                try:
                    arr[tuple(k_id[i, :])] = v[i]
                except IndexError:
                    # TODO: raise warnigns
                    self.error_flag = True

            self.gene_grid[gene] = arr

        if self.cell_data is not None:
            arr = np.zeros((self.grid_params[0], self.grid_params[1]))
            tiles = np.array([(self.cellpos[:, 0] // x_step).astype(int), (self.cellpos[:, 1] // y_step).astype(int)]).T
            k_id, v = np.unique(tiles, return_counts=True, axis=0)

            for i in range(len(v)):
                try:
                    arr[tuple(k_id[i, :])] = v[i]
                except IndexError:
                    self.error_flag = True

            self.cell_grid = arr

        if self.error_flag:
            print("Some of the points were out of bound")

    @staticmethod
    def _img_to_float(img):
        return cv.normalize(img, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)

    @staticmethod
    def _float_to_img(fimg, original_numpy_dtype):
        cv_np_map = {
            np.dtype("uint8"): cv.CV_8U,
            np.dtype("uint16"): cv.CV_16U,
            np.dtype("int8"): cv.CV_8S,
            np.dtype("int16"): cv.CV_16S,
        }
        cv_dtype = cv_np_map[original_numpy_dtype]
        alpha = np.iinfo(original_numpy_dtype).min
        beta = np.iinfo(original_numpy_dtype).max
        return cv.normalize(fimg, None, alpha, beta, cv.NORM_MINMAX, cv_dtype)

    def _diff_of_gaus(self, img, low_sigma: int = 5, high_sigma: int = 9):
        # TODO replace with difference of kernels
        original_dtype = copy.copy(img.dtype)
        if original_dtype != np.float32:
            fimg = self._img_to_float(img)
        else:
            fimg = img
        low_sigma = cv.GaussianBlur(fimg, (0, 0), sigmaX=low_sigma, dst=None, sigmaY=low_sigma)
        high_sigma = cv.GaussianBlur(fimg, (0, 0), sigmaX=high_sigma, dst=None, sigmaY=high_sigma)
        diff = low_sigma - high_sigma
        del low_sigma, high_sigma
        if original_dtype == np.float32:
            # do not need to convert back already in float32
            return diff
        else:
            return self._float_to_img(diff, original_dtype)

    def _preprocess_image(self, img):
        # TODO try to use opencv retina module instead of exposure.rescale_intensity, the last thing to do
        p2, p98 = np.percentile(img, (2, 98))
        processed_img = exposure.rescale_intensity(img, in_range=(p2, p98))

        if processed_img.dtype != np.uint8:
            processed_img = cv.normalize(processed_img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

        processed_img = self._diff_of_gaus(processed_img, 3, 5)

        return processed_img

    @staticmethod
    # TODO depricated
    def _pickle_keypoints(point):
        return cv.KeyPoint, (*point.pt, point.size, point.angle, point.response, point.octave, point.class_id)

    def _find_features(self, img):
        processed_img = self._preprocess_image(img)
        if processed_img.max() == 0:
            return [], []
        # detector = cv.MSER_create()
        detector = cv.FastFeatureDetector_create(1, True)
        descriptor = cv.xfeatures2d.DAISY_create()
        kp = detector.detect(processed_img)

        # get 1000 best points based on feature detector response
        if len(kp) <= 3000:
            pass
        else:
            kp = sorted(kp, key=lambda x: x.response, reverse=True)[:3000]

        kp, des = descriptor.compute(processed_img, kp)

        if kp is None or len(kp) < 3:
            return [], []
        if des is None or len(des) < 3:
            return [], []

        # fix problem with pickle
        temp_kp_storage = []
        for point in kp:
            temp_kp_storage.append((point.pt, point.size, point.angle, point.response, point.octave, point.class_id))
        return temp_kp_storage, des

    def _match_features(self, img1_kp_des, img2_kp_des):
        kp1, des1 = img1_kp_des
        kp2, des2 = img2_kp_des

        matcher = cv.FlannBasedMatcher_create()
        matches = matcher.knnMatch(des2, des1, k=2)

        # Filter out unreliable points
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append(m)

        print("good matches", len(good), "/", len(matches))
        if len(good) < 3:
            return None
        # convert keypoints to format acceptable for estimator
        src_pts = np.float32([kp1[m.trainIdx][0] for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.queryIdx][0] for m in good]).reshape(-1, 1, 2)

        # find out how images shifted (compute affine transformation)
        affine_transform_matrix, mask = cv.estimateAffinePartial2D(dst_pts, src_pts, method=cv.RANSAC, confidence=0.99)
        return affine_transform_matrix

    def transform_points2background(self, img_file, upsampling=5):
        img1 = tifffile.imread(img_file)
        img2 = tifffile.imread(self.image)

        print('image load complete')

        img1_small = cv2.resize(img1, (int(img1.shape[1] / upsampling), int(img1.shape[0] / upsampling)))
        self._scaffold_image = img1_small
        img2_small = cv2.resize(img2, (int(img2.shape[1] / upsampling), int(img2.shape[0] / upsampling)))
        self._original_image = img2_small

        warp_matirx = self._match_features(self._find_features(img2_small), self._find_features(img1_small))
        warp_matirx = transform.EuclideanTransform(
            matrix=np.concatenate([warp_matirx[::, :], np.array([0, 0, 1])[None, :]])).params
        self._tform = warp_matirx
        # don't forget to scale
        warp_matirx = np.array([[upsampling, 0., 0.],
                                [0., upsampling, 0.],
                                [0., 0., 1.]]) @ \
                      np.linalg.inv(warp_matirx) @ \
                      np.array([[1 / upsampling, 0., 0.],
                                [0., 1 / upsampling, 0.],
                                [0., 0., 1.]])

        data = warp_matirx @ np.array([self.data['PosX'], self.data['PosY'], np.ones(self.data['PosY'].shape[0])])
        self.data['PosX'] = data[0, :]
        self.data['PosY'] = data[1, :]

        self.image = img_file
        self.spatial_dims = self.get_img_size(self.image)

    def diagnostic_image_overlay(self):
        img1_small = self._scaffold_image
        img2_small = transform.warp(self._original_image, self._tform, output_shape=img1_small.shape)

        img_RGB = np.zeros(
            (max(img1_small.shape[0], img2_small.shape[0]), max(img1_small.shape[1], img2_small.shape[1]), 3))
        img_RGB[:img1_small.shape[0], :img1_small.shape[1], 0] = img1_small / (img1_small.max() / 2)
        img_RGB[:img2_small.shape[0], :img2_small.shape[1], 1] = img2_small / (img2_small.max() / 2)

        plt.figure(figsize=(16, 16))
        plt.imshow(img_RGB)

    def update_coords(self, warp_matrix_file, resizing_params, small_img_source, small_img_target):

        warp = GridWarpAlignment(
            warp_matrix_file=warp_matrix_file,
            source_coords=np.stack([self.data["PosX"], self.data["PosY"]]),
            resizing_params=resizing_params,
            small_img_source=small_img_source,
            small_img_target=small_img_target,
        )

        self.data["PosX"], self.data["PosY"] = warp.warp_coords()

        warp = GridWarpAlignment(
            warp_matrix_file=warp_matrix_file,
            source_coords=self.cellpos.T,
            resizing_params=resizing_params,
            small_img_source=small_img_source,
            small_img_target=small_img_target,
        )

        self.cellpos[:, 0], self.cellpos[:, 1] = warp.warp_coords()

        self.spatial_dims = resizing_params["target"][0]

    def _get_ducts(self, svg_data):
        svg_path = svg_data

        paths, attributes = svg2paths(svg_path)

        size_source = self.get_img_size(self.image)
        size_svg = self.get_img_size(svg_path)
        scale = (np.array(size_source) / size_svg).mean(axis=0)
        NUM_SAMPLES = 1000
        paths_interpol = []
        for path in paths:
            path_interpol = []
            for i in range(NUM_SAMPLES):
                path_interpol.append(path.point(i / (float(NUM_SAMPLES) - 1)))
            paths_interpol.append(np.array([[j.real for j in path_interpol], [j.imag for j in path_interpol]]).T)

        for i in range(len(paths)):
            paths_interpol[i] = (np.concatenate([paths_interpol[i], np.ones(NUM_SAMPLES)[:, None]], axis=1)
                                @ np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]]))[:, :-1]

        return {"paths": paths_interpol, "linetype": [attributes[i]["class"] for i in range(len(paths))]}

    def filter_by_ducts(self, subset=None):
        if self.masks_svg is None:
            print('no duct info')
            return None
        else:
            paths_interpol = np.array(self.ducts['paths'])

            if subset is None:
                subset = [True] * len(paths_interpol)

            paths_matplot = [mpltPath.Path(paths_interpol[subset][i]) for i in range(len(paths_interpol[subset]))]
            ifcontains_mut = [path.contains_points(np.array([self.data['PosX'], self.data['PosY']]).T) for path in
                              paths_matplot]
            ifcontains_mut = np.array(ifcontains_mut).sum(axis=0).astype(bool)

            ifcontains_cell = [path.contains_points(self.cellpos) for path in paths_matplot]
            ifcontains_cell = np.array(ifcontains_cell).sum(axis=0).astype(bool)

            NewSample = deepcopy(self)
            for k in NewSample.data.keys():
                NewSample.data[k] = NewSample.data[k][ifcontains_mut]
            NewSample.iss_probability = NewSample.iss_probability[ifcontains_mut]

            NewSample.cellpos = NewSample.cellpos[ifcontains_cell]
            NewSample.cell_types = NewSample.cell_types[ifcontains_cell]
            return NewSample

    def add_gene_data(self, sample):
        for k in self.data.keys():
            self.data[k] = np.concatenate([self.data[k], sample.data[k]])

        self.iss_probability = np.concatenate([self.iss_probability, sample.iss_probability])

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


def generate_data4model(samples_list, genes, M, n_aug=1):
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


# code for primary data preprocessing
# fusing split images
new_labels = ["d3_Top", "d3_Bottom", "m2_Top", "m2_Bottom"]
old_labels = ["7", "7", "8", "8"]

old_folders = ["Validation_Mutation", "Validation_Mutation", "Mutation", "Mutation"]
old_prefix = ["Valid_", "Valid_", "", ""]

masks_svgs = ['../contours/R1_PD9694d_contour_only-01.svg', '../contours/R1_PD9694d_contour_only-01.svg', None, None]

sample_list = []
for i in range(len(new_labels)):
    sample_list.append(Sample(
        iss_data=f'/nfs/research1/gerstung/mg617/ISS_data/Mut_PD9694{new_labels[i]}/decoding/Mut_PD9694{new_labels[i]}_GMMdecoding.csv',
        image=f'/nfs/research1/gerstung/artem/projects/spatial_bayes/GMM_restored_DAPI/Mut_PD9694{new_labels[i]}.tif',
        cell_data=f'../ultra_hd_segmentation/Patient_2085/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}_segmented/{old_prefix[i]}2805_{old_labels[i]}_cellpos.csv',
        masks_svg=masks_svgs[i]))

for i, sample in enumerate(sample_list):

    single_flag = True
    if "".join(os.listdir(f"../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/")).find(
            "full") != -1:
        single_flag = False

    for file in os.listdir(f"../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/"):
        if single_flag:
            if file.find("autobright.tif") != -1 or file.find("autbright.tif") != -1:
                break
        else:
            if file.find("full_autobright.tif") != -1:
                break

    sample.transform_points2background(
        f'../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/{file}', upsampling=15)

sample_list[0].add_gene_data(sample_list[1])
sample_list[2].add_gene_data(sample_list[3])

d3_sample = sample_list[0]
m2_sample = sample_list[2]

df_dict = {
    "Name": d3_sample.data["Gene"],
    "Code": np.nan,
    "Probability": d3_sample.iss_probability,
    "X": d3_sample.data["PosX"],
    "Y": d3_sample.data["PosY"],
    "Tile": np.nan,
}
pd.DataFrame(df_dict).to_csv(f"../GMM_decoding/remapped/Mut_PD9694d3_composed_GMMdecoding_remapped.csv", index=False)

df_dict = {
    "Name": m2_sample.data["Gene"],
    "Code": np.nan,
    "Probability": m2_sample.iss_probability,
    "X": m2_sample.data["PosX"],
    "Y": m2_sample.data["PosY"],
    "Tile": np.nan,
}
pd.DataFrame(df_dict).to_csv(f"../GMM_decoding/remapped/Mut_PD9694m2_composed_GMMdecoding_remapped.csv", index=False)
# main mut data
new_labels = ['a3', 'c3', 'l2']
old_labels = ['2', '5', '6.1']

old_folders = ['Validation_Mutation', 'Validation_Mutation', 'Mutation']
old_prefix = ['Valid_', 'Valid_', '']
masks_svgs = [
    "../contours/R1_PD9694a_contours_only-01.svg",
    "../contours/R1_PD9694c_contours_only_corrected-01.svg",
    "../contours/PD9694x_contours_only-01.svg",
]
mut_sample_list = []

for i in range(len(new_labels)):
    mut_sample_list.append(Sample(
        iss_data=f'/nfs/research1/gerstung/mg617/ISS_data/Mut_PD9694{new_labels[i]}/decoding/Mut_PD9694{new_labels[i]}_GMMdecoding.csv',
        image=f'/nfs/research1/gerstung/artem/projects/spatial_bayes/GMM_restored_DAPI/Mut_PD9694{new_labels[i]}.tif',
        cell_data=f'../ultra_hd_segmentation/Patient_2085/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}_segmented/{old_prefix[i]}2805_{old_labels[i]}_cellpos.csv',
        masks_svg=masks_svgs[i]))

for i, sample in enumerate(mut_sample_list):
    single_flag = True
    if ("".join(os.listdir(f"../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/")).find("full")
            != -1
    ):
        single_flag = False

    for file in os.listdir(f"../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/"):
        if single_flag:
            if file.find("autobright.tif") != -1 or file.find("autbright.tif") != -1:
                break
        else:
            if file.find("full_autobright.tif") != -1:
                break
    print(f"../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/{file}")

    sample.transform_points2background(
        f"../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/{file}", upsampling=15
    )

    df_dict = {
        "Name": sample.data["Gene"],
        "Code": np.nan,
        "Probability": sample.iss_probability,
        "X": sample.data["PosX"],
        "Y": sample.data["PosY"],
        "Tile": np.nan,
    }
    pd.DataFrame(df_dict).to_csv(
        f"../GMM_decoding/remapped/Mut_PD9694{new_labels[i]}_GMMdecoding_remapped.csv", index=False
    )

mut_sample_list = [d3_sample] + mut_sample_list[:] + [m2_sample]

# validation data

new_labels = ['d2', 'a2', 'c2']
old_labels = ['7', '2', '5']
old_folders = ['Mutation', 'Mutation', 'Mutation']
old_prefix = ['', '', '']
masks_svgs = [None] * 4

val_sample_list = []
for i in range(len(new_labels)):
    val_sample_list.append(Sample(
        iss_data=f'/nfs/research1/gerstung/mg617/ISS_data/Mut_PD9694{new_labels[i]}/decoding/Mut_PD9694{new_labels[i]}_GMMdecoding.csv',
        image=f'/nfs/research1/gerstung/artem/projects/spatial_bayes/GMM_restored_DAPI/Mut_PD9694{new_labels[i]}.tif',
        cell_data=f'../ultra_hd_segmentation/Patient_2085/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}_segmented/{old_prefix[i]}2805_{old_labels[i]}_cellpos.csv',
        masks_svg=masks_svgs[i]))

for i, sample in enumerate(val_sample_list):

    single_flag = True
    if ''.join(os.listdir(f'../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/')).find(
            'full') != -1:
        single_flag = False

    for file in os.listdir(f'../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/'):
        if single_flag:
            if file.find('autobright.tif') != -1 or file.find('autbright.tif') != -1:
                break
        else:
            if file.find('full_autobright.tif') != -1:
                break

    sample.transform_points2background(
        f'../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/{file}', upsampling=15)

    df_dict = {'Name': sample.data['Gene'], 'Code': np.nan, 'Probability': sample.iss_probability,
               'X': sample.data['PosX'], 'Y': sample.data['PosY'], 'Tile': np.nan}
    pd.DataFrame(df_dict).to_csv(f'../GMM_decoding/remapped/Mut_PD9694{new_labels[i]}_GMMdecoding_remapped.csv',
                                 index=False)

# Exp and Immune import

Images = {
    "7": (
        cv2.imread(
            "../Globus/DAPI_2805/Validation_Mutation/Valid_2805_7/Valid_mut_2805_7_DAPI_full_autobright_mini.jpg",
            cv2.IMREAD_COLOR,
        ),
        cv2.imread("../Globus/DAPI_2805/Expression/2805_7/Exp2805_7_full_autobright_mini.jpg", cv2.IMREAD_COLOR),
        cv2.imread("../Globus/DAPI_2805/Immuno/2805_7/Imm2805_7_full_autobright_mini.jpg", cv2.IMREAD_COLOR),
    ),
    "2": (
        cv2.imread(
            "../Globus/DAPI_2805/Validation_Mutation/Valid_2805_2/Valid_Mut_2805_2_DAPI_base5_c1_ORG_autobright_mini.jpg",
            cv2.IMREAD_COLOR,
        ),
        cv2.imread("../Globus/DAPI_2805/Expression/2805_2/Exp2805_2_full_autobright_mini.jpg", cv2.IMREAD_COLOR),
        cv2.imread("../Globus/DAPI_2805/Immuno/2805_2/Imm2805_2_full_autobright_mini.jpg", cv2.IMREAD_COLOR),
    ),
    "5": (
        cv2.imread(
            "../Globus/DAPI_2805/Validation_Mutation/Valid_2805_5/Valid_Mut_2805_5_DAPI_base5_c1_ORG_autobright_mini.jpg",
            cv2.IMREAD_COLOR,
        ),
        cv2.imread("../Globus/DAPI_2805/Expression/2805_5/Exp2805_5_full_autobright_mini.jpg", cv2.IMREAD_COLOR),
        cv2.imread("../Globus/DAPI_2805/Immuno/2805_5/Imm2805_5_full_autobright_mini.jpg", cv2.IMREAD_COLOR),
    ),
    "4": (
        cv2.imread("../Globus/DAPI_2805/Mutation/2805_4/2805_4_DAPI_full_autobright_mini.jpg", cv2.IMREAD_COLOR),
        cv2.imread("../Globus/DAPI_2805/Expression/2805_4/Exp2805_4_full_autobright_mini.jpg", cv2.IMREAD_COLOR),
        cv2.imread("../Globus/DAPI_2805/Immuno/2805_4/Imm2805_4_full_autobright_mini.jpg", cv2.IMREAD_COLOR),
    ),
    "6.1": (
        cv2.imread("../Globus/DAPI_2805/Mutation/2805_6.1/2805_6.1_full_autobright_mini.jpg", cv2.IMREAD_COLOR),
        cv2.imread("../Globus/DAPI_2805/Expression/2805_6.1/Exp2805_6.1_full_autobright_mini.jpg", cv2.IMREAD_COLOR),
        cv2.imread("../Globus/DAPI_2805/Immuno/2805_6.1/Imm2805_6.1_full_autobright_mini.jpg", cv2.IMREAD_COLOR),
    ),
    "8": (
        cv2.imread("../Globus/DAPI_2805/Mutation/2805_8/2805_8_DAPI_full_autobright_mini.jpg", cv2.IMREAD_COLOR),
        cv2.imread("../Globus/DAPI_2805/Expression/2805_8/Exp2805_8_full_autobright_mini.jpg", cv2.IMREAD_COLOR),
        cv2.imread("../Globus/DAPI_2805/Immuno/2805_8/Imm2805_8_full_autobright_mini.jpg", cv2.IMREAD_COLOR),
    ),
}
Matrices = {
    "7": (
        "../Globus/DAPI_2805/alignment/Expression/Valid_mut_2805_7_DAPI_full_autobright_mini_inverse_transf.txt",
        "../Globus/DAPI_2805/alignment/Immuno/Valid_mut_2805_7_DAPI_full_autobright_mini_inverse_transf.txt",
    ),
    "2": (
        "../Globus/DAPI_2805/alignment/Expression/Valid_Mut_2805_2_DAPI_base5_c1_ORG_autobright_mini_inverse_transf.txt",
        "../Globus/DAPI_2805/alignment/Immuno/Valid_Mut_2805_2_DAPI_base5_c1_ORG_autobright_mini_inverse_transf.txt",
    ),
    "5": (
        "../Globus/DAPI_2805/alignment/Expression/Valid_Mut_2805_5_DAPI_base5_c1_ORG_autobright_mini_inverse_transf.txt",
        "../Globus/DAPI_2805/alignment/Immuno/Valid_Mut_2805_5_DAPI_base5_c1_ORG_autobright_mini_inverse_transf.txt",
    ),
    "4": (
        "../Globus/DAPI_2805/alignment/Expression/2805_4_DAPI_full_autobright_mini_inverse_transf.txt",
        "../Globus/DAPI_2805/alignment/Immuno/2805_4_DAPI_full_autobright_mini_inverse_transf.txt",
    ),
    "6.1": (
        "../Globus/DAPI_2805/alignment/Expression/2805_6.1_full_autobright_mini_inverse_transf.txt",
        "../Globus/DAPI_2805/alignment/Immuno/2805_6.1_full_autobright_mini_inverse_transf.txt",
    ),
    "8": (
        "../Globus/DAPI_2805/alignment/Expression/2805_8_DAPI_full_autobright_mini_inverse_transf.txt",
        "../Globus/DAPI_2805/alignment/Immuno/2805_8_DAPI_full_autobright_mini_inverse_transf.txt",
    ),
}

Im_Sources = {
    "7": (
        "../Globus/DAPI_2805/Validation_Mutation/Valid_2805_7/Valid_mut_2805_7_DAPI_full_autobright.tif",
        "../Globus/DAPI_2805/Expression/2805_7/Exp2805_7_full_autobright.tif",
        "../Globus/DAPI_2805/Immuno/2805_7/Imm2805_7_full_autobright.tif",
    ),
    "2": (
        "../Globus/DAPI_2805/Validation_Mutation/Valid_2805_2/Valid_Mut_2805_2_DAPI_base5_c1_ORG_autobright.tif",
        "../Globus/DAPI_2805/Expression/2805_2/Exp2805_2_full_autobright.tif",
        "../Globus/DAPI_2805/Immuno/2805_2/Imm2805_2_full_autobright.tif",
    ),
    "5": (
        "../Globus/DAPI_2805/Validation_Mutation/Valid_2805_5/Valid_Mut_2805_5_DAPI_base5_c1_ORG_autobright.tif",
        "../Globus/DAPI_2805/Expression/2805_5/Exp2805_5_full_autobright.tif",
        "../Globus/DAPI_2805/Immuno/2805_5/Imm2805_5_full_autobright.tif",
    ),
    "4": (
        "../Globus/DAPI_2805/Mutation/2805_4/2805_4_DAPI_full_autobright.tif",
        "../Globus/DAPI_2805/Expression/2805_4/Exp2805_4_full_autobright.tif",
        "../Globus/DAPI_2805/Immuno/2805_4/Imm2805_4_full_autobright.tif",
    ),
    "6.1": (
        "../Globus/DAPI_2805/Mutation/2805_6.1/2805_6.1_full_autobright.tif",
        "../Globus/DAPI_2805/Expression/2805_6.1/Exp2805_6.1_full_autobright.tif",
        "../Globus/DAPI_2805/Immuno/2805_6.1/Imm2805_6.1_full_autobright.tif",
    ),
    "8": (
        "../Globus/DAPI_2805/Mutation/2805_8/2805_8_DAPI_full_autobright.tif",
        "../Globus/DAPI_2805/Expression/2805_8/Exp2805_8_full_autobright.tif",
        "../Globus/DAPI_2805/Immuno/2805_8/Imm2805_8_full_autobright.tif",
    ),
}
# exp samples
new_labels = ['d2', 'a2', 'c2', 'l2', 'm2']
old_labels = ['7', '2', '5', '6.1', '8']
version = ['', '', '', '', '']

old_folders = ['Expression', 'Expression', 'Expression', 'Expression', 'Expression']
old_prefix = ['', '', '', '', '']
masks_svgs = ['../contours/R1_PD9694d_contour_only-01.svg',
              '../contours/R1_PD9694a_contours_only-01.svg',
              '../contours/R1_PD9694c_contours_only_corrected-01.svg',
              '../contours/PD9694x_contours_only-01.svg',
              None]

exp_sample_list = []
for i in range(len(new_labels)):
    exp_sample_list.append(Sample(
        iss_data=f'/nfs/research1/gerstung/mg617/ISS_data/Exp_PD9694{new_labels[i]}/decoding/Exp_PD9694{new_labels[i]}_GMMdecoding.csv',
        image=f'/nfs/research1/gerstung/artem/projects/spatial_bayes/GMM_restored_DAPI/Exp_PD9694{new_labels[i]}.tif',
        cell_data=f'../ultra_hd_segmentation/Patient_2085/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}_segmented/{old_prefix[i]}2805_{old_labels[i]}_cellpos.csv',
        masks_svg=masks_svgs[i]))

for i, sample in enumerate(exp_sample_list):

    single_flag = True
    if ''.join(os.listdir(f'../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/')).find(
            'full') != -1:
        single_flag = False

    for file in os.listdir(f'../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/'):
        if single_flag:
            if file.find('autobright.tif') != -1 or file.find('autbright.tif') != -1:
                break
        else:
            if file.find('full_autobright.tif') != -1:
                break

    sample.transform_points2background(
        f'../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/{file}', upsampling=15)

    df_dict = {'Name': sample.data['Gene'], 'Code': np.nan, 'Probability': sample.iss_probability,
               'X': sample.data['PosX'], 'Y': sample.data['PosY'], 'Tile': np.nan}
    pd.DataFrame(df_dict).to_csv(f'../GMM_decoding/remapped/Exp_PD9694{new_labels[i]}_GMMdecoding_remapped.csv',
                                 index=False)

    # Legacy code, change when have time
    sample.update_coords(warp_matrix_file=Matrices[old_labels[i]][0],
                         resizing_params={'source': [np.array(sample.get_img_size(Im_Sources[old_labels[i]][1])),
                                                     [Images[old_labels[i]][1].shape[1],
                                                      Images[old_labels[i]][1].shape[0]]],
                                          'target': [np.array(sample.get_img_size(Im_Sources[old_labels[i]][0])),
                                                     [Images[old_labels[i]][0].shape[1],
                                                      Images[old_labels[i]][0].shape[0]]]},
                         small_img_source=Images[old_labels[i]][1],
                         small_img_target=Images[old_labels[i]][0])

# imm samples

new_labels = ['m2_Top', 'm2_Bottom']
old_labels = ['8', '8']

old_folders = ['Immuno', 'Immuno']
old_prefix = ['', '']

# image = [f'/nfs/research1/gerstung/mg617/ISS_data/Mut_PD9694d3_Top/MutPD9694d3_Top_Nilson_prereg_DAPI_restitch_Cycle4_feature_DAPI-optflow_Cy7/cycles_combined_opt_flow_registered_DAPI_c01.tif',
#         f'/nfs/research1/gerstung/mg617/ISS_data/Mut_PD9694d3_Bottom/MutPD9694d3_Bottom_Nilson_prereg_DAPI_restitch_Cycle4_feature_DAPI-optflow_Cy7/cycles_combined_opt_flow_registered_DAPI_c01.tif',
#         f'/nfs/research1/gerstung/mg617/ISS_data/Mut_PD9694m2_Top/MutPD9694m2_Top_nilson-pre-reg/cycles_combined_DAPI_c01.tif',
#         f'/nfs/research1/gerstung/mg617/ISS_data/Mut_PD9694m2_Bottom/MutPD9694m2_Bottom_nilson-pre-reg/cycles_combined_DAPI_c01.tif']


imm_sample_list = []
for i in range(len(new_labels)):
    imm_sample_list.append(Sample(
        iss_data=f'/nfs/research1/gerstung/mg617/ISS_data/Imm_PD9694{new_labels[i]}/decoding/Imm_PD9694{new_labels[i]}_GMMdecoding.csv',
        image=f'/nfs/research1/gerstung/artem/projects/spatial_bayes/GMM_restored_DAPI/Imm_PD9694{new_labels[i]}.tif',
        cell_data=f'../ultra_hd_segmentation/Patient_2085/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}_segmented/{old_prefix[i]}2805_{old_labels[i]}_cellpos.csv'))

for i, sample in enumerate(imm_sample_list):

    single_flag = True
    if ''.join(os.listdir(f'../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/')).find(
            'full') != -1:
        single_flag = False

    for file in os.listdir(f'../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/'):
        if single_flag:
            if file.find('autobright.tif') != -1 or file.find('autbright.tif') != -1:
                break
        else:
            if file.find('full_autobright.tif') != -1:
                break

    sample.transform_points2background(
        f'../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/{file}', upsampling=15)

imm_sample_list[0].add_gene_data(imm_sample_list[1])
m2_imm_sample = imm_sample_list[0]

df_dict = {
    "Name": m2_imm_sample.data["Gene"],
    "Code": np.nan,
    "Probability": m2_imm_sample.iss_probability,
    "X": m2_imm_sample.data["PosX"],
    "Y": m2_imm_sample.data["PosY"],
    "Tile": np.nan,
}
pd.DataFrame(df_dict).to_csv(f"../GMM_decoding/remapped/Imm_PD9694m2_composed_GMMdecoding_remapped.csv", index=False)

m2_imm_sample.update_coords(
    warp_matrix_file=Matrices["8"][1],
    resizing_params={
        "source": [
            np.array(sample.get_img_size(Im_Sources["8"][2])),
            [Images["8"][2].shape[1], Images["8"][2].shape[0]],
        ],
        "target": [
            np.array(sample.get_img_size(Im_Sources["8"][0])),
            [Images["8"][0].shape[1], Images["8"][0].shape[0]],
        ],
    },
    small_img_source=Images["8"][2],
    small_img_target=Images["8"][0],
)

new_labels = ['d2', 'a2', 'c2', 'l2']
old_labels = ['7', '2', '5', '6.1']

old_folders = ['Immuno', 'Immuno', 'Immuno', 'Immuno', 'Immuno']
old_prefix = ['', '', '', '', '']
masks_svgs = [
    "../contours/R1_PD9694d_contour_only-01.svg",
    "../contours/R1_PD9694a_contours_only-01.svg",
    "../contours/R1_PD9694c_contours_only_corrected-01.svg",
    "../contours/PD9694x_contours_only-01.svg",
]

imm_sample_list = []
for i in range(len(new_labels)):
    imm_sample_list.append(Sample(
        iss_data=f'/nfs/research1/gerstung/mg617/ISS_data/Imm_PD9694{new_labels[i]}/decoding/Imm_PD9694{new_labels[i]}_GMMdecoding.csv',
        image=f'/nfs/research1/gerstung/artem/projects/spatial_bayes/GMM_restored_DAPI/Imm_PD9694{new_labels[i]}.tif',
        cell_data=f'../ultra_hd_segmentation/Patient_2085/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}_segmented/{old_prefix[i]}2805_{old_labels[i]}_cellpos.csv',
        masks_svg=masks_svgs[i]))

for i, sample in enumerate(imm_sample_list):
    single_flag = True
    if ''.join(os.listdir(f'../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/')).find(
            'full') != -1:
        single_flag = False

    for file in os.listdir(f'../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/'):
        if single_flag:
            if file.find('autobright.tif') != -1 or file.find('autbright.tif') != -1:
                break
        else:
            if file.find('full_autobright.tif') != -1:
                break

    sample.transform_points2background(
        f'../Globus/DAPI_2805/{old_folders[i]}/{old_prefix[i]}2805_{old_labels[i]}/{file}', upsampling=15)

    df_dict = {'Name': sample.data['Gene'], 'Code': np.nan, 'Probability': sample.iss_probability,
               'X': sample.data['PosX'], 'Y': sample.data['PosY'], 'Tile': np.nan}
    pd.DataFrame(df_dict).to_csv(f'../GMM_decoding/remapped/Imm_PD9694{new_labels[i]}_GMMdecoding_remapped.csv',
                                 index=False)

    sample.update_coords(warp_matrix_file=Matrices[old_labels[i]][1],
                         resizing_params={'source': [np.array(sample.get_img_size(Im_Sources[old_labels[i]][2])),
                                                     [Images[old_labels[i]][2].shape[1],
                                                      Images[old_labels[i]][2].shape[0]]],
                                          'target': [np.array(sample.get_img_size(Im_Sources[old_labels[i]][0])),
                                                     [Images[old_labels[i]][0].shape[1],
                                                      Images[old_labels[i]][0].shape[0]]]},
                         small_img_source=Images[old_labels[i]][2],
                         small_img_target=Images[old_labels[i]][0])

imm_sample_list = imm_sample_list + [m2_imm_sample]

# code for saving

# save all the data

# saved_list = {'imm_sample_list':imm_sample_list, 'exp_sample_list':exp_sample_list, 'mut_sample_list':mut_sample_list, 'val_sample_list':val_sample_list}

# with open('./data/newdata_saved.pkl', 'wb') as file:
#    pkl.dump(saved_list, file)

# code for load
with open('./data/newdata_saved.pkl', 'rb') as file:
    saved_lists = pkl.load(file)
imm_sample_list = saved_lists['imm_sample_list']
exp_sample_list = saved_lists['exp_sample_list']
mut_sample_list = saved_lists['mut_sample_list']
# val_sample_list = saved_lists['val_sample_list']


# Construct model data

genes_to_drop = []

tree = pd.read_csv('./tree_data/final_matrix6_case1.csv')
tree.columns = ['SF3B1mut', 'STUB1mut', 'CREBBPmut', 'ARHGEF28mut', 'KIAA0652mut',
                'OXSMmut', 'CKAP5mut', 'DENND1Amut', 'NOB1mut', 'RELAmut', 'PLXNA2mut',
                'PQLC2mut', 'LRP1Bmut', 'PTEN1mut', 'PTEN2mut', 'TMEM8Amut', 'DSELmut',
                'FGFR1exp', 'KIF14mut', 'AMZ1mut', 'KCNT1mut', 'FZD4mut', 'AP3B22mut',
                'EMILIN2mut', 'CCDC105mut', 'ZNF468mut', 'SF3B1wt', 'STUB1wt',
                'CREBBPwt', 'ARHGEF28wt', 'KIAA0652wt', 'OXSMwt', 'CKAP5wt',
                'DENND1Awt', 'NOB1wt', 'RELAwt', 'PLXNA2wt', 'PQLC2wt', 'LRP1Bwt',
                'PTEN1wt', 'PTEN2wt', 'TMEM8Awt', 'DSELwt', 'KIF14wt', 'AMZ1wt',
                'KCNT1wt', 'FZD4wt', 'AP3B22wt', 'EMILIN2wt', 'CCDC105wt', 'ZNF468wt']

tree = tree[tree.columns[np.isin(tree.columns,
                                 list(filter(lambda x: not ((x[:-3] in genes_to_drop) or (x[:-2] in genes_to_drop) or (
                                         x in genes_to_drop)), tree.columns)))]]

# tree.iloc[3,:] = 0
# tree['AMZ1mut'][3]=12 # Amp/OE?
# tree['TMEM8Awt'][3]=2 # Gained copy?
# tree['STUB1wt'][3]=2 # Gained copy?
#
# tree['LRP1Bmut'][3]=1 # Should be in magenta because in blue
#
# tree['EMILIN2mut'][3]=1 # Seems likely
#
# tree["LRP1Bmut"][3]=1 # Restore blue copies in magenta
#
# tree["CKAP5mut"][3]=1 # Restore blue copies in magenta
tree['AMZ1wt'][1] = 3
tree['AMZ1wt'][3] = 3
tree['AMZ1wt'][4] = 3
# tree["DSELwt"][1]=2 # Was 1 in green, but no mut?
tree["KIAA0652wt"] = 2  # Set WT to 2 copies ..
tree["KIAA0652wt"][1] = 1  # .. but bot in green
tree["PTEN2wt"][2] = 1
tree["PTEN1wt"][5] = 1

tree.iloc[0, :] = tree.iloc[4, :]  # Make proper grey
tree.loc[0, "CKAP5mut"] = 0
tree.loc[0, "CKAP5wt"] = 2

tree.loc[0, "LRP1Bmut"] = 0
tree.loc[0, "LRP1Bwt"] = 2

tree.loc[6, "FGFR1exp"] = 12
tree.loc[5, "FGFR1exp"] = 16

M = np.array(tree)
# M = np.delete(M, [0], axis=0) # remove grey
M = np.delete(M, [3], axis=0)  # remove magenta
M_wt = np.zeros(M.shape[1])
M_wt[np.where((tree.columns == "FGFR1exp") | ([x.endswith("wt") for x in tree.columns]))[0]] = 2
n_wt = 2
M = np.concatenate([M] + n_wt * [M_wt[None, :]], axis=0)

wgs_data = pd.read_csv("./data/PD9694_genome_data_dec_2020.csv")
wgs_data = wgs_data[np.logical_not(np.isin(wgs_data.ISS_id, genes_to_drop))]
wgs_names = np.array(wgs_data.ISS_id)
wgs_names[np.where(wgs_names == "AP3B2")[0]] = "AP3B22"
# wgs_names[np.where(wgs_names=='AMZ1')[0]] = 'AMZ1non'
wgs_wt = []
wgs_mut = []
for i, s in enumerate("dac"):
    wgs_mut.append(np.array(wgs_data[f"VAL_MtAll_PD9694{s}"] + wgs_data[f"DIS_MtAll_PD9694{s}"]))
    wgs_wt.append(
        np.array(
            wgs_data[f"VAL_sample_PD9694{s}_depth"]
            + wgs_data[f"DIS_sample_PD9694{s}_depth"]
            - (wgs_data[f"VAL_MtAll_PD9694{s}"] + wgs_data[f"DIS_MtAll_PD9694{s}"])
        )
    )

# wgs_wt.append(np.array([M[:,np.where(tree.columns == gene + 'wt')[0][0]].mean(axis=0) for gene in wgs_names]))
# wgs_mut.append(np.array([M[:,np.where(tree.columns == gene + 'mut')[0][0]].mean(axis=0) for gene in wgs_names]))
wgs_wt.append(wgs_wt[0] / 5)
wgs_mut.append(wgs_mut[0] / 5)
wgs_wt.append(wgs_wt[0] / 5)
wgs_mut.append(wgs_mut[0] / 5)

# plot priors
plt.figure(figsize=(25, 5))
for s in range(5):
    plt.subplot(1, 5, s + 1)
    for i in range(10):  # len(wgs_names)):
        plt.plot(
            np.linspace(0.0, 1, 500),
            stats.beta(wgs_mut[s][i] / 5 + 1, wgs_wt[s][i] / 5 + 1).pdf(np.linspace(0.0, 1, 500)),
            label=wgs_names[i],
        )
        # plt.ylim(0,100)
        plt.legend()
        plt.rcParams["figure.facecolor"] = "w"
    plt.title("daclm"[s])
# Create data for model
scale = 3

mut_mask = mask_infeasible(mut_sample_list, scale, probability=0.6, plot=False)
data4model = generate_data4model(samples_list=mut_sample_list, genes=tree.columns, M=M, n_aug=1)

n_samples = data4model["n_samples"]
n_genes = data4model["n_genes"]
iss_data = data4model["iss_data"]
tiles_axes = data4model["tiles_axes"]
cells_counts = data4model["cells_counts"]
sample_dims = data4model["sample_dims"]
n_factors = data4model["n_factors"]
n_aug = data4model["n_aug"]

mask = mut_mask
# Actual model definition and run
class Beta_sum(pm.Beta):
    def __init__(self, n=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n

    def logp(self, value):
        return super().logp(value) * self.n


class Beta_sum(pm.Beta):
    def __init__(self, n=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n

    def logp(self, value):
        return super().logp(value) * self.n
    
    
with pm.Model() as model_hierarchical_errosion:
    xi = [pm.Gamma('xi_{}'.format(s), mu=0.5, sigma=1, shape=n_genes) for s in range(n_samples)]
    r_mu = pm.Gamma('r', mu=0.5, sigma=1, shape=n_genes)
    # r = [pm.Gamma('r_{}'.format(s), mu=r_mu, sigma=r_mu/20, shape=n_genes) for s in range(n_samples)]
    r_de = [pm.Lognormal('r_de_{}'.format(s), mu=0, sigma=0.05, shape=n_genes) for s in range(n_samples)]
    r = [r_mu * r_de_i for r_de_i in r_de]

    cov_func1_f = [[pm.gp.cov.ExpQuad(1, ls=2.5 * np.sqrt(scale)) for i in range(n_factors - 1 + n_aug)] for s in
                   range(n_samples)]
    cov_func2_f = [[pm.gp.cov.ExpQuad(1, ls=2.5 * np.sqrt(scale)) for i in range(n_factors - 1 + n_aug)] for s in
                   range(n_samples)]

    gp_f = [[pm.gp.LatentKron(cov_funcs=[cov_func1_f[s][i], cov_func2_f[s][i]]) for i in range(n_factors - 1 + n_aug)]
            for s in range(n_samples)]
    f_f = [[gp_f[s][i].prior(name, Xs=tiles_axes[s]) for i, name in
            enumerate(['f_f_{}_{}'.format(i, s) for i in range(n_factors - 1 + n_aug)])] for s in range(n_samples)]
    # f_f = [[pm.Normal(name, mu=0, sigma=1, shape=len(cells_counts[s]))  for i, name in enumerate(['f_f_{}_{}'.format(i, s) for i in range(n_factors - 1 + n_aug)])] for s in range(n_samples)]
    f_f = [f_f[s] + [np.ones(len(cells_counts[s])) * (-1.7)] for s in range(n_samples)]
    F_matrix = [tt.stack(f_f[s], axis=1) for s in range(n_samples)]

    F = [pm.Deterministic('F_{}'.format(s), tt.exp(F_matrix[s] / 2) / tt.exp(F_matrix[s] / 2).sum(axis=1)[:, None]) for
         s in range(n_samples)]

    lm_n = [pm.Gamma('lm_n_{}'.format(s), mu=50, sigma=100, shape=len(cells_counts[s])) for s in range(n_samples)]
    pois_n = [pm.Poisson('n_{}'.format(s), lm_n[s], observed=cells_counts[s]) for s in range(n_samples)]

    # Frac_obs = [tt.sum(F[s][:,:] * lm_n[s][:,np.newaxis], axis=0) / tt.sum(F[s][:,:] * lm_n[s][:,np.newaxis]) for s in range(n_samples)]

    # for s in range(n_samples):
    #    Frac_obs[s] = tt.set_subtensor(Frac_obs[s][2], Frac_obs[s][0] + Frac_obs[s][1] + Frac_obs[s][2])
    #    Frac_obs[s] = tt.set_subtensor(Frac_obs[s][4], Frac_obs[s][3] + Frac_obs[s][4])
    # Frac_obs[s] = tt.set_subtensor(Frac_obs[s][5], Frac_obs[s][5] + Frac_obs[s][6])

    # beta_prior = [pm.Beta('beta_prior_{}'.format(s), alphas[s], betas[s], observed=Frac_obs[s][:]) for s in range(3)]
    # beta_prior = [pm.Beta('beta_prior_{}'.format(s), wt_alphas[s-3], wt_betas[s-3], observed=Frac_obs[s][5]) for s in range(3,n_samples)]

    # E = np.zeros((n_aug, n_genes))
    E0 = 4  # + (M.max(0) > 7)*3 ## ie 1 for most loci; 4 for amps
    E = [E0 * pm.Beta(f'E_{s}', 0.01, 1, shape=(n_aug, n_genes)) for s in range(n_samples)]  # (M - M[-1,:]).max(0)
    # E = [pm.Gamma(f'E_{s}',mu=0.1,sigma=0.1, shape=(n_aug, n_genes)) for s in range(n_samples)]
    # E = pm.Uniform('E', lower=0, upper=6, shape = (n_aug, n_genes))

    # M_aug = tt.concatenate([M * M_de, E], axis=0)

    M_de = pm.Lognormal('M_de', mu=0.0, sigma=0.05, shape=M.shape)
    M_de = tt.pow(M_de, tt.concatenate([tt.ones((n_factors - n_wt, 1)), 2 * tt.ones((n_wt, 1))], axis=0))
    M_aug = M * M_de
    M_aug_de = [pm.Lognormal(f'M_aug_{s}', mu=0.0, sigma=0.05, shape=(n_factors, n_genes)) for s in range(n_samples)]

    theta_pure = [pm.Deterministic('theta_{}'.format(s), tt.dot(F[s], tt.concatenate([M, E[s]]))) for s in
                  range(n_samples)]

    # M_aug = tt.concatenate([M_aug[:n_factors-1,:], M_aug[None,-1,:]], axis=0)
    theta = [
        pm.Deterministic('theta_mod_{}'.format(s), tt.dot(F[s], tt.concatenate([M_aug * M_aug_de[s], E[s]], axis=0)))
        for s in range(n_samples)]  # * M_aug_de[s]

    z = 1 - pm.Beta('z', 1, 25, shape=n_genes) / 4  ## ie ~1% probe confusion

    confusion_matrix = tt.eye(n_genes)

    for i in range(n_genes):
        for j in range(n_genes):
            if tree.columns[i].endswith('mut') and tree.columns[j].endswith('mut') and tree.columns[i][:-3] == \
                    tree.columns[j][:-3]:
                confusion_matrix = tt.set_subtensor(confusion_matrix[i, j], z[i])

            if tree.columns[i].endswith('wt') and tree.columns[j].endswith('wt') and tree.columns[i][:-2] == \
                    tree.columns[j][:-2]:
                confusion_matrix = tt.set_subtensor(confusion_matrix[i, j], z[i])  #

            if tree.columns[i][:-3] == tree.columns[j][:-2]:
                confusion_matrix = tt.set_subtensor(confusion_matrix[i, j], 1 - z[i])  #

            if tree.columns[i][:-2] == tree.columns[j][:-3]:
                confusion_matrix = tt.set_subtensor(confusion_matrix[i, j], 1 - z[i])

    lm = [theta[s] * lm_n[s][:, None] * r[s][None, :] for s in range(n_samples)]

    # T = [theta[s] .sum(axis=0) for s in range(n_samples)]
    # M0 = tt.concatenate([M, E], axis=0)
    T = [tt.dot(F[s].sum(axis=0), tt.concatenate([M, E[s]], axis=0)) for s in
         range(n_samples)]  ## Just sum actual genotypes
    T_freq = [T[s][[np.where(tree.columns == gene + 'mut')[0][0] for gene in wgs_names]] \
              / (T[s][[np.where(tree.columns == gene + 'wt')[0][0] for gene in wgs_names]] \
                 + T[s][[np.where(tree.columns == gene + 'mut')[0][0] for gene in wgs_names]]) for s in
              range(n_samples)]

    # T_freq = [tt[T[s][:,np.where(tree.columns == gene + 'mut')[0][0]] \
    #           / (T[s][:,np.where(tree.columns == gene + 'wt')[0][0]] \
    #              + T[s][:,np.where(tree.columns == gene + 'mut')[0][0]]) for gene in wgs_names] for s in range(n_samples)]

    # beta_prior = [Beta_sum('beta_prior_{}'.format(s), n=(sample_dims[s][0] * sample_dims[s][1])/100, alpha=(wgs_mut[s] / (wgs_wt[s] + wgs_mut[s])), beta=((1 - wgs_mut[s] / (wgs_wt[s] + wgs_mut[s]))), observed=T_freq[s][:]) for s in range(n_samples)]
    # beta_prior = [Beta_sum('beta_prior_{}'.format(s), n=(sample_dims[s][0] * sample_dims[s][1]), alpha=(wgs_mut[s]/5 + 1), beta=(wgs_wt[s]/5 + 1), observed=T_freq[s][:]) for s in range(n_samples)]
    beta_prior = [
        Beta_sum('beta_prior_{}'.format(s), n=(mask[s].sum()), alpha=(wgs_mut[s] / 5 + 1), beta=(wgs_wt[s] / 5 + 1),
                 observed=T_freq[s][:]) for s in range(n_samples)]

    lm_er = [pm.Deterministic('lm_er_{}'.format(s), tt.dot(lm[s], confusion_matrix) + xi[s][None, :]) for s in
             range(n_samples)]

    # o = [pm.Exponential('o_{}'.format(s), 7, shape=n_genes) for s in range(n_samples)]
    o = pm.Gamma('o', mu=100, sd=10, shape=n_genes)

    # signal = [pm.NegativeBinomial('exp_{}'.format(s), mu = lm_er[s][mask[s],:], alpha=1/(o[s][None,:])**2, observed=iss_data[s][mask[s],:]) for s in range(n_samples)]
    signal = [
        pm.NegativeBinomial('exp_{}'.format(s), mu=lm_er[s][mask[s], :], alpha=o, observed=iss_data[s][mask[s], :]) for
        s in range(n_samples)]
    # signal = [pm.Poisson('exp_{}'.format(s), mu = lm_er[s][mask[s],:], observed=iss_data[s][mask[s],:]) for s in range(n_samples)]

# run_model

np.random.seed(1234)
pm.set_tt_rng(1234)

with model_hierarchical_errosion:
    advi = pm.ADVI()
    approx_hierarchical_errosion = advi.fit(n=15000, obj_optimizer=pm.adam(learning_rate=0.01))
    # approx_hierarchical_errosion = advi.fit(n=2000, obj_optimizer=pm.adam(learning_rate=0.01))

plt.plot(approx_hierarchical_errosion.hist[1000:])
samples_hierarchical_errosion = approx_hierarchical_errosion.sample(300)

# Plot fields

from matplotlib.colors import ListedColormap

alphas = np.concatenate((np.abs(np.linspace(0, 0, 256 - 200)), np.abs(np.linspace(0, 1.0, 256 - 56))))
N = 256

vals = np.ones((N, 4))
vals[:, 0] = np.linspace(1, 240 / 256, N)
vals[:, 1] = np.linspace(1, 228 / 256, N)
vals[:, 2] = np.linspace(1, 66 / 256, N)
vals[:, 3] = alphas
YellowCM = ListedColormap(vals)

plt.figure(figsize=(20, 30))
c = 0
# names = ['grey','green', 'purple', 'magenta', 'blue', 'red', 'orange'] + n_wt * ['wt'] + n_aug *['residuals']
names = ['grey', 'green', 'purple', 'blue', 'red', 'orange'] + n_wt * ['wt'] + n_aug * ['residuals']

# names = names[1:]

cmaps = {'grey': "Greys", 'green': "Greens", 'purple': "Purples", 'magenta': "RdPu", 'blue': "Blues", 'red': "Reds",
         'orange': "YlOrBr", 'wt': "Greys", 'residuals': "Greys"}
for i, ide in enumerate(np.arange(n_factors + n_aug)):
    for s in range(n_samples):
        plt.subplot(n_factors + n_aug, n_samples, c + 1)
        # plt.imshow(resized_img_list[s])
        plt.imshow(cv2.resize((np.percentile(
            samples_hierarchical_errosion['F_{}'.format(s)][:, :, ide] * samples_hierarchical_errosion[
                                                                             'lm_n_{}'.format(s)][:, :], 50,
            axis=0)).reshape(*sample_dims[s]).T[::-1, :], tuple(np.array(sample_dims[s]) * 4)),
                   cmap=plt.get_cmap(cmaps[names[ide]]), vmin=5, vmax=50)
        if s == 1:
            plt.title(names[ide])
        c += 1
    plt.colorbar()

plt.rcParams['figure.facecolor'] = 'w'
plt.tight_layout()
plt.show()

# Plot composition with densities

from scipy.interpolate import make_interp_spline


def plot_density_stacked(sampleID, site, save=False, ax=None, flipped=False, rescale_y=1):
    names = ['grey', 'green', 'purple', 'blue', 'red', 'orange', 'wt']

    SampleID = sampleID
    site = site
    if flipped:
        site = -site
    grid_mm2 = (Sample.get_img_size(mut_sample_list[SampleID].image)[0] / sample_dims[SampleID][0]) ** 2 / 1e6
    data = samples_hierarchical_errosion['F_{}'.format(sampleID)][:, :, :].reshape(300, *sample_dims[sampleID], 9)[:,
           ::, -site, :] * \
           cells_counts[sampleID].reshape(sample_dims[sampleID])[None, ::, -site, None] / grid_mm2
    data = np.concatenate([data[:, :, [0, 1, 2, 3, 4, 5]], data[:, :, [6, 7]].sum(axis=2)[:, :, None]], axis=2)
    # print(SampleID, site)
    # plt.figure(figsize=(12,6))
    # plt.imshow(cells_counts[sampleID].reshape(sample_dims[sampleID]))
    # plt.gca().axis('off')
    # print(cells_counts[sampleID].reshape(sample_dims[sampleID]).shape)
    # plt.axvline(sample_dims[sampleID][1]-site, color='white')
    # plt.show()
    # plt.figure(figsize=(12,6))
    # plt.imshow(mut_sample_list[3]._scaffold_image.T, vmax=50)
    # plt.axvline(mut_sample_list[3]._scaffold_image.shape[0] - ((site-1) * 33.559), color='white')
    # plt.gca().axis('off')
    # plt.show()
    CI = (2.5, 97.5)
    color_list = [get_cmap(cmaps[name])(150) for name in names]
    line_id_list = [0, 1, 2, 3, 4, 5, 6]
    # 1e3 / 0.325 / 15
    xold = np.linspace(0, sample_dims[SampleID][0], sample_dims[SampleID][0])
    xnew = np.linspace(0, (sample_dims[SampleID][0]), 5000)

    lines_smooth = []
    for i, idx in enumerate(line_id_list):
        line = data.mean(axis=0)[:, idx].T
        line_low = np.percentile(data, CI[0], axis=0)[:, idx].T
        line_up = np.percentile(data, CI[1], axis=0)[:, idx].T

        spl_line = make_interp_spline(xold, line, k=2)
        line_smooth = spl_line(xnew)
        line_smooth[line_smooth < 0] = 0
        lines_smooth.append(line_smooth)

    cum = np.zeros(lines_smooth[0].shape)
    if ax is not None:
        for i in range(len(line_id_list) - 1):
            ax.fill_between(xnew * rescale_y, cum, cum + lines_smooth[i], color=color_list[i], alpha=1)
            cum += lines_smooth[i]
        ax.plot(xnew * rescale_y, cum + lines_smooth[i + 1], color='black', alpha=1)

        ax.set_xlim(-1)
        ax.set_ylim(0, 2000)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_visible(False)
    else:
        # plt.figure(figsize=(int(Sample.get_img_size(mut_sample_list[SampleID].image)[0]/2000),2))
        for i in range(len(line_id_list) - 1):
            plt.fill_between(xnew * rescale_y, cum, cum + lines_smooth[i], color=color_list[i], alpha=1)
            cum += lines_smooth[i]
        plt.plot(xnew * rescale_y, cum + lines_smooth[i + 1], color='black', alpha=1)
        # cum += lines_smooth[i]

        plt.xlim(-1)
        plt.ylim(0, 2000)

        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().get_xaxis().set_visible(False)
    # plt.savefig(f'./images/2085-{SampleID}_line_{site}.pdf'.format(i))
    # plt.show()


stackplot_params = {0: [19, 33], 3: [16, 27], 4: [18, 31], 2: [10, 30], 1: [15, 44]}

stackplot_params = {k: list((np.array(stackplot_params[k]) * 3 / 2).astype(int)) for k in stackplot_params.keys()}


# stackplot_params = {1: [15, 42]}
# for s in [3]:
#    for i in stackplot_params[s]:
#        plt.figure(figsize=(12,2))
#        plot_density_stacked(sampleID=s, site=i, save=True)
#        plt.show()

def format_number(x, dec=1):
    x = float(x)
    if x % 1 == 0:
        return int(x)
    else:
        return round(x, dec)


import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42


def format_number(x, dec=1):
    x = float(x)
    if x % 1 == 0:
        return int(x)
    else:
        return round(x, dec)


from matplotlib.cm import get_cmap

fixed_y_gridsize = np.array([[int(x) for x in list(mut_sample_list[0]._scaffold_image.shape)[::-1]][1], 300, 300])
fixed_y_size = fixed_y_gridsize.sum() / 300

pixel2um = 0.325
cmaps = {
    'grey': "Greys",
    'green': "Greens",
    'purple': "Purples",
    'magenta': "RdPu",
    'blue': "Blues",
    'red': "Reds",
    'orange': "YlOrBr",
    'wt': "Greys",
    'residuals': "Greys",
}

names = ['grey', 'green', 'purple', 'blue', 'red', 'orange'] + 1 * ['wt']
c = [get_cmap(cmaps[n])(150) for n in names]
# c[0] = get_cmap("cool")(10)
F = [samples_hierarchical_errosion[f'F_{s}'].mean(0).reshape(*sample_dims[s], -1) for s in range(n_samples)]
Fmap = [(f[:, :, : n_factors - 2]).argmax(2) for f in F]
Fn = [(f[:, :, n_factors - 2 :]).sum(2) > 0.75 for f in F]
n = [
    samples_hierarchical_errosion['lm_n_{}'.format(s)].mean(0).reshape(*sample_dims[s]).T[::-1, :]
    for s in range(n_samples)
]

for i in range(5):
    img = mut_sample_list[i]._scaffold_image  # tifffile.imread(mut_sample_list[i].image)
    s = img.shape
    s = tuple([int(x) for x in list(s)[::-1]])
    p35, p90 = np.percentile(img, (35, 90))
    processed_img = exposure.rescale_intensity(img, in_range=(p35, p90))
    b = cv.resize(processed_img, s)[::-1, :] / 255.0

    b = np.maximum(np.minimum(b, 1), 0)
    Fc = (
        np.array([c[int(i)] for i in Fmap[i].flatten()]).reshape((*Fmap[i].shape, -1)).transpose((1, 0, 2))[::-1, :, :3]
    )
    Fc[Fn[i].T[::-1, :], :] = 1.0

    scale = s[0] / Fc.shape[1]
    scaffold_image_rescale = 15
    x_axis_rescale = (fixed_y_gridsize[0]) / s[1] * s[0] / 300

    fig, axs = plt.subplots(
        3, 1, figsize=(x_axis_rescale, fixed_y_size * 1.5), gridspec_kw={"height_ratios": fixed_y_gridsize}, sharex=True
    )
    # axs[0].plot([b.shape[1]*0.9 - 2.5e3 / pixel2um / scaffold_image_rescale, b.shape[1]*0.9], [b.shape[0]*0.95, b.shape[0]*0.95], color='white', lw=5)

    axs[0].axhline((stackplot_params[i][0] - 1) * (s[1] / sample_dims[i][1]), linestyle="dotted", lw=3, color="white")
    axs[0].axhline((stackplot_params[i][1] - 1) * (s[1] / sample_dims[i][1]), linestyle="dotted", lw=3, color="white")
    if i == 1:
        axs[0].imshow((cv.resize(Fc, s) * b.reshape(*b.shape, 1))[::-1])  # , alpha=n[0]/n[0].max())
        flipped = True
    else:
        axs[0].imshow(cv.resize(Fc, s) * b.reshape(*b.shape, 1))
        flipped = False
    axs[0].axis("off")
    plot_density_stacked(i, stackplot_params[i][0], ax=axs[1], flipped=flipped, rescale_y=scale)
    plot_density_stacked(i, stackplot_params[i][1], ax=axs[2], flipped=flipped, rescale_y=scale)
    axs[2].set_xticks(np.arange(0, s[0], 2.5 / (scaffold_image_rescale * 0.325 / 1e3)))
    xticklabels = np.arange(0, s[0] * scaffold_image_rescale * 0.325 / 1e3, 2.5)
    xticklabels = [format_number(x) for x in xticklabels]
    axs[2].set_xticklabels(xticklabels)
    axs[2].set_xlabel("Distance, $mm$")
    axs[2].set_ylabel("Cell density, $mm^{-2}$")
    axs[1].set_ylabel("Cell density, $mm^{-2}$")
    axs[2].get_xaxis().set_visible(True)
    plt.xlim(0, s[0])
    plt.savefig(f"./images/2085-{[7, 2, 5, 6, 8][i]}_stacked_densities_small.pdf".format(i))
    plt.show()
# Validation plot (not the model)

import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42


def format_number(x, dec=1):
    x = float(x)
    if x % 1 == 0:
        return int(x)
    else:
        return round(x, dec)


fixed_y_gridsize = np.array([[int(x) for x in list(mut_sample_list[0]._scaffold_image.shape)[::-1]][1], 300, 300])
fixed_y_size = fixed_y_gridsize.sum() / 300

pixel2um = 0.325
cmaps = {'grey': "Greys", 'green': "Greens", 'purple': "Purples", 'magenta': "RdPu", 'blue': "Blues", 'red': "Reds",
         'orange': "YlOrBr", 'wt': "Greys", 'residuals': "Greys"}

names = ['grey', 'green', 'purple', 'blue', 'red', 'orange'] + 1 * ['wt']
c = [get_cmap(cmaps[n])(150) for n in names]
# c[0] = get_cmap("cool")(10)
F = [samples_hierarchical_errosion[f'F_{s}'].mean(0).reshape(*sample_dims[s], -1) for s in range(n_samples)]
Fmap = [(f[:, :, :n_factors - 2]).argmax(2) for f in F]
Fn = [(f[:, :, n_factors - 2:]).sum(2) > 0.75 for f in F]
n = [samples_hierarchical_errosion['lm_n_{}'.format(s)].mean(0).reshape(*sample_dims[s]).T[::-1, :] for s in
     range(n_samples)]
for i in range(5):
    img = mut_sample_list[i]._scaffold_image  # tifffile.imread(mut_sample_list[i].image)
    s = img.shape
    s = tuple([int(x) for x in list(s)[::-1]])
    p35, p90 = np.percentile(img, (35, 90))
    processed_img = exposure.rescale_intensity(img, in_range=(p35, p90))
    b = cv.resize(processed_img, s)[::-1, :] / 255.

    b = np.maximum(np.minimum(b, 1), 0)
    Fc = np.array([c[int(i)] for i in Fmap[i].flatten()]).reshape((*Fmap[i].shape, -1)).transpose((1, 0, 2))[::-1, :,
         :3]
    Fc[Fn[i].T[::-1, :], :] = 1.0

    scale = s[0] / Fc.shape[1]
    scaffold_image_rescale = 15
    x_axis_rescale = (fixed_y_gridsize[0]) / s[1] * s[0] / 300

    fig, axs = plt.subplots(3, 1, figsize=(x_axis_rescale, fixed_y_size * 1.5),
                            gridspec_kw={'height_ratios': fixed_y_gridsize}, sharex=True)
    # axs[0].plot([b.shape[1]*0.9 - 2.5e3 / pixel2um / scaffold_image_rescale, b.shape[1]*0.9], [b.shape[0]*0.95, b.shape[0]*0.95], color='white', lw=5)

    axs[0].axhline((stackplot_params[i][0] - 1) * (s[1] / sample_dims[i][1]), linestyle='dotted', lw=3, color='white')
    axs[0].axhline((stackplot_params[i][1] - 1) * (s[1] / sample_dims[i][1]), linestyle='dotted', lw=3, color='white')
    if i == 1:
        axs[0].imshow((cv.resize(Fc, s) * b.reshape(*b.shape, 1))[::-1])  # , alpha=n[0]/n[0].max())
        flipped = True
    else:
        axs[0].imshow(cv.resize(Fc, s) * b.reshape(*b.shape, 1))
        flipped = False
    axs[0].axis('off')
    plot_density_stacked(i, stackplot_params[i][0], ax=axs[1], flipped=flipped, rescale_y=scale)
    plot_density_stacked(i, stackplot_params[i][1], ax=axs[2], flipped=flipped, rescale_y=scale)
    axs[2].set_xticks(np.arange(0, s[0], 2.5 / (scaffold_image_rescale * 0.325 / 1e3)))
    xticklabels = np.arange(0, s[0] * scaffold_image_rescale * 0.325 / 1e3, 2.5)
    xticklabels = [format_number(x) for x in xticklabels]
    axs[2].set_xticklabels(xticklabels)
    axs[2].set_xlabel('Distance, $mm$')
    axs[2].set_ylabel('Cell density, $mm^{-2}$')
    axs[1].set_ylabel('Cell density, $mm^{-2}$')
    axs[2].get_xaxis().set_visible(True)
    plt.xlim(0, s[0])
    plt.savefig(f'./images/2085-{[7, 2, 5, 6, 8][i]}_stacked_densities_small.pdf'.format(i))
    plt.show()


c = [get_cmap(cmaps[n])(150) for n in names]

data4model = generate_data4model(samples_list=val_sample_list, M=M, n_aug=1)
sample_dimsR0 = data4model['sample_dims']

data4model = generate_data4model(samples_list=mut_sample_list, M=M, n_aug=1)
sample_dimsR1 = data4model['sample_dims']

FR0 = [val_samples_model[f'F_{s}'].reshape(300, *sample_dimsR0[s], -1).copy() for s in range(3)]
FR1 = [mut_samples_model[f'F_{s}'].reshape(300, *sample_dimsR1[s], -1).copy() for s in range(3)]
for i in range(3):
    FR0[i][:, :, :, 7] += FR0[i][:, :, :, 8]
    FR1[i][:, :, :, 7] += FR1[i][:, :, :, 8]
# Fmap = [(f[:,:,:n_factors-2]).argmax(2) for f in F]
# Fn = [(f[:,:,n_factors-2:]).sum(2) > 0.8 for f in F]
# n = [samples_hierarchical_errosion['lm_n_{}'.format(s)].mean(0).reshape(*sample_dims[s]).T[::-1, :] for s in range(n_samples)]
for i in range(3):
    imgR0 = val_sample_list[i]._scaffold_image
    imgR1 = mut_sample_list[i]._scaffold_image  # tifffile.imread(mut_sample_list[i].image)
    sR0 = imgR0.shape
    sR0 = tuple([int(x / 2) for x in list(sR0)[::-1]])
    sR1 = imgR1.shape
    sR1 = tuple([int(x / 2) for x in list(sR1)[::-1]])

    p35, p90 = np.percentile(imgR0, (35, 90))
    processed_img = exposure.rescale_intensity(imgR0, in_range=(p35, p90))
    b0 = cv.resize(processed_img, sR0)[::-1, :] / 255.0

    p35, p90 = np.percentile(imgR1, (35, 90))
    processed_img = exposure.rescale_intensity(imgR1, in_range=(p35, p90))
    b1 = cv.resize(processed_img, sR1)[::-1, :] / 255.0

    counter = 0
    plt.figure(figsize=(14, 6 * 6))

    for factor in range(7):
        for percentile in [50]:
            if factor == 7:
                vmax = 0.7
            else:
                vmax = 0.20
            plt.subplot(7, 2, counter + 1)
            b0 = np.maximum(np.minimum(b0, 1), 0)
            Fc = np.minimum(np.percentile(FR0[i][:, :, :, factor], percentile, axis=0), vmax) / vmax
            Fc = (Fc.T[::-1, :][:, :, None] * (1 - np.array(c[factor])))[:, :, :3]
            # plt.imshow(cv.resize(Fc, sR0))
            plt.imshow(1 - cv.resize(Fc, sR0) * b0.reshape(*b0.shape, 1))
            plt.title(f"{['D2', 'A2', 'C2'][s]} facotor{factor + 1} {percentile}%")
            plt.plot(
                [b0.shape[1] * 0.1, b0.shape[1] * 0.1 + 1e3 / 0.325 / 15],
                [b0.shape[0] * 1.05, b0.shape[0] * 1.05],
                color="black",
                lw=4,
            )
            plt.gca().axis("off")

            plt.subplot(7, 2, counter + 2)
            b1 = np.maximum(np.minimum(b1, 1), 0)
            Fc = np.minimum(np.percentile(FR1[i][:, :, :, factor], percentile, axis=0), vmax) / vmax
            Fc = (Fc.T[::-1, :][:, :, None] * (1 - np.array(c[factor])))[:, :, :3]
            plt.imshow(1 - cv.resize(Fc, sR1) * b1.reshape(*b1.shape, 1))
            plt.title(f"{['D3', 'A3', 'C3'][s]} facotor{factor + 1} {percentile}%")
            plt.plot(
                [b1.shape[1] * 0.1, b1.shape[1] * 0.1 + 1e3 / 0.325 / 15],
                [b1.shape[0] * 1.05, b1.shape[0] * 1.05],
                color="black",
                lw=4,
            )
            plt.gca().axis("off")
            counter += 2

    plt.rcParams["figure.facecolor"] = "w"
    # plt.imshow(255-cv.resize(sample_list[i]._scaffold_image, s)[::-1,:], cmap='Greys')
    # Fc = np.array([c[int(i)] for i in Fmap[i].flatten()]).reshape((*Fmap[i].shape,-1)).transpose((1,0,2))[::-1,:,:3]
    # Fc[:,:,3] = n[i]/n[i].max()
    # Fc[Fn[i].T[::-1,:],:]=1.0
    # plt.imshow(cv.resize(Fc, s) * b.reshape(*b.shape,1))#, alpha=n[0]/n[0].max())
    plt.show()

# Expression model

# DCIS vs INV regions

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
s = 1
df_mut = pd.DataFrame(mut_sample_list[s].data)
df_exp = pd.DataFrame(exp_sample_list[s].data)
df_imm = pd.DataFrame(imm_sample_list[s].data)
colors = {"cls-1": "green", "cls-2": "grey", "cls-3": "red", "cls-4": "purple", "cls-5": "black"}
paths = exp_sample_list[s].ducts["paths"]
duct_type = exp_sample_list[s].ducts["linetype"]
for i in range(len(paths)):
    plt.text(paths[i][:, 0].mean(), paths[i][:, 1].mean(), i)
    plt.plot(paths[i][:, 0], paths[i][:, 1], color=colors[duct_type[i]])
plt.scatter(df_exp.PosX[df_exp.Gene == "CD163"], df_exp.PosY[df_exp.Gene == "CD163"], color="red", s=10, alpha=1)
plt.scatter(df_imm.PosX[df_imm.Gene == "CD163"], df_imm.PosY[df_imm.Gene == "CD163"], color="blue", s=1, alpha=1)
plt.gca().set_aspect("equal", adjustable="box")

plt.subplot(1, 2, 2)
s = 2
df_mut = pd.DataFrame(mut_sample_list[s].data)
df_exp = pd.DataFrame(exp_sample_list[s].data)
df_imm = pd.DataFrame(imm_sample_list[s].data)
colors = {'cls-1': 'green', 'cls-2': 'grey', 'cls-3': 'red', 'cls-4': 'purple', 'cls-5': 'black'}
paths = exp_sample_list[s].ducts['paths']
duct_type = exp_sample_list[s].ducts['linetype']
for i in range(len(paths)):
    plt.text(paths[i][:, 0].mean(), paths[i][:, 1].mean(), i)
    plt.plot(paths[i][:, 0], paths[i][:, 1], color=colors[duct_type[i]])
plt.scatter(df_exp.PosX[df_exp.Gene == 'CD163'], df_exp.PosY[df_exp.Gene == 'CD163'], color='red', s=10, alpha=1)
plt.scatter(df_imm.PosX[df_imm.Gene == 'CD163'], df_imm.PosY[df_imm.Gene == 'CD163'], color='blue', s=1, alpha=1)
plt.gca().set_aspect('equal', adjustable='box')

condition_a_inv = (np.array(exp_sample_list[1].ducts['linetype']) == 'cls-3')
condition_a_dcis = (np.isin(np.arange(len(np.array(exp_sample_list[1].ducts['linetype']))), [0, 1, 2, 6, 5]))
condition_c_inv = (np.array(exp_sample_list[2].ducts['linetype']) == 'cls-3')
condition_c_dcis = (
    np.isin(np.arange(len(np.array(exp_sample_list[2].ducts['linetype']))), [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))

exp_dcis_a = exp_sample_list[1].filter_by_ducts(condition_a_inv | condition_a_dcis)
exp_dcis_c = exp_sample_list[2].filter_by_ducts(condition_c_inv | condition_c_dcis)

imm_dcis_a = imm_sample_list[1].filter_by_ducts(condition_a_inv | condition_a_dcis)
imm_dcis_c = imm_sample_list[2].filter_by_ducts(condition_c_inv | condition_c_dcis)

exp_dcis_inv_list = [exp_dcis_a, exp_dcis_c]
imm_dcis_inv_list = [imm_dcis_a, imm_dcis_c]


def softmax(x, t=2, axis=1):
    end_shape = list(x.shape)
    end_shape[axis] = -1
    return np.exp(x / t) / np.exp(x / t).sum(axis=axis).reshape(*end_shape)


samples_id = [1, 2]
n_samples = len(exp_dcis_inv_list)

mask_expimm = []
scale = 3
for i in range(n_samples):
    exp_dcis_inv_list[i].data_to_grid(scale_factor=scale, probability=0.6)
    imm_dcis_inv_list[i].data_to_grid(scale_factor=scale, probability=0.6)
    t_exp = np.array([s for s in exp_dcis_inv_list[i].gene_grid.values()]).sum(0)
    t_imm = np.array([s for s in imm_dcis_inv_list[i].gene_grid.values()]).sum(0)

    mask_infisiable_expimm = (exp_dcis_inv_list[i].gene_grid['infeasible'] / t_exp < 0.1) * (
            imm_dcis_inv_list[i].gene_grid['infeasible'] / t_imm < 0.1)

    plt.figure(figsize=(8, 4))
    # plt.subplot(1,2,1)
    # plt.imshow(mask_infeasible.T[::-1,:])
    plt.subplot(1, 2, 2)
    plt.imshow(mask_infisiable_expimm.T[::-1, :])
    plt.show()
    # mask.append(mask_infeasible.flatten())
    mask_expimm.append(mask_infisiable_expimm.flatten())

# renaming inconsitent genes
for i in range(n_samples):
    exp_dcis_inv_list[i].data_to_grid(scale_factor=scale, probability=0.6)
    imm_dcis_inv_list[i].data_to_grid(scale_factor=scale, probability=0.6)
    if 'Oct-04' in exp_dcis_inv_list[i].gene_grid.keys():
        exp_dcis_inv_list[i].gene_grid['OCT-4'] = exp_sample_list[i].gene_grid.pop('Oct-04')
        exp_dcis_inv_list[i].genes[np.where(exp_sample_list[i].genes == 'Oct-04')] = 'OCT-4'

imm_genes = np.concatenate([imm_dcis_inv_list[0].genes[:-4], np.array(['perforin'])])
exp_genes = exp_dcis_inv_list[0].genes[:-3]

n_imm_genes = len(imm_genes)
n_exp_genes = len(exp_genes)

# create pseudosubclones which represent cancer cells in DCIS and INV regions
n_factors_bassis = 8
subclone_proportions = [
    np.stack([samples_hierarchical_errosion[f'f_f_{i}_{s}'] for i in range(n_factors_bassis)], axis=2) for s in
    samples_id]
# subclone_proportions_dcis_inv_wt = [np.zeros((subclone_proportions[s].shape[0],subclone_proportions[s].shape[1], 2)) for s in range(n_samples)]
subclone_means = [np.zeros((subclone_proportions[s][:, :, :].shape[1], 2)) for s in range(n_samples)]

for s in range(n_samples):
    # inv
    tmp = exp_dcis_inv_list[s].filter_by_ducts([condition_a_inv, condition_c_inv][s])
    tmp.data_to_grid(scale_factor=scale, probability=0.6)
    subclone_means[s][tmp.cell_grid.flatten() > 0, 0] = 1
    # subclone_proportions_dcis_inv_wt[s][:,tmp.cell_grid.flatten()>0,2] = 2*np.log(np.exp(subclone_proportions[s][:,tmp.cell_grid.flatten()>0,6:]/2).sum(axis=2))
    # dcis
    tmp = exp_dcis_inv_list[s].filter_by_ducts([condition_a_dcis, condition_c_dcis][s])
    tmp.data_to_grid(scale_factor=scale, probability=0.6)
    subclone_means[s][tmp.cell_grid.flatten() > 0, 1] = 1
    # subclone_proportions_dcis_inv_wt[s][:,tmp.cell_grid.flatten()>0,2] = 2*np.log(np.exp(subclone_proportions[s][:,tmp.cell_grid.flatten()>0,6:]/2).sum(axis=2))

# subclone_means = [subclone_proportions_dcis_inv_wt[s].mean(axis=0) for s in range(n_samples)]
# subclone_vars = [subclone_proportions_dcis_inv_wt[s].var(axis=0) for s in range(n_samples)]


exp_iss_data = [
    np.transpose(np.array([exp_dcis_inv_list[i].gene_grid[k] for k in exp_genes]), [1, 2, 0]).reshape(-1, n_exp_genes)
    for i in range(n_samples)]
imm_iss_data = [
    np.transpose(np.array([imm_dcis_inv_list[i].gene_grid[k] for k in imm_genes]), [1, 2, 0]).reshape(-1, n_imm_genes)
    for i in range(n_samples)]

tiles_axes = [exp_dcis_inv_list[i].tile_axis for i in range(n_samples)]

exp_cells_counts = [exp_dcis_inv_list[i].cell_grid.flatten() for i in range(n_samples)]
imm_cells_counts = [imm_dcis_inv_list[i].cell_grid.flatten() for i in range(n_samples)]
expimm_masks = [mask_expimm[i] for i in range(n_samples)]

sample_dims = [(int(tiles_axes[i][0][-1] + 1), int(tiles_axes[i][1][-1] + 1)) for i in range(n_samples)]
# n_factors = subclone_means[0].shape[-1] - 2
factors_id = [0, 1]
n_factors = len(factors_id)
# n_aug=1

n_aug = 0

sample_separate_flag = True

for s in range(n_samples):
    mask_expimm[s] *= subclone_means[s].sum(axis=1).astype(bool)

# Model for expression

with pm.Model() as model_exp_bi_prior:
    if sample_separate_flag:
        r_mu_exp = [pm.Gamma(f'r_mu_exp_{s}', mu=0.5, sigma=1, shape=n_exp_genes) for s in range(n_samples)]
        r_mu_imm = [pm.Gamma(f'r_mu_imm_{s}', mu=0.5, sigma=1, shape=n_imm_genes) for s in range(n_samples)]
    else:
        r_mu_exp = pm.Gamma('r_mu_exp_', mu=0.5, sigma=1, shape=n_exp_genes)
        r_mu_imm = pm.Gamma('r_mu_imm_', mu=0.5, sigma=1, shape=n_imm_genes)

    r_xi_exp = [pm.Gamma('r_xi_exp_{}'.format(s), mu=0.5, sigma=1, shape=n_exp_genes) for s in range(n_samples)]
    r_xi_imm = [pm.Gamma('r_xi_imm_{}'.format(s), mu=0.5, sigma=1, shape=n_imm_genes) for s in range(n_samples)]

    lm_n_exp = [pm.Gamma('lm_n_exp_{}'.format(s), mu=50, sigma=100, shape=len(exp_cells_counts[s])) for s in
                range(n_samples)]
    pois_n_exp = [pm.Poisson('n_exp_{}'.format(s), lm_n_exp[s], observed=exp_cells_counts[s]) for s in range(n_samples)]
    lm_n_imm = [pm.Gamma('lm_n_imm_{}'.format(s), mu=50, sigma=100, shape=len(imm_cells_counts[s])) for s in
                range(n_samples)]
    pois_n_imm = [pm.Poisson('n_imm_{}'.format(s), lm_n_imm[s], observed=imm_cells_counts[s]) for s in range(n_samples)]

    # cov_func1_f = [[pm.gp.cov.ExpQuad(1, ls=1*scale) for i in range(n_factors-1 + n_aug)] for s in range(n_samples)]
    # cov_func2_f = [[pm.gp.cov.ExpQuad(1, ls=1*scale) for i in range(n_factors-1 + n_aug)] for s in range(n_samples)]

    # gp_f = [[pm.gp.LatentKron(cov_funcs=[cov_func1_f[s][i], cov_func2_f[s][i]]) for i in range(n_factors - 1 + n_aug)] for s in range(n_samples)]
    # f_f = [[gp_f[s][i].prior(name, Xs=tiles_axes[s]) * (subclone_vars[s][:,i] ** 0.5)/5 + subclone_means[s][:,i]  for i, name in enumerate(['f_f_{}_{}'.format(i, s) for i in range(n_factors - 1 + n_aug)])] for s in range(n_samples)]
    # f_f = [f_f[s] + [np.zeros(len(cells_counts[s]))] for s in range(n_samples)]
    # F_matrix = [pm.Deterministic('F_raw_{}'.format(s), tt.stack(f_f[s], axis=1)) for s in range(n_samples)]

    # prior_proportions = [pm.Normal('prop_prior_{}'.format(s), subclone_means[s] , subclone_vars[s]**0.5/2, shape=(len(exp_cells_counts[s]),n_factors)) for s in range(n_samples)]
    prior_proportions = [subclone_means[s] for s in range(n_samples)]
    # F_pri = [tt.concatenate([prior_proportions[s], np.zeros(len(exp_cells_counts[s]))[:,None]], axis=1) for s in range(n_samples)]

    # F = [pm.Deterministic('F_{}'.format(s), tt.exp(F_pri[s] / 2) / tt.exp(F_pri[s] / 2).sum(axis=1)[:,None]) for s in range(n_samples)]
    F = [pm.Deterministic('F_{}'.format(s), theano.shared(prior_proportions[s])) for s in range(n_samples)]

    if sample_separate_flag:
        E_exp = [pm.Dirichlet(f'E_exp_{s}', np.ones(n_factors + n_aug)[np.newaxis, :] * 5,
                              shape=(n_exp_genes, n_factors + n_aug)) for s in range(n_samples)]
        E_imm = [pm.Dirichlet(f'E_imm_{s}', np.ones(n_factors + n_aug)[np.newaxis, :] * 5,
                              shape=(n_imm_genes, n_factors + n_aug)) for s in range(n_samples)]
        theta_E_exp = [pm.Deterministic('theta_E_exp_{}'.format(s), tt.dot(F[s][:, factors_id], E_exp[s].T)) for s in
                       range(n_samples)]
        theta_E_imm = [pm.Deterministic('theta_E_imm_{}'.format(s), tt.dot(F[s][:, factors_id], E_imm[s].T)) for s in
                       range(n_samples)]
        lm_exp = [lm_n_exp[s][:, np.newaxis] * theta_E_exp[s] * r_mu_exp[s][None, :] + r_xi_exp[s][None, :] for s in
                  range(n_samples)]
        lm_imm = [lm_n_imm[s][:, np.newaxis] * theta_E_imm[s] * r_mu_imm[s][None, :] + r_xi_imm[s][None, :] for s in
                  range(n_samples)]

    else:
        E_exp = pm.Dirichlet('E_exp', np.ones(n_factors + n_aug)[np.newaxis, :] * 5,
                             shape=(n_exp_genes, n_factors + n_aug))
        E_imm = pm.Dirichlet('E_imm', np.ones(n_factors + n_aug)[np.newaxis, :] * 5,
                             shape=(n_imm_genes, n_factors + n_aug))
        theta_E_exp = [pm.Deterministic('theta_E_exp_{}'.format(s), tt.dot(F[s][:, factors_id], E_exp.T)) for s in
                       range(n_samples)]
        theta_E_imm = [pm.Deterministic('theta_E_imm_{}'.format(s), tt.dot(F[s][:, factors_id], E_imm.T)) for s in
                       range(n_samples)]
        lm_exp = [lm_n_exp[s][:, np.newaxis] * theta_E_exp[s] * r_mu_exp[None, :] + r_xi_exp[s][None, :] for s in
                  range(n_samples)]
        lm_imm = [lm_n_imm[s][:, np.newaxis] * theta_E_imm[s] * r_mu_imm[None, :] + r_xi_imm[s][None, :] for s in
                  range(n_samples)]

    o_exp = pm.Gamma('o_exp', mu=100, sd=10, shape=n_exp_genes)
    o_imm = pm.Gamma('o_imm', mu=100, sd=10, shape=n_imm_genes)

    expression_genes_exp = [pm.NegativeBinomial('exp_genes_{}'.format(s), mu=lm_exp[s][expimm_masks[s], :], alpha=o_exp,
                                                observed=exp_iss_data[s][expimm_masks[s], :]) for s in range(n_samples)]
    expression_genes_imm = [pm.NegativeBinomial('imm_genes_{}'.format(s), mu=lm_imm[s][expimm_masks[s], :], alpha=o_imm,
                                                observed=imm_iss_data[s][expimm_masks[s], :]) for s in range(n_samples)]

np.random.seed(1234)
pm.set_tt_rng(1234)

with model_exp_bi_prior:
    advi = pm.ADVI()
    approx_exp_bi_prior = advi.fit(n=10000, obj_optimizer=pm.adam(learning_rate=0.01))

plt.plot(approx_exp_bi_prior.hist[1000:])
samples_exp_bi_prior = approx_exp_bi_prior.sample(300)

# working with the output
if sample_separate_flag:
    E_data_pre = np.concatenate([np.stack([samples_exp_bi_prior['E_exp_{s}'] for s in range(n_samples)], axis=3),
                                 np.stack([samples_exp_bi_prior['E_imm_{s}'] for s in range(n_samples)], axis=3)],
                                axis=1)
    mu_data_pre = np.concatenate([np.stack([samples_exp_bi_prior['r_mu_exp_{s}'] for s in range(n_samples)], axis=2),
                                  np.stack([samples_exp_bi_prior['r_mu_imm_{s}'] for s in range(n_samples)], axis=2)],
                                 axis=1)
else:
    E_data_pre = np.concatenate([samples_exp_bi_prior['E_exp'], samples_exp_bi_prior['E_imm']], axis=1)
    mu_data_pre = np.concatenate([samples_exp_bi_prior['r_mu_exp'], samples_exp_bi_prior['r_mu_imm']], axis=1)

E_data = E_data_pre
E_names = np.concatenate([exp_genes, imm_genes])
E_pannel_from = np.array(['exp'] * n_exp_genes + ['imm'] * n_imm_genes)
E_pannel_color = np.array(['orange'] * n_exp_genes + ['skyblue'] * n_imm_genes)

# chagne gene names in plots
gene_name2good = {'PTPRC': 'CD45', 'CD274': 'PD-L1', 'Ki-67': 'MKI67'}
for k in gene_name2good.keys():
    E_names[np.where(E_names == k)] = gene_name2good[k]

# remove PTPRC_trans5

ptprc_trans5_loc = np.where(E_names == 'PTPRC_trans5')[0]

E_data = np.delete(E_data, ptprc_trans5_loc, axis=1)
E_names = np.delete(E_names, ptprc_trans5_loc)
E_pannel_from = np.delete(E_pannel_from, ptprc_trans5_loc)
E_pannel_color = np.delete(E_pannel_color, ptprc_trans5_loc)

sample_names = ['a', 'c']
data_dict = {sample_names[i]: E_data[:, :, :, n] for i, n in enumerate([0, 1])}
# look at d and l
data_dict = {k: data_dict[k] for k in ['a', 'c']}

dat = [data_dict[k][:, :, 0] / data_dict[k][:, :, 1] for k in ['a', 'c']]
dat = [np.log10(dat[i]) for i, k in enumerate(['a', 'c'])]
p_vals = [pval_baes(data_dict[k][:, :, 0], data_dict[k][:, :, 1], 1.5) for k in ['a', 'c']]
gene_filtered = np.where(np.logical_or(p_vals[0] < 2, p_vals[1] < 2))[0]
dat = [dat[i][:, gene_filtered] for i, k in enumerate(['a', 'c'])]

# plotting

from matplotlib.ticker import MultipleLocator, FixedLocator, FuncFormatter

###### Locators for Y-axis
# set tickmarks at multiples of 1.
majorLocator = MultipleLocator(1.)
# create custom minor ticklabels at logarithmic positions
ra = np.array([[n + (1. - np.log10(i))] for n in range(-3, 3) for i in [2, 3, 4, 5, 6, 7, 8, 9][::-1]]).flatten() * -1.
minorLocator = FixedLocator(ra)
###### Formatter for Y-axis (chose any of the following two)
# show labels as powers of 10 (looks ugly)
# majorFormatter= FuncFormatter(lambda x,p: "{:.1e}".format(10**x) )
# or using MathText (looks nice, but not conform to the rest of the layout)
majorFormatter = FuncFormatter(lambda x, p: r"$10^{" + "{x:d}".format(x=int(x)) + r"}$")

percentiles_dat = [[np.percentile(dat[s], pct, axis=0) for pct in [2.5, 50, 97.5]] for s in range(2)]

for pannel_id in range(2):

    subset = np.where(E_pannel_from[gene_filtered] == ["imm", "exp"][pannel_id])[0]
    sub_percentiles_dat = [[percentiles_dat[s][i][subset] for i in range(3)] for s in range(2)]

    log_sum_pval = []
    for i in range(len(p_vals[0][gene_filtered][subset])):
        if np.sign(sub_percentiles_dat[0][1][i]) == np.sign(sub_percentiles_dat[1][1][i]):
            val = -2 * (np.log(p_vals[0][gene_filtered][subset][i]) + np.log(p_vals[1][gene_filtered][subset][i]))
        else:
            p1 = p_vals[0][gene_filtered][subset][i]
            p2 = p_vals[1][gene_filtered][subset][i]

            p1, p2 = np.minimum(p1, p2), np.maximum(p1, p2)
            val = -2 * (np.log(p1) + np.log(1))
        log_sum_pval.append(val)

    # p_vals_chi2 = np.minimum(p_vals[0][gene_filtered][subset], p_vals[1][gene_filtered][subset])
    p_vals_chi2 = np.round(stats.chi2.sf(log_sum_pval, 4), 5)

    ranks = -np.round(1 / p_vals_chi2 * np.sign(sub_percentiles_dat[0][1] + sub_percentiles_dat[1][1]), 2)
    order = []
    for r in np.unique(ranks):
        ranked_order = np.where(ranks == r)[0]
        magnitude_order = list(
            ranked_order[
                np.argsort(((sub_percentiles_dat[0][1] + sub_percentiles_dat[1][1]))[np.where(ranks == r)[0]])[::-1]
            ]
        )
        order += magnitude_order

    boundaries = [0] + list(np.cumsum([len(np.where(ranks == r)[0]) for r in np.unique(ranks)]))
    boundaries_pval = np.array([0] + list([p_vals_chi2[np.where(ranks == r)[0]][0] for r in np.unique(ranks)]))

    p_val_th = 0.05

    # color = [['darkgrey' if i < p_val_th else 'lightgrey' for i in p_vals[s][gene_filtered][subset][order]] for s in range(2)]
    color = ['tomato', 'darkviolet']

    bpv = np.array(boundaries)[np.where(np.diff(boundaries_pval < p_val_th))[0]]
    xposdiff = np.ones(len(order))
    squeezed = 0.2

    if len(bpv) == 2:
        xposdiff[bpv[0] + 1 : bpv[1]] = squeezed
    elif len(bpv) == 1:
        if (boundaries_pval < p_val_th)[0]:
            xposdiff[: bpv[0] - 1] = squeezed
        else:
            xposdiff[bpv[0] + 1 :] = squeezed
    elif len(bpv) == 0:
        xposdiff[:] = squeezed
    else:
        raise IndexError("what the hell")

    xpos = np.cumsum(xposdiff) * 2
    widths = xposdiff.copy()
    widths[bpv[0]] = squeezed
    for s in range(2):
        for i in range(3):
            sub_percentiles_dat[s][i] = sub_percentiles_dat[s][i][order]

    plt.figure(figsize=(xpos[-1] * 0.3, 3))
    plt.bar(xpos - 0.4 * widths, sub_percentiles_dat[0][1],
            yerr=[np.abs(sub_percentiles_dat[0][0] - sub_percentiles_dat[0][1]),
                  np.abs(sub_percentiles_dat[0][2] - sub_percentiles_dat[0][1])], width=0.7 * widths,
            color=color[0], error_kw={'linewidth': widths * 1, 'alpha': 0.8})
    plt.bar(xpos + 0.4 * widths, sub_percentiles_dat[1][1],
            yerr=[np.abs(sub_percentiles_dat[1][0] - sub_percentiles_dat[1][1]),
                  np.abs(sub_percentiles_dat[1][2] - sub_percentiles_dat[1][1])], width=0.7 * widths,
            color=color[1], error_kw={'linewidth': widths * 1, 'alpha': 0.8})

    plt.gca().set_xticks(xpos)
    gene_names = E_names[gene_filtered][subset][order]
    gene_names[widths == squeezed] = ''
    plt.gca().set_xticklabels(gene_names, rotation=90)
    plt.gca().set_title(['immune', 'expression'][pannel_id])
    plt.xlim(xpos[0] - 2, xpos[-1] + 2)
    plt.gca().yaxis.set_major_locator(majorLocator)
    plt.gca().yaxis.set_minor_locator(minorLocator)
    plt.gca().yaxis.set_major_formatter(majorFormatter)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.ylim(-1.1, 1.1)
    plt.savefig(f'./images/2085_InvVsDcis_{["imm", "exp"][pannel_id]}_bars_xi2.pdf'.format(i))
    plt.show()

# Green vs Orange comparison

# get data

samples_id = [0, 3]
factors_id = [1, 5]

n_samples = len(samples_id)
n_factors = len(factors_id)

mask_expimm = []
scale = 3
for i in samples_id:
    exp_sample_list[i].data_to_grid(scale_factor=scale, probability=0.6)
    imm_sample_list[i].data_to_grid(scale_factor=scale, probability=0.6)
    t_exp = np.array([s for s in exp_sample_list[i].gene_grid.values()]).sum(0)
    t_imm = np.array([s for s in imm_sample_list[i].gene_grid.values()]).sum(0)

    mask_infisiable_expimm = (exp_sample_list[i].gene_grid['infeasible'] / t_exp < 0.1) * (
            imm_sample_list[i].gene_grid['infeasible'] / t_imm < 0.1)

    plt.figure(figsize=(8, 4))
    # plt.subplot(1,2,1)
    # plt.imshow(mask_infeasible.T[::-1,:])
    plt.subplot(1, 2, 2)
    plt.imshow(mask_infisiable_expimm.T[::-1, :])
    plt.show()
    # mask.append(mask_infeasible.flatten())
    mask_expimm.append(mask_infisiable_expimm.flatten())

# renaming inconsitent genes
for i in samples_id:
    exp_sample_list[i].data_to_grid(scale_factor=scale, probability=0.6)
    imm_sample_list[i].data_to_grid(scale_factor=scale, probability=0.6)
    if 'Oct-04' in exp_sample_list[i].gene_grid.keys():
        exp_sample_list[i].gene_grid['OCT-4'] = exp_sample_list[i].gene_grid.pop('Oct-04')
        exp_sample_list[i].genes[np.where(exp_sample_list[i].genes == 'Oct-04')] = 'OCT-4'

imm_genes = np.concatenate([imm_sample_list[0].genes[:-4], np.array(['perforin'])])
exp_genes = exp_sample_list[0].genes[:-3]

n_imm_genes = len(imm_genes)
n_exp_genes = len(exp_genes)

subclone_proportions = [
    np.stack([samples_hierarchical_errosion[f'f_f_{i}_{s}'] for i in range(n_factors_bassis)], axis=2) for s in
    samples_id]
# subclone_proportions_wtjoined = [subclone_proportions[s][:,:,:].copy() for s in range(n_samples)]

subcl_locations = []
for s in range(n_samples):
    subcl_locations.append([])
    for i, f in enumerate(factors_id):
        subcl_locations[s].append(softmax(subclone_proportions[s], axis=2)[:, :, f].mean(axis=0))
        subcl_locations[s][i] = subcl_locations[s][i] > np.percentile(subcl_locations[s][i], 80)
    subcl_intersection = np.zeros(subcl_locations[s][0].shape)
    for i in range(n_factors):
        subcl_intersection += subcl_locations[s][i].astype(int)
    subcl_intersection = subcl_intersection > 1
    for i in range(n_factors):
        subcl_locations[s][i] *= np.logical_not(subcl_intersection)

subclone_means = [np.array(subcl_locations[s]).T for s in
                  range(n_samples)]  # [subclone_proportions_wtjoined[s].mean(axis=0) for s in range(n_samples)]
# subclone_vars = [subclone_proportions_wtjoined[s].var(axis=0) for s in range(n_samples)]


exp_iss_data = [
    np.transpose(np.array([exp_sample_list[i].gene_grid[k] for k in exp_genes]), [1, 2, 0]).reshape(-1, n_exp_genes) for
    i in samples_id]
imm_iss_data = [
    np.transpose(np.array([imm_sample_list[i].gene_grid[k] for k in imm_genes]), [1, 2, 0]).reshape(-1, n_imm_genes) for
    i in samples_id]

tiles_axes = [exp_sample_list[i].tile_axis for i in samples_id]

exp_cells_counts = [exp_sample_list[i].cell_grid.flatten() for i in samples_id]
imm_cells_counts = [imm_sample_list[i].cell_grid.flatten() for i in samples_id]
expimm_masks = [mask_expimm[i] for i in range(n_samples)]

sample_dims = [(int(tiles_axes[i][0][-1] + 1), int(tiles_axes[i][1][-1] + 1)) for i in range(n_samples)]
# n_factors = subclone_means[0].shape[-1] - 2
# n_factors = 7
# n_aug=1

n_aug = 0

sample_separate_flag = True

for s in range(n_samples):
    mask_expimm[s] *= subclone_means[s].sum(axis=1).astype(bool)

with pm.Model() as model_exp_bi_prior:
    if sample_separate_flag:
        r_mu_exp = [pm.Gamma(f'r_mu_exp_{s}', mu=0.5, sigma=1, shape=n_exp_genes) for s in range(n_samples)]
        r_mu_imm = [pm.Gamma(f'r_mu_imm_{s}', mu=0.5, sigma=1, shape=n_imm_genes) for s in range(n_samples)]
    else:
        r_mu_exp = pm.Gamma('r_mu_exp_', mu=0.5, sigma=1, shape=n_exp_genes)
        r_mu_imm = pm.Gamma('r_mu_imm_', mu=0.5, sigma=1, shape=n_imm_genes)

    r_xi_exp = [pm.Gamma('r_xi_exp_{}'.format(s), mu=0.5, sigma=1, shape=n_exp_genes) for s in range(n_samples)]
    r_xi_imm = [pm.Gamma('r_xi_imm_{}'.format(s), mu=0.5, sigma=1, shape=n_imm_genes) for s in range(n_samples)]

    lm_n_exp = [pm.Gamma('lm_n_exp_{}'.format(s), mu=50, sigma=100, shape=len(exp_cells_counts[s])) for s in
                range(n_samples)]
    pois_n_exp = [pm.Poisson('n_exp_{}'.format(s), lm_n_exp[s], observed=exp_cells_counts[s]) for s in range(n_samples)]
    lm_n_imm = [pm.Gamma('lm_n_imm_{}'.format(s), mu=50, sigma=100, shape=len(imm_cells_counts[s])) for s in
                range(n_samples)]
    pois_n_imm = [pm.Poisson('n_imm_{}'.format(s), lm_n_imm[s], observed=imm_cells_counts[s]) for s in range(n_samples)]

    # cov_func1_f = [[pm.gp.cov.ExpQuad(1, ls=1*scale) for i in range(n_factors-1 + n_aug)] for s in range(n_samples)]
    # cov_func2_f = [[pm.gp.cov.ExpQuad(1, ls=1*scale) for i in range(n_factors-1 + n_aug)] for s in range(n_samples)]

    # gp_f = [[pm.gp.LatentKron(cov_funcs=[cov_func1_f[s][i], cov_func2_f[s][i]]) for i in range(n_factors - 1 + n_aug)] for s in range(n_samples)]
    # f_f = [[gp_f[s][i].prior(name, Xs=tiles_axes[s]) * (subclone_vars[s][:,i] ** 0.5)/5 + subclone_means[s][:,i]  for i, name in enumerate(['f_f_{}_{}'.format(i, s) for i in range(n_factors - 1 + n_aug)])] for s in range(n_samples)]
    # f_f = [f_f[s] + [np.zeros(len(cells_counts[s]))] for s in range(n_samples)]
    # F_matrix = [pm.Deterministic('F_raw_{}'.format(s), tt.stack(f_f[s], axis=1)) for s in range(n_samples)]

    # prior_proportions = [pm.Normal('prop_prior_{}'.format(s), subclone_means[s] , subclone_vars[s]**0.5/2, shape=(len(exp_cells_counts[s]),n_factors)) for s in range(n_samples)]
    prior_proportions = [subclone_means[s] for s in range(n_samples)]
    # F_pri = [tt.concatenate([prior_proportions[s], np.zeros(len(exp_cells_counts[s]))[:,None]], axis=1) for s in range(n_samples)]

    F = [pm.Deterministic('F_{}'.format(s), theano.shared(prior_proportions[s])) for s in range(
        n_samples)]  # [pm.Deterministic('F_{}'.format(s), tt.exp(F_pri[s] / 2) / tt.exp(F_pri[s] / 2).sum(axis=1)[:,None]) for s in range(n_samples)]

    if sample_separate_flag:
        E_exp = [pm.Dirichlet(f'E_exp_{s}', np.ones(n_factors + n_aug)[np.newaxis, :] * 5,
                              shape=(n_exp_genes, n_factors + n_aug)) for s in range(n_samples)]
        E_imm = [pm.Dirichlet(f'E_imm_{s}', np.ones(n_factors + n_aug)[np.newaxis, :] * 5,
                              shape=(n_imm_genes, n_factors + n_aug)) for s in range(n_samples)]
        theta_E_exp = [pm.Deterministic('theta_E_exp_{}'.format(s), tt.dot(F[s][:, :], E_exp[s].T)) for s in
                       range(n_samples)]
        theta_E_imm = [pm.Deterministic('theta_E_imm_{}'.format(s), tt.dot(F[s][:, :], E_imm[s].T)) for s in
                       range(n_samples)]
        lm_exp = [lm_n_exp[s][:, np.newaxis] * theta_E_exp[s] * r_mu_exp[s][None, :] + r_xi_exp[s][None, :] for s in
                  range(n_samples)]
        lm_imm = [lm_n_imm[s][:, np.newaxis] * theta_E_imm[s] * r_mu_imm[s][None, :] + r_xi_imm[s][None, :] for s in
                  range(n_samples)]

    else:
        E_exp = pm.Dirichlet('E_exp', np.ones(n_factors + n_aug)[np.newaxis, :] * 5,
                             shape=(n_exp_genes, n_factors + n_aug))
        E_imm = pm.Dirichlet('E_imm', np.ones(n_factors + n_aug)[np.newaxis, :] * 5,
                             shape=(n_imm_genes, n_factors + n_aug))
        theta_E_exp = [pm.Deterministic('theta_E_exp_{}'.format(s), tt.dot(F[s][:, :], E_exp.T)) for s in
                       range(n_samples)]
        theta_E_imm = [pm.Deterministic('theta_E_imm_{}'.format(s), tt.dot(F[s][:, :], E_imm.T)) for s in
                       range(n_samples)]
        lm_exp = [lm_n_exp[s][:, np.newaxis] * theta_E_exp[s] * r_mu_exp[None, :] + r_xi_exp[s][None, :] for s in
                  range(n_samples)]
        lm_imm = [lm_n_imm[s][:, np.newaxis] * theta_E_imm[s] * r_mu_imm[None, :] + r_xi_imm[s][None, :] for s in
                  range(n_samples)]

    o_exp = pm.Gamma('o_exp', mu=100, sd=10, shape=n_exp_genes)
    o_imm = pm.Gamma('o_imm', mu=100, sd=10, shape=n_imm_genes)

    expression_genes_exp = [pm.NegativeBinomial('exp_genes_{}'.format(s), mu=lm_exp[s][expimm_masks[s], :], alpha=o_exp,
                                                observed=exp_iss_data[s][expimm_masks[s], :]) for s in range(n_samples)]
    expression_genes_imm = [pm.NegativeBinomial('imm_genes_{}'.format(s), mu=lm_imm[s][expimm_masks[s], :], alpha=o_imm,
                                                observed=imm_iss_data[s][expimm_masks[s], :]) for s in range(n_samples)]

np.random.seed(1234)
pm.set_tt_rng(1234)

with model_exp_bi_prior:
    advi = pm.ADVI()
    approx_exp_bi_prior = advi.fit(n=10000, obj_optimizer=pm.adam(learning_rate=0.01))

plt.plot(approx_exp_bi_prior.hist[1000:])
samples_exp_bi_prior = approx_exp_bi_prior.sample(300)

if sample_separate_flag:
    E_data_pre = np.concatenate([np.stack([samples_exp_bi_prior[f'E_exp_{s}'] for s in range(n_samples)], axis=3),
                                 np.stack([samples_exp_bi_prior[f'E_imm_{s}'] for s in range(n_samples)], axis=3)],
                                axis=1)
    mu_data_pre = np.concatenate([np.stack([samples_exp_bi_prior[f'r_mu_exp_{s}'] for s in range(n_samples)], axis=2),
                                  np.stack([samples_exp_bi_prior[f'r_mu_imm_{s}'] for s in range(n_samples)], axis=2)],
                                 axis=1)
else:
    E_data_pre = np.concatenate([samples_exp_bi_prior[f'E_exp'], samples_exp_bi_prior[f'E_imm']], axis=1)
    mu_data_pre = np.concatenate([samples_exp_bi_prior[f'r_mu_exp'], samples_exp_bi_prior[f'r_mu_imm']], axis=1)

E_data = E_data_pre
E_names = np.concatenate([exp_genes, imm_genes])
E_pannel_from = np.array(['exp'] * n_exp_genes + ['imm'] * n_imm_genes)
E_pannel_color = np.array(['orange'] * n_exp_genes + ['skyblue'] * n_imm_genes)

# chagne gene names in plots
gene_name2good = {'PTPRC': 'CD45', 'CD274': 'PD-L1', 'Ki-67': 'MKI67'}
for k in gene_name2good.keys():
    E_names[np.where(E_names == k)] = gene_name2good[k]

# remove PTPRC_trans5

ptprc_trans5_loc = np.where(E_names == 'PTPRC_trans5')[0]

E_data = np.delete(E_data, ptprc_trans5_loc, axis=1)
E_names = np.delete(E_names, ptprc_trans5_loc)
E_pannel_from = np.delete(E_pannel_from, ptprc_trans5_loc)
E_pannel_color = np.delete(E_pannel_color, ptprc_trans5_loc)

sample_names = ['d', 'l']
data_dict = {sample_names[i]: E_data[:, :, :, n] for i, n in enumerate([0, 1])}
# look at d and l
data_dict = {k: data_dict[k] for k in ['d', 'l']}

exp_gene_groups = pd.read_csv('./data/exp_genes_groups.csv')
imm_gene_groups = pd.read_csv('./data/imm_genes_group.csv')
OncotypeDX = exp_gene_groups[(exp_gene_groups.OncotypeDX == 'down') | (exp_gene_groups.OncotypeDX == 'up')]

imm_gene2type = {imm_gene_groups['ISS target name '].values[i]: imm_gene_groups['Group'].values[i] for i in
                 range(imm_gene_groups.shape[0])}
exp_gene2type = {exp_gene_groups['ISS Target name'].values[i]: exp_gene_groups['Group'].values[i] for i in
                 range(exp_gene_groups.shape[0])}
exp_gene2oncotype = {OncotypeDX['ISS Target name'].values[i]: OncotypeDX['OncotypeDX'].values[i] for i in
                     range(OncotypeDX.shape[0])}


def gene2funcgroup(gene, pannel):
    if pannel == 'imm':
        try:
            return imm_gene2type[gene]
        except KeyError:
            return None
    elif pannel == 'exp':
        try:
            return exp_gene2type[gene]
        except KeyError:
            return None
    else:
        raise KeyError('either exp or imm')


def gene2funccolor(gene, pannel):
    colors = {
        "Bcell": "yellow",
        "CD8_cytotoxic": "magenta",  # , 'CD8_naive':'forestgreen', 'CD8_Tcell':'forestgreen',
        # 'DC': 'purple',
        # 'fibroblast':'blue','Fibroblast markers':'blue',
        # 'hypoxia_angiogenesis':'red','Angiogenesis/growth factors':'red',
        # 'immune_tolerance': 'skyblue',
        # 'NK':'purple',
        "macrophage": "dimgrey",
        "Macrophage markers": "dimgrey",
        "monocyte": "tomato",
        "pan_immune_cell_marker": "black",
        "Lymphocyte marker": "black",
        # 'proliferation': 'violet', 'Proliferation':'violet', 'HER2':'violet',
        # 'Tcell':'brown', 'Tcell_CD4':'brown',
        "Treg": "dodgerblue",
    }
    # 'Stemness/differentiation': 'darkgreen'}

    default_color = "lightgrey"
    group = gene2funcgroup(gene, pannel)
    try:
        return colors[group]
    except KeyError:
        return default_color


def gene2oncocolor(gene):
    color = {"up": "salmon", "down": "dodgerblue"}
    default_color = "lightgrey"
    try:
        oncotype = exp_gene2oncotype[gene]
        return color[oncotype]
    except KeyError:
        return default_color

dat = [data_dict[k][:, :, 0] / data_dict[k][:, :, 1] for k in ['d', 'l']]
dat = [np.log10(dat[i]) for i, k in enumerate(['d', 'l'])]
p_vals = [pval_baes(data_dict[k][:, :, 0], data_dict[k][:, :, 1], 1.5) for k in ['d', 'l']]
gene_filtered = np.where(np.logical_or(p_vals[0] < 2, p_vals[1] < 2))[0]
dat = [dat[i][:, gene_filtered] for i, k in enumerate(['d', 'l'])]

from matplotlib.ticker import MultipleLocator, FixedLocator, FuncFormatter

###### Locators for Y-axis
# set tickmarks at multiples of 1.
majorLocator = MultipleLocator(1.)
# create custom minor ticklabels at logarithmic positions
ra = np.array([[n + (1. - np.log10(i))] for n in range(-3, 3) for i in [2, 3, 4, 5, 6, 7, 8, 9][::-1]]).flatten() * -1.
minorLocator = FixedLocator(ra)
###### Formatter for Y-axis (chose any of the following two)
# show labels as powers of 10 (looks ugly)
# majorFormatter= FuncFormatter(lambda x,p: "{:.1e}".format(10**x) )
# or using MathText (looks nice, but not conform to the rest of the layout)
majorFormatter = FuncFormatter(lambda x, p: r"$10^{" + "{x:d}".format(x=int(x)) + r"}$")

percentiles_dat = [[np.percentile(dat[s], pct, axis=0) for pct in [2.5, 50, 97.5]] for s in range(2)]

for pannel_id in range(2):

    subset = np.where(E_pannel_from[gene_filtered] == ['imm', 'exp'][pannel_id])[0]
    sub_percentiles_dat = [[percentiles_dat[s][i][subset] for i in range(3)] for s in range(2)]

    log_sum_pval = []
    for i in range(len(p_vals[0][gene_filtered][subset])):
        if np.sign(sub_percentiles_dat[0][1][i]) == np.sign(sub_percentiles_dat[1][1][i]):
            val = -2 * (np.log(p_vals[0][gene_filtered][subset][i]) + np.log(p_vals[1][gene_filtered][subset][i]))
        else:
            p1 = p_vals[0][gene_filtered][subset][i]
            p2 = p_vals[1][gene_filtered][subset][i]

            p1, p2 = np.minimum(p1, p2), np.maximum(p1, p2)
            val = -2 * (np.log(p1) + np.log(1))
        log_sum_pval.append(val)

    # p_vals_chi2 = np.minimum(p_vals[0][gene_filtered][subset], p_vals[1][gene_filtered][subset])
    p_vals_chi2 = np.round(stats.chi2.sf(log_sum_pval, 4), 5)

    ranks = -np.round(1 / p_vals_chi2 * np.sign(sub_percentiles_dat[0][1] + sub_percentiles_dat[1][1]), 2)
    order = []
    for r in np.unique(ranks):
        ranked_order = np.where(ranks == r)[0]
        magnitude_order = list(ranked_order[np.argsort(
            ((sub_percentiles_dat[0][1] + sub_percentiles_dat[1][1]))[np.where(ranks == r)[0]])[::-1]])
        order += magnitude_order

    boundaries = [0] + list(np.cumsum([len(np.where(ranks == r)[0]) for r in np.unique(ranks)]))
    boundaries_pval = np.array([0] + list([p_vals_chi2[np.where(ranks == r)[0]][0] for r in np.unique(ranks)]))

    p_val_th = 0.05
    # color = [['darkgrey' if i < p_val_th else 'lightgrey' for i in p_vals[s][gene_filtered][subset][order]] for s in range(2)]
    gene_names = E_names[gene_filtered][subset][order]
    if pannel_id == 1:
        color = ['darkgrey', 'lightgrey']
        # color = [[gene2oncocolor(gene) for gene in gene_names], [gene2oncocolor(gene) for gene in gene_names]]
    else:
        color = ['darkgrey', 'lightgrey']
    bpv = np.array(boundaries)[np.where(np.diff(boundaries_pval < p_val_th))[0]]
    xposdiff = np.ones(len(order))
    squeezed = 0.2
    xposdiff[bpv[0] + 1:bpv[1]] = squeezed
    xpos = np.cumsum(xposdiff) * 2
    widths = xposdiff.copy()
    widths[bpv[0]] = squeezed
    for s in range(2):
        for i in range(3):
            sub_percentiles_dat[s][i] = sub_percentiles_dat[s][i][order]

    plt.figure(figsize=(xpos[-1] * 0.3, 3))
    plt.bar(xpos - 0.4 * widths, sub_percentiles_dat[0][1],
            yerr=[np.abs(sub_percentiles_dat[0][0] - sub_percentiles_dat[0][1]),
                  np.abs(sub_percentiles_dat[0][2] - sub_percentiles_dat[0][1])], width=0.7 * widths,
            color=color[0], error_kw={'linewidth': widths * 1, 'alpha': 0.8})
    plt.bar(xpos + 0.4 * widths, sub_percentiles_dat[1][1],
            yerr=[np.abs(sub_percentiles_dat[1][0] - sub_percentiles_dat[1][1]),
                  np.abs(sub_percentiles_dat[1][2] - sub_percentiles_dat[1][1])], width=0.7 * widths,
            color=color[1], error_kw={'linewidth': widths * 1, 'alpha': 0.8})

    plt.gca().set_xticks(xpos)
    gene_names[widths == squeezed] = ''
    plt.gca().set_xticklabels(gene_names, rotation=90)
    plt.gca().set_title(['immune', 'expression'][pannel_id])
    plt.xlim(xpos[0] - 2, xpos[-1] + 2)
    plt.ylim(-1.3, 1.3)
    plt.gca().yaxis.set_major_locator(majorLocator)
    plt.gca().yaxis.set_minor_locator(minorLocator)
    plt.gca().yaxis.set_major_formatter(majorFormatter)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.savefig(f'./images/2085_GreenVsOrange_{["imm", "exp"][pannel_id]}_bars_xi2.pdf'.format(i))
    plt.show()
