from itertools import chain

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate as si


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
