import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
from scipy import stats
import scipy as sp
import pandas as pd
from theano import tensor as tt
import theano
import pickle as pkl
import functools
import cv2
import cv2 as cv
import scipy.interpolate as si
import scipy.stats as st
from tqdm import tqdm
import re 
import subprocess
import itertools
from matplotlib.colors import ListedColormap
import tifffile 
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcls
import seaborn as sns

from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float
from skimage import transform
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
from skimage.filters import window, difference_of_gaussians
from scipy.fftpack import fft2, fftshift
from skimage import exposure
from svgpathtools import svg2paths, wsvg
import matplotlib.path as mpltPath
from copy import copy, deepcopy


from matplotlib.legend_handler import HandlerPatch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.cluster.hierarchy as sch
from matplotlib.cm import get_cmap
from matplotlib.ticker import FormatStrFormatter, FuncFormatter



import copy


from itertools import chain
import itertools


class GridWarp_alingment:
    def __init__(self, warp_matrix_file, source_coords, resizing_params, small_img_source, small_img_target, approx=100):
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
            np.linspace( -(self.img_source_dim[1] / (self.n_spl-3)), self.img_source_dim[1] + (self.img_source_dim[1] / (self.n_spl-3)), self.n_spl),
            np.linspace( -(self.img_source_dim[0] / (self.n_spl-3)), self.img_source_dim[0] + (self.img_source_dim[0] / (self.n_spl-3)), self.n_spl)))
        

    def _to_float(self, x):
        if x == '':
            return ''
        else:
            return float(x)
        
    def _read_file(self, filename):
        with open(filename, 'r') as file:
            raw_transform_file = file.readlines()
            raw_transform = {'X':[], 'Y':[]}

            for i in range(len(raw_transform_file)):
                if raw_transform_file[i] == 'X Coeffs -----------------------------------\n':
                    c = i + 1
                    while raw_transform_file[c] != '\n':
                        raw_transform['X'].append(list(map(lambda x: self._to_float(x), raw_transform_file[c].strip().split())))
                        c += 1
                elif raw_transform_file[i] == 'Y Coeffs -----------------------------------\n':
                    c = i + 1
                    while c < len(raw_transform_file) and raw_transform_file[c] != '\n':
                        raw_transform['Y'].append(list(map(lambda x: self._to_float(x), raw_transform_file[c].strip().split())))
                        c += 1

            raw_transform['X'] = list(chain(*(raw_transform['X'])))
            raw_transform['Y'] = list(chain(*(raw_transform['Y'])))

            raw_transform['X'] = list(filter(lambda x: x != '', raw_transform['X']))
            raw_transform['Y'] = list(filter(lambda x: x != '', raw_transform['Y']))
            
            self.n_spl = int(np.array(raw_transform['X']).shape[0]**0.5)
        return np.array([np.array(raw_transform['X']).reshape(self.n_spl,self.n_spl),
                         np.array(raw_transform['Y']).reshape(self.n_spl,self.n_spl)])
    
    
    def _get_spline_coords_interpolation(self, approx = None):
        
        if type(approx) == type(None):
            approx=self.approx
            
        tx = np.clip(np.arange(self.n_spl+3+1)-3,0,self.n_spl-3)
        
        intact_x = si.bisplev(np.linspace(0,self.n_spl-3,approx), np.linspace(0,self.n_spl-3,approx),
                              (tx,tx,self.norm_matrix[:,:,:].ravel(),3,3))
        intact_y = si.bisplev(np.linspace(0,self.n_spl-3,approx), np.linspace(0,self.n_spl-3,approx),
                              (tx,tx,np.transpose(self.norm_matrix[::-1,:,:], (0, 2, 1)).ravel(),3,3))
        
        #print(intact_x)
        interp_intact_x = si.interp1d(intact_x[0], np.linspace(0,self.n_spl-3,approx), kind='cubic', fill_value="extrapolate")
        interp_intact_y = si.interp1d(intact_y[0], np.linspace(0,self.n_spl-3,approx), kind='cubic', fill_value="extrapolate")
        
        return interp_intact_x, interp_intact_y
    
    def resize_coords(self, source_coords = None, resize_from_to = [[1,1], [1, 1]]):
        
        if type(source_coords) == type(None):
            source_coords=self.source_coords
        
        M = np.array([[resize_from_to[1][0] / resize_from_to[0][0], 0],
                        [0, resize_from_to[1][1] / resize_from_to[0][1]]])
        
        source_coords_rescaled = np.array(source_coords).T @ M
        
        return source_coords_rescaled
        
    
    def warp_coords(self, source_coords = None):
        
        if type(source_coords) == type(None):
            source_coords=self.source_coords
            
        tx = np.clip(np.arange(self.n_spl+3+1)-3,0,self.n_spl-3)
        
        rescaled_coords = self.resize_coords(self.source_coords, resize_from_to = self.resizing_params['source']).T
        X, Y = rescaled_coords[0], rescaled_coords[1]
        #print(X, Y)
        interp_intact_x, interp_intact_y = self._get_spline_coords_interpolation()
        
        coord_spline_space_X = interp_intact_x(X)
        coord_spline_space_Y = interp_intact_y(Y)
        
        transformed_coord_X = list(map(lambda x: si.bisplev(coord_spline_space_Y[x],
                                                            coord_spline_space_X[x],
                                                            (tx,tx,self.trans_matrix[:,:,:].ravel(),3,3)),
                                       list(range(len(X)))))
        
        transformed_coord_Y = list(map(lambda x: si.bisplev(coord_spline_space_X[x],
                                                            coord_spline_space_Y[x],
                                                            (tx,tx,np.transpose(self.trans_matrix[::-1,:,:], (0, 2, 1)).ravel(),3,3)),
                                       list(range(len(X)))))
        
        resized_transformed_coords = self.resize_coords([transformed_coord_X, transformed_coord_Y],
                                                        [self.resizing_params['target'][1],
                                                         self.resizing_params['target'][0]]).T
        
        return(resized_transformed_coords[0], resized_transformed_coords[1])
    
    def plot_transformation(self, approx = None, alpha=0.1, before=False):
        
        if type(approx) == type(None):
            approx=self.approx
            
        transformed_coord_X, transformed_coord_Y = self.warp_coords()
        transformed_coord_X, transformed_coord_Y = self.resize_coords([transformed_coord_X,
                                                                       transformed_coord_Y],
                                                                       resize_from_to = self.resizing_params['target']).T
            

        
        #plt.figure(figsize=(20, 15))
        tx = np.clip(np.arange(self.n_spl+3+1)-3,0,self.n_spl-3)
        plt.imshow(self.img_target)
        
        if before:
            rescaled_coords = self.resize_coords(self.source_coords,
                                                 resize_from_to = self.resizing_params['source']).T
            plt.scatter(rescaled_coords[0], rescaled_coords[1], alpha=alpha)
            
            plt.plot(np.linspace(self.norm_matrix[0].min(),self.norm_matrix[0].max(),approx),
                     si.bisplev(np.linspace(0,self.n_spl-3,approx), np.linspace(0,self.n_spl-3,approx),
                                (tx,tx,np.transpose(self.norm_matrix[::-1,:,:], (0, 2, 1)).ravel(),3,3)), 'k-', alpha=0.3)
            plt.plot(si.bisplev(np.linspace(0,self.n_spl-3,approx), np.linspace(0,self.n_spl-3,approx),
                                (tx,tx,self.norm_matrix[::,:,:].ravel(),3,3)),
                     np.linspace(self.norm_matrix[1].min(),self.norm_matrix[1].max(),approx),'k-', alpha=0.3)
        else:
            plt.scatter(transformed_coord_X, transformed_coord_Y, alpha=alpha)
        
            plt.plot(np.linspace(self.trans_matrix[0].min(),self.trans_matrix[0].max(),approx),
                     si.bisplev(np.linspace(0,self.n_spl-3,approx), np.linspace(0,self.n_spl-3,approx),
                                (tx,tx,np.transpose(self.trans_matrix[::-1,:,:], (0, 2, 1)).ravel(),3,3)), 'k-', alpha=0.3)
            plt.plot(si.bisplev(np.linspace(0,self.n_spl-3,approx), np.linspace(0,self.n_spl-3,approx),
                                (tx,tx,self.trans_matrix[::,:,:].ravel(),3,3)),
                     np.linspace(self.trans_matrix[1].min(),self.trans_matrix[1].max(),approx),'k-', alpha=0.3)
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
        self.genes = np.unique(self.data['Gene'])
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
            self.cellpos = np.array(self._cell_density_df_raw[['x', 'y']])
            if 'cell_type' in self._cell_density_df_raw.columns:
                self.cell_types = np.array(self._cell_density_df_raw['cell_type'])
            else:
                self.cell_types = np.array(['core'] * self.cellpos.shape[0])
            
            
        else:
            self._cell_density_df_raw = None
            self.cellpos = None
            
        if masks_svg is not None:
            self.ducts = self._get_ducts(masks_svg)
            
    
    @staticmethod
    def get_img_size(filename):
        image_size = list(map(lambda x: int(x),
                              re.findall('\ (\d+)x(\d+)\ ',
                                         subprocess.getoutput("identify " + filename))[-1]))
        return image_size
    
    def standardise_data(self):
        aliases = {'Gene': 'Gene', 'Name': 'Gene', 'gene': 'Gene', 'name': 'Gene',
                   'PosX': 'PosX', 'PosY': 'PosY', 'X': 'PosX', 'Y': 'PosY', 'global_X_pos': 'PosX', 'global_Y_pos': 'PosY'}
        
        data = {}
        for col in self._iss_df_raw.columns:
            if col in aliases.keys():
                data[aliases[col]] = np.array(list(self._iss_df_raw[col]))
        
        if 'Probability' in self._iss_df_raw.columns:
            self.iss_probability = np.array(self._iss_df_raw['Probability'])
        else:
            self.iss_probability = None

        
        return data
                
        
    def data_to_grid(self, scale_factor=4, gene_list='all', probability=None):
        
        #TODO creates parameters outside of the constructor, deal with it if necessery 
        self.grid_params = (np.array(self.spatial_dims) / 1000 * scale_factor).astype(int)
        x_step = self.spatial_dims[0] / (self.grid_params[0]-1)
        y_step = self.spatial_dims[1] / (self.grid_params[1]-1)
        
        if gene_list=='all':
            gene_list = self.genes
            
        self.gene_grid = {}
    
        self.tile_axis = [np.arange(self.grid_params[0])[:,None], np.arange(self.grid_params[1])[:,None]]
            
        
        for gene in tqdm(gene_list):
            arr = np.zeros((self.grid_params[0], self.grid_params[1]))
            if (probability is not None) and (self.iss_probability is not None):
                cur_gene = np.where(np.logical_and(self.data['Gene'] == gene, self.iss_probability >= probability))
            else:
                cur_gene = np.where(self.data['Gene'] == gene)
            tiles = np.array([(self.data['PosX'][cur_gene]//x_step).astype(int), (self.data['PosY'][cur_gene]//y_step).astype(int)]).T
            k_id, v = np.unique(tiles, return_counts=True, axis=0)
            
            for i in range(len(v)):
                try:
                    arr[tuple(k_id[i,:])] = v[i]
                except IndexError:
                    # TODO: raise warnigns  
                    self.error_flag = True
                
            self.gene_grid[gene] = arr
            
                
        if self.cell_data is not None:
            arr = np.zeros((self.grid_params[0], self.grid_params[1]))
            tiles = np.array([(self.cellpos[:,0]//x_step).astype(int), (self.cellpos[:,1]//y_step).astype(int)]).T
            k_id, v = np.unique(tiles, return_counts=True, axis=0)

            for i in range(len(v)):
                try:
                    arr[tuple(k_id[i,:])] = v[i]
                except IndexError:
                    self.error_flag = True

            self.cell_grid = arr

            
        if self.error_flag:
            print('Some of the points were out of bound')


    @staticmethod
    #TODO Depricated
    def _find_angle(img_original, img_deformed):
        # First, band-pass filter both images
        image = img_original
        rts_image = img_deformed
        image = difference_of_gaussians(image, 5, 15)
        rts_image = difference_of_gaussians(rts_image, 5, 15)
        # window images
        wimage = image * window('hann', image.shape)
        rts_wimage = rts_image * window('hann', rts_image.shape)

        # work with shifted FFT magnitudes
        image_fs = np.abs(fftshift(fft2(wimage)))
        rts_fs = np.abs(fftshift(fft2(rts_wimage)))

        # Create log-polar transformed FFT mag images and register
        shape_or = image_fs.shape
        shape_def = rts_fs.shape

        radius_or = shape_or[0] // 8
        radius_def = shape_def[0] // 8 # only take lower frequencies
        warped_image_fs = warp_polar(image_fs, radius=radius_or, output_shape=shape_or,
                                     scaling='log', order=0)
        warped_rts_fs = warp_polar(rts_fs, radius=radius_def, output_shape=shape_or,
                                   scaling='log', order=0)

        warped_image_fs = warped_image_fs[:shape_or[0], :]  # only use half of FFT
        warped_rts_fs = warped_rts_fs[:shape_or[0], :]
        shifts, error, phasediff = phase_cross_correlation(warped_image_fs,
                                                           warped_rts_fs,
                                                           upsample_factor=10)

        # Use translation parameters to calculate rotation and scaling parameters
        shiftr, shiftc = shifts[:2]
        recovered_angle = (360 / shape_or[0]) * shiftr
        return recovered_angle
    @staticmethod
    def _img_to_float(img):
        return cv.normalize(img, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)

    @staticmethod
    def _float_to_img(fimg, original_numpy_dtype):
        cv_np_map = {np.dtype('uint8'): cv.CV_8U,
                     np.dtype('uint16'): cv.CV_16U,
                     np.dtype('int8'): cv.CV_8S,
                     np.dtype('int16'): cv.CV_16S
                     }
        cv_dtype = cv_np_map[original_numpy_dtype]
        alpha = np.iinfo(original_numpy_dtype).min
        beta = np.iinfo(original_numpy_dtype).max
        return cv.normalize(fimg, None, alpha, beta, cv.NORM_MINMAX, cv_dtype)

    
    def _diff_of_gaus(self, img, low_sigma: int = 5, high_sigma: int = 9):
        #TODO replace with difference of kernels
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
    #TODO depricated 
    def _pickle_keypoints(point):
        return cv.KeyPoint, (*point.pt, point.size, point.angle,
                              point.response, point.octave, point.class_id)

    def _find_features(self, img):

        processed_img = self._preprocess_image(img)
        if processed_img.max() == 0:
            return [], []
        #detector = cv.MSER_create()
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

        #fix problem with pickle
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

        print('good matches', len(good), '/', len(matches))
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

        img1_small = cv2.resize(img1, (int(img1.shape[1]/upsampling), int(img1.shape[0]/upsampling)))
        self._scaffold_image = img1_small
        img2_small = cv2.resize(img2, (int(img2.shape[1]/upsampling), int(img2.shape[0]/upsampling)))
        self._original_image = img2_small
        
        
        warp_matirx = self._match_features(self._find_features(img2_small), self._find_features(img1_small))
        warp_matirx = transform.EuclideanTransform(matrix=np.concatenate([warp_matirx[::,:], np.array([0, 0, 1])[None,:]])).params
        self._tform = warp_matirx
        #don't forget to scale
        warp_matirx = np.array([[upsampling,  0.,  0.],
                                [ 0., upsampling,  0.],
                                [ 0.,  0.,  1.]]) @\
                      np.linalg.inv(warp_matirx) @\
                      np.array([[1/upsampling,  0.,  0.],
                                [ 0., 1/upsampling,  0.],
                                [ 0.,  0.,  1.]])
            


        data = warp_matirx @ np.array([self.data['PosX'], self.data['PosY'], np.ones(self.data['PosY'].shape[0]) ])
        self.data['PosX'] = data[0,:]
        self.data['PosY'] = data[1,:]
        
        self.image = img_file
        self.spatial_dims = self.get_img_size(self.image) 
        if self.masks_svg is not None:
            self.ducts = self._get_ducts(self.masks_svg)
            
    def diagnostic_image_overlay(self):
        img1_small = self._scaffold_image
        img2_small = transform.warp(self._original_image, self._tform, output_shape=img1_small.shape)
        
        img_RGB = np.zeros((max(img1_small.shape[0], img2_small.shape[0]), max(img1_small.shape[1], img2_small.shape[1]), 3))
        img_RGB[:img1_small.shape[0],:img1_small.shape[1],0] = img1_small / (img1_small.max() / 2)
        img_RGB[:img2_small.shape[0],:img2_small.shape[1],1] = img2_small / (img2_small.max() / 2)

        plt.figure(figsize=(16,16))
        plt.imshow(img_RGB)
        
    def update_coords(self, warp_matrix_file, source_img, target_img, small_img_source, small_img_target):
        
        resizing_params = {'source': [np.array(self.get_img_size(source_img)), [small_img_source.shape[1], small_img_source.shape[0]]],
                           'target': [np.array(self.get_img_size(target_img)), [small_img_target.shape[1], small_img_target.shape[0]]]}
               
        warp = GridWarp_alingment(warp_matrix_file=warp_matrix_file,
                      source_coords=np.stack([self.data['PosX'], self.data['PosY']]),
                      resizing_params = resizing_params,
                      small_img_source = small_img_source,
                      small_img_target = small_img_target)
        
        self.data['PosX'], self.data['PosY'] = warp.warp_coords()
     
        warp = GridWarp_alingment(warp_matrix_file=warp_matrix_file,
                      source_coords=self.cellpos.T,
                      resizing_params = resizing_params,
                      small_img_source = small_img_source,
                      small_img_target = small_img_target)
        
        self.cellpos[:,0], self.cellpos[:,1] = warp.warp_coords()
        
        
        self.spatial_dims = resizing_params['target'][0]
        
        self.image = target_img
        self.spatial_dims = self.get_img_size(self.image) 
        if self.masks_svg is not None:
            self.ducts = self._get_ducts(self.masks_svg)
        
    def _get_ducts(self, svg_data):
        svg_path = svg_data
        
        paths, attributes = svg2paths(svg_path)
        
        size_source = self.get_img_size(self.image)
        size_svg = self.get_img_size(svg_path)
        scale = (np.array(size_source ) /  size_svg).mean(axis=0)
        #from svgpathtools import Path, Line, CubicBezier
        NUM_SAMPLES = 1000
        paths_interpol = []
        for path in paths:
            path_interpol = []
            for i in range(NUM_SAMPLES):
                path_interpol.append(path.point(i/(float(NUM_SAMPLES)-1)))
            paths_interpol.append(np.array([[j.real for j in path_interpol], [j.imag for j in path_interpol]]).T)

        for i in range(len(paths)):
            paths_interpol[i] = (np.concatenate([paths_interpol[i], np.ones(NUM_SAMPLES)[:,None]],axis=1) @  np.array([[scale,  0.,  0.],
                                        [ 0., scale,  0.],
                                        [ 0.,  0.,  1.]]))[:,:-1]
    
    
        return {'paths':paths_interpol, 'linetype':[attributes[i]['class'] for i in range(len(paths))]}
        
            
    def filter_by_ducts(self, subset=None):
        if self.masks_svg is None:
            print('no duct info')
            return None
        else:
            paths_interpol = np.array(self.ducts['paths'])
            
            if subset is None:
                subset = [True] * len(paths_interpol)
                
            paths_matplot = [mpltPath.Path(paths_interpol[subset][i]) for i in range(len(paths_interpol[subset]))]
            ifcontains_mut = [path.contains_points(np.array([self.data['PosX'], self.data['PosY']]).T) for path in paths_matplot]
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
        

def mask_infisble(mut_sample_list, scale, probability=0.6, critical_genes=False, plot=False):
    mask = []
    for i in range(len(mut_sample_list)):
        mut_sample_list[i].data_to_grid(scale_factor=scale, probability=0.6)
        t = np.array([s for s in mut_sample_list[i].gene_grid.values()])[:-3].sum(0)
        mask_infisiable = mut_sample_list[i].gene_grid['infeasible']/t < 0.1
        mask_infisiable *= mut_sample_list[i].cell_grid > 5
        
        if critical_genes:
            if i == 0:
                mask_infisiable *= mut_sample_list[i].gene_grid["PTEN2mut"] + mut_sample_list[i].gene_grid["LRP1Bmut"] + mut_sample_list[i].gene_grid["NOB1wt"] <= 3 

        if plot:
            plt.figure(figsize=(8,4))
            plt.imshow(mask_infisiable.T[::-1,:])
            
        mask.append(mask_infisiable.flatten())
            
    return mask
    
def generate_data4model(samples_list, genes, M, n_aug=1):
    n_samples = len(samples_list)
    n_genes = len(genes)

    iss_data = [np.transpose(np.array([samples_list[i].gene_grid[k] for k in genes]), [1,2,0]).reshape(-1, n_genes) for i in range(n_samples)]

    tiles_axes = [samples_list[i].tile_axis for i in range(n_samples)]

    cells_counts = [samples_list[i].cell_grid.flatten() for i in range(n_samples)]
    sample_dims = [(int(tiles_axes[i][0][-1]+1), int(tiles_axes[i][1][-1]+1)) for i in range(n_samples)]
    n_factors = M.shape[0]
    n_aug=1
    
    return {'iss_data': iss_data, 'tiles_axes': tiles_axes, 'cells_counts': cells_counts, 'sample_dims': sample_dims,
            'n_factors': n_factors, 'n_aug': n_aug, 'tree_matrix':M, 'n_samples': n_samples, 'n_genes': n_genes, 'genes': genes}

def softmax(x, t=2, axis=1):
    end_shape = list(x.shape)
    end_shape[axis] = -1
    return np.exp(x / t) / np.exp(x / t).sum(axis=axis).reshape(*end_shape)
    
    
    
def stickbreaking(y_):
    y = y_.T
    y = np.concatenate([y, -np.sum(y, 0, keepdims=True)])
    e_y = np.exp(y - np.max(y, 0, keepdims=True))
    x = e_y / np.sum(e_y, 0, keepdims=True)
    return x.T

def multilogit(y_):
    y = y_.T
    y = np.concatenate([y, np.zeros_like(y)])
    e_y = np.exp(y)
    x = e_y / np.sum(e_y, 0, keepdims=True)
    return x.T


def pplr(X, Y, by=1):
    #Y = Y+1e-10
    n = X.shape[0]
    pr = 1-np.stack([(X / Y > by).sum(axis=0) / n, (X / Y < 1/by).sum(axis=0) / n], axis=1).max(axis=1)
    #pr = 1-np.stack([(X / Y > by).sum(axis=0) / n, (X / Y < 1/by).sum(axis=0) / n], axis=1)
    ##pr[pr > 0.5] = 1 - pr[pr > 0.5]
    pr[pr == 0] = 1/n
    return pr

def qval(pv, verbose=True):
    
    m = pv.size
    if True:# pv.size < 100:
        pi0 = 1
    else:
        # evaluate pi0 for different lambdas
        pi0 = []
        lam = np.arange(0, 0.90, 0.01)
        counts = np.array([(pv > i).sum() for i in np.arange(0, 0.9, 0.01)])
        for l in range(len(lam)):
            pi0.append(counts[l]/(m*(1-lam[l])))

        pi0 = np.array(pi0)

        # fit natural cubic spline
        tck = sp.interpolate.splrep(lam, pi0, k=3)
        pi0 = sp.interpolate.splev(lam[-1], tck)
        if verbose:
            print("qvalues pi0=%.3f, estimated proportion of null features " % pi0)

        if pi0 > 1:
            if verbose:
                print("got pi0 > 1 (%.3f) while estimating qvalues, setting it to 1" % pi0)
            pi0 = 1.0

        assert(pi0 >= 0 and pi0 <= 1), "pi0 is not between 0 and 1: %f" % pi0

    p_ordered = np.argsort(pv)
    pv = pv[p_ordered]
    qv = pi0*m/len(pv) * pv
    qv[-1] = min(qv[-1], 1.0)

    for i in range(len(pv)-2, -1, -1):
        qv[i] = min(pi0*m*pv[i]/(i+1.0), qv[i+1])

    # reorder qvalues
    qv_temp = qv.copy()
    qv = np.zeros_like(qv)
    qv[p_ordered] = qv_temp
    return qv

def bonferonni(pv, m=None):
    if m is None:
        m = len(pv)
    return np.minimum(pv*m, 1)


def fdr(p_vals):

    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr


def add_comparison_to_dict(dict2add, comp_name, dat, p_val, gene_names, panel):
    n = len(gene_names)
    dict2add['gene'] += list(gene_names)
    dict2add['panel'] += list(panel)
    dict2add['comparison'] += [comp_name] * n
    dict2add['50pct'] += list(np.percentile(dat, 50, axis=0))
    dict2add['2.5pct'] += list(np.percentile(dat, 2.5, axis=0))
    dict2add['97.5pct'] += list(np.percentile(dat, 97.5, axis=0))
    dict2add['p_vals'] += list(p_val)
    dict2add['p_vals_adj'] += list(bonferonni(p_val, p_val.size))
    
    return dict2add

class Beta_sum(pm.Beta):
    def __init__(self, n=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
    def logp(self,value):
        return super().logp(value) * self.n
    