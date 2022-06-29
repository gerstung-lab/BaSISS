import numpy as np
import pandas as pd 
import re

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
import cv2 as cv

from svgpathtools import svg2paths, wsvg
import matplotlib.path as mpltPath
from copy import copy, deepcopy
import tifffile

from tqdm import tqdm

import subprocess
import matplotlib.pyplot as plt

class Sample:
    """A strucutral class to store bassis/iss experiment data.

    ...

    Attributes
    ----------
    image : str
        path to background image
    data : dict
        iss data with 'Gene', 'PosX', 'PosY' information
    spatial_dims : tuple
        age of the person
    genes : list
        list of genes found in the sample
    cell_data : list 
        segmented nulcei positions
    masks_svg : str
        paht to svg with regions 
    ducts : dict
        region's ids and approximated contour
    tile_axis : list
        x, y indices of grid
    gene_grid : dict
        binned counts for each genes
    cell_grid : np.array
        binned nuclei counts
    
    
    Methods
    -------
    get_img_size(filename):
        determines size of the image
        
    data_to_grid(scale_factor=4, gene_list='all', probability=None):
        defines the square grid and bins data
        
    transform_points2background(img_file, upsampling=5):
        registers image used for decoding on the bachround and moves iss signals
        
    diagnostic_image_overlay():
        diagnostics if images have been correctly registerd
        
    add_gene_data():
        merges with another sample, used when samples were too large for decoding and had to be split
    
    """
    def __init__(self, iss_data, image, cell_data=None, masks_svg=None):
        """
        Consturctor
        
        Parameters
        ----------
            iss_data : str
                path to iss data
            image : str
                path to background image
            cell_data : str
                path to segmented nuclei data
            
        """
        
        self._iss_df_raw = pd.read_csv(iss_data)
        self.image = image
        self.spatial_dims = self.get_img_size(self.image)
        self.data = self._standardise_data()
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
    
    def _standardise_data(self):
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
        original_dtype = copy(img.dtype)
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

        img1_small = cv.resize(img1, (int(img1.shape[1]/upsampling), int(img1.shape[0]/upsampling)))
        self._scaffold_image = img1_small
        img2_small = cv.resize(img2, (int(img2.shape[1]/upsampling), int(img2.shape[0]/upsampling)))
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
           
    def add_gene_data(self, sample):
        for k in self.data.keys():
            self.data[k] = np.concatenate([self.data[k], sample.data[k]])
            
        self.iss_probability = np.concatenate([self.iss_probability, sample.iss_probability])
