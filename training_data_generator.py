#!/usr/bin/python3.6
"""
	This script creates a training dataset based on the output of the crop detection algorithm.
	The input should be the file DATA.pickle, which contains the points and contours for each detected crop,
	as well as a dictionary containing the simulation parameters under the key 'metadata'.
"""

import os
import numpy as np 
import matplotlib.pyplot as plt 
import skimage
from PIL import Image
import cv2
import pickle
import rasterio
from shapely.geometry import Polygon, Point, mapping, shape, MultiPolygon
import fiona
from fiona.crs import from_epsg
from PIL import Image, ImageDraw

import tif_functions

def get_mask(xs, ys, box_size, fill=1):
	poly = [(xs[k], ys[k]) for k in range(len(xs))]
	tmp = Image.new('L', (box_size, box_size), 0)
	ImageDraw.Draw(tmp).polygon(poly, outline=1, fill=fill)
	return tmp

def generate_data(data_path, target_dir='.', crop='Broccoli', min_confidence=0.99, mask_fill=1):
	with open(data_path, 'rb') as p:
		data_dict = pickle.load(p)

	params = data_dict['metadata']
	img_path, dem_path, clp_path = params['input_tif'], params['input_dem'], params['input_clp']
	del data_dict['metadata']
	box_size = params['box_size']

	funcs = tif_functions.get_functions(img_path, dem_path, clp_path)
	get_block = funcs['get_block']

	dir_color  = target_dir+'/Training Data Color/'+crop+'/'
	dir_height = target_dir+'/Training Data Height/'+crop+'/'
	dir_masks  = target_dir+'/Training Data Mask/'+crop+'/'
	for direc in [dir_color, dir_masks, dir_height]:
		if not os.path.exists(direc):
			os.makedirs(direc)

	count = 0
	for (i,j) in data_dict:
		contours  = data_dict[(i,j)]['contours']
		centroids = data_dict[(i,j)]['centroids']
		(i_ad, j_ad, height, width) = data_dict[(i,j)]['block']
		probs = data_dict[(i,j)]['confidence']

		for (k, pnt) in enumerate(centroids):
			if probs[k] > min_confidence:
				row = int(pnt[1] + i_ad - box_size/2.)
				col = int(pnt[0] + j_ad - box_size/2.)
				c_im, h_im = get_block(row, col, box_size, box_size)
				h_im = h_im - h_im.min()
				h_im *= 255/h_im.max()			# normalize height
				c_im = c_im.astype(np.uint8)
				h_im = h_im.astype(np.uint8)
				cimim = Image.fromarray(c_im, 'RGB')
				himim = Image.fromarray(h_im, 'L')
				cimim.save(dir_color +'c_{}_{}_{}.png'.format(i,j,k))
				himim.save(dir_height+'h_{}_{}_{}.png'.format(i,j,k))
				xs = contours[k][:,1] - pnt[0] + box_size/2. 
				ys = contours[k][:,0] - pnt[1] + box_size/2.
				mask = get_mask(xs, ys, box_size, fill=mask_fill)
				mask.save(dir_masks+'m_{}_{}_{}.png'.format(i,j,k))
				count += 1

	print('{} images saved'.format(count))

if __name__ == "__main__":
	name = 'c01_verdonk-Wever oost-201907240707' # 'c01_verdonk-Wever oost-201907240707' #
	GR = True
	img_path = r"../../Orthomosaics/"+name+GR*'-GR'+r'/'+name+GR*'-GR'+r".tif"
	dem_path = r"../../Orthomosaics/"+name+GR*'-GR'+r'/'+name+r"_DEM"+GR*'-GR'+".tif"
	clp_path = r"../../Orthomosaics/"+name+GR*'-GR'+r'/'+name+GR*'-GR'+r"_FIELD.shp"
	generate_data('Plant Count/DATA.pickle')#, img_path, dem_path, clp_path)