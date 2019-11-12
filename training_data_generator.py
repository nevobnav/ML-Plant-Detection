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

def get_empty_mask(box_size):
	# poly = [(xs[k], ys[k]) for k in range(len(xs))]
	tmp = Image.new('L', (box_size, box_size), 0)
	# ImageDraw.Draw(tmp).polygon(poly, outline=1, fill=fill)
	return tmp

def save_ims(ims, targets, label):
	c_im, h_im, mask  = ims
	dir_color, dir_height, dir_masks = targets
	h_im = (h_im - h_im.min())
	h_im *= 255/h_im.max()			# normalize height
	c_im = c_im.astype(np.uint8)
	h_im = h_im.astype(np.uint8)
	cimim = Image.fromarray(c_im, 'RGB')
	himim = Image.fromarray(h_im, 'L')
	c_save_path = dir_color +'c_'+label+'.png'
	h_save_path = dir_height+'h_'+label+'.png'
	m_save_path = dir_masks +'m_'+label+'.png'
	cimim.save(c_save_path)
	himim.save(h_save_path)
	mask.save(m_save_path)

def generate_data_from_pickle(data_path, target_dir='.', crop_name='Broccoli', min_confidence=0.99, mask_fill=1):
	with open(data_path, 'rb') as p:
		data_dict = pickle.load(p)

	params = data_dict['metadata']
	img_path, dem_path, clp_path = params['input_tif'], params['input_dem'], params['input_clp']
	del data_dict['metadata']
	box_size = params['box_size']

	funcs = tif_functions.get_functions(img_path, dem_path, clp_path)
	get_block = funcs['get_block']

	dir_color  = target_dir+'/Training Data Color/'+crop_name+'/'
	dir_height = target_dir+'/Training Data Height/'+crop_name+'/'
	dir_masks  = target_dir+'/Training Data Mask/'+crop_name+'/'
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
				xs = contours[k][:,1] - pnt[0] + box_size/2.
				ys = contours[k][:,0] - pnt[1] + box_size/2.
				mask = get_mask(xs, ys, box_size, fill=mask_fill)
				save_ims((c_im, h_im, mask), (dir_color, dir_height, dir_masks), label=os.path.basename(img_path).split('.')[0]+'_{}_{}_{}'.format(row,col,k))

				count += 1

	print('{} images saved'.format(count))
	return count

def generate_background_data(BG_DATA, img_path, dem_path, clp_path, box_size, target_dir='.', min_confidence=0.99, max_count=np.infty):
	with open(BG_DATA, 'rb') as p:
		data_dict = pickle.load(p)

	crop_name = 'Background'
	dir_color  = target_dir+'/Training Data Color/'+crop_name+'/'
	dir_height = target_dir+'/Training Data Height/'+crop_name+'/'
	dir_masks  = target_dir+'/Training Data Mask/'+crop_name+'/'
	for direc in [dir_color, dir_masks, dir_height]:
		if not os.path.exists(direc):
			os.makedirs(direc)

	print('{} crop images saved'.format(count))
	max_count = int(bg_factor*count)
	count = 0

	for (i,j) in data_dict:
		boxes = data_dict[(i,j)]['background_boxes']
		(i_ad, j_ad, height, width) = data_dict[(i,j)]['block']
		probs = data_dict[(i,j)]['background_confidence']

		for (k, box) in enumerate(boxes):
			if probs[k] > min_confidence:
				row = box[1] + i_ad
				col = box[0] + j_ad
				c_im, h_im = get_block(row, col, box_size, box_size)
				mask = get_empty_mask(box_size)
				save_ims((c_im, h_im, mask), (dir_color, dir_height, dir_masks), label=os.path.basename(img_path).split('.')[0]+'_{}_{}_{}'.format(row,col,k))

				count += 1
				if count >= max_count:
					print('{} bg images saved'.format(count))
					return count

	print('{} images saved'.format(count))

def generate_data(CONTOURS, POINTS, BG_DATA, img_path, dem_path, clp_path, box_size, crop_name='Broccoli', bg_factor=0.5, target_dir='.', min_confidence=0.99, max_count=np.infty):
	dir_color  = target_dir+'/Training Data Color/'+crop_name+'/'
	dir_height = target_dir+'/Training Data Height/'+crop_name+'/'
	dir_masks  = target_dir+'/Training Data Mask/'+crop_name+'/'
	for direc in [dir_color, dir_masks, dir_height]:
		if not os.path.exists(direc):
			os.makedirs(direc)

	spatial_functions = tif_functions.get_functions(img_path, dem_path, clp_path)
	img_rowcol = spatial_functions['img_rowcol']
	get_block  = spatial_functions['get_block']
	shp_cnts = fiona.open(CONTOURS)
	shp_pnts = fiona.open(POINTS)

	count = 0
	for k in range(len(shp_pnts)):
		pnt, cnt = shp_pnts[k], shp_cnts[k]
		if pnt['properties']['name'] != cnt['properties']['name']:
			raise IndexError('Point and Contour do not have the same name; datasets not of the same form.')
		pnt_xy = pnt['geometry']['coordinates']
		cnt_xy = cnt['geometry']['coordinates'][0]
		pnt_ij = img_rowcol(*pnt_xy)
		cnt_ij = np.array([list(img_rowcol(x, y)) for (x,y) in cnt_xy])
		relative_cnt = cnt_ij - np.ones(cnt_ij.shape)*(pnt_ij[0]-box_size/2, pnt_ij[1]-box_size/2)
		row, col = (np.round(pnt_ij)-np.array([box_size/2, box_size/2])).astype(int)

		c_im, h_im = get_block(row, col, box_size, box_size)
		mask = get_mask(relative_cnt[:,1], relative_cnt[:,0], box_size, fill=1)
		save_ims((c_im, h_im, mask), (dir_color, dir_height, dir_masks), label=os.path.basename(img_path).split('.')[0]+'_{}_{}_{}'.format(row,col,k))

		count += 1
		if count >= max_count:
			break

	with open(BG_DATA, 'rb') as p:
		data_dict = pickle.load(p)

	print('saved {} images of class "{}"'.format(count, crop_name))


	crop_name = 'Background'
	dir_color  = target_dir+'/Training Data Color/'+crop_name+'/'
	dir_height = target_dir+'/Training Data Height/'+crop_name+'/'
	dir_masks  = target_dir+'/Training Data Mask/'+crop_name+'/'
	for direc in [dir_color, dir_masks, dir_height]:
		if not os.path.exists(direc):
			os.makedirs(direc)

	max_count = int(bg_factor*count)
	count = 0

	for (i,j) in data_dict:
		boxes = data_dict[(i,j)]['background_boxes']
		(i_ad, j_ad, height, width) = data_dict[(i,j)]['block']
		probs = data_dict[(i,j)]['background_confidence']

		for (k, box) in enumerate(boxes):
			if probs[k] > min_confidence:
				row = box[1] + i_ad
				col = box[0] + j_ad
				c_im, h_im = get_block(row, col, box_size, box_size)
				mask = get_empty_mask(box_size)
				save_ims((c_im, h_im, mask), (dir_color, dir_height, dir_masks), label=os.path.basename(img_path).split('.')[0]+'_{}_{}_{}'.format(row,col,k))

				count += 1
				if count >= max_count:
					# print('{} bg images saved'.format(count))
					return count

	print('saved {} images of class "{}"'.format(count, 'Background'))

def merge_training_sets(path1, path2):
	"""path1 and path2 should contain a number of class folders, which in turn contain the follwing folders:
		- Training Data Color
		- Training Data Height
		- Training Data Mask """
	src_color,  dst_color  = path1 + '/Training Data Color/',  path2 + '/Training Data Color/'
	src_height, dst_height = path1 + '/Training Data Height/', path2 + '/Training Data Height/'
	src_mask,   dst_mask   = path1 + '/Training Data Mask/',   path2 + '/Training Data Mask/'
	for (src, dst) in [(src_color, dst_color), (src_height, dst_height), (src_mask, dst_mask)]:
		for class_name in os.listdir(src):
			for im_name in os.listdir(src+class_name):
			    full_file_name = os.path.join(src+class_name, im_name)
			    if os.path.isfile(full_file_name):
			        shutil.move(full_file_name, dst+class_name)

if __name__ == "__main__":
	# name = 'c01_verdonk-Wever oost-201907240707' # 'c01_verdonk-Wever oost-201907240707' #
	# GR = True
	# img_path = r"../../Orthomosaics/"+name+GR*'-GR'+r'/'+name+GR*'-GR'+r".tif"
	# dem_path = r"../../Orthomosaics/"+name+GR*'-GR'+r'/'+name+r"_DEM"+GR*'-GR'+".tif"
	# clp_path = r"../../Orthomosaics/"+name+GR*'-GR'+r'/'+name+GR*'-GR'+r"_FIELD.shp"
	# input_contours = r"../../Orthomosaics/"+name+GR*'-GR'+r'/Plant Count/CONTOURS.shp'
	# input_centers  = r"../../Orthomosaics/"+name+GR*'-GR'+r'/Plant Count/POINTS.shp'
	# picklepath = r"../../Orthomosaics/"+name+GR*'-GR'+r'/Plant Count/BG_DATA.pickle'

	name = "c01_verdonk-Rijweg stalling 2-201907170908-GR"
	img_path = r"D:\\Old GR\\c01_verdonk-Rijweg stalling 2-201907170908-GR.tif"
	dem_path = r"D:\\Old GR\\c01_verdonk-Rijweg stalling 2-201907170908_DEM-GR.tif"
	clp_path = r".\\Field Shapefiles\\c01_verdonk-Rijweg stalling 2-201907170908-GR_FIELD.shp"
	input_contours = r"..\\PLANT COUNT - "+name+r"\\CONTOURS.shp"
	input_centers  = r"..\\PLANT COUNT - "+name+r"\\POINTS.shp"
	picklepath     = r"..\\PLANT COUNT - "+name+r"\\BG_DATA.pickle"

	target = r"..\\GeneratedTrainingData"
	generate_data(input_contours, input_centers, picklepath, img_path, dem_path, clp_path, 50, bg_factor=1, target_dir=target, max_count=1000)
	# generate_background_data(picklepath, img_path, dem_path, clp_path, 50, target_dir=target, max_count=1000)
