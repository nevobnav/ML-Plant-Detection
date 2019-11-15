#!/usr/bin/python3.6

import os
import gdal
import numpy as np
import rasterio
from rasterio.windows import Window

R_EARTH = 6371000.		# radius of earth in meters

def clip_ortho2shp_array(input_file, clip_shp, nodata=None, out=''):
	"""Converts clip_shape to GDAL object."""
	output_file = out
	shape_path = clip_shp
	shape_name = os.path.basename(clip_shp)[:-4]
	input_object = gdal.Open(input_file)	
	ds = gdal.Warp(output_file,
                   input_object,
                   format = 'VRT',
                   cutlineDSName = shape_path,
                   cutlineLayer = shape_name,
                   warpOptions=['NUM_THREADS=ALL_CPUS'],
                   multithread=True,
                   warpMemoryLimit=3000,
                   dstNodata = nodata,
                   transformerOptions=['NUM_THREADS=ALL_CPUS'])
	return ds

def get_clip_slice(clip, i_img, j_img, num_rows, num_cols):
	r = np.array(clip.GetRasterBand(1).ReadAsArray(j_img, i_img, num_cols, num_rows), dtype=np.uint8)
	g = np.array(clip.GetRasterBand(2).ReadAsArray(j_img, i_img, num_cols, num_rows), dtype=np.uint8)
	b = np.array(clip.GetRasterBand(3).ReadAsArray(j_img, i_img, num_cols, num_rows), dtype=np.uint8)
	return np.dstack((r,g,b))

def get_functions(img_path, dem_path, clip_path=None):
	if clip_path != None:
		clip_ortho2shp_array(img_path, clip_path, out='.tmp.vrt')
		img = rasterio.open('.tmp.vrt')
	else:
		img = rasterio.open(img_path)

	dem = rasterio.open(dem_path)

	def img_rowcol(x, y):
		return img.index(x, y, op=(lambda x : x))

	def get_dem_pixel(i_img, j_img):
		return dem.index(*img.xy(i_img, j_img))

	def get_color_slice(i_img, j_img, num_rows, num_cols):
		"""Returns (num_rows, num_cols, 3) numpy array represting the RGB-slice i_img:i_img+num_rows, j_img:j_img+num_cols."""
		r = img.read(1, window=Window(j_img, i_img, num_cols, num_rows))#, dtype = np.uint(8))
		g = img.read(2, window=Window(j_img, i_img, num_cols, num_rows))#, dtype = np.uint(8))
		b = img.read(3, window=Window(j_img, i_img, num_cols, num_rows))#, dtype = np.uint(8))
		return np.dstack((r,g,b))

	def get_height_slice(i_img, j_img, num_rows, num_cols):
		"""Returns the H-slice corresponding to the RGB slice returned by get_color_slice(i_img, j_img, num_rows, num_cols)."""
		i_dem, j_dem = get_dem_pixel(i_img, j_img)
		rows2 = int(np.round(num_rows*img.transform[0]/dem.transform[0]))
		cols2 = int(np.round(num_cols*img.transform[4]/dem.transform[4]))
		h = dem.read(1, window=Window(j_dem, i_dem, cols2, rows2))#, dtype=np.uint(8))
		return h

	def get_slices(i_img, j_img, num_rows, num_cols):
		"""Get both RGB and H slices. (i_img,j_img) is the center point in the RGB image, where i_img is the row and j_img in the column."""
		c_im =  get_color_slice(i_img-num_rows//2, j_img-num_cols//2, num_rows, num_cols)
		h_im = get_height_slice(i_img-num_rows//2, j_img-num_cols//2, num_rows, num_cols)
		return c_im, h_im

	def get_adjusted_window(i_img, j_img, num_rows, num_cols):
		"""If part of the window is outside of the RGB tif, the window is adjusted. The new window extent is returned."""
		if i_img<0:											# north border
			num_rows = num_rows-abs(i_img)
			i_img = max(0, i_img)
		elif i_img+num_rows>img.height:					# sourth border
			num_rows = min(img.height - i_img, num_rows)
		if j_img<0:											# west border
			num_cols = num_cols-abs(j_img)
			j_img = max(0, j_img)
		elif j_img+num_cols>img.width:					# east border
			num_cols = min(img.width-j_img, num_cols)
		return i_img, j_img, num_rows, num_cols

	def get_block(i_img, j_img, num_rows, num_cols):
		"""Gets the window [i_img:i_img+num_rows, j_img:j_img+num_cols] from RGB tif and the corresponding H slice."""
		c_im =  get_color_slice(i_img, j_img, num_rows, num_cols)
		h_im = get_height_slice(i_img, j_img, num_rows, num_cols)
		return c_im, h_im

	func_dict = dict()
	for f in [get_dem_pixel, get_slices, get_adjusted_window, get_block, img_rowcol]:
		func_dict[f.__name__] = f
	func_dict['transform'] = img.transform
	func_dict['scale_factor'] = img.transform[0]/dem.transform[0]
	return func_dict


if __name__ == "__main__":
	name = 'c01_verdonk-Wiering Boerderij-201907230919'
	GR = False
	img_path = r"../../Orthomosaics/"+name+GR*'-GR'+r'/'+name+GR*'-GR'+r".tif"
	dem_path = r"../../Orthomosaics/"+name+GR*'-GR'+r'/'+name+r"_DEM"+GR*'-GR'+".tif"
	clp_path = r"../../Orthomosaics/"+name+GR*'-GR'+r'/'+name+GR*'-GR'+r"_FIELD.shp"


	fs = get_functions_rasterio(img_path, dem_path)#, clp_path)
	print(fs['get_dem_pixel'](10000,10000))

