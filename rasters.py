#!/usr/bin/python3.6

"""
Script containing the RasterCommunicator class, that allows one to
communicate between RGB and DEM recordings of different resolutions
"""

import os

import numpy as np
import gdal
import fiona
import rasterio
import shapely.geometry as geometry

__author__ = "Duncan den Bakker"

def clip_ortho2shp_array(input_file, clip_shp, nodata=None, out=''):
	"""Clips input_file to shapes in clip_shp.

	It might print something along the lines of
	pj_obj_create: Open of ... failed
	This can be ignored.

	Arguments
	---------
	input_file : str (or path object)
		File to be clipped.
	clip_shp : str (or path object)
		Shapefile containing clip shape(s).
	nodata : ?, optional
		No-data value to be used. Default is None.
	out : str (or path object), optional
		Save the resulting clipped VRT to this location.
		Default is ''.

	Returns
	-------
	ds : gdal.Dataset
		Gdal representation of clipped file.
	"""
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

class RasterCommunicator(object):
	"""
	Object that allows for communication between RGB, DEM and clip files.

	Attributes
	----------
	rgb_path : str (or path object)
		Path to RGB file. Should point to either a .tif or .vrt file.
	dem_path : str (or path object)
		Path to DEM file. Should point to either a .tif or .vrt file.
	clip_path : str (or path object)
		Path to clipped field file. Should point to a .shp file. The
		shapefile should contain Polygons covering the part of the
		RGB file that should be analyzed.
	rgb : rasterio.io.DatasetReader
		Rasterio object from which RGB data can be read.
	dem : rasterio.io.DatasetReader
		Rasterio object from which DEM data can be read.
	transform : affine.Affine
		RGB geotransform.
	scale_factor : float
		Relative scale difference between lengths in RGB and DEM.

	Methods
	-------
	rgb_index(x, y)
		Returns RGB pixel location corresponding to a lon-lat point.
	rgb_to_dem(i_rgb, j_rgb)
		Returns DEM-pixel location corresponding to RGB-pixel location.
	get_rgb_block(i_rgb, j_rgb, rows_rgb, cols_rgb)
		Gets RGB block data.
	get_dem_block(i_rgb, j_rgb, rows_rgb, cols_rgb)
		Gets DEM block data.
	get_blocks(i_rgb, j_rgb, rows_rgb, cols_rgb)
		Gets both RGB and DEM block data.
	adjust_window(i_rgb, j_rgb, rows_rgb, cols_rgb)
		Returns an adjusted window that lies completely within RGB raster.
	get_field_blocks(block_size, block_overlap, max_count=None):
		Generates a dictionary containing blocks that intersect the specified
		clipped field.
	get_clip_polygons()
		Returns a MultiPolygon representing the clipped field.
	"""

	def __init__(self, rgb_path, dem_path, clip_path):
		"""
		Loads RGB and DEM files such that they can be accessed.

		Clips the input RGB file according to the polygons in the clip
		file. A temporary (hidden) file .tmp.vrt is created containing
		clipped view of RGB file, using the function clip_ortho2shp_array.

		Arguments
		---------
		rgb_path : str (or path object)
			Path to RGB file. Should point to either a .tif or .vrt file.
		dem_path : str (or path object)
			Path to DEM file. Should point to either a .tif or .vrt file.
		clip_path : str (or path object)
			Path to clipped field file. Should point to a .shp file. The
			shapefile should contain Polygons covering the part of the
			RGB file that should be analyzed.
		"""

		self.rgb_path = rgb_path
		self.dem_path = dem_path
		self.clip_path = clip_path

		clip_ortho2shp_array(self.rgb_path, self.clip_path, out='.tmp.vrt')
		self.rgb = rasterio.open('.tmp.vrt')
		self.dem = rasterio.open(dem_path)

		self.transform = self.rgb.transform
		self.scale_factor = self.rgb.transform[0]/self.dem.transform[0]

	def rgb_index(self, x, y):
		"""
		Returns RGB pixel location corresponding to a lon-lat point.

		Arguments
		---------
		x : float
			Longitude.
		y : float
			Latitude.

		Returns
		-------
		i_rgb : float
			Fractional RGB pixel location.
		j_rgb : float
			Fractional RGB pixel location.
		"""

		return self.rgb.index(x, y, op=(lambda x : x))

	def rgb_to_dem(self, i_rgb, j_rgb):
		"""
		Returns DEM-pixel location corresponding to RGB-pixel location.

		Arguments
		---------
		i_rgb : int
			Row index in RGB raster.
		j_rgb : int
			Column index in RGB raster.

		Returns
		-------
		i_dem : int
			Row index in DEM raster.
		j_dem : int
			Column index in DEM raster.
		"""

		return self.dem.index(*self.rgb.xy(i_rgb, j_rgb))

	def get_rgb_block(self, i_rgb, j_rgb, rows_rgb, cols_rgb):
		"""
		Gets RGB block data.

		Arguments
		---------
		i_rgb : int
			Row index of block in RGB raster.
		j_rgb : int
			Column index of block in DEM raster.
		rows_rgb : int
		 	Number of rows the block spans in terms of RGB-pixels.
		cols_rgb : int
			Number of columns the block spans in terms of RGB-pixels.

		Returns
		-------
		rgb_block : (?,?,3) array
			Numpy array containing RGB data of block.
		"""

		r = self.rgb.read(1, window=rasterio.windows.Window(j_rgb, i_rgb, cols_rgb, rows_rgb))
		g = self.rgb.read(2, window=rasterio.windows.Window(j_rgb, i_rgb, cols_rgb, rows_rgb))
		b = self.rgb.read(3, window=rasterio.windows.Window(j_rgb, i_rgb, cols_rgb, rows_rgb))
		return np.dstack((r,g,b))

	def get_dem_block(self, i_rgb, j_rgb, rows_rgb, cols_rgb):
		"""
		Gets DEM block data from corresponding RGB-block dimensions.

		Arguments
		---------
		i_rgb : int
			Row index of block in RGB raster.
		j_rgb : int
			Column index of block in DEM raster.
		rows_rgb : int
		 	Number of rows the block spans in terms of RGB-pixels.
		cols_rgb : int
			Number of columns the block spans in terms of RGB-pixels.

		Returns
		-------
		dem_block : (?,?) array
			Numpy array containing DEM data of block. Typically of a
			different (spatial) size as rgb_block due to resolution
			differences between RGB and DEM files.
		"""

		i_dem, j_dem = self.rgb_to_dem(i_rgb, j_rgb)
		rows2 = int(np.round(rows_rgb*self.rgb.transform[0]/self.dem.transform[0]))
		cols2 = int(np.round(cols_rgb*self.rgb.transform[4]/self.dem.transform[4]))
		dem_block = self.dem.read(1, window=rasterio.windows.Window(j_dem, i_dem, cols2, rows2))#, dtype=np.uint(8))
		return dem_block

	def get_blocks(self, i_rgb, j_rgb, rows_rgb, cols_rgb):
		"""
		Gets blocks from both RGB and DEM files.

		Arguments
		---------
		i_rgb : int
			Row index of block in RGB raster.
		j_rgb : int
			Column index of block in DEM raster.
		rows_rgb : int
		 	Number of rows the block spans in terms of RGB-pixels.
		cols_rgb : int
			Number of columns the block spans in terms of RGB-pixels.

		Returns
		-------
		rgb_block : (?,?,3) array
			Numpy array containing RGB data of block.
		dem_block : (?,?) array
			Numpy array containing DEM data of block. Typically of a
			different (spatial) size as rgb_block due to resolution
			differences between RGB and DEM files.
		"""

		rgb_block = self.get_rgb_block(i_rgb, j_rgb, rows_rgb, cols_rgb)
		dem_block = self.get_dem_block(i_rgb, j_rgb, rows_rgb, cols_rgb)
		return rgb_block, dem_block

	def adjust_window(self, i_rgb, j_rgb, rows_rgb, cols_rgb):
		"""
		Adjusts window dimensions if it crosses the RGB file boundaries.

		Arguments
		---------
		i_rgb : int
			Row index of block in RGB raster.
		j_rgb : int
			Column index of block in DEM raster.
		rows_rgb : int
		 	Number of rows the block spans in terms of RGB-pixels.
		cols_rgb : int
			Number of columns the block spans in terms of RGB-pixels.

		Returns
		-------
		(i_adj, j_adj, rows_adj, cols_adj) : tuple
			Adjusted window dimensions.
		"""

		if i_rgb < 0:											# north border
			rows_rgb = rows_rgb-abs(i_rgb)
			i_rgb = max(0, i_rgb)
		elif i_rgb+rows_rgb > self.rgb.height:					# sourth border
			rows_rgb = min(self.rgb.height - i_rgb, rows_rgb)
		if j_rgb < 0:											# west border
			cols_rgb = cols_rgb-abs(j_rgb)
			j_rgb = max(0, j_rgb)
		elif j_rgb+cols_rgb > self.rgb.width:					# east border
			cols_rgb = min(self.rgb.width-j_rgb, cols_rgb)
		return i_rgb, j_rgb, rows_rgb, cols_rgb

	def get_field_blocks(self, block_size, block_overlap, max_count=None):
		"""
		Generates a dictionary containing blocks that intersect the specified
		clipped field.

		Arguments
		---------
		block_size : int
			Width and height of block in terms of RGB-pixels.
		block_overlap : int
			Width or height of the overlapping region between two neighbouring
			blocks in terms of RGB-pixels. Used to prevent missing crops near
			box boundaries. Typically set to 2 times the bounding box size.
		max_count : int, optional
			Maximum number of blocks to return. Default value is None, in this
			case all the valid blocks are returned.

		Returns
		-------
		valid_blocks : dict
			Dictionary containing blocks that intersect the clipped field. Its
			keys are tuples representing the block location within a grid. For
			example, block (n,m-1) is the east neighbour of block (n,m). The
			corresponding value is a 4-tuple of the form (i, j, rows, cols),
			where (i,j) is the top left corner of the block, and rows, cols
			are its height and width, all in terms of RGB-pixels.
		"""

		field_shape = fiona.open(self.clip_path)
		field_polygons = []
		for feature in field_shape:
			poly = geometry.shape(feature['geometry'])
			field_polygons.append(poly)
		cols_rgb = self.rgb.width//block_size
		rows_rgb = self.rgb.height//block_size

		valid_blocks = dict()

		if max_count == None:
			max_count = (rows_rgb+1)*(cols_rgb+1)

		count = 0
		for i in range(0,rows_rgb+1):
			for j in range(0,cols_rgb+1):
				i_ad, j_ad, height, width = self.adjust_window(i*block_size-block_overlap//2,
				 											   j*block_size-block_overlap//2,
														       block_size+block_overlap,
															   block_size+block_overlap)
				block_vertices = [(i_ad, j_ad), (i_ad+height, j_ad), (i_ad+height, j_ad+width), (i_ad, j_ad+width)]
				transformed_vertices = geometry.Polygon([self.transform*(a,b) for (b,a) in block_vertices])

				for field_poly in field_polygons:
					if field_poly.intersects(transformed_vertices):
						valid_blocks[(i,j)] = (i_ad, j_ad, height, width)
						count += 1
						break

				if count >= max_count:
					print('Found {} valid blocks of a total of {}'.format(len(valid_blocks), (rows_rgb+1)*(cols_rgb+1)))
					return valid_blocks
					
		print('Found {} valid blocks of a total of {}'.format(len(valid_blocks), (rows_rgb+1)*(cols_rgb+1)))
		return valid_blocks


	def get_clip_polygons(self):
		"""
		Returns a shapely MultiPolygon representing the clipped field.
		"""

		field_shape = fiona.open(self.clip_path)
		field_polygons = []
		for feature in field_shape:
			poly = geometry.shape(feature['geometry'])
			field_polygons.append(poly)
		field = geometry.MultiPolygon(field_polygons)
		return field
