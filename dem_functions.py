#!/usr/bin/python3.6

import os
import gdal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp2d
from PIL import Image

R_EARTH = 6371000.		# radius of earth in meters

def get_offset(path):
	"""Looks at all files in folder specified by path. Files should end with a number, 
	the highest of these numbers is returned. Useful for determining the next number
	in a training set."""
	nums = []
	contents = os.listdir(path)
	if len(contents)==0:
		return 0
	for elt in contents:
		numstr = elt.split('_')[1].split('.')[0]
		nums.append(int(numstr))
	return max(nums)

def get_functions(img_path, dem_path):
	"""Returns a dict containing functions and constants useful for jumping between Color image and DEM."""
	img = gdal.Open(img_path)
	dem = gdal.Open(dem_path)

	trans_img = img.GetGeoTransform()
	trans_dem = dem.GetGeoTransform()
	east_img, north_img, pixel_size_img = trans_img[0], trans_img[3], trans_img[1]
	east_dem, north_dem, pixel_size_dem = trans_dem[0], trans_dem[3], trans_dem[1]
	scale_factor = pixel_size_img/pixel_size_dem

	band1 = img.GetRasterBand(1)

	def xy_img(i_img, j_img):
		"""Geotransform (i_img,j_img)_RGB -> (x,y)."""
		return np.array([east_img+pixel_size_img*j_img, north_img-pixel_size_img*i_img])

	def xy_dem(i_dem, j_dem):
		"""Geotransform (i_dem,j_dem)_H -> (x,y)."""
		return np.array([east_dem+pixel_size_dem*j_dem, north_dem-pixel_size_dem*i_dem])

	def rowcol_img(x, y):
		"""Inverse geotransform (x,y) -> (i_img,j_img)."""
		return 1./pixel_size_img*np.array([north_img-y, x-east_img])

	def rowcol_dem(x, y):
		"""Inverse geotransform (x,y) -> (i_dem,j_dem)."""
		return 1./pixel_size_dem*np.array([north_dem-y, x-east_dem])

	def get_dem_pixel(i_img, j_img):
		"""Composition of rowcol_dem with xy_img, to compute H-pixel from RGB-pixel."""
		i_dem, j_dem = rowcol_dem(*tuple(xy_img(i_img,j_img)))
		i_dem, j_dem = int(np.round(i_dem)), int(np.round(j_dem))
		return i_dem, j_dem

	def get_color_slice(i_img, j_img, num_rows, num_cols):
		"""Returns (num_rows, num_cols, 3) numpy array represting the RGB-slice i_img:i_img+num_rows, j_img:j_img+num_cols."""
		r = np.array(img.GetRasterBand(1).ReadAsArray(j_img, i_img, num_cols, num_rows), dtype = np.uint(8))
		g = np.array(img.GetRasterBand(2).ReadAsArray(j_img, i_img, num_cols, num_rows), dtype = np.uint(8))
		b = np.array(img.GetRasterBand(3).ReadAsArray(j_img, i_img, num_cols, num_rows), dtype = np.uint(8))
		return np.dstack((r,g,b))

	def get_height_slice(i_img, j_img, num_rows, num_cols):
		"""Returns the H-slice corresponding to the RGB slice returned by get_color_slice(i_img, j_img, num_rows, num_cols)."""
		i_dem, j_dem = get_dem_pixel(i_img, j_img)
		rows2 = int(np.round(num_rows*scale_factor))
		cols2 = int(np.round(num_cols*scale_factor))
		h = np.array(dem.GetRasterBand(1).ReadAsArray(j_dem, i_dem, cols2, rows2))#, dtype=np.uint(8))
		return h

	def get_slices(i_img, j_img, num_rows, num_cols):
		"""Get both RGB and H slices. (i_img,j_img) is the center point in the RGB image, where i_img is the row and j_img in the column."""
		c_im =  get_color_slice(i_img-num_rows//2, j_img-num_cols//2, num_rows, num_cols)
		h_im = get_height_slice(i_img-num_rows//2, j_img-num_cols//2, num_rows, num_cols)
		return c_im, h_im

	def get_window(i_img, j_img, num_rows, num_cols):
		"""Gets the window [i_img:i_img+num_rows, j_img:j_img+num_cols] from RGB tif and the corresponding H slice. If part of the window is 
		outside of the RGB tif, the window gets cropped. The new window extent is also returned."""
		if i_img<0:											# north border
			num_rows = num_rows-abs(i_img)
			i_img = max(0, i_img)
		elif i_img+num_rows>band1.YSize:					# sourth border
			num_rows = min(band1.YSize - i_img, num_rows)
		if j_img<0:											# west border
			num_cols = num_cols-abs(j_img)
			j_img = max(0, j_img)
		elif j_img+num_cols>band1.XSize:					# east border
			num_cols = min(band1.XSize - j_img, num_cols)
		c_im =  get_color_slice(i_img, j_img, num_rows, num_cols)
		h_im = get_height_slice(i_img, j_img, num_rows, num_cols)
		adjusted_window = (i_img, j_img, num_rows, num_cols)
		return c_im, h_im, adjusted_window

	func_dict = dict()
	for f in [xy_img, xy_dem, rowcol_img, rowcol_dem, get_dem_pixel, get_color_slice, get_height_slice, get_slices, get_window]:
		func_dict[f.__name__] = f
	func_dict['scale_factor'] = scale_factor
	func_dict['delta_c'] 	  = pixel_size_img 							# long-lat pixel step size in RGB image (degrees)
	func_dict['delta_h'] 	  = pixel_size_dem 							# long-lat step size in H image (degrees)
	func_dict['delta_c_m']    = R_EARTH*np.sin(np.pi-north_img/180*np.pi)*np.pi*pixel_size_img/180		# pixel size in meters
	func_dict['delta_h_m']    = R_EARTH*np.sin(np.pi-north_dem/180*np.pi)*np.pi*pixel_size_dem/180
	return func_dict

def select(I, J, L, class_name, func_dict, paths, size=50):
	"""Function for selecting training data.
	Input: 	- I : row in rasterband
			- J : column in rasterbad
			- L : width and height of area; we analyze slice I:I+L, J:J+L
			- class_name : one of 'broccoli' and 'background'
			- func_dict  : output of get_functions, contains relevant functions to jump between RGB and H images
			- paths : tuple (c_path, h_path) containing the paths to locations where RGB and H crops should be saved
			- size  : size of box to put around clicked point
	Click on a location to put a box of size 'size' around this point. The contents of this box are saved to the 
	RGB path, and the corresponding contents of the box in the height map are saved to the H path."""
	c_path, h_path = paths
	if class_name.lower() != 'lettuce' and class_name.lower() != 'background':
		raise NameError('Please choose a class using the arg class_name; it should be either lettuce or background')
	
	scale_factor = func_dict['scale_factor']
	get_slices = func_dict['get_slices']

	img_crop, h_crop = get_slices(I, J, L, L)

	fig, ax = plt.subplots(1, figsize=(10,10))
	ax.imshow(img_crop)
	ax.set_title(class_name)
	plt.axis('off')

	global coords
	coords = []

	def draw_rect(x, y):
		box = patches.Rectangle((x-0.5*size, y-0.5*size), size, size, edgecolor='w', facecolor='none', lw=0.5)
		ax.add_patch(box)
		ax.plot(x, y, '.', color='w')

	def onclick(event):
	    global ix, iy
	    if event.button==3:
	    	fig.canvas.mpl_disconnect(cid)
	    	plt.close()
	    else:
		    ix, iy = event.xdata, event.ydata
		    # print('x = %d, y = %d'%(ix, iy))		
		    global coords
		    coords.append([ix, iy])
		    draw_rect(ix, iy)
		    fig.canvas.draw()

	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	plt.show()

	coords = np.array(coords, dtype=int)
	hsize = int(np.round(scale_factor*size))
	offset = get_offset(c_path)

	for (count, coord) in enumerate(coords):
		x, y = coord						# we need i_img, j_img to be python ints, not numpy ints
		i1, i2 = int(y-size//2), int(y+size//2)
		j1, j2 = int(x-size//2), int(x+size//2)
		i_dem, j_dem = int(np.round(scale_factor*y)), int(np.round(scale_factor*x))
		k1, k2 = i_dem-hsize//2-1, i_dem+hsize//2+1
		l1, l2 = j_dem-hsize//2-1, j_dem+hsize//2+1
		c_im = img_crop[i1:i2, j1:j2, :]
		h_im =   h_crop[k1:k2, l1:l2]
		try:
			h_im = h_im - h_im.min()
			h_im *= 255/h_im.max()			# normalize height

			c_im = c_im.astype(np.uint8)
			h_im = h_im.astype(np.uint8)

			cimim = Image.fromarray(c_im, 'RGB')
			himim = Image.fromarray(h_im, 'L')
			cimim.save(c_path+'c_%s.png'%(count+offset+1))
			himim.save(h_path+'h_%s.png'%(count+offset+1))
		except:
			print('not able to save box {}'.format(count+offset+1))

if __name__ == "__main__":
	# =============================================================================================================
	img_path = '/home/duncan/Documents/VanBoven/Orthomosaics/c08_biobrass-C49-201906151558-GR.tif'
	dem_path = '/home/duncan/Documents/VanBoven/Orthomosaics/c08_biobrass-C49-201906151558_DEM-GR.tif'
	# =============================================================================================================
	plot = 'C49'
	c_path_broc = '/home/duncan/Documents/VanBoven/DL Datasets/Lettuce Height/'+plot+'/Training Data Color/Lettuce/'
	h_path_broc = '/home/duncan/Documents/VanBoven/DL Datasets/Lettuce Height/'+plot+'/Training Data Height/Lettuce/'
	# =============================================================================================================
	c_path_bg = '/home/duncan/Documents/VanBoven/DL Datasets/Lettuce Height/'+plot+'/Training Data Color/Background/'
	h_path_bg = '/home/duncan/Documents/VanBoven/DL Datasets/Lettuce Height/'+plot+'/Training Data Height/Background/'
	# =============================================================================================================

	funcs = get_functions(img_path, dem_path)
	I, J = funcs['rowcol_img'](5.47798739,52.48525566)
	I, J = int(I), int(J)
	# I, J = 15000+300, 20000												# row, column location in tif in pixels
	
	slice_size = 600
	box_size = 55

	func_dict = get_functions(img_path, dem_path)
	select(I, J, slice_size, 'lettuce',    func_dict, (c_path_broc, h_path_broc), size=box_size)
	select(I, J, slice_size, 'background', func_dict, (c_path_bg, h_path_bg), size=box_size)


