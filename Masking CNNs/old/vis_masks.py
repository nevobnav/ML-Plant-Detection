#!/usr/bin/python3.6

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import matplotlib.patches as patches
import shapely.geometry
import csv
import cv2
from PIL import Image, ImageDraw
import scipy.ndimage.filters as filters

parent_path = '/home/duncan/Documents/VanBoven/DL Datasets/Broccoli Height/Stage 1/Training Data Color/Broccoli/'
target_path = '/home/duncan/Documents/VanBoven/DL Datasets/Crop Mask/'

def get_mask(xs, ys, shape):
	poly = [(xs[k], ys[k]) for k in range(len(xs))]
	xs.append(xs[0])
	ys.append(ys[0])
	tmp = Image.new('L', shape, 0)
	ImageDraw.Draw(tmp).polygon(poly, outline=1, fill=1)
	mask = np.array(tmp)
	return mask

csv_filename = 'via_export_csv.csv'
with open(csv_filename) as csvfile:
	reader = csv.DictReader(csvfile)
	bigdict = dict()
	for row in reader:
		imname = row['filename']
		exec('d = '+row['region_shape_attributes'])
		bigdict[imname] = (d['all_points_x'], d['all_points_y'])

imname = 'c_299.png'
im = np.array(Image.open(parent_path+imname))
xs, ys = bigdict[imname]

mask = get_mask(xs, ys, (im.shape[0], im.shape[1]))
# mask = filters.gaussian_filter(mask.astype(float), 1)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(im)#, interpolation='bilinear')
ax1.axis('off')
ax2.axis('off')
ax1.plot(xs, ys, 'w.-', lw=1)
ax2.imshow(cv2.resize(mask, (20,20)), cmap='Greys_r')
# plt.fill_between(xs, ys, alpha=0.2)
plt.show()