#!/usr/bin/python3.6

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.preprocessing as prep

import shapely.geometry
import csv
import cv2
from PIL import Image, ImageDraw

import os
import numpy as np 
import matplotlib.pyplot as plt 
import skimage

im_shape = (60, 60)
mask_shape = (20, 20)

def get_mask(xs, ys, in_shape, out_shape):
	poly = [(xs[k], ys[k]) for k in range(len(xs))]
	xs.append(xs[0])
	ys.append(ys[0])
	tmp = Image.new('L', in_shape, 0)
	ImageDraw.Draw(tmp).polygon(poly, outline=1, fill=1)
	mask = np.array(tmp)
	return cv2.resize(mask, out_shape)

def show(im_arr, mask_vec, pred_mask):
	fig, axs = plt.subplots(1,3)
	axs[0].imshow(im_arr, interpolation='bilinear')
	axs[1].imshow(mask_vec.reshape(mask_shape), cmap='Greys_r')
	axs[2].imshow(pred_mask.reshape(mask_shape)>0.5, cmap='Greys_r')
	for ax in axs:
		ax.axis('off')
	# plt.show()

data_path = '/home/duncan/Documents/VanBoven/DL Datasets/Crop Mask/'
images = os.listdir(data_path)

csv_filename = 'via_export_csv.csv'
with open(csv_filename) as csvfile:
	reader = csv.DictReader(csvfile)
	bigdict = dict()
	for row in reader:
		imname = row['filename']
		exec('d = '+row['region_shape_attributes'])
		bigdict[imname] = (d['all_points_x'], d['all_points_y'])

NUM = 200				# number of images used for training
training_im_tensor   = np.zeros((NUM, *im_shape, 3), dtype=np.uint8)
testing_im_tensor    = np.zeros((len(images)-training_im_tensor.shape[0], *im_shape, 3), dtype=np.uint8)
training_mask_tensor = np.zeros((NUM, mask_shape[0]*mask_shape[1]))
testing_mask_tensor  = np.zeros((len(images)-training_mask_tensor.shape[0], mask_shape[0]*mask_shape[1]))

for (i, im_name) in enumerate(list(bigdict.keys())[:NUM]):
	im_arr = np.array(Image.open(data_path+im_name))
	training_im_tensor[i,:,:,:] = cv2.resize(im_arr, im_shape)
	xs, ys = bigdict[im_name]
	mask = get_mask(xs, ys, (im_arr.shape[0], im_arr.shape[1]), mask_shape)
	training_mask_tensor[i,:] = mask.flatten()

for (i, im_name) in enumerate(list(bigdict.keys())[NUM:]):
	im_arr = np.array(Image.open(data_path+im_name))
	testing_im_tensor[i,:,:,:] = cv2.resize(im_arr, im_shape)
	xs, ys = bigdict[im_name]
	mask = get_mask(xs, ys, (im_arr.shape[0], im_arr.shape[1]), mask_shape)
	testing_mask_tensor[i,:] = mask.flatten()

input1 = layers.Input(shape=(60,60,3))
x = layers.Conv2D(32, (3,3), activation='relu')(input1)
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Flatten()(x)
x = layers.Dense(200, activation='relu')(x)
x = layers.Dense(200, activation='relu')(x)
x = layers.Dense(mask_shape[0]*mask_shape[1], activation='sigmoid')(x)
model = keras.Model(inputs=input1, outputs=x)

model.compile(optimizer='rmsprop',
			  loss='mean_squared_error',
			  metrics=['accuracy'])

model.fit(training_im_tensor, training_mask_tensor,
			epochs=1,
			steps_per_epoch=50,
			shuffle=True)

model.save('masker.h5')

# model = models.load_model('masker.h5')
predictions = model.predict(testing_im_tensor)
for k in [0, 2 ,4, 6, 9, 56, 67]:
	show(testing_im_tensor[k,:,:,:], testing_mask_tensor[k,:], predictions[k,:])
plt.show()

