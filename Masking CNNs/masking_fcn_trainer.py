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

#==================================== Functions =====================================

im_shape   = (64, 64)
mask_shape = im_shape #(20, 20)

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

#==================================== Read CSV =====================================

data_path = '/home/duncan/Documents/VanBoven/DL Datasets/Lettuce Mask/'
images = os.listdir(data_path)

csv_filename = data_path+'via_export_csv.csv'
with open(csv_filename) as csvfile:
	reader = csv.DictReader(csvfile)
	bigdict = dict()
	for row in reader:
		imname = row['filename']
		exec('d = '+row['region_shape_attributes'])
		bigdict[imname] = (d['all_points_x'], d['all_points_y'])

#=============================== Create Data Tensors ================================

NUM = 350				# number of images used for training
training_im_tensor   = np.zeros((NUM, *im_shape, 3), dtype=np.uint8)
testing_im_tensor    = np.zeros((len(images)-training_im_tensor.shape[0], *im_shape, 3), dtype=np.uint8)
training_mask_tensor = np.zeros((NUM, mask_shape[0], mask_shape[1], 1))
testing_mask_tensor  = np.zeros((len(images)-training_mask_tensor.shape[0], mask_shape[0], mask_shape[1], 1))

shuffled_keys = list(np.random.permutation(list(bigdict.keys())))			# introduce randomness into training set

for (i, im_name) in enumerate(shuffled_keys[:NUM]):
	im_arr = np.array(Image.open(data_path+im_name)).astype(np.uint8)
	training_im_tensor[i,:,:,:] = cv2.resize(im_arr, im_shape)
	xs, ys = bigdict[im_name]
	mask = get_mask(xs, ys, (im_arr.shape[0], im_arr.shape[1]), mask_shape)
	training_mask_tensor[i,:,:,0] = mask

for (i, im_name) in enumerate(shuffled_keys[NUM:]):
	im_arr = np.array(Image.open(data_path+im_name)).astype(np.uint8)
	testing_im_tensor[i,:,:,:] = cv2.resize(im_arr, im_shape)
	xs, ys = bigdict[im_name]
	mask = get_mask(xs, ys, (im_arr.shape[0], im_arr.shape[1]), mask_shape)
	testing_mask_tensor[i,:,:,0] = mask

#==================================== FCN model =====================================

def make_fcn(input_model):
	"""Load input model, extract its layers and construct a FCN."""
	model = models.load_model(input_model)

	input_c   = model.get_layer('input_1')
	conv1_fcn = model.get_layer('conv2d')
	conv2_fcn = model.get_layer('conv2d_1')
	pool1_fcn = model.get_layer('max_pooling2d')
	conv3_fcn = model.get_layer('conv2d_2')
	conv4_fcn = model.get_layer('conv2d_3')
	pool2_fcn = model.get_layer('max_pooling2d_1')
	conv5_fcn = model.get_layer('conv2d_4')
	conv6_fcn = model.get_layer('conv2d_5')
	conv7_fcn = model.get_layer('conv2d_6')

	in1 = layers.Input(shape=(64,64,3))
	c1 = conv1_fcn(in1)
	c2 = conv2_fcn(c1)
	p1 = pool1_fcn(c2)
	c3 = conv3_fcn(p1)
	c4 = conv4_fcn(c3)
	p2 = pool2_fcn(c4)
	c5 = conv5_fcn(p2)
	c6 = conv6_fcn(c5)
	c7 = conv7_fcn(c6)
	
	fcn8 = layers.Conv2D(filters=1, kernel_size=1, name='fcn8')(c7)
	fcn9 = layers.Conv2DTranspose(filters=c4.get_shape().as_list()[-1], kernel_size=4, strides=(2,2), padding='same', name='fcn9')(fcn8)
	fcn9_skip = layers.Add()([fcn9, c4]) #, name='skip')
	fcn10 = layers.Conv2DTranspose(filters=1, kernel_size=4, strides=(2,2), padding='same', name='fcn10')(fcn9_skip)

	y = keras.Model(inputs = in1, outputs = fcn10)

	return y

def run(steps, save_name):
	"""Create a new FCN masking model and train it on the data."""
	model_mask = make_fcn('../CNNs/CROP7+SAHN4_same_pad.h5')
	model_mask.compile(optimizer='adam',
			  loss='mean_squared_error',
			  metrics=['accuracy'])
	model_mask.fit(training_im_tensor, training_mask_tensor,
				epochs=1,
				steps_per_epoch=steps,
				shuffle=True)
	model_mask.save(save_name)

def train_pretrained(model_path, output, steps):
	"""Load a pre-trained model and train it further."""
	model_mask = models.load_model(model_path)
	model_mask.compile(optimizer='adam',
			  loss='mean_squared_error',
			  metrics=['accuracy'])
	model_mask.fit(training_im_tensor, training_mask_tensor,
				epochs=1,
				steps_per_epoch=steps,
				shuffle=True)
	model_mask.save(output)


if __name__ == "__main__":
	# run(NUM, 'lettuce_masker_100.h5')
	train_pretrained('lettuce_masker_80.h5', 'lettuce_masker_350.h5', NUM)

	model = models.load_model('lettuce_masker_80.h5')
	predictions = model.predict(testing_im_tensor)
	print(predictions.shape)
	for k in [0,1,2]:
		show(testing_im_tensor[k,:,:,:], testing_mask_tensor[k,:], predictions[k,:])
	plt.show()