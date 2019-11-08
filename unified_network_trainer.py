#!/usr/bin/python3.6

from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
import keras.backend as K

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
# import json

c_size = (64, 64)
h_size = (20, 20)

def windows_load(path):
	# name = path.split(r'\\')[-1].split('.')[0]
	with open(path+'.json', 'r') as f:
		json_string = f.read()
	new_model = models.model_from_json(json_string)
	new_model.load_weights(path+'_weights.h5')
	return new_model

def save_separate(model, out_name):
	json_string = model.to_json()
	with open(out_name+'.json', 'w') as f:
		f.write(json_string)
	model.save_weights(out_name+'_weights.h5')

def customloss(y_true, y_pred):
	return K.categorical_crossentropy(y_true[0], y_pred[0]) + 0.0*K.mean(K.square(y_true[1]-y_pred[1]))

def create_network(num_classes):
	# Color image convolutional network
	input_RGB = layers.Input(shape=(c_size[0], c_size[1], 3))
	x1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_RGB)
	x2 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x1)
	x3 = layers.MaxPooling2D((2,2))(x2)
	x4 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x3)
	c4 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x4)
	x5 = layers.MaxPooling2D((2,2))(c4)
	x6 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x5)
	x7 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x6)
	c7 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x7)
	x8 = layers.MaxPooling2D((2,2))(c7)
	x = layers.Flatten()(x8)

	# Height image convolutional network
	input_H = layers.Input(shape=(h_size[0], h_size[1], 1))
	y1 = layers.Conv2D(16, (3,3), activation='relu')(input_H)
	y2 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(y1)
	y3 = layers.MaxPooling2D((2,2))(y2)
	y4 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(y3)
	y5 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(y4)
	y6 = layers.MaxPooling2D((2,2))(y5)
	y7 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(y6)
	y8 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(y7)
	y9 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(y8)			#added
	y = layers.Flatten()(y9)

	# Combine Color and Height networks into FC layers
	combined = layers.concatenate([x, y])
	z1 = layers.Dense(128, activation='relu')(combined)
	z = layers.Dense(32,  activation='relu')(z1)
	class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(z)

	# FCN Masking part
	fcn1 = layers.Conv2D(filters=1, kernel_size=1, name='fcn1')(c7)
	fcn2 = layers.Conv2DTranspose(filters=c4.get_shape().as_list()[-1], kernel_size=4, strides=(2,2), padding='same', name='fcn2')(fcn1)
	fcn3 = layers.Add()([fcn2, c4])
	mask_output = layers.Conv2DTranspose(filters=1, kernel_size=4, strides=(2,2), padding='same', name='mask_output')(fcn3)

	model = keras.Model(inputs=[input_RGB, input_H], outputs=[class_output, mask_output])

	losses = {"class_output": 'categorical_crossentropy', "mask_output":'mean_squared_error'}
	loss_weights = {"class_output": 1.0, "mask_output": 0.5}
	model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights, metrics=['accuracy'])

	# model.compile(optimizer='adam', loss=customloss, metrics=['accuracy'])

	return model

def init_data_tensors(c_size, h_size, master, c_dir, h_dir, mask_dir):
	class_names = os.listdir(master+c_dir)
	color_filenames, height_filenames, mask_filenames = [], [], []
	label_idxs = []
	for (k, class_name) in enumerate(class_names):
		color_titles = os.listdir(master+c_dir+class_name)
		height_titles = os.listdir(master+h_dir+class_name)
		mask_titles = os.listdir(master+mask_dir+class_name)
		color_titles.sort()
		height_titles.sort()
		mask_titles.sort()
		color_filenames  += [class_name+'/'+title for title in color_titles]
		height_filenames += [class_name+'/'+title for title in height_titles]
		mask_filenames   += [class_name+'/'+title for title in mask_titles]
		label_idxs += [class_names.index(class_name) for title in color_titles]
		print('Found {} training images for class "{}"'.format(len(color_titles), class_name))

	if not (len(color_filenames) == len(height_filenames) == len(mask_filenames)):
		raise IndexError('Incompatible number of input files:\n {} RGB, {} H, {} mask'.format(len(color_filenames), len(height_filenames), len(mask_filenames)))

	input_length = len(label_idxs)
	RGB_tensor   = np.zeros((input_length, c_size[0], c_size[1], 3), dtype=np.uint8)
	H_tensor     = np.zeros((input_length, h_size[0], h_size[1], 1), dtype=np.uint8)
	mask_tensor  = np.zeros((input_length, c_size[0], c_size[1], 1), dtype=np.uint8)
	label_tensor = np.zeros((input_length, num_classes), dtype=np.uint8)

	for k in range(input_length):
		rgb_arr  = np.array(Image.open(master+c_dir  + color_filenames[k] )).astype(np.uint8)
		h_arr    = np.array(Image.open(master+h_dir + height_filenames[k])).astype(np.uint8)
		RGB_tensor[k,:,:,:]  = cv2.resize(rgb_arr, c_size)
		H_tensor[k,:,:,0]    = cv2.resize(h_arr,   h_size)
		label_tensor[k, label_idxs[k]] = 1
		# if label_idxs[k] > 0:					# non-background, if background the mask is empty
		mask_arr = np.array(Image.open(master+mask_dir   + mask_filenames[k]  )).astype(np.uint8)
		mask_tensor[k,:,:,0] = cv2.resize(mask_arr, c_size)

	print('Tensors inintialized')
	p = np.random.permutation(input_length)		# shuffle data
	# p = np.arange(input_length)
	RGB_tensor, H_tensor, mask_tensor, label_tensor = RGB_tensor[p,...], H_tensor[p,...], mask_tensor[p,...], label_tensor[p,...]
	return [RGB_tensor, H_tensor], [label_tensor, mask_tensor]

def show_prediction(model, RGB_tensor, H_tensor, class_names=None):
	"""Testing function to show if output makes sense"""
	labels, masks = model.predict([RGB_tensor, H_tensor])
	for n in range(RGB_tensor.shape[0]):
		f, (ax1, ax2, ax3) = plt.subplots(1,3)
		ax1.imshow(RGB_tensor[n,...])
		ax2.imshow(H_tensor[n,:,:,0])
		ax3.imshow(masks[n,:,:,0]>0.5)
		if class_names==None:
			ax1.set_title(np.argmax(labels[n,:]))
		else:
			ax1.set_title(class_names[np.argmax(labels[n,:])])
		for ax in (ax1, ax2, ax3):
			ax.axis('off')
	plt.show()

if __name__ == "__main__":
	# master_dir = '/home/duncan/Documents/VanBoven/Code/Git Folder/testing/'
	# c_dir  	 = 'Training Data Color/'
	# h_dir 	 = 'Training Data Height/'
	# mask_dir = 'Training Data Mask/'

	master_dir = r'.\\testing\\'
	c_dir  	 = r'Training Data Color\\'
	h_dir 	 = r'Training Data Height\\'
	mask_dir = r'Training Data Mask\\'
	num_classes = len(os.listdir(master_dir+c_dir))
	input_tensors, output_tensors = init_data_tensors(c_size, h_size, master_dir, c_dir, h_dir, mask_dir)

	model = create_network(num_classes)
	es = keras.callbacks.EarlyStopping(monitor='mask_output_acc', mode='max')
	model.fit(input_tensors, {"mask_output":output_tensors[1], "class_output": output_tensors[0]}, \
			  epochs=1, steps_per_epoch=input_tensors[0].shape[0], callbacks=[es])
	model.save('Unified CNNs/broccoli_unified_GPU.h5')
	save_separate(model, 'Unified CNNs/broccoli_unified_GPU')

	model = windows_load('Unified CNNs/broccoli_unified_GPU')
	idxs = [0, 1, 2, 3, 4]
	show_prediction(model, input_tensors[0][idxs,...], input_tensors[1][idxs,...])
