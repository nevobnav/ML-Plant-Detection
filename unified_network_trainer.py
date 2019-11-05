#!/usr/bin/python3.6

from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

import os
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
import cv2

master = '/home/duncan/Documents/VanBoven/DL Datasets/Unified Broccoli/'
train_dir_color  = 'Training Data Color/'
train_dir_height = 'Training Data Height/'
train_dir_mask	 = 'Training Data Mask/'
class_names = os.listdir(master+train_dir_color)

c_size = (64, 64)
h_size = (20, 20)

def create_network(num_classes):
	# Color image convolutional network 
	input_RGB = layers.Input(shape=(c_size[0], c_size[1], 3))
	x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_RGB)
	x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
	x = layers.MaxPooling2D((2,2))(x)
	x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
	c4 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
	x = layers.MaxPooling2D((2,2))(c4)
	x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
	x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
	c7 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
	x = layers.MaxPooling2D((2,2))(c7)
	x = layers.Flatten()(x)

	# FCN Masking part
	fcn1 = layers.Conv2D(filters=1, kernel_size=1, name='fcn1')(c7)
	fcn2 = layers.Conv2DTranspose(filters=c4.get_shape().as_list()[-1], kernel_size=4, strides=(2,2), padding='same', name='fcn2')(fcn1)
	fcn3 = layers.Add()([fcn2, c4])
	mask_output = layers.Conv2DTranspose(filters=1, kernel_size=4, strides=(2,2), padding='same', name='mask_output')(fcn3)

	# Height image convolutional network
	input_H = layers.Input(shape=(h_size[0], h_size[1], 1))
	y = layers.Conv2D(16, (3,3), activation='relu')(input_H)
	y = layers.Conv2D(16, (3,3), activation='relu', padding='same')(y)
	y = layers.MaxPooling2D((2,2))(y)
	y = layers.Conv2D(32, (3,3), activation='relu', padding='same')(y)
	y = layers.Conv2D(32, (3,3), activation='relu', padding='same')(y)
	y = layers.MaxPooling2D((2,2))(y)
	y = layers.Conv2D(64, (3,3), activation='relu', padding='same')(y)
	y = layers.Conv2D(64, (3,3), activation='relu', padding='same')(y)
	y = layers.Conv2D(128, (3,3), activation='relu', padding='same')(y)			#added
	y = layers.Flatten()(y)

	# Combine Color and Height networks into FC layers
	combined = layers.concatenate([x, y])
	z = layers.Dense(128, activation='relu')(combined)
	z = layers.Dense(32, activation='relu')(z)
	class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(z)

	model = keras.Model(inputs=[input_RGB, input_H], outputs=[class_output, mask_output])

	losses = {"class_output": "categorical_crossentropy", "mask_output": "mean_squared_error"}
	loss_weights = {"class_output": 1.0, "mask_output": 0.5}
	model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights, metrics=['accuracy'])

	return model

def init_data_tensors():
	color_filenames, height_filenames, mask_filenames = [], [], []
	label_idxs = []
	for (k, class_name) in enumerate(class_names):
		color_titles = os.listdir(master+train_dir_color+class_name)
		height_titles = os.listdir(master+train_dir_height+class_name)
		mask_titles = os.listdir(master+train_dir_mask+class_name)
		color_titles.sort()
		height_titles.sort()
		mask_titles.sort()
		color_filenames  += [class_name+'/'+title for title in color_titles]
		height_filenames += [class_name+'/'+title for title in height_titles]
		mask_filenames   += [class_name+'/'+title for title in mask_titles]
		label_idxs += [class_names.index(class_name) for title in color_titles]

	if not (len(color_filenames) == len(height_filenames) == len(mask_filenames)):
		raise IndexError('Incompatible number of input files:\n {} RGB, {} H, {} mask'.format(len(color_filenames), len(height_filenames), len(mask_filenames)))

	input_length = len(label_idxs)

	RGB_tensor   = np.zeros((input_length, c_size[0], c_size[1], 3), dtype=np.uint8)
	H_tensor     = np.zeros((input_length, h_size[0], h_size[1], 1), dtype=np.uint8)
	mask_tensor  = np.zeros((input_length, c_size[0], c_size[1], 1), dtype=np.uint8)
	label_tensor = np.zeros((input_length, num_classes), dtype=np.uint8)

	for k in range(input_length):
		rgb_arr  = np.array(Image.open(master+train_dir_color  + color_filenames[k] )).astype(np.uint8)
		h_arr    = np.array(Image.open(master+train_dir_height + height_filenames[k])).astype(np.uint8)
		mask_arr = np.array(Image.open(master+train_dir_mask   + mask_filenames[k]  )).astype(np.uint8)
		RGB_tensor[k,:,:,:]  = cv2.resize(rgb_arr, c_size)
		H_tensor[k,:,:,0]    = cv2.resize(h_arr,   h_size)
		mask_tensor[k,:,:,0] = cv2.resize(mask_arr, c_size)
		label_tensor[k, label_idxs[k]] = 1

	p = np.random.permutation(input_length)
	RGB_tensor, H_tensor, mask_tensor, label_tensor = RGB_tensor[p,...], H_tensor[p,...], mask_tensor[p,...], label_tensor[p,...]
	return [RGB_tensor, H_tensor], [label_tensor, mask_tensor]

def show_prediction(model, RGB_tensor, H_tensor):
	labels, masks = model.predict([RGB_tensor, H_tensor])
	for n in range(RGB_tensor.shape[0]):
		f, (ax1, ax2, ax3) = plt.subplots(1,3)
		ax1.imshow(RGB_tensor[n,...])
		ax2.imshow(H_tensor[n,:,:,0])
		ax3.imshow(masks[n,:,:,0]>0.5)
		ax1.set_title(class_names[np.argmax(labels[n,:])])
		for ax in (ax1, ax2, ax3):
			ax.axis('off')
	plt.show()

if __name__ == "__main__":
	num_classes = len(os.listdir(master+train_dir_color))
	model = create_network(num_classes)
	input_tensors, output_tensors = init_data_tensors((64,64), (20,20))

	model = models.load_model('unified.h5')
	model.fit(input_tensors, output_tensors,
				epochs=1,
				steps_per_epoch=40)
				# shuffle=True)
	model.save('unified.h5')

	idxs = [0, 1, 2, 3, 4]
	show_prediction(model, input_tensors[0][idxs,...], input_tensors[1][idxs,...])


