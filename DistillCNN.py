#!/usr/bin/env python3

import os
import numpy as np
import theano
import theano.tensor as T
import lasagne

#Loading data from MNIST

import urllib.request
import gzip

def load_dataset():
	def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
		print('Downloading {}'.format(filename))
		urllib.request.urlretrieve(source + filename, filename)

	def load_mnist_images(filename):
		if not os.path.exists(filename):
			download(filename)
		with gzip.open(filename, 'rb') as f:
			data = np.frombuffer(f.read(), np.uint8, offset=16)
		data = data.reshape(-1, 1, 28, 28)
#normalize the data
		return data / np.float32(256)

	def load_mnist_labels(filename):
		if not os.path.exists(filename):
			download(filename)
		with gzip.open(filename, 'rb') as f:
			data = np.frombuffer(f.read(), np.uint8, offset=8)
		return data

	X_train = load_mnist_images('train-images-idx3-ubyte.gz')
	y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
	X_test  = load_mnist_images('t10k-images-idx3-ubyte.gz')
	y_test  = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

#Obtain the validation set
	X_train, X_val = X_train[:-1000],  X_train[-1000:]
	y_train, y_val = y_train[:-1000],  y_train[-1000:]

	return X_train, y_train, X_val, y_val, X_test, y_test

load_dataset()

def build_cnn(input_var=None):
	network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)

	network = lasagne.layers.Conv2DLayer(
			network, num_filters=32, filter_size=(3,3),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform())
	network = lasagne.layers.Conv2DLayer(
			network, num_filters=32, filter_size=(3,3),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform())
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2))

	network = lasagne.layers.Conv2DLayer(
			network, num_filters=64, filter_size=(3,3),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform())
	network = lasagne.layers.Conv2DLayer(
			network, num_filters=64, filter_size=(3,3),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform())
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2,2))

	network = lasagne.layers.DenseLayer(
			network, num_units=200,
			nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.DenseLayer(
			num_units=200,
			nonlinearity=lasagne.nonlinearities.rectify)

	network = lasagne.layers.DenseLayer(
			network, num_units=10,
			nonlinearity=lasagne.nonlinearities.softmax)

return network




