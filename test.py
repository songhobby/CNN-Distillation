#!/usr/bin/env python3                                                      
import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from six.moves import cPickle

import matplotlib
import matplotlib.pyplot as plt

if not os.path.isdir('./Test'):
	os.mkdir('./Test')
def display(input_array, filename, title, prediction):
	if not os.path.isdir('./Test/Pics'):
		os.mkdir('./Test/Pics')
	fig=plt.figure(1)
	ax=plt.subplot(111)
	plot=plt.imshow(input_array, cmap=matplotlib.cm.Greys)
	plt.title('actual: ' + title + '    predicted: '+prediction)
	fig.savefig('./Test/Pics/' + filename)

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

	X_test  = load_mnist_images('t10k-images-idx3-ubyte.gz')
	y_test  = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

#Obtain the validation set
	return X_test, y_test

def gen_samples(inputs, targets, batchsize):
	assert len(inputs) == len(targets)
	indices = np.arange(len(inputs))
	np.random.shuffle(indices)
	excerpt = indices[start:start + batchsize]
	return inputs[excerpt], targets[excerpt]
		
#loading data
X_test, y_test = load_dataset()
X_sample, y_sample = gen_samples(X_test, y_test, 1000)

#loading functions
f=open('./Std/std_f', 'rb')
std_f = cPickle.load(f)
f.close()
f=open('/Std/std_grad_f', 'rb')
std_grad_f = cPickle.load(f)
f.close()

f=open('./Distill/distill_f', 'rb')
distill_f = cPickle.load(f)
f.close()
f=open('/Distill/distill_grad_f', 'rb')
distill_grad_f = cPickle.load(f)
f.close()

data = np.arange(28)
Data = []
for i in range(28):
	Data = np.append(Data, data)
Data = np.float32(Data/27).reshape(1,1,28,28)
print(std_f(Data))
print(std_grad_f(Data))
display(Data[0][0], "test_pics", 'no', '1')
theano.printing.pydotprint(std_f, outfile="std_graph", var_with_name_simple=True)

for item in range (1000):
	saliency = np.arange(28*28)
	predict = np.argmax(std_f(X_sample[item]))
	for i in range(28):
		for j in range(28):
			saliency[i*28,j] = std_grad_f(X_sample[item], y_sample[item])



