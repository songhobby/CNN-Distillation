#!/usr/bin/env python3                                                      
import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import gzip
from six.moves import cPickle

import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
	sample = 5
	limit = 100
	if len(sys.argv) > 1:
		sample = int(sys.argv[1])
	if len(sys.argv) > 2:
		limit = int(sys.argv[2])


if not os.path.isdir('./Test'):
	os.mkdir('./Test')
if not os.path.isdir('./Test/Pics'):
	os.mkdir('./Test/Pics')
if not os.path.isdir('./Test/Pics/Std'):
	os.mkdir('./Test/Pics/Std')
if not os.path.isdir('./Test/Pics/Dis'):
	os.mkdir('./Test/Pics/Dis')
if not os.path.isdir('./Test/Pics/Org'):
	os.mkdir('./Test/Pics/Org')
def display(input_array, filename, title, prediction):
	fig=plt.figure(1)
	ax=plt.subplot(111)
	plot=plt.imshow(input_array, cmap=matplotlib.cm.Greys)
	plt.title('actual: ' + str(title) + '    predicted: '+str(prediction))
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
	excerpt = indices[:batchsize]
	return inputs[excerpt], targets[excerpt]
		
#loading data
X_test, y_test = load_dataset()
X_sample, y_sample = gen_samples(X_test, y_test, 1000)

#loading functions
f=open('./Std/std_f', 'rb')
std_f = cPickle.load(f)
f.close()
f=open('./Std/std_grad_f', 'rb')
std_grad_f = cPickle.load(f)
f.close()
f=open('./Std/std_loss_f', 'rb')
std_loss_f = cPickle.load(f)
f.close()

f=open('./Distill/distill_f', 'rb')
distill_f = cPickle.load(f)
f.close()
f=open('./Distill/distill_grad_f', 'rb')
distill_grad_f = cPickle.load(f)
f.close()
f=open('./Distill/distill_loss_f', 'rb')
distill_loss_f = cPickle.load(f)
f.close()

def load_distilled (filename='./Pre/distilled_labels'):
	return np.load(filename + '.npz')['arr_0']

total_std = 0
total_dis = 0
num_std = 0
num_fail = 0
for item in range (sample):
	print('sample number',item+1)
	saliency_std = np.arange(28*28, dtype=np.float32).reshape(28,28)
	saliency_dis = np.arange(28*28, dtype=np.float32).reshape(28,28)
	predict_std = np.argmax(std_f([X_sample[item]])[0])
	predict_dis = np.argmax(distill_f([X_sample[item]])[0])
	if predict_std != y_sample[item] or predict_dis != y_sample[item]:
		print("bad prediction")
		continue
	loss_std = std_loss_f([X_sample[item]], [y_sample[item]])[0]
	loss_dis = distill_loss_f([X_sample[item]], [y_sample[item]])[0]

	print('building saliency map')
	for i in range(28):
		for j in range(28):
			change = np.copy(X_sample[item])
			change[0][i][j] = 1
			saliency_std[i][j] = std_loss_f([change], [y_sample[item]])[0]-loss_std
			saliency_dis[i][j] = distill_loss_f([change], [y_sample[item]])[0]-loss_dis
	print("std:")
	print(saliency_std)
	print("dis:")
	print(saliency_dis)
	print("mean std:", saliency_std.mean())
	print("mean dis:", saliency_dis.mean())
	print("max std:", np.max(saliency_std))
	print("max dis:", np.max(saliency_dis))
	arr_std = np.copy(X_sample[item][0])
	arr_dis = np.copy(X_sample[item][0])
	for i in range(limit):

		index_std = np.argmax(saliency_std)

		x_std = index_std // 28
		y_std = index_std % 28
		saliency_std[x_std][y_std] = float('-inf')
		arr_std[x_std][y_std] = 1

		result_std = np.argmax(std_f([[arr_std]]))
		if i == limit - 1:
			print('not suitable')
			break
		elif result_std != y_sample[item]:
			num_std += 1
			total_std += i + 1
			print("Perturbed:", i + 1)
			display(X_sample[item][0],'Org/actual_{}_index_{}.png'.format(str(y_sample[item]),str(item)), y_sample[item], y_sample[item])
			display(arr_std,'Std/actual_{}_predict_{}_index_{}.png'.format(str(y_sample[item]), str(result_std), str(item)),y_sample[item], result_std)

			print ('found good sample {}'.format(str(num_std)))
			for j in range(limit):

				index_dis = np.argmax(saliency_dis)
				x_dis = index_dis // 28
				y_dis = index_dis % 28
				saliency_dis[x_dis][y_dis] = float('-inf')
				arr_dis[x_dis][y_dis] = 1
				result_dis = np.argmax(distill_f([[arr_dis]]))
				if j == limit -1:
					num_fail += 1
					total_dis += j + 1
					print("failed for distilled CNN")
					display(arr_dis,'Dis/actual_{}_predict_{}_index_{}_fail.png'.format(str(y_sample[item]), str(result_dis), str(item)),y_sample[item], result_dis)
				elif result_dis != y_sample[item]:
					total_dis += j + 1
					display(arr_dis,'Dis/actual_{}_predict_{}_index_{}.png'.format(str(y_sample[item]), str(result_dis), str(item)),y_sample[item], result_dis)
					print("Perturbed:", j+1)
					break
			break
	print()



total_std=total_std/num_std
total_dis=total_dis/(num_std - num_fail)
print ("For standard CNN, the average number of pixels perturbed is:")
print (total_std)
print ("For distilled CNN, the average number of pixels perturbed is:")
print (total_dis)
print ("For distilled CNN, fail:")
print (num_fail)
print ("success rate for standard CNN")
print (num_std/sample*100, "%")
print ("success rate for distilled CNN")
print ((num_std-num_fail)/sample*100, "%")

'''
print("Standard input gradient:")
print(saliency_std)
print("Distilled input gradient:")
print(saliency_dis)
'''
theano.printing.pydotprint(std_f, outfile="std_graph", var_with_name_simple=True)
theano.printing.pydotprint(distill_f, outfile="dis_graph", var_with_name_simple=True)
