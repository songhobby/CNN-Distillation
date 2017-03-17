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

f=open('./Std/std_f', 'rb')
std_f = cPickle.load(f)
f.close()
f=open('/Std/grad_f', 'rb')
std_grad_f = cPickle.load(f)
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

