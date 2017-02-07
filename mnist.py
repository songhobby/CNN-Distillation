#!/usr/bin/env python3

import tensorflow as tf
import numpy
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

x_sample = tf.placeholder(tf.float32, [None,784])
y_sample = tf.placeholder(tf.float32, [None, 10])
Re = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#y_expect = tf.nn.softmax(tf.matmul(x_sample, Re) + b)
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_sample * tf.log(y_expect), reduction_indices=[1]))
#more accurate
y_expect = tf.matmul(x_sample, Re) +b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_expect, y_sample))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict = {x_sample: batch_xs, y_sample: batch_ys})

correct_prediction = tf.equal (tf.argmax(y_expect,1), tf.argmax(y_sample,1))
accuracy = tf.reduce_mean (tf.cast (correct_prediction, tf.float32))

print (sess.run (accuracy, feed_dict = {x_sample: mnist.test.images, y_sample: mnist.test.labels}))

#numpy.set_printoptions(threshold=numpy.nan)
#print (sess.run(Re))
