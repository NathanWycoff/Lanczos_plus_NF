#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/playground.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 05.03.2019

#TODO: Biases; non-square weights.

# See if we can get Multiple Gaussian peaks, using naive determinant calculation.
import tensorflow as tf
import keras
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import scipy
from scipy.stats import multivariate_normal
from tensorflow.python.ops.parallel_for.gradients import jacobian
tf.enable_eager_execution()

P = 2

mvn1 = multivariate_normal(mean = np.zeros(P))
mvn2 = multivariate_normal(mean = np.zeros(P) + 6)
mvn3 = multivariate_normal(mean = np.zeros(P) - 6)
mvns = [mvn1, mvn2, mvn3]

mix_pdf = lambda x: np.mean([mvn.pdf(x) for mvn in mvns])
mix_pdf([1,2])

res = 50
ticks = np.linspace(-9,9,res)
density = np.empty([res,res])
for i,x in enumerate(ticks):
    for j,y in enumerate(ticks):
        density[i,j] = mix_pdf([x,y])

plt.imshow(density)
plt.savefig('temp.pdf')

# Define an invertible neural network
W = tf.Variable(np.random.normal(size=[P,P]))
nonlin = lambda x: 1. / (1. + tf.exp(-x))
x = tf.Variable(np.random.normal(size=[1,P]))

# Test get_jac
#h = 1e-6
#xh = np.empty([1,P])
#xh[:] = x
#xh[0,1] += h
#
#(nonlin(xh.dot(W)) - nonlin(x.dot(W))) / h

def get_jac(x):
    a = nonlin(tf.linalg.matmul(x, W)) * (1 - nonlin(tf.linalg.matmul(x, W)))
    return a * W

def neural_density(x):
    J = get_jac(x)
    return -tf.linalg.logdet(tf.matmul(J,tf.transpose(J))) * 0.5

with tf.GradientTape() as t:
    dens = neural_density(x)
t.gradient(dens, W)

density = np.empty([res,res])
for i,x in enumerate(ticks):
    for j,y in enumerate(ticks):
        density[i,j] = neural_density(np.array([x,y]).reshape([1,P])).numpy()

plt.imshow(density)
plt.savefig('temp.pdf')

## Optimize to fit the target density
