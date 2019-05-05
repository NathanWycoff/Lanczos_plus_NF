#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/playground.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 05.03.2019

#TODO: This currently assumes the output lives in [0,1]^P. I'll address this by not requring the final layer to have an activation.

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

plt.imshow(density, origin = 'lower')
plt.savefig('temp.pdf')

# Define an invertible neural network
L = 10
nonlin = lambda x: 1. / (1. + tf.exp(-x))
dnonlin = lambda x: nonlin(x) * (1 - nonlin(x))

# A test vector
x = tf.Variable(np.random.normal(size=[1,P]))

# Define our weight tensor.
# Warning: Due to the way we've configured things, each weight matrix is really its tranpose.
W = tf.Variable(np.random.normal(scale = 0.5, size=[P,P,L]))

def fpass(x):
    z = x
    for l in range(L):
        # If we are at the last layer, we don't have a nonlinearity (so that the range of the last layer is all of R).
        if l < (L-1):
            z = nonlin(tf.linalg.matmul(z,W[:,:,l]))
        else:
            z = tf.linalg.matmul(z,W[:,:,l])
    return z

def get_jac(x):
    z = x
    jac = tf.eye(P, dtype = tf.float64)
    for l in range(L):
        zW = tf.linalg.matmul(z,W[:,:,l])
        # If we are at the last layer, we don't have a nonlinearity (so that the range of the last layer is all of R).
        if l < (L-1):
            al = dnonlin(zW)
            jac = al * (tf.linalg.matmul(jac, W[:,:,l]))
            z = nonlin(zW)
        else:
            jac = tf.linalg.matmul(jac, W[:,:,l])
            z = zW
    return jac

# Verify that our jacobian is still gucci
xn = x.numpy()
xh = np.empty(shape=[1,P])
xh[:] = xn
h = 1e-6
xh[0,0] += h

((fpass(xh) - fpass(x)) / h).numpy()
get_jac(x)

def neural_density(x):
    J = get_jac(x)
    return -tf.linalg.logdet(tf.matmul(J,tf.transpose(J))) * 0.5

## Can we compute a gradient WRT weights for backprop?
with tf.GradientTape() as t:
    dens = neural_density(x)
t.gradient(dens, W)

# See what the prior density looks like.
density = np.empty([res,res])
for i,x in enumerate(ticks):
    for j,y in enumerate(ticks):
        density[i,j] = neural_density(np.array([x,y]).reshape([1,P])).numpy()

plt.imshow(density, origin = 'lower')
plt.savefig('temp.pdf')
