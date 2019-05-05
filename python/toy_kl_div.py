#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/toy_kl_div.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 05.05.2019

## Practice doing VB with MC on an intractable ELBO.
# x ~ N(0,I); f(x) = Ax; find A such that f(x) best matches N(0, SIGMA) for some SIGMA.

#TODO: Do true SGD on it (rn our batch size is so large it may as well be GD).

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
import tensorflow_probability as tfp
tf.enable_eager_execution()
tfd = tfp.distributions

P = 2#Dim of space
B = 100000 #Num of MC samples for KL estimation.

# Sample true transform
SIGMA_L = np.random.normal(size=[P,P])
SIGMA = np.dot(SIGMA_L.T, SIGMA_L)

## Define distributions.
# The variational base
rho = tfd.MultivariateNormalDiag(loc = np.repeat(0.,P), scale_diag = np.repeat(1.,P))
# The true posterior
post = tfd.MultivariateNormalFullCovariance(loc=np.repeat(0.,P), covariance_matrix=SIGMA)

A = tf.Variable(np.random.normal(size=[P,P]))

print("Init Est:")
An = A.numpy()
print(An.dot(An.T))

eta = 0.1
iters = 1000

for it in range(iters):
    x = np.random.normal(size=[P,B])

    with tf.GradientTape() as t:
        y = tf.linalg.matmul(A,x)
        klmc = -tf.reduce_mean(post.log_prob(tf.transpose(y))) - 0.5*tf.linalg.logdet(tf.linalg.matmul(A, tf.transpose(A))) 
    grad = t.gradient(klmc, A)

    A.assign_sub(eta*grad)

    #print("Achieved:")
    #print(tf.matmul(A, tf.transpose(A)).numpy())
    #print("Desired:")
    #print(SIGMA)
    #print("KL Estimate:")
    #print(klmc.numpy())

print("Final Est:")
An = A.numpy()
print(An.dot(An.T))
