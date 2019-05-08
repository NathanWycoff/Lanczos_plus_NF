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
from tqdm import tqdm
plt.ion()
import numpy as np
import scipy
from scipy.stats import multivariate_normal
from tensorflow.python.ops.parallel_for.gradients import jacobian
import tensorflow_probability as tfp
tf.enable_eager_execution()
tfd = tfp.distributions

P = 2#Dim of space
B = 10 #Num of MC samples for KL estimation.
eta = 0.1
half_every = int(1e10)
iters = 1000

# Sample true transform
SIGMA_L = np.random.normal(size=[P,P])
SIGMA = np.dot(SIGMA_L.T, SIGMA_L)

## Define distributions.
# The variational base
rho = tfd.MultivariateNormalDiag(loc = np.repeat(0.,P), scale_diag = np.repeat(1.,P))
# The true posterior
post = tfd.MultivariateNormalFullCovariance(loc=np.repeat(0.,P), covariance_matrix=SIGMA)

############################################################
## Optimize using a handmade GD
A = tf.Variable(np.random.normal(size=[P,P]))

print("Init Est:")
An = A.numpy()
print(An.dot(An.T))


for it in range(iters):
    x = tf.transpose(rho.sample(B))

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

############################################################
## Optimize using a TF's builtin GD.
A = tf.Variable(np.random.normal(size=[P,P]))

print("Init Est:")
An = A.numpy()
print(An.dot(An.T))

def klmc():
    x = tf.transpose(rho.sample(B))
    return(-tf.reduce_mean(post.log_prob(tf.transpose(tf.linalg.matmul(A,x)))) - 0.5*tf.linalg.logdet(tf.linalg.matmul(A, tf.transpose(A))))

opt = tf.train.GradientDescentOptimizer(learning_rate=eta)

for it in range(iters):
    if (it+1) % half_every == 0:
        opt._learning_rate /= 2.0
    opt_op = opt.minimize(klmc, var_list=[A])

print("Final Est:")
An = A.numpy()
print(An.dot(An.T))

############################################################
## Try to fit to a mixture of mvnormals.

P = 2
mix_count = 3
mix_means = [np.repeat(-3., P), np.repeat(0., P), np.repeat(3., P)]
post = tfd.Mixture(
  cat=tfd.Categorical(probs=np.repeat(1./mix_count, mix_count)),
  components=[
    tfd.MultivariateNormalDiag(loc = mix_means[0], scale_diag = np.repeat(1.,P)),
    tfd.MultivariateNormalDiag(loc = mix_means[1], scale_diag = np.repeat(1.,P)),
    tfd.MultivariateNormalDiag(loc = mix_means[2], scale_diag = np.repeat(1.,P))
])

#A = tf.Variable(np.random.normal(size=[P,P]))
A = tf.Variable(np.array([[1,0.9],[0.9,1]]))

print("Init Est:")
An = A.numpy()
print(An.dot(An.T))

def transformer(x):
    return(tf.linalg.matmul(A,x))

def inv_transformer(x):
    return(tf.linalg.solve(A,x))

def get_jac(x):
    return(A)

def klmc():
    x = tf.transpose(rho.sample(B))
    y = transformer(x)
    J = get_jac(x)
    return(tf.reduce_mean(-post.log_prob(tf.transpose(y)) - 0.5*tf.linalg.logdet(tf.linalg.matmul(J, tf.transpose(J)))))

opt = tf.train.GradientDescentOptimizer(learning_rate=eta)

for it in tqdm(range(iters)):
    if (it+1) % half_every == 0:
        opt._learning_rate /= 2.0
    opt_op = opt.minimize(klmc, var_list=[A])

print("Final Est:")
An = A.numpy()
print(An.dot(An.T))

v_dens = lambda y: rho.prob(tf.transpose(inv_transformer(y))) *  tf.exp(- 0.5*tf.linalg.logdet(tf.linalg.matmul(A, tf.transpose(A))))

#vd = tfd.MultivariateNormalFullCovariance(loc = np.repeat(0.0, P), covariance_matrix = A.numpy().dot(A.numpy().T))
#v_dens = lambda x: vd.prob(x)

# Plot the resulting densities.
res = 50
ticks = np.linspace(-9,9,res)
post_density = np.empty([res,res])
for i,x in enumerate(ticks):
    for j,y in enumerate(ticks):
        post_density[i,j] = post.prob([x,y])

# Plot the resulting densities.
res = 50
ticks = np.linspace(-9,9,res)
var_density = np.empty([res,res])
for i,x in enumerate(ticks):
    for j,y in enumerate(ticks):
        var_density[i,j] = v_dens(np.array([x,y]).reshape([P,1])).numpy()

plt.subplot(1,2,1)
plt.imshow(post_density, origin = 'lower')
plt.subplot(1,2,2)
plt.imshow(var_density, origin = 'lower')
plt.savefig('temp.pdf')
