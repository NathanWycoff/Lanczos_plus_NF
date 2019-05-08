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
import tensorflow_probability as tfp
from tqdm import tqdm
tfd = tfp.distributions

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
L = 5
#nonlin = lambda x: 1. / (1. + tf.exp(-x))
#inv_nonlin = lambda x: tf.log(1./(1./x - 1.))
#dnonlin = lambda x: nonlin(x) * (1 - nonlin(x))
alpha = 0.5
nonlin = lambda x: tf.math.maximum(0,x) - alpha * tf.math.maximum(0,-x)
inv_nonlin = lambda x: tf.math.maximum(0,x) - 1/alpha * tf.math.maximum(0.,-x)
dnonlin = lambda x: tf.where(tf.less(-x, 0.), tf.reshape(tf.zeros(P, dtype = tf.float64), [1,P])+1., tf.reshape(tf.zeros(P, dtype = tf.float64), [1,P])+alpha)

# Define our weight tensor.
# Warning: Due to the way we've configured things, each weight matrix is really its tranpose.
W = tf.Variable(np.random.normal(scale = 1, size=[P,P,L]))
bias = tf.Variable(np.random.normal(scale = 1, size=[P,L]))

def transformer(x):
    z = x
    for l in range(L):
        # If we are at the last layer, we don't have a nonlinearity (so that the range of the last layer is all of R).
        if l < (L-1):
            z = nonlin(tf.linalg.matmul(z,W[:,:,l]) + bias[:,l])
        else:
            z = tf.linalg.matmul(z,W[:,:,l]) + bias[:,l]
    return z

def inv_transformer(x):
    z = x
    for lr in range(L):
        l = L - lr - 1
        # If we are at the last layer, we don't have a nonlinearity (so that the range of the last layer is all of R).
        if l == (L-1):
            z = tf.linalg.transpose(tf.linalg.solve(tf.linalg.transpose(W[:,:,l]), tf.linalg.transpose(z - bias[:,l])))
        else:
            z = tf.linalg.transpose(tf.linalg.solve(tf.linalg.transpose(W[:,:,l]), inv_nonlin(tf.linalg.transpose(z - bias[:,l]))))
    return z

def get_jac(x):
    z = x
    jac = tf.eye(P, dtype = tf.float64)
    for l in range(L):
        zW = tf.linalg.matmul((z),W[:,:,l])+bias[:,l]
        # If we are at the last layer, we don't have a nonlinearity (so that the range of the last layer is all of R).
        if l < (L-1):
            al = dnonlin(zW)
            jac = al * (tf.linalg.matmul(jac, W[:,:,l]))
            z = nonlin(zW)
        else:
            jac = tf.linalg.matmul(jac, W[:,:,l])
            z = zW
    return jac

P = 2#Dim of space
B = 10 #Num of MC samples for KL estimation.
eta = 0.1
half_every = int(1e10)
iters = 1000

rho = tfd.MultivariateNormalDiag(loc = np.repeat(0.,P), scale_diag = np.repeat(1.,P))

# Verify that our jacobian is still gucci
x = tf.Variable(np.random.normal(size=[1,P]))
xn = x.numpy()
xh = np.empty(shape=[1,P])
xh[:] = xn
h = 1e-6
xh[0,0] += h

((transformer(xh) - transformer(x)) / h).numpy()
get_jac(x)

def log_neural_density(y):
    lbase_dens = rho.log_prob(inv_transformer(y))
    J = get_jac(y)
    return lbase_dens - tf.linalg.logdet(tf.matmul(J,tf.transpose(J))) * 0.5

## Can we compute a gradient WRT weights for backprop?
with tf.GradientTape() as t:
    dens = log_neural_density(x)
t.gradient(dens, W)

# See what the prior density looks like.
density = np.empty([res,res])
for i,x in enumerate(ticks):
    for j,y in enumerate(ticks):
        density[i,j] = log_neural_density(np.array([x,y]).reshape([1,P])).numpy()

fig = plt.figure()
plt.imshow(density, origin = 'lower')
plt.savefig('temp.pdf')

## See if we can hit the mixture target.

mix_count = 3
mix_means = [np.repeat(-3., P), np.repeat(0., P), np.repeat(3., P)]
post = tfd.Mixture(
  cat=tfd.Categorical(probs=np.repeat(1./mix_count, mix_count)),
  components=[
    tfd.MultivariateNormalDiag(loc = mix_means[0], scale_diag = np.repeat(1.,P)),
    tfd.MultivariateNormalDiag(loc = mix_means[1], scale_diag = np.repeat(1.,P)),
    tfd.MultivariateNormalDiag(loc = mix_means[2], scale_diag = np.repeat(1.,P))
])

def klmc():
    x = rho.sample(B)
    y = transformer(x)

    sum_kl = 0
    for b in range(B):
        xi = tf.reshape(x[b,:], [1,P])
        yi = tf.reshape(y[b,:], [1,P])
        J = get_jac(xi)
        sum_kl += -post.log_prob(yi) - 0.5*tf.linalg.logdet(tf.linalg.matmul(J, tf.transpose(J)))
    return(sum_kl / float(B))

opt = tf.train.GradientDescentOptimizer(learning_rate=eta)


print("Init Est:")
Wn = W[:,:,0].numpy()
print(Wn.dot(Wn.T))

for it in tqdm(range(iters)):
    if (it+1) % half_every == 0:
        opt._learning_rate /= 2.0
    opt_op = opt.minimize(klmc, var_list=[W,bias])

print("Final Est:")
Wn = W[:,:,0].numpy()
print(Wn.dot(Wn.T))

# Plot the resulting densities.
res = 50
ticks = np.linspace(-6,6,res)
post_density = np.empty([res,res])
for i,x in enumerate(ticks):
    for j,y in enumerate(ticks):
        post_density[i,j] = post.prob([x,y])

# Plot the resulting densities.
res = 50
ticks = np.linspace(-6,6,res)
var_density = np.empty([res,res])
for i,x in enumerate(ticks):
    for j,y in enumerate(ticks):
        var_density[i,j] = np.exp(log_neural_density(np.array([x,y]).reshape([1,P])).numpy())

plt.subplot(1,2,1)
plt.imshow(post_density, origin = 'lower')
plt.subplot(1,2,2)
plt.imshow(var_density, origin = 'lower')
plt.savefig('temp.pdf')
