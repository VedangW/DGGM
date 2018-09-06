#!usr/bin/python

""" Module that defines the utility functions for reading
    and selecting data along with functions that define the
    initialiser, layers and prior for the tf model.
"""

import config
import numpy as np
import tensorflow as tf

# Load the train and test files

train = config.train
test = config.test

# Preprocessing and batch generation functions

def batch_gen(data, batch_n):
    inds = list(range(data.shape[0]))
    np.random.shuffle(inds)

    for i in range(int(data.shape[0] / batch_n)):
        ii = inds[i*batch_n:(i+1)*batch_n]
        yield data[ii, :]

def buffered_gen(f, batch_n=1024, buffer_size=2000):
    inp = open(f)
    data = []

    for i, line in enumerate(inp):
        data.append(np.array(list(map(float, line.strip().split('\t')[1]))))
        if (i+1) % (buffer_size * batch_n) == 0:
            bgen = batch_gen(np.vstack(data), batch_n)
            for batch in bgen:
                yield batch
            data = []
            
    else:
        bgen = batch_gen(np.vstack(data[:-1]), batch_n)

        for batch in bgen:
            yield batch

def load_test():
    with open(test) as inp:
        data = [np.array(list(map(float, line.strip().split('\t')[1]))) for line in inp]
    return np.vstack(data)

# Utility functions for class AAE

def he_initializer(size):
    return tf.random_normal_initializer(mean=0.0, stddev=np.sqrt(1. / size), seed=None, dtype=tf.float32)

def linear_layer(tensor, input_size, out_size, init_fn=he_initializer,):
    W = tf.get_variable('W', shape=[input_size, out_size], initializer=init_fn(input_size))
    b = tf.get_variable('b', shape=[out_size], initializer=tf.constant_initializer(0.1))
    return tf.add(tf.matmul(tensor, W), b)

# NOTE: Currently not in use
def bn_layer(tensor, size, epsilon=0.0001):
    batch_mean, batch_var = tf.nn.moments(tensor, [0])
    scale = tf.get_variable('scale', shape=[size], initializer=tf.constant_initializer(1.))
    beta = tf.get_variable('beta', shape=[size], initializer=tf.constant_initializer(0.))
    return tf.nn.batch_normalization(tensor, batch_mean, batch_var, beta, scale, epsilon)

def sample_prior(loc=0., scale=1., size=(64, 10)):
    return np.random.normal(loc=loc, scale=scale, size=size)