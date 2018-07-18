# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


size = 400



inp = tf.placeholder(tf.float32, shape=(None, 5))

seed = tf.get_variable('seed', (5,), initializer=tf.random_uniform_initializer(minval=-10, maxval=10))

batch_size = tf.shape(inp)[0]

h = tf.concat([inp, tf.tile(tf.expand_dims(seed, 0), [batch_size, 1])], 1)
h = tf.layers.dense(h, 20, activation=tf.nn.sigmoid,
                    kernel_initializer=tf.glorot_normal_initializer(),
                    bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)) * 100
h = tf.layers.dense(h, 10, activation=tf.nn.sigmoid,
                    kernel_initializer=tf.glorot_normal_initializer(),
                    bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)) * 8
h = tf.layers.dense(h, 1, activation=tf.sin,
                    kernel_initializer=tf.glorot_normal_initializer(),
                    bias_initializer=tf.glorot_normal_initializer())

if True:
    h = h / 2. + 0.5
else:
    h = h ** 2


sess = tf.Session()

def gen():
    sess.run(tf.global_variables_initializer())

    x = [
        ((x-size/2)*0.2, (y-size/2)*0.2, float(r == 0), float(r == 1), float(r == 2))
        for x in range(size)
        for y in range(size)
        for r in range(3)
    ]
    y = sess.run(h, {inp: x})

    return np.reshape(y, (size, size, 3))

if __name__ == '__main__':
    plt.gca().axison = False
    plt.imshow(gen())
    if False:
        plt.show()
    else:
        for x in range(10):
            plt.savefig('img{}.png'.format(x))
