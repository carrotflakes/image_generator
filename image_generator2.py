# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


inp_w = tf.placeholder(tf.int32, shape=(None))
inp_h = tf.placeholder(tf.int32, shape=(None))

xs = tf.range(inp_w) / (inp_w - 1) * 2 - 1
ys = tf.range(inp_h) / (inp_h - 1) * 2 - 1

tf.tile(xs)


seed = tf.get_variable('seed', (1,), initializer=tf.random_uniform_initializer(minval=-10, maxval=10))

batch_size = tf.shape(inp)[0]

xyr = tf.concat()

h = tf.concat([
    inp,
    f.tile(tf.expand_dims(seed, 0), [batch_size, 1])
], 1)
h = tf.layers.dense(h, 30, activation=tf.nn.sigmoid,
                    kernel_initializer=tf.glorot_normal_initializer(),
                    bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)) * 15
h = tf.layers.dense(h, 1, activation=tf.sin,
                    kernel_initializer=tf.glorot_normal_initializer(),
                    bias_initializer=tf.glorot_normal_initializer())

h = h / 2. + 0.5
#h = h ** 2

h = tf.reshape(h, [-1, 3])

sess = tf.Session()

sess.run(tf.global_variables_initializer())

size = 200

x = [
    ((x-size/2)*0.2, (y-size/2)*0.2)
    for x in range(size)
    for y in range(size)
]
y = sess.run(h, {inp: x})

print(y[0])
plt.imshow(np.reshape(y, (size, size, 3)))
if True:
    plt.show()
else:
    plt.savefig('img.png')
