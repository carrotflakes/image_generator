# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


inp = tf.placeholder(tf.float32, shape=(None, 2))

seed = tf.get_variable('seed', (1,), initializer=tf.random_uniform_initializer(minval=-10, maxval=10))

batch_size = tf.shape(inp)[0]

h = tf.concat([inp, tf.tile(tf.expand_dims(seed, 0), [batch_size, 1])], 1)
h = tf.layers.dense(h, 30, activation=tf.nn.sigmoid,
                    kernel_initializer=tf.glorot_normal_initializer(),
                    bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)) * 15
h = tf.layers.dense(h, 3, activation=tf.sin,
                    kernel_initializer=tf.glorot_normal_initializer(),
                    bias_initializer=tf.glorot_normal_initializer())

h = h / 2. + 0.5
#h = h ** 2

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
plt.show()
#plt.savefig('img.png')
