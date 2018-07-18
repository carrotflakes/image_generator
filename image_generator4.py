# -*- coding: utf-8 -*-

import tensorflow as tf
from PIL import Image
import numpy as np


size = 600



inp = tf.placeholder(tf.float32, shape=(None, 5))

seed = tf.get_variable('seed', (3,), initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

batch_size = tf.shape(inp)[0]

h = tf.concat([inp, tf.tile(tf.expand_dims(seed, 0), [batch_size, 1])], 1) * 0.4
h = tf.layers.dense(h, 20, activation=tf.nn.sigmoid,
                    kernel_initializer=tf.glorot_normal_initializer(),
                    bias_initializer=tf.random_uniform_initializer(minval=-0.5, maxval=0.5)) * 100
h = tf.layers.dense(h, 10, activation=tf.nn.tanh,
                    kernel_initializer=tf.glorot_normal_initializer(),
                    bias_initializer=tf.random_uniform_initializer(minval=-0.5, maxval=0.5)) * 100
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

    scale = 0.2
    x = [
        ((x-size/2)*scale, (y-size/2)*scale, float(r == 0), float(r == 1), float(r == 2))
        for x in range(size)
        for y in range(size)
        for r in range(3)
    ]
    y = sess.run(h, {inp: x})

    return Image.fromarray(np.uint8(np.reshape(y, (size, size, 3)) * 255))

if __name__ == '__main__':
    if False:
        gen().show()
    else:
        for i in range(10):
            gen().save('img{}.png'.format(i))
