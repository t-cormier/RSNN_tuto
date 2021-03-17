import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import models as m


fc_k = tf.random.uniform(shape=(15,))
w = tf.random.uniform(shape=(1, 700, 100, 100))

z_t = tf.random.uniform(shape=(1,100))
wp_t = tf.random.uniform(shape=(1,100,1))
W_p = tf.stack([2. for i in range(700)])

def vector_shape(vector):

    return tf.reshape(vector, [1, int(vector.shape[1]), 1])

def kernel_shape(kernel):
    print(kernel.shape)
    return tf.reshape(kernel, [int(kernel.shape[0]), 1, 1])


print(len(fc_k))
