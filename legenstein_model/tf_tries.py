import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


etrace = tf.cast(tf.random.uniform(shape=(1,1000,100,100))>0.8, tf.float32)
# d = tf.cast(tf.random.uniform(shape=(1,1000,1))>0.8, tf.float32)
z = tf.cast(tf.random.uniform(shape=(1,1000,100))>0.5, tf.float32)

def reward_kernel(time, A_p=1.379, A_m=0.27, tau_p=0.2, tau_m=1.):
    t = time/250 # very important hyper parameter in order to make the dopamine model work
    kernel_p = A_p * t / tau_p * np.exp(1 - t / tau_p)
    kernel_m = A_m * t / tau_m * np.exp(1 - t / tau_m)
    return kernel_p - kernel_m


def compute_dopamine(z, idx_cn, r_kernel=reward_kernel):
    # switch to time major
    shp = z.get_shape()
    seq_len = shp[1]
    r_shp = range(len(shp))
    transpose_perm = [1, 0] + list(r_shp)[2:]
    z_time_major = tf.transpose(z, perm=transpose_perm)

    def xi_r(t):
        return tf.constant([r_kernel(i) for i in range(t)], shape=(t,1), dtype=tf.float32)

    d = tf.constant([tf.reduce_sum(xi_r(t) * z_time_major[t:0:-1, :, idx_cn]).numpy() for t in range(seq_len)], shape=(1,seq_len,1), dtype=tf.float32)

    return d

d = compute_dopamine(z, 2)
print(d)
