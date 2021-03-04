import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


z = tf.cast(np.arange(10), tf.float32)
er = tf.cast(np.arange(10, 1, -1), tf.float32)
d = tf.zeros_like(z)
print(z, er, d)
for t in range(10):
    int = 0
    for r in range(t):
        int += er[r]*z[t-r]
    d.numpy()[t] = int

print(d)
