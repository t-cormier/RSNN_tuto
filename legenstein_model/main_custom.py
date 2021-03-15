import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np

import models as m

tf.config.run_functions_eagerly(True)

seq_len=1000
n_input=20
n_recurrent=100
cn_idx = 1
epochs = 1
idx_cn = 1
num_itr = 20

train_writer = tf.summary.create_file_writer("logs/train/")
test_writer = tf.summary.create_file_writer("logs/test/")

exp_model = m.Exp_model(n_recurrent, n_input, seq_len)
dataset = m.create_data_set(seq_len, n_input, itr=num_itr)
optimizer = keras.optimizers.Adam(lr=1e-3)


for epoch in range(epochs):
    for batch_idx, (x,y) in enumerate(dataset) :
        print(f'\rEpoch : {epoch}, {batch_idx}/{num_itr}', end="")
        with tf.GradientTape() as tape :
            v, z = exp_model(x)
            regularization_loss, _ = m.reg_loss(z)

        # compute the metric and losses(here it is the activity of the conditioned neuron)
        avg_act = m.compute_avg_activity(exp_model, z)
        avg_act_cn = avg_act[idx_cn]
        mask = np.zeros(len(avg_act.numpy()))
        mask[idx_cn] = 1
        mask = tf.cast(mask, dtype=bool)
        avg_act_nocn = tf.reduce_mean(tf.where(mask, tf.zeros_like(avg_act), avg_act))

        # compute the gradients ( grad = - delta w_ji = - d(t) * e_ji )
        d = m.compute_dopamine(idx_cn, z)
        etrace = m.compute_etrace(exp_model, v, z)

        leg_grads = tf.reduce_sum(d[:, :, None, None] * etrace, axis=(0, 1), name='leg_grads')
        vars = exp_model.trainable_variables

        reg_grads = tape.gradient(regularization_loss, vars)

        grads = reg_grads + leg_grads*10

        # tensorboard writer
        with train_writer.as_default():
            tf.summary.scalar("Cn activity", avg_act_cn, step=batch_idx)
            tf.summary.scalar("Average activity", avg_act_nocn, step=batch_idx)
        # Apply the gradients
        optimizer.apply_gradients(zip(grads, vars))
