import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

import models as m

tf.config.run_functions_eagerly(True)

######## Constants #######
seq_len=1000
n_input=20
n_recurrent=100
cn_idx = 10
epochs = 1



######## Train ###########
# init
exp_model = m.Exp_model(n_recurrent, n_input, seq_len)
dataset = m.create_data_set(seq_len, n_input)



# define the training model
leg = m.Leg_fit(exp_model, cn_idx)
opt = keras.optimizers.Adam(lr=1e-3)
leg.compile(optimizer = opt)

leg.fit(dataset, epochs=epochs)
