import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

import models as m



######## Constants #######
seq_len=1000
n_input=20
n_recurrent=100

epochs = 20



######## Train ###########
# init
exp_model = m.Exp_model(n_recurrent)
dataset = m.create_data_set(seq_len, n_input)
it = iter(dataset)
test_example = next(it)

# define the training model
eprop = m.Eprop_fit(exp_model, method="random")
opt = m.keras.optimizers.Adam(lr=1e-3)
eprop.compile(optimizer = opt)

# define Callbacks
plt.ion()
fig, axes = plt.subplots(4, figsize=(6, 8), sharex=True)
plot_callback = m.PlotCallback(test_example, fig, axes)
callbacks = [plot_callback]

eprop.fit(dataset, epochs=epochs, callbacks=callbacks)

while True:
    wait = input('Press q to quit : ')
    if wait=='q':
        break
    else :
        pass
