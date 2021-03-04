import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import models


def create_model(seq_len=1000, n_input=20, n_recurrent=100):
    inputs = tf.keras.layers.Input(shape=(seq_len, n_input))

    cell = models.LIFCell(n_recurrent)
    rnn = tf.keras.layers.RNN(cell, return_sequences=True)

    batch_size = tf.shape(inputs)[0]
    initial_state = cell.zero_state(batch_size)
    rnn_output = rnn(inputs, initial_state=initial_state)
    regularization_layer = models.SpikeVoltageRegularization(cell)
    voltages, spikes = regularization_layer(rnn_output)
    voltages = tf.identity(voltages, name='voltages')
    spikes = tf.identity(spikes, name='spikes')

    weighted_out_projection = tf.keras.layers.Dense(1)
    weighted_out = weighted_out_projection(spikes)

    prediction = models.exp_convolve(weighted_out, axis=1)
    prediction = tf.keras.layers.Lambda(lambda _a: _a, name='output')(prediction)

    return tf.keras.Model(inputs=inputs, outputs=[voltages, spikes, prediction])


def create_data_set(seq_len=1000, n_input=20, n_batch=1):
    x = tf.random.uniform(shape=(seq_len, n_input))[None] * .5
    y = tf.sin(tf.linspace(0., 4 * np.pi, seq_len))[None, :, None]

    return tf.data.Dataset.from_tensor_slices((x, dict(output=y))).repeat(count=20).batch(n_batch)


class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_example, fig, axes):
        super().__init__()
        self.test_example = test_example
        self.fig = fig
        self.axes = axes

    def on_epoch_end(self, epoch, logs=None):
        output = self.model(self.test_example[0])
        [ax.clear() for ax in self.axes]
        self.axes[0].pcolormesh(self.test_example[0].numpy()[0].T, cmap='cividis')
        self.axes[0].set_ylabel('input')
        v = output[0].numpy()[0]
        z = output[1].numpy()[0]
        out = output[2].numpy()[0, :, 0]
        abs_max = np.abs(v).max()
        self.axes[1].pcolormesh(v.T, cmap='seismic', vmin=-abs_max, vmax=abs_max)
        self.axes[1].set_ylabel('voltage')
        self.axes[2].pcolormesh(z.T, cmap='Greys')
        self.axes[2].set_ylabel('spike')
        self.axes[3].plot(self.test_example[1]['output'][0, :, 0], 'k--', lw=2, alpha=.7, label='target')
        self.axes[3].plot(out, 'b', lw=2, alpha=.7, label='prediction')
        self.axes[3].set_ylabel('output')
        self.axes[3].legend(frameon=False)
        [ax.yaxis.set_label_coords(-.05, .5) for ax in self.axes]
        plt.draw()
        plt.pause(.2)



def main(args):
    model = create_model()
    data_set = create_data_set()
    it = iter(data_set)
    test_example = next(it)

    if args.do_plot:
        plt.ion()
        fig, axes = plt.subplots(4, figsize=(6, 8), sharex=True)
        plot_callback = PlotCallback(test_example, fig, axes)
        callbacks = [plot_callback]
    else:
        callbacks = []

    # train the model
    opt = tf.keras.optimizers.Adam(lr=1e-3)
    mse = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=opt, loss=dict(output=mse))
    model.fit(data_set, epochs=20, callbacks=callbacks)


    # analyse the model
    inputs = test_example[0]
    targets = test_example[1]['output'].numpy()
    voltage, spikes, prediction = model(inputs)

    voltage = voltage.numpy()
    spikes = spikes.numpy()
    prediction = prediction.numpy()

    print(f'inputs:            array with shape {inputs.shape}')
    print(f'membrane voltages: array with shape {voltage.shape}')
    print(f'spikes:            array with shape {spikes.shape}')
    print(f'prediction:        array with shape {prediction.shape}')
    print(f'targets:           array with shape {targets.shape}')

    plt.ioff()
    plt.show(block=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_plot', action='store_true', default=False)
    _args = parser.parse_args()
    main(_args)
