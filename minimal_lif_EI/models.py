import numpy as np
import tensorflow as tf


def pseudo_derivative(v_scaled, dampening_factor):
    return dampening_factor * tf.maximum(1 - tf.abs(v_scaled), 0)


@tf.custom_gradient
def spike_function(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad


def exp_convolve(tensor, decay=.8, reverse=False, initializer=None, axis=0):
    rank = len(tensor.get_shape())
    perm = np.arange(rank)
    perm[0], perm[axis] = perm[axis], perm[0]
    tensor = tf.transpose(tensor, perm)

    if initializer is None:
        initializer = tf.zeros_like(tensor[0])

    def scan_fun(_acc, _t):
        return _acc * decay + _t

    filtered = tf.scan(scan_fun, tensor, reverse=reverse, initializer=initializer)

    filtered = tf.transpose(filtered, perm)
    return filtered


class CellConstraint(tf.keras.constraints.Constraint):
    def __init__(self, connectivity_mask, EI_mask, disconnect_mask):
        self.connectivity_mask = connectivity_mask
        self.EI_mask = EI_mask
        self.disconnect_mask = disconnect_mask

    def __call__(self, w):
        w = tf.where(self.connectivity_mask, w, tf.zeros_like(w))
        w = tf.where(self.EI_mask, tf.nn.relu(w), -tf.nn.relu(-w))
        w = tf.where(self.disconnect_mask, tf.zeros_like(w), w)
        return w


class LIFCell(tf.keras.layers.Layer):
    def __init__(self, units, connectivity=0.2, ei_ratio=0.2, tau=20., thr=1., dt=1, n_refractory=5, dampening_factor=.3):
        super().__init__()
        self.units = units
        self._n_inh = int(units * ei_ratio)
        self._n_exc = units - self._n_inh
        self._connect = connectivity

        self._dt = float(dt)
        self._decay = tf.exp(-dt / tau)
        self._n_refractory = n_refractory

        self.input_weights = None
        self.bias_currents = None
        self.recurrent_weights = None # weight matrix
        self.connectivity_mask = None
        self.EI_mask = None
        self.disconnect_mask = None

        self.threshold = thr
        self._dampening_factor = dampening_factor

        #                  voltage, refractory, previous spikes
        self.state_size = (units, units, units)

    def zero_state(self, batch_size, dtype=tf.float32):
        v0 = tf.zeros((batch_size, self.units), dtype)
        r0 = tf.zeros((batch_size, self.units), tf.int32)
        z_buf0 = tf.zeros((batch_size, self.units), tf.float32)
        return v0, r0, z_buf0

    def build(self, input_shape):
        self.input_weights = self.add_weight(shape=(input_shape[-1], self.units),
                                             initializer=tf.keras.initializers.RandomNormal(
                                                 stddev=1. / np.sqrt(input_shape[-1] + self.units)),
                                             name='input_weights')
        nump_connectivity_mat = (np.random.uniform(size=(self.units, self.units)) < self._connect)
        self.connectivity_mask = tf.cast(nump_connectivity_mat, tf.bool)
        nump_EI_mat = np.concatenate((np.ones((self.units, self._n_exc), dtype=np.bool),
                                     np.zeros((self.units, self._n_inh), dtype=np.bool)),
                                     axis=1)
        self.EI_mask = tf.cast(nump_EI_mat, tf.bool)
        self.disconnect_mask = tf.cast(np.diag(np.ones(self.units, dtype=np.bool)), tf.bool)
        self.recurrent_weights = self.add_weight(
            shape=(self.units, self.units),
            initializer=tf.keras.initializers.Orthogonal(gain=.7),
            name='recurrent_weights')
        self.bias_currents = self.add_weight(shape=(self.units,),
                                             initializer=tf.keras.initializers.Zeros(),
                                             name='bias_currents')
        super().build(input_shape)

    def call(self, inputs, state):
        old_v = state[0]
        old_r = state[1]
        old_z = state[2]

        constraint = CellConstraint(self.connectivity_mask, self.EI_mask, self.disconnect_mask)
        corrected_w = constraint(self.recurrent_weights)
        #corrected_w = tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights)

        i_in = tf.matmul(inputs, self.input_weights)
        i_rec = tf.matmul(old_z, corrected_w)
        i_reset = -self.threshold * old_z
        input_current = i_in + i_rec + i_reset + self.bias_currents[None]

        new_v = self._decay * old_v + input_current

        is_refractory = tf.greater(old_r, 0)
        v_scaled = (new_v - self.threshold) / self.threshold
        new_z = spike_function(v_scaled, self._dampening_factor)
        new_z = tf.where(is_refractory, tf.zeros_like(new_z), new_z)
        new_r = tf.clip_by_value(
            old_r - 1 + tf.cast(new_z * self._n_refractory, tf.int32),
            0,
            self._n_refractory)

        new_state = (new_v, new_r, new_z)
        output = (new_v, new_z)

        return output, new_state


class SpikeVoltageRegularization(tf.keras.layers.Layer):
    def __init__(self, cell, rate_cost=.1, voltage_cost=.01, target_rate=.02):
        self._rate_cost = rate_cost
        self._voltage_cost = voltage_cost
        self._target_rate = target_rate
        self._cell = cell
        super().__init__()

    def call(self, inputs, **kwargs):
        voltage = inputs[0]
        spike = inputs[1]
        upper_threshold = self._cell.threshold

        rate = tf.reduce_mean(spike, axis=(0, 1))
        global_rate = tf.reduce_mean(rate)
        self.add_metric(global_rate, name='rate', aggregation='mean')

        reg_loss = tf.reduce_sum(tf.square(rate - self._target_rate)) * self._rate_cost
        self.add_loss(reg_loss)
        self.add_metric(reg_loss, name='rate_loss', aggregation='mean')

        v_pos = tf.square(tf.clip_by_value(tf.nn.relu(voltage - upper_threshold), 0., 1.))
        v_neg = tf.square(tf.clip_by_value(tf.nn.relu(-voltage - self._cell.threshold), 0., 1.))
        voltage_loss = tf.reduce_mean(tf.reduce_sum(v_pos + v_neg, -1)) * self._voltage_cost
        self.add_loss(voltage_loss)
        self.add_metric(voltage_loss, name='voltage_loss', aggregation='mean')
        return inputs
