import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt




######### Spiking functions ###########@
def pseudo_derivative(v_scaled, dampening_factor): # checked
    return dampening_factor * tf.maximum(1 - tf.abs(v_scaled), 0)


@tf.custom_gradient
def spike_function(v_scaled, dampening_factor): # checked
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
    # switch to time major (instead of batch_size major)
    perm = np.arange(rank)
    perm[0], perm[axis] = perm[axis], perm[0]
    tensor = tf.transpose(tensor, perm)

    if initializer is None:
        initializer = tf.zeros_like(tensor[0])

    def scan_fun(_acc, _t):
        return _acc * decay + _t

    filtered = tf.scan(scan_fun, tensor, reverse=reverse, initializer=initializer)

    # Switch back to batch_size major
    filtered = tf.transpose(filtered, perm)
    return filtered




def shift_by_one_time_step(tensor, initializer=None):
    assert tensor.dtype in [tf.float16, tf.float32, tf.float64]
    r_shp = range(len(tensor.get_shape()))
    transpose_perm = [1, 0] + list(r_shp)[2:]
    tensor_time_major = tf.transpose(tensor, perm=transpose_perm)

    if initializer is None:
        initializer = tf.zeros_like(tensor_time_major[0])

    shifted_tensor = tf.concat([initializer[None, :, :], tensor_time_major[:-1]], axis=0)

    shifted_tensor = tf.transpose(shifted_tensor, perm=transpose_perm)

    return shifted_tensor





####### Layer definition #########
class CellConstraint(keras.constraints.Constraint):
    def __init__(self, connectivity_mask, EI_mask, disconnect_mask):
        self.connectivity_mask = connectivity_mask
        self.EI_mask = EI_mask
        self.disconnect_mask = disconnect_mask

    def __call__(self, w):
        w = tf.where(self.connectivity_mask, w, tf.zeros_like(w))
        w = tf.where(self.EI_mask, tf.nn.relu(w), -tf.nn.relu(-w))
        w = tf.where(self.disconnect_mask, tf.zeros_like(w), w)
        return w


class LIFCell(layers.Layer):
    """RSNN model for the Experiemnt (LIF)"""

    def __init__(self, units, connectivity=0.2, ei_ratio=0.2, tau=20., thr=1., dt=1, n_refractory=5, dampening_factor=.3):
        super().__init__()
        self.units = units
        self.n_exc = int(self.units * ei_ratio)
        self.n_inh = self.units - self.n_exc

        self._dt = float(dt)
        self._decay = tf.exp(-dt / tau)
        self._n_refractory = n_refractory
        self._connect = connectivity


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
        # Weights
        self.input_weights = self.add_weight(shape=(input_shape[-1], self.units),
                                             initializer=tf.keras.initializers.RandomNormal(
                                                 stddev=1. / np.sqrt(input_shape[-1] + self.units)),
                                             name='input_weights')

        self.recurrent_weights = self.add_weight(shape=(self.units, self.units),
                                                 initializer=tf.keras.initializers.Orthogonal(gain=.7),
                                                 name='recurrent_weights')

        # Masks
        self.disconnect_mask = tf.cast(np.diag(np.ones(self.units, dtype=np.bool)), tf.bool)
        nump_connectivity_mat = (np.random.uniform(size=(self.units, self.units)) < self._connect)
        self.connectivity_mask = tf.cast(nump_connectivity_mat, tf.bool)
        nump_EI_mat = np.concatenate((np.ones((self.units, self.n_exc), dtype=np.bool),
                                     np.zeros((self.units, self.n_inh), dtype=np.bool)),
                                     axis=1)
        self.EI_mask = tf.cast(nump_EI_mat, tf.bool)

        # Constraint
        self.constraint = CellConstraint(self.connectivity_mask, self.EI_mask, self.disconnect_mask)
        super().build(input_shape)

    def call(self, inputs, state):
        old_v = state[0]
        old_r = state[1]
        old_z = state[2]

        corrected_w = self.constraint(self.recurrent_weights)

        i_in = tf.matmul(inputs, self.input_weights)
        i_rec = tf.matmul(old_z, corrected_w)
        i_reset = -self.threshold * old_z
        input_current = i_in + i_rec + i_reset

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

        return output, new_state #v,r,z




############# Metrics and gradients ################
def compute_cn_activity(arg):
    pass

def compute_etrace(model, v, z):
    v_scaled = tf.identity((v - model.cell.threshold) / model.cell.threshold, name='v_scaled')
    post_term = tf.identity(pseudo_derivative(v_scaled, model.cell._dampening_factor) / model.cell.threshold,name='psi')
    z_previous_time = shift_by_one_time_step(z)

    pre_term_w_rec = tf.identity(exp_convolve(z_previous_time, decay=model.cell._decay), name = 'z_bar')

    # Eligibility traces
    eligibility_traces_w_rec = tf.identity(post_term[:, :, None, :] * pre_term_w_rec[:, :, :, None], name='etrace_rec')
    eligibility_traces_convolved_w_rec = tf.identity(exp_convolve(eligibility_traces_w_rec), name='fetrace_rec')

    return eligibility_traces_convolved_w_rec

def reward_kernel(time, A_p=1.379, A_m=0.27, tau_p=0.2, tau_m=1.):
    kernel_p = A_p * time / tau_p * np.exp(1 - time / tau_p)
    kernel_m = A_m * time / tau_m * np.exp(1 - time / tau_m)
    return kernel_p - kernel_m

def compute_dopamine(z):
    # switch to time major
    # define the scan fun (z and reward kernel)
    # define initializer
    # scan the time major z with the scan fun
    # switch back to batch_size major
    pass


############ Experiment model ##############
class Exp_model(keras.Model):
    """__init__ and passforward (__call__) of the model of the experiment"""

    def __init__(self, n_recurrent, n_input, seq_len):
        super(Exp_model, self).__init__()
        self.cell = LIFCell(n_recurrent)
        self.rnn = layers.RNN(self.cell, return_sequences=True)

    def __call__(self, inputs):
        batch_size = tf.shape(inputs)[0]
        initial_state = self.cell.zero_state(batch_size)
        voltages, spikes = self.rnn(inputs, initial_state=initial_state)

        return [voltages, spikes]



class Leg_fit(keras.model):
    """Custom model.fit for (Legenstein and al., 2008) learning rule"""

    def __init__(self, model):
        super(Leg_fit, self).__init__()
        self.model = model

    def train_step(self, data):
        x, y = data

        v, z = self.model(x)

        # compute the metric (here it is the activity of the conditioned neuron)

        # compute the gradients ( grad = - delta w_ji = - d(t) * e_ji )

        # Apply the gradients
