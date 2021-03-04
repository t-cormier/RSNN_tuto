import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt




######## Useful functions #######
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


def compute_refractory_count(z, n_ref):

    def update_refractory(refractory_count, z):
        return tf.where(z > 0,tf.ones_like(refractory_count) * (n_ref - 1),tf.maximum(0, refractory_count - 1))

    # Switch to time major :
    z = tf.transpose(z, perm=[1,0,2])

    refractory_count_init = tf.zeros_like(z[0], dtype=tf.int32)
    refractory_count = tf.scan(update_refractory, z[:-1], initializer=refractory_count_init)
    refractory_count = tf.concat([[refractory_count_init], refractory_count], axis=0)

    # Switch back to batch_size major
    refractory_count = tf.transpose(refractory_count, perm=[1,0,2])

    return refractory_count






####### Dataset #######
def create_data_set(seq_len, n_input, n_batch=1):
    x = tf.random.uniform(shape=(seq_len, n_input))[None] * .5
    y = tf.sin(tf.linspace(0., 4 * np.pi, seq_len))[None, :, None]
    return tf.data.Dataset.from_tensor_slices((x, y)).repeat(count=20).batch(n_batch)






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

    def __init__(self, units, connectivity=0.2, ei_ratio=0.2, tau=20., thr=1., dt=1, n_refractory=5, dampening_factor=.3, stop_gradients=False):
        super().__init__()
        self.units = units
        self.n_exc = int(self.units * ei_ratio)
        self.n_inh = self.units - self.n_exc

        self._dt = float(dt)
        self._decay = tf.exp(-dt / tau)
        self._n_refractory = n_refractory
        self._stop_gradients = stop_gradients
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

        if self._stop_gradients:
            old_z = tf.stop_gradient(old_z)

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





######### Loss and gradients #############
def loss_fn(error):
    return 0.5 * tf.reduce_sum((error) ** 2)

def reg_loss(z, target_rate=0.02):
    av = tf.reduce_mean(z, axis=(0, 1))
    average_firing_rate_error = av - target_rate
    return loss_fn(average_firing_rate_error), average_firing_rate_error

def compute_learning_signals(error, w_out):
    return tf.einsum('btk,jk->btj', error, w_out, name='learning_signals')

def compute_eprop_gradients(model, v, z, x , error1, error2, learning_signals, reg):
    v_scaled = tf.identity((v - model.cell.threshold) / model.cell.threshold, name='v_scaled')
    z_previous_time = shift_by_one_time_step(z)
    print(z)

    # Compute psi considering refractory period neurons
    psi_no_ref = pseudo_derivative(v_scaled, model.cell._dampening_factor) / model.cell.threshold
    ref_count = compute_refractory_count(z, model.cell._n_refractory)
    is_ref = ref_count > 0
    post_term = tf.where(is_ref, tf.zeros_like(psi_no_ref), psi_no_ref)

    # Compute psi not considering ref period
    # post_term = tf.identity(pseudo_derivative(v_scaled, model.cell._dampening_factor) / model.cell.threshold,name='psi')


    pre_term_w_in = tf.identity(exp_convolve(x, decay=model.cell._decay), name = 'x_bar')
    pre_term_w_rec = tf.identity(exp_convolve(z_previous_time, decay=model.cell._decay), name = 'z_bar')
    pre_term_w_out = exp_convolve(z, decay=model.cell._decay)

    # Eligibility traces
    eligibility_traces_w_in = tf.identity(post_term[:, :, None, :] * pre_term_w_in[:, :, :, None], name='etrace_in')
    eligibility_traces_w_rec = tf.identity(post_term[:, :, None, :] * pre_term_w_rec[:, :, :, None], name='etrace_rec')

    eligibility_traces_convolved_w_in = tf.identity(exp_convolve(eligibility_traces_w_in), name='fetrace_in')
    eligibility_traces_convolved_w_rec = tf.identity(exp_convolve(eligibility_traces_w_rec), name='fetrace_rec')

    eligibility_traces_averaged_w_in = tf.reduce_mean(eligibility_traces_w_in, axis=(0, 1), name='fetrace_in_reg')
    eligibility_traces_averaged_w_rec = tf.reduce_mean(eligibility_traces_w_rec, axis=(0, 1), name='fetrace_rec_reg')

    # Gradients
    dloss_dw_out = tf.reduce_sum(error1[:, :, None, :] * pre_term_w_out[:, :, :, None], axis=(0, 1))
    dloss_dw_in = tf.reduce_sum(learning_signals[:, :, None, :] * eligibility_traces_convolved_w_in, axis=(0, 1))
    dloss_dw_rec = tf.reduce_sum(learning_signals[:, :, None, :] * eligibility_traces_convolved_w_rec, axis=(0, 1))

    dreg_loss_dw_in = error2 * eligibility_traces_averaged_w_in
    dreg_loss_dw_rec = error2 * eligibility_traces_averaged_w_rec

    dloss_dw_in += dreg_loss_dw_in * reg
    dloss_dw_rec += dreg_loss_dw_rec * reg

    # zeros on the diagonal
    mask_autotapse = np.diag(np.ones(model.cell.units, dtype=bool))
    dloss_dw_rec = tf.where(mask_autotapse, tf.zeros_like(dloss_dw_rec), dloss_dw_rec)

    return [dloss_dw_in, dloss_dw_rec, dloss_dw_out]







###### Custom models ##########
class Exp_model(keras.Model):
    """__init__ and passforward (__call__) of the model of the experiment"""

    def __init__(self, n_recurrent):
        super(Exp_model, self).__init__()
        self.cell = LIFCell(n_recurrent)
        self.rnn = layers.RNN(self.cell, return_sequences=True)
        self.weighted_out_projection = layers.Dense(1, use_bias=False)


    def __call__(self, inputs):
        batch_size = tf.shape(inputs)[0]
        initial_state = self.cell.zero_state(batch_size)
        voltages, spikes = self.rnn(inputs, initial_state=initial_state)
        weighted_out = self.weighted_out_projection(spikes)
        prediction = exp_convolve(weighted_out, axis=1)

        return [voltages, spikes, prediction]



class Eprop_fit(keras.Model):
    """Custom model.fit for eprop gradients"""

    def __init__(self, model, method="symmetric", test=False):
        super(Eprop_fit, self).__init__()
        self.reg = 300
        self.model = model
        self.test = test
        if self.test :
            self.method = 'symmetric'
            self.model.cell._stop_gradients = True
        else :
            self.method = method

        assert method in ['symmetric', 'random', 'autodiff'], "Eprop method not defined properly : symmetric, random or autodiff"

        if self.method == 'autodiff' :
            self.model.cell._stop_gradients = True


    def train_step(self, data):
        x, y = data

        # Tape gradients :
        with tf.GradientTape() as tape:
            v, z, y_pred = self.model(x)
            output_error = tf.subtract(y_pred, y, name='output_error')

            # Compute main loss
            main_loss = loss_fn(output_error)

            # Compute reg_loss
            regularization_loss, average_firing_rate_error = reg_loss(z)

            # Compute overall loss
            overall_loss = regularization_loss * self.reg + main_loss


        return_dict = {'main_loss' : main_loss, 'reg_loss' : regularization_loss, 'overall_loss' : overall_loss}

        # Compute eprop Gradients
        if self.method == "symmetric" :
            w_out = self.model.weighted_out_projection.trainable_weights[0]
            learning_signals = compute_learning_signals(output_error, w_out)
        elif self.method == "random" :
            B_random = tf.constant(np.random.randn(self.model.cell.units, 1) / np.sqrt(self.model.cell.units), dtype=tf.float32, name='B_random')
            learning_signals = compute_learning_signals(output_error, w_out)

        vars = self.model.trainable_weights

        if self.method == "autodiff" :
            grads = tape.gradient(overall_loss, vars)
        else :
            grads = compute_eprop_gradients(self.model, v, z, x, output_error, average_firing_rate_error, learning_signals, self.reg)

        # Test if eprop and autodiff compute the same Gradients
        if self.test :
            assert self.model.cell._stop_gradients == True, f'stop_gradients is {self.model.cell._stop_gradients} and should be True'
            grads_autodiff = tape.gradient(overall_loss, vars)
            grads_eprop = compute_eprop_gradients(self.model, v, z, x, output_error, average_firing_rate_error, learning_signals, self.reg)
            delta_grads = tf.reduce_mean([tf.reduce_mean(g_auto - g_eprop) for g_auto, g_eprop in zip(grads_autodiff, grads_eprop)])
            return_dict = {'main_loss' : main_loss, 'reg_loss' : regularization_loss, 'overall_loss' : overall_loss, 'delta_grads' : delta_grads}

        # Apply Gradients
        self.optimizer.apply_gradients(zip(grads, vars))

        return return_dict







########## Custom Callbacks ############@
class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_example, fig, axes):
        super().__init__()
        self.test_example = test_example
        self.fig = fig
        self.axes = axes


    def on_epoch_end(self, epoch, logs=None):
        output = self.model.model(self.test_example[0])
        [ax.clear() for ax in self.axes]
        self.axes[0].pcolormesh(self.test_example[0].numpy()[0].T, cmap='cividis')
        self.axes[0].set_ylabel('input')
        self.axes[0].set_title(f'Epoch {epoch+1}')
        v = output[0].numpy()[0]
        z = output[1].numpy()[0]
        out = output[2].numpy()[0, :, 0]
        abs_max = np.abs(v).max()
        self.axes[1].pcolormesh(v.T, cmap='seismic', vmin=-abs_max, vmax=abs_max)
        self.axes[1].set_ylabel('voltage')
        self.axes[2].pcolormesh(z.T, cmap='Greys')
        self.axes[2].set_ylabel('spike')
        self.axes[3].plot(self.test_example[1][0, :, 0], 'k--', lw=2, alpha=.7, label='target')
        self.axes[3].plot(out, 'b', lw=2, alpha=.7, label='prediction')
        self.axes[3].set_ylabel('output')
        self.axes[3].legend(frameon=False)
        [ax.yaxis.set_label_coords(-.05, .5) for ax in self.axes]
        plt.draw()
        plt.pause(.2)
