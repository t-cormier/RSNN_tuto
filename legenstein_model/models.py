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




######### Create Dataset for Experiment###############
def create_data_set(seq_len, n_input, batch_size=700):
    n_batch = seq_len // batch_size
    x = tf.random.uniform(shape=(n_batch, batch_size, n_input)) * 0.25
    y = tf.zeros(shape=(n_batch, batch_size, 1))
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(1)
    return dataset



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

    def __init__(self, units, tau=20., tau_readout=50., thr=1., dt=1, n_refractory=5, dampening_factor=.3):
        super().__init__()
        self.units = units

        self._dt = float(dt)
        self._decay = tf.exp(-dt / tau)
        self._readout_decay = tf.exp(-dt / tau_readout)
        self._n_refractory = n_refractory


        self.threshold = thr
        self._dampening_factor = dampening_factor

        #                  voltage, refractory, previous spikes
        self.state_size = (units, units, units)


    def zero_state(self, dtype=tf.float32):
        v0 = tf.zeros((1, self.units), dtype)
        r0 = tf.zeros((1, self.units), tf.int32)
        z_buf0 = tf.zeros((1, self.units), dtype)
        return v0, r0, z_buf0

    def build(self, input_shape):
        # Weights
        self.input_weights = self.add_weight(shape=(input_shape[-1], self.units),
                                             initializer=tf.keras.initializers.RandomNormal(
                                                 stddev=1. / np.sqrt(input_shape[-1] + self.units)),
                                             trainable=False,
                                             name='input_weights')

        self.recurrent_weights = self.add_weight(shape=(self.units, self.units),
                                                 initializer=tf.keras.initializers.Orthogonal(gain=.7),
                                                 name='recurrent_weights')


        self.disconnect_mask = tf.cast(np.diag(np.ones(self.units, dtype=np.bool)), tf.bool)

        super().build(input_shape)

    def call(self, inputs, state):
        old_v = state[0]
        old_r = state[1]
        old_z = state[2]



        no_autapse_w_rec = tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights)

        i_in = tf.matmul(inputs, self.input_weights, name='I_in')

        i_rec = tf.matmul(old_z, no_autapse_w_rec, name='I_rec')
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
        output = (new_v, new_r, new_z)

        return output, new_state #v,r,z




############# Metrics and gradients ################

def compute_avg_activity(model, z):
    av = tf.reduce_mean(z, axis=(0, 1))
    return av # shape=(1,100)


def compute_etrace(model, v, z):

    v_scaled = tf.identity((v - model.cell.threshold) / model.cell.threshold, name='v_scaled')
    post_term = tf.identity(pseudo_derivative(v_scaled, model.cell._dampening_factor) / model.cell.threshold,name='psi')
    z_previous_time = shift_by_one_time_step(z)


    pre_term_w_rec = tf.identity(exp_convolve(z_previous_time, decay=model.cell._decay), name = 'z_bar')


    # Eligibility traces                                 # adding None as a dimension grants the right product
    eligibility_traces_w_rec = tf.identity(post_term[:, :, None, :] * pre_term_w_rec[:, :, :, None], name='etrace_rec')
    eligibility_traces_convolved_w_rec = tf.identity(exp_convolve(eligibility_traces_w_rec), name='fetrace_rec')

    # Warning : Only LTP implemented here
    return eligibility_traces_convolved_w_rec #shape=(None, 1000, 100, 100)







def compute_dopamine(idx_cn, z):
    # switch to time major
    shp = z.get_shape()
    seq_len = shp[1]
    r_shp = range(len(shp))
    transpose_perm = [1, 0] + list(r_shp)[2:]
    z_time_major = tf.transpose(z, perm=transpose_perm)
    z_time_major = tf.cast(z_time_major, dtype=tf.float32)

    def reward(time, A_p=1.379, A_m=0.27, tau_p=0.2, tau_m=1., scale=500):
        t = time / scale # very important hyper parameter in order to make the dopamine model work
        kernel_p = A_p * t / tau_p * np.exp(1 - t / tau_p)
        kernel_m = A_m * t / tau_m * np.exp(1 - t / tau_m)
        return kernel_p - kernel_m

    r_k = tf.stack([reward(t) for t in range(seq_len,0,-1)])
    z_cn = z_time_major[:, :, idx_cn]

    z_cn = tf.reshape(z_cn, [1, int(z_cn.shape[0]), 1], name='z_rev')
    r_k = tf.reshape(r_k, [int(r_k.shape[0]), 1, 1], name='reward_kernel')

    dopa = tf.nn.conv1d(z_cn, r_k, stride=1, padding='SAME')

    return dopa #shape=(1,1000)


def reg_loss(z, cn_idx, target_rate):
    av = tf.reduce_mean(z, axis=(0, 1))
    average_firing_rate_error = target_rate - av[cn_idx]
    regularization_loss = tf.maximum(average_firing_rate_error, 0)
    return regularization_loss


############### metrics ###################
class Activity_metric(tf.keras.metrics.Metric):
    """ Activity metrics for the conditioned neuron and the whole network """
    def __init__(self, cn_idx=None, name="activity_metric", **kwargs):
        super(Activity_metric, self).__init__(name=name, **kwargs)
        self.cn_idx = cn_idx
        self.avg_act = None
        self.avg_act_cn = None


    def update_state(self, y_true, z_pred, sample_weight=None):
        avg_activity = tf.reduce_mean(z_pred, axis=(0, 1))
        if self.cn_idx == None :
            self.avg_act = tf.reduce_mean(avg_activity)
        else :
            self.avg_act_cn = avg_activity[self.cn_idx]
            #assert self.avg_act_cn > 0, "Conditioned neuron has no spontaneous activity"

    def result(self):
        if self.cn_idx == None :
            return self.avg_act
        else :
            return self.avg_act_cn

    def reset_states(self):
        if self.cn_idx == None :
            self.avg_act = 0.
        else :
            self.avg_act_cn = 0.










############ Experiment model ##############
class Exp_model(keras.Model):
    """__init__ and passforward (__call__) of the model of the experiment"""

    def __init__(self, n_recurrent, n_input, seq_len, batch_size):
        super(Exp_model, self).__init__()
        self.cell = LIFCell(n_recurrent)
        self.rnn = layers.RNN(self.cell, return_sequences=True, return_state=True)
        self.init_volt = tf.Variable(self.cell.zero_state()[0], trainable=False, name='init_voltage')
        self.init_refrac = tf.Variable(self.cell.zero_state()[1], trainable=False, name='init_refractory')
        self.init_spike = tf.Variable(self.cell.zero_state()[2], trainable=False, name='init_spike')


    def call(self, inputs):
        init_state = (self.init_volt.value(), self.init_refrac.value(), self.init_spike.value())
        outputs, s_volt, s_refr, s_spike = self.rnn(inputs, initial_state=init_state)
        voltages, _, spikes = outputs
        self.init_volt.assign(s_volt)
        self.init_refrac.assign(s_refr)
        self.init_spike.assign(s_spike)

        return [voltages, spikes]



class Leg_fit(keras.Model):
    """Custom model.fit for (Legenstein and al., 2008) learning rule"""

    def __init__(self, model, cn_idx, target_rate=0.0001):
        super(Leg_fit, self).__init__()
        self.model = model
        assert cn_idx in np.arange(self.model.cell.units)
        self.cn = cn_idx
        self.target_rate = target_rate
        self.stop_training = False


    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape :
            v, z = self.model(x)
            regularization_loss_cn = reg_loss(z, self.cn, target_rate=self.target_rate)

        #self.metrics.reset_states()
        self.compiled_metrics.update_state(y, z)

        # compute the gradients ( grad = - delta w_ji = - d(t) * e_ji )
        vars = self.model.trainable_variables

        dopa = compute_dopamine(self.cn, z)

        etrace = compute_etrace(self.model, v, z)
        leg_grads = tf.reduce_sum(dopa[:, :, None, None] * etrace, axis=(0, 1), name='leg_grads')
        reg_grads = tape.gradient(regularization_loss_cn, vars)
        grads = reg_grads # + leg_grads

        # show the gradients as Metrics
        metric_leg_grads = tf.reduce_mean(leg_grads)
        metric_reg_grads = tf.reduce_mean(tf.math.abs(reg_grads))

        # Apply the gradients
        self.optimizer.apply_gradients(zip(grads, vars))

        return {'CN average activity' : self.metrics[0].result(),
                'CN regularization loss' : regularization_loss_cn,
                'average network activity' : self.metrics[1].result()}
                # 'Leg grads' : metric_leg_grads,
                # 'Reg grads' : metric_reg_grads}

        #return {m.name: m.result() for m in self.metrics}
