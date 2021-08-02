import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import io

import LIFcell_leg as l




########## Create Dataset for Experiment###############
def gather_cols(params, indices, name=None):

    p_shape = tf.shape(params)
    p_flat = tf.reshape(params, [-1])
    i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
    return tf.reshape(tf.gather(p_flat, i_flat),
                          [p_shape[0], -1])


def exp_convolve(tensor, decay=.8, gain=1., reverse=False, initializer=None, axis=0):
    rank = len(tensor.get_shape())
    # switch to time major (instead of batch_size major)
    perm = np.arange(rank)
    perm[0], perm[axis] = perm[axis], perm[0]
    tensor = tf.transpose(tensor, perm)

    if initializer is None:
        initializer = tf.zeros_like(tensor[0])

    def scan_fun(_acc, _t):
        return _acc * decay + gain * _t

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


def create_data_set(seq_len, batch_size=700):
    n_batch = seq_len // batch_size
    x = tf.zeros(shape=(n_batch, batch_size, 1))
    y = tf.zeros(shape=(n_batch, batch_size, 1))
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(1)
    return dataset

def create_data_set_OU(seq_len, n_recurrent, batch_size=700):
    n_batch = seq_len // batch_size
    tf.random.set_seed(666)
    x = tf.random.normal(shape=(n_batch, batch_size, 2, n_recurrent), seed=1234)
    y = tf.zeros(shape=(n_batch, batch_size, 1))
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(1)
    return dataset

############# Metrics and gradients ################


def leg_etrace(model, z):
    wmax = model.cell.wmax
    shp = z.get_shape()
    batch_size = shp[1]

    A_p = 0.01 * wmax
    A_m = 1.05 * A_p
    tau_p = 30
    tau_m = 30
    decay_p = tf.exp(-1/tau_p)
    decay_m = tf.exp(-1/tau_m)
    tau_e = 400
    decay_e = tf.exp(-1/tau_e)

    def build_z_pad(z, z_pre):
        n_pre = z_pre.get_shape()[1]
        z_pre_flat = z_pre[:,0,:, :]
        for n in range(1,n_pre):
            z_pre_flat = tf.concat([z_pre_flat, z_pre[:, n, :, :]], axis=1)
        z_pad = tf.concat([z_pre_flat, z], axis=1 )
        return z[:,1:, :]




    ########################################################
    # simple eligibility trace computation
    z_previous = shift_by_one_time_step(z)  # (1, 700, 100)
    STDP_m = exp_convolve(z_previous, decay=decay_m, gain=-A_m, axis=1)
    STDP_p = exp_convolve(z, decay=decay_p, gain=A_p, axis=1)
    STDP = tf.matmul(z_previous[:, :, :, None], STDP_p[:, :, None, :]) + tf.matmul(STDP_m[:, :, :, None], z[:, :, None, :]) # (1,700, 100, 100)
    etrace = exp_convolve(STDP, decay = decay_e, gain=1/100, axis=1)
    ##########################################################


    ######################################################
    # more complex eligibility trace computation (formula from the article)
    # z_memory = build_z_pad(z, z_pre)
    # z_previous = shift_by_one_time_step(z_memory)  # (1, 700, 100)
    # STDP_m = exp_convolve(z_previous, decay=decay_m, gain=-A_m, axis=1)
    # STDP_p = exp_convolve(z, decay=decay_p, gain=A_p, axis=1)
    # STDP = tf.matmul(z_previous[:, :, :, None], STDP_p[:, :, None, :]) + tf.matmul(STDP_m[:, :, :, None], z_memory[:, :, None, :]) # (1,700, 100, 100)
    # def f_c(time, tau_e=200):
    #     return time/tau_e * tf.exp(time/tau_e)
    #
    # def vector_shape(vector):
    #     vector = tf.reshape(vector, [1, int(vector.shape[1]), 1])
    #     return tf.pad(vector, [[0,0], [batch_size-1,0], [0,0]])
    #
    # def kernel_shape(kernel):
    #     return tf.reshape(kernel, [int(kernel.shape[0]), 1, 1])
    #
    # fc = tf.stack( [ f_c(s) for s in range(STDP.shape[1],0,-1) ] ) # (700,)
    # fc_k = kernel_shape(fc)
    #
    # etrace = tf.stack([
    #     tf.stack([
    #         tf.nn.conv1d(vector_shape(elem_ji), fc_k, 1, padding='VALID')  for elem_ji in tf.unstack(elem_j, axis=2)
    #     ], axis=2)  for  elem_j in tf.unstack(STDP, axis=2)
    # ], axis=2) # (1, 700, 100, 100, 1)
    #
    # etrace = tf.reshape(etrace, etrace.shape[:-1]) # (1, 700, 100, 100)
    #######################################################################

    return etrace


def new_leg_dopamine(idx_cn, z, z_cn_pre):

    def reward(time, A_p=1.379, A_m=0.27, tau_p=200, tau_m=1023, scale=1):
        t = time / scale
        kernel_p = A_p * t / tau_p * np.exp(1 - t / tau_p)
        kernel_m = A_m * t / tau_m * np.exp(1 - t / tau_m)
        return kernel_p - kernel_m

    def build_z_cn_pad(z_cn, z_cn_pre):
        n_pre = z_cn_pre.get_shape()[1]
        z_cn_pre_flat = z_cn_pre[:,0,:]
        for n in range(1,n_pre):
            z_cn_pre_flat = tf.concat([z_cn_pre_flat, z_cn_pre[:, n, :]], axis=1)
        z_cn = tf.concat([z_cn_pre_flat, z_cn], axis=1 )
        return z_cn[:,1:]


    z_cn = z[:, :, idx_cn]
    z_cn_pad = build_z_cn_pad(z_cn, z_cn_pre)


    len_z = z_cn.get_shape()[1]  # batch_size
    len_r = z_cn_pad.get_shape()[1]
    r_k = tf.stack([reward(t) for t in range(len_r-len_z+1,0,-1)])


    z_cn_pad = tf.reshape(z_cn_pad, [1, int(z_cn_pad.shape[1]), 1], name='z_rev')
    r_k = tf.reshape(r_k, [int(r_k.shape[0]), 1, 1], name='reward_kernel')

    dopa = tf.nn.conv1d(z_cn_pad, r_k, stride=1, padding='VALID')
    dopa = tf.reshape(dopa, dopa.shape[:-1])

    return dopa #shape=(1,700)

def leg_gradients(model, dopamine, etrace):
    grads =  tf.stack([
        tf.stack([
            dopamine * etrace_ji for etrace_ji in tf.unstack(etrace_i, axis=2)
        ], axis=2) for etrace_i in tf.unstack(etrace, axis=2)
    ], axis=2) # (1, 700, 100, 100)
    dw_rec_dt = tf.reduce_sum(grads, axis=[0,1])[:model.cell.n_exc, :model.cell.n_exc] # (80, 80)
    # bring to zero the gradients where the weights are out of bound
    w_rec_ex = model.cell.recurrent_weights_EE.value()
    up_bound = tf.greater(w_rec_ex, model.cell.wmax)
    low_bound = tf.less(w_rec_ex, 0.)
    dw_rec_dt = tf.where(up_bound, tf.minimum(tf.zeros_like(dw_rec_dt), dw_rec_dt), dw_rec_dt)
    dw_rec_dt = tf.where(low_bound, tf.maximum(tf.zeros_like(dw_rec_dt), dw_rec_dt), dw_rec_dt)
    return [-dw_rec_dt]


############### metrics ###################
class Activity_metric(tf.keras.metrics.Metric):
    """ Activity metrics for the conditioned neuron and the whole network """
    def __init__(self, cn_bool=False, name="activity_metric", **kwargs):
        super(Activity_metric, self).__init__(name=name, **kwargs)
        self.cn_bool = cn_bool
        self.avg_act = None
        self.avg_act_cn = None


    def update_state(self, cn_idx, z_pred, sample_weight=None):
        avg_activity = tf.reduce_mean(z_pred, axis=(0, 1))
        if self.cn_bool :
            self.avg_act_cn = avg_activity[cn_idx]*1000

        else :
            self.avg_act = tf.reduce_mean(avg_activity)*1000

    def result(self):
        if self.cn_bool :
            return self.avg_act_cn
        else :
            return self.avg_act







class Buffer_metric(tf.keras.metrics.Metric):
    """Buffer to store all the values of the neurons activity"""
    def __init__(self, batch_size, cn_bool=False, name="activity_metric", **kwargs):
        super(Buffer_metric, self).__init__(name=name, **kwargs)
        self.cn_bool = cn_bool
        self.batch_size = batch_size
        self.avg_act = tf.Variable(tf.zeros(self.batch_size), trainable=False)
        self.avg_act_cn = tf.Variable(tf.zeros(self.batch_size), trainable=False)
        self.batch_idx = tf.Variable(0, trainable=False)


    def update_state(self, cn_idx, z_pred, sample_weight=None):
        avg_activity = tf.reduce_mean(z_pred, axis=(0, 1))

        if self.cn_bool :
            self.avg_act_cn.assign_add(  tf.one_hot( indices=self.batch_idx, depth=self.batch_size, on_value=avg_activity[cn_idx]*1000 )   )
        else :
            self.avg_act.assign_add(  tf.one_hot( indices=self.batch_idx, depth=self.batch_size, on_value=tf.reduce_mean(avg_activity)*1000 )  )

        self.batch_idx.assign_add(1)


    def result(self):
        if self.cn_bool :
            return self.avg_act_cn.value()
        else :
            return self.avg_act.value()












############ Experiment model ##############
class Exp_model(keras.Model):
    """__init__ and passforward (__call__) of the model of the experiment"""

    def __init__(self, n_recurrent, n_batch, batch_size, connectivity=[0.02, 0.02, 0.024, 0.016], w=[10.7, 211.6], noise_factor=0.2, con_factor=0.4):
        super(Exp_model, self).__init__()
        self.n_batch = n_batch
        self.batch_size = batch_size
        self.cell = l.LIFCell(n_recurrent, connectivity=connectivity, w_init_exc=w[0], w_init_inh=w[1], noise_factor=noise_factor, con_factor=con_factor)
        self.rnn = layers.RNN(self.cell, return_sequences=True, return_state=True)
        self.cn = tf.Variable(self.cell.units+1, trainable=False, dtype=tf.int32)


    def call(self, inputs):
        init_state = self.cell.zero_state()
        outputs, _, _, _, _, _, _ = self.rnn(inputs, initial_state=init_state)
        voltages, spikes = outputs

        return [voltages, spikes]

    def cn_search(self, dataset, time_length):
        # pass in the dataset and run the avergae over a few seconds ( like 10s ) of simulation
        it = iter(dataset)
        avg_act_ex = tf.zeros(self.cell.n_exc)
        for time in range(time_length):
            trial, _ = next(it)
            v, z = self(trial)
            avg_act_ex_t = tf.reduce_mean(z, axis=(0,1))[:self.cell.n_exc] # (100)
            avg_act_ex += avg_act_ex_t
        avg_act_ex /= time_length
        mean_act = tf.reduce_mean(avg_act_ex)
        print(f'Excitatory mean avg act : {mean_act * 1000} Hz')

        cond = tf.logical_and(avg_act_ex >= 0.011, avg_act_ex <= 0.013) # (1,100)
        cond = tf.logical_and(cond, self.cell.pool_cond[:self.cell.n_exc])
        assert_cn = tf.debugging.assert_none_equal(tf.reduce_sum(tf.cast(cond, tf.int32)), 0, message='No neuron suitable, reroll')
        eligible = tf.cast(tf.where(cond), tf.int32)[0]
        self.cn.assign(eligible[0])
        return z, self.cell.pool_cond ,v, self.cell.Vthresh, self.cn.value()




class Leg_fit(keras.Model):
    """Custom model.fit for (Legenstein and al., 2008) learning rule"""

    def __init__(self, model):
        super(Leg_fit, self).__init__()
        self.model = model
        z_pre_init = tf.cast( tf.random.uniform((1,8,self.model.batch_size)) < 0.010 , tf.float32 )
        self.z_cn_pre = tf.Variable(z_pre_init, trainable=False)


    def update_z_pre(self, new_z):
        old_z_cn = self.z_cn_pre.value()
        new_z_cn = new_z[:,:,self.model.cn.value()]
        new = tf.concat([old_z_cn[:, 1:, :], new_z_cn[:, None, :]] , axis=1)
        self.z_cn_pre.assign(new)




    def train_step(self, data):
        x, y = data
        v, z = self.model(x) # z : (1, 700, 100)
        vars = self.model.trainable_variables


        # compute the gradients ( grad = - delta w_ji = - d(t) * e_ji )
        print('\rGraph computing .     ', end='')
        dopa = new_leg_dopamine(self.model.cn.value(), z, self.z_cn_pre)

        print('\rGraph computing . .   ', end='')
        etrace = leg_etrace(self.model, z)
        self.update_z_pre(z)

        print('\rGraph computing . . . ', end='')
        grads = leg_gradients(self.model, dopa, etrace)
        print('\rGraph computed        ')

        # Apply the gradients
        self.optimizer.apply_gradients(zip(grads, vars))

        self.compiled_metrics.update_state(self.model.cn.value(), z)

        return {'CN average activity (Hz)' : self.metrics[0].result(),
                'average network activity (Hz)' : self.metrics[1].result()}







############## Callbacks #######################

class Tb_ActivityPlot(keras.callbacks.Callback):
    """Tensorboard callback to plot the conditioned neuron activity"""
    def __init__(self, tensorboard_dir):
        super(Tb_ActivityPlot, self).__init__()
        self.writer = tf.summary.create_file_writer(tensorboard_dir)


    def on_batch_end(self, batch, logs=None):

        def plot_to_image(figure):
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(figure)
            buf.seek(0)
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)
            return image

        def smooth_fn(data):
            len_k = 10
            kernel = tf.ones(shape=(len_k, 1, 1))/len_k
            data_s = tf.reshape( data, (1, int(data.shape[0]), 1) )
            data_s = tf.pad(data_s, [[0,0], [len_k-1, 0], [0, 0]], constant_values=data_s[0, 0, 0])
            data_s = tf.squeeze(tf.nn.conv1d(data_s, kernel, 1, padding='VALID'))
            return data_s


        time = np.arange(self.model.model.n_batch) * self.model.model.batch_size /1000
        cn_activity = self.model.metrics[2].result()
        avg_activity = self.model.metrics[3].result()

        cn_activity_smoothed = smooth_fn(cn_activity)
        avg_activity_smoothed = smooth_fn(avg_activity)

        figure = plt.figure()
        plt.plot(time, cn_activity_smoothed, color = 'blue', linewidth=1, label='CN activity')
        plt.plot(time, avg_activity_smoothed, color = 'orange', linestyle='dotted', linewidth=1, label='Average activity')
        plt.plot(time, cn_activity, color = 'blue', linewidth=1, alpha=0.2)
        plt.plot(time, avg_activity, color = 'orange', linestyle='dotted', linewidth=1, alpha=0.2)
        plt.legend()
        plt.title(f'Conditioned Neuron firing rate')
        plt.xlabel('Time (s)')
        plt.ylabel('Firing rate (Hz)')

        plot_image = plot_to_image(figure)

        with self.writer.as_default():
            tf.summary.image(f'Activity plot', plot_image, step=batch)


class Tb_AvgWeightsPlot(keras.callbacks.Callback):
    """Tensorboard callback to plot the reinforced neuron's synapses weight and avg weights (excitatory neurons only)"""

    def __init__(self, tensorboard_dir, n_batch):
        super(Tb_AvgWeightsPlot, self).__init__()
        self.writer = tf.summary.create_file_writer(tensorboard_dir)
        self.w_avg_buffer_EE = tf.Variable(tf.zeros(n_batch), trainable=False)
        self.w_avg_buffer_EI = tf.Variable(tf.zeros(n_batch), trainable=False)
        self.w_tocn_buffer = tf.Variable(tf.zeros(n_batch), trainable=False)
        self.w_lownoise_buffer = tf.Variable(tf.zeros(n_batch), trainable=False)
        self.w_highnoise_buffer = tf.Variable(tf.zeros(n_batch), trainable=False)
        self.batch_idx = tf.Variable(0, trainable=False)


    def on_batch_end(self, batch, logs=None):

        def plot_to_image(figure):
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(figure)
            buf.seek(0)
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)
            return image

        n_exc = self.model.model.cell.n_exc
        cn_idx = self.model.model.cn.value()
        connection =  tf.multiply( tf.cast(self.model.model.cell.connectivity_mask_EE, tf.float32),
                                   1-tf.cast(self.model.model.cell.disconnect_mask[:n_exc, :n_exc], tf.float32)) # (80, 80)

        conn_EI = tf.multiply( tf.cast(self.model.model.cell.connectivity_mask_EI, tf.float32),
                               1 - tf.cast(self.model.model.cell.disconnect_mask[:n_exc, n_exc:], tf.float32)) # (80, 20)

        pool_idx_E = tf.cast(tf.squeeze(tf.where(self.model.model.cell.pool_cond[:n_exc])), tf.int32)
        unpool_idx_E = tf.cast(tf.squeeze(tf.where(tf.cast(1 - tf.cast(self.model.model.cell.pool_cond[:n_exc], tf.int32), tf.bool))), tf.int32)

        n_tocn = tf.reduce_sum(connection[:, cn_idx])
        n_lownoise = tf.reduce_sum(gather_cols(connection, pool_idx_E))
        n_highnoise = tf.reduce_sum(gather_cols(connection, unpool_idx_E))
        n_rec_EE = tf.reduce_sum(connection[:, :cn_idx]) + tf.reduce_sum(connection[:, cn_idx+1:])
        n_rec_EI = tf.reduce_sum(conn_EI)

        # get the weights at the end of the batch

        w_rec_batch = self.model.model.cell.recurrent_weights_EE.value() # (80,80)
        #w_rec_batch = tf.clip_by_value(w_rec_batch, 0., self.model.model.cell.wmax)
        w_rec_batch = tf.where(tf.cast(connection, tf.bool), w_rec_batch, tf.zeros_like(w_rec_batch))  #  (80,80)

        w_rec_EI = self.model.model.cell.recurrent_weights_EI.value()
        w_rec_EI = tf.clip_by_value(w_rec_EI, 0., self.model.model.cell.wmax)
        w_rec_EI = tf.where( tf.cast(conn_EI, tf.bool), w_rec_EI, tf.zeros_like(w_rec_EI)) # (80, 20)

        # get the weights of the synapses to the cn
        w_rec_tocn = tf.reduce_sum(w_rec_batch[:,cn_idx]) / n_tocn


        # get the low and high noise neurons synapses
        w_rec_lownoise = tf.reduce_sum(gather_cols(w_rec_batch, pool_idx_E)) / n_lownoise
        w_rec_highnoise = tf.reduce_sum(gather_cols(w_rec_batch, unpool_idx_E)) / n_highnoise

        # get the all the other excitatory weights
        w_rec_avg_EE =   (tf.reduce_sum(w_rec_batch[:, :cn_idx]) + tf.reduce_sum(w_rec_batch[:, cn_idx+1:])) / n_rec_EE

        # get the EI weights
        w_rec_avg_EI = tf.reduce_sum(w_rec_EI) / n_rec_EI

        # assign the new variables to the buffer
        self.w_tocn_buffer.assign_add(  tf.one_hot( indices=self.batch_idx,
                                                    depth=self.model.model.n_batch,
                                                    on_value=w_rec_tocn )   )

        self.w_avg_buffer_EE.assign_add(  tf.one_hot( indices=self.batch_idx,
                                                    depth=self.model.model.n_batch,
                                                    on_value=w_rec_avg_EE )   )

        self.w_avg_buffer_EI.assign_add(  tf.one_hot( indices=self.batch_idx,
                                                    depth=self.model.model.n_batch,
                                                    on_value=w_rec_avg_EI )   )

        self.w_lownoise_buffer.assign_add(  tf.one_hot( indices=self.batch_idx,
                                                    depth=self.model.model.n_batch,
                                                    on_value=w_rec_lownoise )   )

        self.w_highnoise_buffer.assign_add(  tf.one_hot( indices=self.batch_idx,
                                                    depth=self.model.model.n_batch,
                                                    on_value=w_rec_highnoise )   )



        self.batch_idx.assign_add(1)

        time = np.arange(self.model.model.n_batch)*(self.model.model.batch_size)/1000
        w_tocn_plot = self.w_tocn_buffer.value()/self.model.model.cell.wmax
        w_rec_plot_EE = self.w_avg_buffer_EE.value()/self.model.model.cell.wmax
        w_rec_plot_EI = self.w_avg_buffer_EI.value()/self.model.model.cell.wmax
        w_rec_plot_lownoise = self.w_lownoise_buffer.value()/self.model.model.cell.wmax
        w_rec_plot_highnoise = self.w_highnoise_buffer.value()/self.model.model.cell.wmax



        figure = plt.figure()
        plt.plot(time, w_tocn_plot, color = 'blue', linewidth=1, label=f'Excitatory synapses to CN {n_tocn}')
        plt.plot(time,
                 w_rec_plot_EE,
                 color = 'black',
                 linestyle='dotted',
                 linewidth=1.5,
                 label=f'EE synapses {n_rec_EE}')
        plt.plot(time,
                 w_rec_plot_EI,
                 color = 'orange',
                 linestyle='dotted',
                 linewidth=1,
                 label=f'EI synapses {n_rec_EI}')
        plt.plot(time,
                 w_rec_plot_lownoise,
                 color = 'red',
                 linestyle='dotted',
                 linewidth=1,
                 label=f'Low noise synapses {n_lownoise}')
        plt.plot(time,
                 w_rec_plot_highnoise,
                 color = 'green',
                 linestyle='dotted',
                 linewidth=1,
                 label=f'High noise synapses {n_highnoise}')
        plt.legend()
        plt.title(f'Excitatory weights evolution')
        plt.xlabel('Time (s)')
        plt.ylabel('Average weight (w/wmax)')
        plt.ylim((0., 1.1))

        plot_image = plot_to_image(figure)

        with self.writer.as_default():
            tf.summary.image(f'Weight plot', plot_image, step=batch)





# todo when the model is stable enough
class Tb_SpikePlot(keras.callbacks.Callback):
    """Tensorboard callback to plot the spikes of neurons in the network"""

    def __init__(self, tensorboard_dir):
        super(Tb_SpikePlot, self).__init__()
        self.writer = tf.summary.create_file_writer(tensorboard_dir)


    def on_batch_end(self, batch, logs=None):

        def plot_to_image(figure):
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(figure)
            buf.seek(0)
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)
            return image

        def process_z(z):
            z = tf.squeeze(z)
            seq_len = z.shape[0]
            n_neurons = z.shape[1]
            events = [[] for neuron in range(n_neurons)]
            for time in range(seq_len):
                for neuron in range(n_neurons):
                    if z[time, neuron]==True :
                        events[neuron].append(time)

            return events

        time = np.arange(self.model.model.n_batch)*self.model.model.batch_size/1000 # time in ms
        cn_activity = self.model.metrics[2].result()
        avg_activity = self.model.metrics[3].result()

        figure = plt.figure(figsize=(16,10))
        plt.plot(time, cn_activity, color = 'blue', linewidth=1, label='CN activity')
        plt.plot(time, avg_activity, color = 'orange', linestyle='dotted', linewidth=1, label='Average activity')
        plt.legend()
        plt.title(f'Conditioned Neuron firing rate')
        plt.xlabel('Time (s)')
        plt.ylabel('Firing rate (Hz)')

        plot_image = plot_to_image(figure)

        with self.writer.as_default():
            tf.summary.image(f'Activity plot', plot_image, step=batch)
