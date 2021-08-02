import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np

import models_leg as m


######## Constants #######
n_batch = 100 # length of the simulation
n_recurrent= 500
epochs = 1
batch_size = 1000 # time lapse between gradient applying


percent_con = 0.1
w_ratio = 0.7
con_factor = 0.7
noise_factor = 0.5

############## Initialize ##########################
conn, w = tf.multiply([1., 1., 1.2, 0.8], percent_con), tf.multiply([10.7e-3, 211.6e-3], w_ratio)
seq_len = n_batch * batch_size # (ms)
dataset = m.create_data_set_OU(seq_len, n_recurrent, batch_size=batch_size)
exp_model = m.Exp_model(n_recurrent, n_batch, batch_size, connectivity=conn, w=w, noise_factor=noise_factor, con_factor=con_factor )
print("Exp model created")




####### pre training behavior check ###############
def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

writer = tf.summary.create_file_writer('logs_g100_tau/image')
writer2 = tf.summary.create_file_writer('logs_g100_tau/image')

z, cond, v, thr, cn = exp_model.cn_search(dataset, 10)
print("CN search finished")

def process_z(z):
    z = tf.squeeze(z)
    seq_len = z.shape[0]
    n_neurons = z.shape[1]
    events = [[] for neuron in range(n_neurons)]
    for time in range(seq_len):
        for neuron in range(n_neurons):
            if z[time, neuron]==1. :
                events[neuron].append(time)

    return events

print("Processing z")
z_proc = process_z(z)
print("z processed")
uncond = tf.cast(  1 - tf.cast(cond, tf.int32) , tf.bool) # poolcond of highnoise neurons
pool_idx = tf.squeeze(tf.where(cond))
unpool_idx = tf.squeeze(tf.where(uncond))
#                  #lownoise  highnoise
idxs = tf.concat([pool_idx, unpool_idx], axis=0)
colors = [ "red" if i<=len(cond)//2 else "blue" for i in range(len(cond)) ]
z_plot = [ z_proc[i] for i in idxs ]


z_low = [ z[:,:,i] for i in pool_idx]
z_high = [ z[:,:,i] for i in unpool_idx]
havg = tf.reduce_mean(z_high) *1000
lavg = tf.reduce_mean(z_low) *1000
print(f"highnoise avg = {havg}")
print(f"lownoise avg = {lavg}")

figure1 = plt.figure()
plt.eventplot(z_plot, color=colors, linewidth=1)
plt.legend(["low noise", "high noise"], labelcolor=["red", "blue"])
plt.xlabel("Time (ms)")
plt.ylabel("Neuron")
plt.title("Network activity before training")
plot_image1 = plot_to_image(figure1)

avg = tf.squeeze(tf.reduce_mean(z, axis=(0,1))) * 1000

figure2 = plt.figure()
plt.hist(avg.numpy(), bins=60)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Number of neurons')
plt.title('Activity distribution before training')


plot_image2 = plot_to_image(figure2)

with writer.as_default():
    tf.summary.image(f'spike plot', plot_image1, step=1)

with writer2.as_default():
    tf.summary.image(f'activity distribution', plot_image2, step=1)


####### Tensorboard callback #########
tb_callbacks = tf.keras.callbacks.TensorBoard(log_dir = 'logs_g100_tau',
                                                  histogram_freq=0,
                                                  write_graph=False,
                                                  update_freq='batch')

tb_act_plot = m.Tb_ActivityPlot('logs_g100_tau/image')

tb_w_plot = m.Tb_AvgWeightsPlot('logs_g100_tau/image', n_batch)





######### Train ####################
leg = m.Leg_fit(exp_model)
cn_activity = m.Activity_metric(cn_bool=True, name='CN activity (Hz)')
activity = m.Activity_metric(name='avg activity (Hz)')
cn_buffer = m.Buffer_metric(n_batch, cn_bool=True, name='CN buffer')
avg_buffer = m.Buffer_metric(n_batch, name='Avg buffer')
opt = keras.optimizers.SGD(lr=1.)
leg.compile(optimizer = opt, metrics=[cn_activity, activity, cn_buffer, avg_buffer])
print('Model ready for training')
leg.fit(dataset, epochs=epochs, callbacks=[tb_callbacks, tb_w_plot, tb_act_plot])
print('Model trained')






######### post training behavior plots ##########@
it = iter(dataset)
trial, _ = next(it)
_, z_post = exp_model(trial)


print("Processing z")
z_proc = process_z(z_post)
print("z processed")
uncond = tf.cast(  1 - tf.cast(cond, tf.int32) , tf.bool) # poolcond of highnoise neurons
pool_idx = tf.squeeze(tf.where(cond))
unpool_idx = tf.squeeze(tf.where(uncond))
#                  #lownoise  highnoise
idxs = tf.concat([pool_idx, unpool_idx], axis=0)
colors = [ "red" if i<=len(cond)//2 else "blue" for i in range(len(cond)) ]
z_plot = [ z_proc[i] for i in idxs ]


z_low = [ z_post[:,:,i] for i in pool_idx]
z_high = [ z_post[:,:,i] for i in unpool_idx]
havg = tf.reduce_mean(z_high) *1000
lavg = tf.reduce_mean(z_low) *1000
print(f"highnoise avg = {havg}")
print(f"lownoise avg = {lavg}")

figure3 = plt.figure()
plt.eventplot(z_plot, color=colors, linewidth=1)
plt.legend(["low noise", "high noise"], labelcolor=["red", "blue"])
plt.xlabel("Time (ms)")
plt.ylabel("Neuron")
plt.title("Network activity after training")
plot_image3 = plot_to_image(figure3)

avg = tf.squeeze(tf.reduce_mean(z_post, axis=(0,1))) * 1000

figure4 = plt.figure()
plt.hist(avg.numpy(), bins=60)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Number of neurons')
plt.title('Activity distribution after training')


plot_image4 = plot_to_image(figure4)

with writer.as_default():
    tf.summary.image(f'spike plot', plot_image3, step=2)

with writer2.as_default():
    tf.summary.image(f'activity distribution', plot_image4, step=2)
