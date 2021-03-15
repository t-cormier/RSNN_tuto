import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

import models as m






#with tf.device("GPU") :


######## Hyperparameters ########

cn_target_rate = 0.001
num_sim = 10


######## Constants #######
time_sec = 1000
n_input=20
n_recurrent=100
cn_idx = 10
epochs = 1
batch_size = 700 # time lapse between gradient applying (and length of eligibility trace)


####### search loop ############
for idx in range(num_sim) :

    print(f'Sim number {idx}')

    ######## Init experiment ###########
    seq_len = 1000 * time_sec
    exp_model = m.Exp_model(n_recurrent, n_input, seq_len, batch_size)
    dataset = m.create_data_set(seq_len, n_input, batch_size=batch_size)
    print('Dataset created')



    ####### Tensorboard callback #########
    tb_callbacks = tf.keras.callbacks.TensorBoard(log_dir = f'baseline_logs/logs_{idx}',
                                                      histogram_freq=0,
                                                      write_graph=False,
                                                      update_freq='batch')


    ######### Train #####################
    leg = m.Leg_fit(exp_model, cn_idx, target_rate=cn_target_rate)
    cn_activity = m.Activity_metric(cn_idx, name='CN activity')
    activity = m.Activity_metric(name='avg activity')
    opt = keras.optimizers.Adam(lr=1e-3)
    leg.compile(optimizer = opt, metrics=[cn_activity, activity])
    print('Model ready for training')
    leg.fit(dataset, epochs=epochs, callbacks=[tb_callbacks])
    print('Model trained')



    ######### save the exp_model ########
    exp_model.save_weights(f'baseline_models_w/mod_{idx}/')
