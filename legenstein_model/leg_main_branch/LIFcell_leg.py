import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import io





def make_mat_EI(xs, n_exc, n_inh):
    mat_EE = tf.ones((n_exc, n_exc), dtype=tf.float32) * xs[0]
    mat_EI = tf.ones((n_exc, n_inh), dtype=tf.float32) * xs[1]
    mat_IE = tf.ones((n_inh, n_exc), dtype=tf.float32) * xs[2]
    mat_II = tf.ones((n_inh, n_inh), dtype=tf.float32) * xs[3]
    mat_E = tf.concat([mat_EE, mat_EI], axis=1)
    mat_I = tf.concat([mat_IE, mat_II], axis=1)
    mat = tf.concat([mat_E, mat_I], axis=0)
    return mat

def connection_mask(n_exc, n_inh, pool_cond, connectivity, con_factor):
    np.random.seed(0)
    _connect_EE, _connect_EI, _connect_IE, _connect_II = connectivity
    nump_conn_mat_EE = np.cast['float32'](np.random.uniform(size=(n_exc, n_exc)) < _connect_EE)
    nump_conn_mat_EI = np.cast['float32'](np.random.uniform(size=(n_exc, n_inh)) < _connect_EI)
    nump_conn_mat_IE = np.cast['float32'](np.random.uniform(size=(n_inh, n_exc)) < _connect_IE)
    nump_conn_mat_II = np.cast['float32'](np.random.uniform(size=(n_inh, n_inh)) < _connect_II)
    nump_connectivity_mat_I = np.concatenate((nump_conn_mat_IE, nump_conn_mat_II), axis=1)

    unpool_cond = tf.cast(  1 - tf.cast(pool_cond, tf.int32), tf.bool  ) # select the high noise neurons to reduce their connectivty
    pool_idx = tf.squeeze(tf.where(unpool_cond))
    pool_idx_E = tf.squeeze(tf.where(unpool_cond[:n_exc]))
    pool_idx_I = tf.squeeze(tf.where(unpool_cond[n_exc: ]))

    # Prune connections coming to high noise neurons by a 0.4 factor (con_factor)
    con_factor = 1 - con_factor
    nump_conn_mat_EE[:, pool_idx_E] *= np.random.uniform(size=(n_exc, len(pool_idx_E)))
    nump_conn_mat_EI[:, pool_idx_I] *= np.random.uniform(size=(n_exc, len(pool_idx_I)))
    nump_connectivity_mat_I[:, pool_idx] *= np.random.uniform(size=(n_inh, len(pool_idx)))
    nump_conn_mat_EE = nump_conn_mat_EE > con_factor
    nump_conn_mat_EI = nump_conn_mat_EI > con_factor
    nump_connectivity_mat_I = nump_connectivity_mat_I > con_factor

    return nump_conn_mat_EE, nump_conn_mat_EI, nump_connectivity_mat_I




####### Layer definition #########
class CellConstraint(keras.constraints.Constraint):
    def __init__(self, connectivity_mask_EE, connectivity_mask_EI, connectivity_mask_I, disconnect_mask, wmax):
        self.connectivity_mask_EE = connectivity_mask_EE
        self.connectivity_mask_EI = connectivity_mask_EI
        self.connectivity_mask_I = connectivity_mask_I
        self.disconnect_mask = disconnect_mask
        self.wmax = wmax

    def __call__(self, w_EE, w_EI, w_inh):
        w_EE = tf.clip_by_value(w_EE, 0., self.wmax)
        w_EE = tf.where(self.connectivity_mask_EE, w_EE, tf.zeros_like(w_EE))
        w_EI = tf.where(self.connectivity_mask_EI, w_EI, tf.zeros_like(w_EI))
        w_inh = tf.where(self.connectivity_mask_I, w_inh, tf.zeros_like(w_inh))
        w_E = tf.concat([w_EE, w_EI], axis=1)
        w = tf.concat([w_E, w_inh], axis=0)
        w = tf.where(self.disconnect_mask, tf.zeros_like(w), w)
        return w



class LIFCell(layers.Layer):
    """RSNN model for the Experiemnt (LIF)"""
                                                                                    # 0.005
    def __init__(self, units, connectivity=[0.02, 0.02, 0.024, 0.16], ei_ratio=0.2, w_init_exc=10.7, w_init_inh=211.6, noise_factor=0.2,  tau_syn=5., dt=1, n_refractory=5, con_factor=0.4):
        super().__init__()
        self.units = units
        self.n_exc = int(self.units * (1-ei_ratio))
        self.n_inh = self.units - self.n_exc
        self.connectivity = connectivity
        self.con_factor = con_factor
        self.noise_factor = noise_factor
        self.w_init_exc = w_init_exc
        self.w_init_inh = w_init_inh

        self._dt = float(dt)
        self._decay_syn =tf.exp(-dt/tau_syn)
        self._n_refractory = n_refractory

        self.recurrent_weights_EE = tf.Variable(tf.ones((self.n_exc, self.n_exc)) * self.w_init_exc, name="W_rec_EE")
        self.recurrent_weights_EI = tf.Variable(tf.ones((self.n_exc, self.n_inh)) * self.w_init_exc, trainable=False, name="W_rec_EI")
        self.recurrent_weight_inh = tf.Variable(tf.ones((self.n_inh, self.units)) * self.w_init_inh, trainable=False, name="W_rec_inh" )

        self.wmax = tf.constant(2. * self.w_init_exc, dtype = tf.float32)

        self.disconnect_mask = tf.cast(np.diag(np.ones(self.units, dtype=np.bool)), tf.bool)
        h_n_exc = int(self.n_exc / 2)
        h_n_inh = int(self.n_inh / 2)
        tf.random.set_seed(1234)
        rand_ex = tf.random.shuffle(tf.concat([tf.ones(h_n_exc), tf.zeros(self.n_exc - h_n_exc)], axis=0), seed=0)
        rand_inh = tf.random.shuffle(tf.concat([tf.ones(h_n_inh), tf.zeros(self.n_inh - h_n_inh)], axis=0), seed=0)
        self.pool_cond = tf.cast(tf.concat([rand_ex, rand_inh], axis=0), tf.bool) # idx of the chosen low noise neurons ( cn in it )

        conn_mat_EE, conn_mat_EI, conn_mat_I = connection_mask(self.n_exc, self.n_inh, self.pool_cond, self.connectivity, self.con_factor)
        self.connectivity_mask_EE = tf.cast(conn_mat_EE, tf.bool)
        self.connectivity_mask_EI = tf.cast(conn_mat_EI, tf.bool)
        self.connectivity_mask_I = tf.cast(conn_mat_I, tf.bool)

        # Constraint
        self.constraint = CellConstraint(self.connectivity_mask_EE,
                                         self.connectivity_mask_EI,
                                         self.connectivity_mask_I,
                                         self.disconnect_mask,
                                         self.wmax)


        Us = [0.5, 0.25, 0.05, 0.32]
        Ds = [1.1, 0.7, 0.125, 0.144]
        Fs = [0.02, 0.02, 1.2, 0.06]

        self.U = make_mat_EI( Us, self.n_exc, self.n_inh )
        self.D = make_mat_EI( Ds, self.n_exc, self.n_inh )
        self.F = make_mat_EI( Fs, self.n_exc, self.n_inh )

        self.Vrest = -70
        self.Ee = 0.
        self.Ei = -75.
        self.Rm = 100
        self.Cm = 0.3
        self.Vthresh = -59.

        # I_noise constants
        self.ge_0 = 0.012
        self.gi_0 = 0.057
        self.tau_e = 2.7
        self.tau_i = 10.5
        self.sigma_e = 0.003
        self.sigma_i = 0.0066
        self.Ae = tf.constant( self.sigma_e *  tf.sqrt( (1-tf.exp(-2 * self._dt / (self.tau_e))) / 2 ) )
        self.Ai = tf.constant( self.sigma_i *  tf.sqrt( (1-tf.exp(-2 * self._dt / (self.tau_i))) / 2 ) )

        #                  voltage, refractory, previous spikes, previous u, previous R, previous g, ge, gi
        self.state_size = (units, units, units, tf.TensorShape([units,units]), units, units)


    def zero_state(self, dtype=tf.float32):
        v_0 = tf.random.uniform((1, self.units)) * (self.Vthresh - self.Vrest) + self.Vrest
        delta_0 = tf.ones((1, self.units), dtype) * (self._n_refractory+1) # interspike interval
        z_0 = tf.zeros((1, self.units), dtype)

        ###########################################
        # if short term synaptic behavior
        # u_0 = self.U[None, :, :]
        # R_0 = tf.ones((1, self.units, self.units), dtype)
        ############################################

        g_0 = tf.zeros((1, self.units, self.units), dtype)
        #ge_0 = tf.ones((1, self.units), dtype) * self.ge_0 # 0.017, 0.007
        ge_0 = tf.random.uniform((1, self.units)) * (0.017 - 0.007) + 0.007
        #gi_0 = tf.ones((1, self.units), dtype) * self.gi_0 # 0.067, 0.047
        gi_0 = tf.random.uniform((1, self.units)) * (0.067 - 0.047) + 0.047

        return v_0, delta_0, z_0, g_0, ge_0, gi_0

    def compute_UR(self,old_U, old_R, old_delta):
        deltaF_mat = tf.stack([tf.exp(tf.divide(tf.squeeze(old_delta), -F_j)) for F_j in tf.unstack(self.F, axis=0)], axis=0)
        deltaD_mat = tf.stack([tf.exp(tf.divide(tf.squeeze(old_delta), -D_j)) for D_j in tf.unstack(self.D, axis=0)], axis=0)
        new_U = self.U + tf.multiply(tf.multiply(tf.squeeze(old_U), (1-self.U)), deltaF_mat)
        new_R = 1 + tf.multiply(  tf.squeeze(old_R) - tf.multiply(tf.squeeze(old_U),tf.squeeze(old_R)) -1 , deltaD_mat)
        return new_U[None, :, :], new_R[None, :, :]

    def compute_A(self, new_U, new_R, w):
        UR = tf.multiply(tf.squeeze(new_U), tf.squeeze(new_R))
        return tf.multiply(w, UR) # (100, 100)

    def compute_g(self, A, old_z, old_g):
        g = old_g * self._decay_syn
        stacked_z = tf.stack([old_z for i in range(self.units)], axis=1)
        new_g = g + tf.multiply(A, tf.transpose(stacked_z, perm = [0, 2, 1]))
        return new_g # (100, 100)

    def compute_gegi(self, old_ge, old_gi, ne, ni):
        noise_reg = tf.ones(self.units) - tf.cast(self.pool_cond, tf.float32) * (1-self.noise_factor)
        new_ge = self.ge_0*noise_reg + tf.multiply(old_ge - self.ge_0*noise_reg, tf.exp(- self._dt / self.tau_e)) + self.Ae * ne * noise_reg
        new_gi = self.gi_0*noise_reg + tf.multiply(old_gi - self.gi_0*noise_reg, tf.exp(- self._dt / self.tau_i)) + self.Ai * ni * noise_reg
        return new_ge, new_gi

    def compute_v(self, old_v, new_g, ge, gi) :
        E = tf.concat( [tf.ones(self.n_exc) * self.Ee, tf.ones(self.n_inh) * self.Ei], axis=0 )
        denominator = 1/self.Rm + tf.reduce_sum(new_g, axis=1) + ge + gi
        tau = self.Cm / denominator
        mult = tf.matmul(E[None, :], new_g)[0, :, :]
        v_inf = self.Vrest / self.Rm + mult + self.Ee * ge + self.Ei * gi
        v_inf = tf.divide(v_inf, denominator)
        new_v = v_inf + tf.multiply((old_v - v_inf), tf.exp(-self._dt / tau))
        return new_v


    def call(self, inputs, state):
        old_v = state[0]
        old_delta = state[1]
        old_z = state[2]

        #############################
        # old_U = state[3]
        # old_R = state[4]
        ##############################@


        old_g = state[3]
        old_ge = state[4]
        old_gi = state[5]

        ne = inputs[:,0,:]
        ni = inputs[:,1,:]

        corrected_w = self.constraint(self.recurrent_weights_EE, self.recurrent_weights_EI, self.recurrent_weight_inh)

        ###########################################################
        # new_U, new_R = self.compute_UR(old_U, old_R, old_delta)
        # print(f'U,R : {new_U}, {new_R}')
        # A = self.compute_A(new_U, new_R, corrected_w)
        # print(f'A : {A}')
        ###################################################

        new_g = self.compute_g(corrected_w, old_z, old_g)
        #print(f'g : {new_g}')
        new_ge, new_gi = self.compute_gegi(old_ge, old_gi, ne, ni)
        #print(f'ge gi : {new_ge}, {new_gi}')
        new_v = tf.where(tf.cast(old_z, tf.bool), tf.ones_like(old_v) * self.Vrest, old_v)
        new_v = self.compute_v(new_v, new_g, new_ge, new_gi)
        #print(f'v : {new_v}')
        new_v = tf.where(tf.greater(old_delta, self._n_refractory), new_v, tf.ones_like(new_v) * self.Vrest)
        #print(f'v refract : {new_v}')
        new_z = tf.cast(new_v>=self.Vthresh, tf.float32)
        #print(f'spikes : {new_z}')
        new_delta = tf.multiply(old_delta + (1 - new_z), 1 - new_z)
        #print(f'new_delta : {new_delta}')

        #################################################
        # new_state = (new_v, new_delta, new_z, new_U, new_R, new_g, new_ge, new_gi)
        ##################################################

        new_state = (new_v, new_delta, new_z, new_g, new_ge, new_gi)
        output = (new_v, new_z)
        return output, new_state
