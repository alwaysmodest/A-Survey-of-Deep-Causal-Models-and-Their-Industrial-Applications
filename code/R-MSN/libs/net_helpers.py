# -*- coding: utf-8 -*-
"""
[20180320 Treatment Effects with RNNs] net_helpers
Created on 22/3/2018 2:23 PM

@author: limsi
"""

import os
import logging
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

""" 
General
"""

def create_folder_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def reshape_for_sklearn(ip):
    return ip.reshape([-1, ip.shape[-1]])

def linear(input_, output_size, scope=None, bias_start=0.0, with_w=False):

    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear", reuse=tf.AUTO_REUSE) as cur_scope:
        matrix = tf.get_variable("Matrix", [shape[-1], output_size], tf.float32,
                                 tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))

        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def randomise_minibatch_index(Y, minibatch_size):
    batch_num, target_num = Y.shape

    # Randomise rows
    rdm_idx = [i for i in range(batch_num)]
    np.random.shuffle(rdm_idx)

    max_idx = len(rdm_idx)
    num_minibatches = int(max_idx / minibatch_size)
    minibatch_numbers = [j for j in range(num_minibatches)]

    tmp = []
    for count in range(len(minibatch_numbers)):
        j = minibatch_numbers[count]

        ptr = j * minibatch_size
        end_idx = min(minibatch_size + ptr, max_idx)
        minibatch_idx = rdm_idx[ptr:end_idx]

        tmp.append(minibatch_idx)
    return tmp

def get_optimization_graph(loss, learning_rate, max_global_norm, global_step,
                           optimisation_function=tf.train.AdamOptimizer):
    # Optimisation step

    optimizer = optimisation_function(learning_rate)

    # Clip gradients to prevent them from blowing up
    trainables = tf.trainable_variables()
    grads = tf.gradients(loss, trainables)
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
    grad_var_pairs = zip(grads, trainables)
    global_step = global_step

    minimize = optimizer.apply_gradients(grad_var_pairs,
                                         global_step=global_step)

    return minimize


def calc_binary_cross_entropy(probs, outputs, weights=1):

    return -tf.reduce_mean((outputs * tf.log(probs +1e-8)
             + (1-outputs)* tf.log(1-probs +1e-8)) * weights)

def last_relevant_time_slice(output, sequence_length):

    shape = output.get_shape()

    if len(shape) == 3:
        # Get last valid time per batch for batch x time step x feature tensor

        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])

        # Make sure that we get the final index, which is 1 less than sequence length
        index = tf.range(0, batch_size) * max_length + tf.subtract(sequence_length, 1)  # length should be batchsize as well
        flat = tf.reshape(output, [-1, out_size])  # flattens the index into batch * batchsize + length
        relevant = tf.gather(flat, index)

    elif len(shape) == 2:

        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]

        # Make sure that we get the final index, which is 1 less than sequence length
        index = tf.range(0, batch_size) * max_length + tf.subtract(sequence_length,
                                                                   1)  # length should be batchsize as well
        flat = tf.reshape(output, [-1])  # flattens the index into batch * batchsize + length
        relevant = tf.gather(flat, index)

    else:
        raise ValueError("Illegal shape type {0}".format(shape))
    return relevant


def randomise_minibatch_index(num_patients, minibatch_size):
    # Randomise rows
    rdm_idx = [i for i in range(num_patients)]
    np.random.shuffle(rdm_idx)

    max_idx = len(rdm_idx)
    num_minibatches = int(max_idx / minibatch_size)
    minibatch_numbers = [j for j in range(num_minibatches)]

    tmp = []
    for count in range(len(minibatch_numbers)):
        j = minibatch_numbers[count]

        ptr = j * minibatch_size
        end_idx = min(minibatch_size + ptr, max_idx)
        minibatch_idx = rdm_idx[ptr:end_idx]

        tmp.append(minibatch_idx)
    return tmp



""" 
Serialisation
"""

def save_network(tf_session, model_folder, cp_name, optimisation_summary):

    # Save model
    saver = tf.train.Saver(max_to_keep=100000)
    save_path = saver.save(tf_session, os.path.join(model_folder, "{0}.ckpt".format(cp_name)))
    logging.info("Model saved to: {0}".format(save_path))
    # Save opt summary
    opt_summary_path = os.path.join(model_folder, "{0}_optsummary.csv".format(cp_name))
    optimisation_summary.to_csv(opt_summary_path)
    logging.info("Optimisation Summary saved in file: {0}".format(opt_summary_path))


def load_network(tf_session, model_folder, cp_name):

    # Load model proper
    load_path = os.path.join(model_folder, "{0}.ckpt".format(cp_name))

    logging.info('Loading model from {0}'.format(load_path))

    print_weights_in_checkpoint(model_folder, cp_name)

    initial_vars = set([v.name for v in tf.get_default_graph().as_graph_def().node])
    saver = tf.train.Saver()
    saver.restore(tf_session, load_path)
    all_vars = set([v.name for v in tf.get_default_graph().as_graph_def().node])

    logging.info('Restored {0}'.format(','.join(initial_vars.difference(all_vars))))
    logging.info('Existing {0}'.format(','.join(all_vars.difference(initial_vars))))
    logging.info('All {0}'.format(','.join(all_vars)))

    # Load optimisation summary
    opt_summary_path = os.path.join(model_folder, "{0}_optsummary.csv".format(cp_name))
    logging.info('Loading optimisation summary from {0}'.format(load_path))
    optimisation_summary = pd.read_csv(opt_summary_path, index_col=0)
    logging.info("Done.")

    return optimisation_summary

def add_hyperparameter_results(optimisation_summary,
                               model_folder,
                               net_name,
                               serialisation_name,
                               validation_loss_col=None):

    srs = optimisation_summary.copy()

    if validation_loss_col != None:
        srs = srs[validation_loss_col]

    srs = srs.dropna()

    if srs.empty:
        return

    min_loss = srs.min()

    best_idx = list(srs[srs == min_loss].index)[0]

    df = load_hyperparameter_results(model_folder,
                                       net_name)

    df[serialisation_name] = pd.Series({'best_epoch': best_idx,
                                        'validation_loss': min_loss})
    save_name = os.path.join(model_folder, net_name+".csv")
    df.to_csv(save_name)


def load_hyperparameter_results(model_folder,
                               net_name):

    save_name = os.path.join(model_folder, net_name+".csv")

    if os.path.exists(save_name):

        return pd.read_csv(save_name, index_col=0)
    else:
        return pd.DataFrame()

def hyperparameter_result_exists(model_folder,
                                 net_name,
                                 serialisation_name):

    df = load_hyperparameter_results(model_folder, net_name)

    cols = set(df.columns)

    return serialisation_name in cols

def print_weights_in_checkpoint(model_folder, cp_name):

    load_path = os.path.join(model_folder, "{0}.ckpt".format(cp_name))

    print_tensors_in_checkpoint_file(file_name=load_path, tensor_name='',
                                     all_tensors=True, all_tensor_names=True)

# In[*]: main

if __name__ == "__main__":

    model_folder = r"D:\MODELS\treatment_effects\rnn_propensity_weighted_seq2seq_feedback_training_mode"
    checkpoint_name = r"treatment_effects_0.5_16_100_1024_0.0001_4.0_60_optimal"

    print_weights_in_checkpoint(model_folder, checkpoint_name)