from model import *
import os
import numpy as np
import glob
import argparse
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.optimizers import rmsprop, SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, TerminateOnNaN


def _split_output(yt_hat, t, y, y_scaler, x, index):
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].copy())
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].copy())
    g = yt_hat[:, 2].copy()

    if yt_hat.shape[1] > 3:
        eps = yt_hat[:, 3][0]
    else:
        eps = np.zeros_like(yt_hat[:, 2])

    y = y_scaler.inverse_transform(y.copy())

    return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x, 'index': index, 'eps': eps}


def donut(t, y_unscaled, x, regularization='orthogonal', output_dir='', 
          knob_loss=factual_loss, ratio=1., model='', val_split=0.2, batch_size=64, test_size = 0.1):
    verbose = 0
    y_scaler = StandardScaler().fit(y_unscaled)
    y = y_scaler.transform(y_unscaled)
    train_outputs = []
    test_outputs = []

    print("Donut")
    net = make_donut(x.shape[1], 0.01)

    metrics = [regression_loss, binary_classification_loss, treatment_accuracy, track_epsilon]

    if regularization == 'orthogonal':
        loss = make_orthogonal_loss(ratio=ratio, fn = net, global_input = x, net_loss=knob_loss)
    else:
        loss = knob_loss

    # for reporducing the experiments
    i = 0
    tf.random.set_random_seed(i)
    np.random.seed(i)
    train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=test_size, random_state=1)

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    t_train, t_test = t[train_index], t[test_index]

    yt_train = np.concatenate([y_train, t_train], 1)
        
    import time;
    start_time = time.time()
 
    sgd_callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=40, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                          min_delta=0., cooldown=0, min_lr=0)
    ]
    
    sgd_lr = 1e-5
    momentum = 0.9
    net.compile(optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True), loss=loss,
                metrics=metrics)
    net.fit(x_train, yt_train, callbacks=sgd_callbacks, 
            validation_split=val_split,
            epochs=300,
            batch_size=batch_size, verbose=verbose)

    elapsed_time = time.time() - start_time
    print("***************************** elapsed_time is: ", elapsed_time)

    yt_hat_test = net.predict(x_test)
    yt_hat_train = net.predict(x_train)

    test_outputs += [_split_output(yt_hat_test, t_test, y_test, y_scaler, x_test, test_index)]
    train_outputs += [_split_output(yt_hat_train, t_train, y_train, y_scaler, x_train, train_index)]
    K.clear_session()

    return test_outputs, train_outputs


