import tensorflow as tf
import keras.backend as K
from keras.engine.topology import Layer
from keras.metrics import binary_accuracy
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from keras.models import Model
from keras import regularizers


def binary_classification_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002
    losst = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    return losst


def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))

    return loss0 + loss1


def factual_loss(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)


def treatment_accuracy(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    return binary_accuracy(t_true, t_pred)


def track_epsilon(concat_true, concat_pred):
    epsilons = concat_pred[:, 3]
    return tf.abs(tf.reduce_mean(epsilons))


class EpsilonLayer(Layer):

    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 1],
                                       initializer='RandomNormal',
                                       trainable=True)
        super(EpsilonLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]


def make_orthogonal_reg_loss(ratio, fn, global_input, f_loss=factual_loss):
    def orthogonal_ATE_loss(concat_true, concat_pred):
        unreg_loss = f_loss(concat_true, concat_pred)
        
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        
        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]
        
        epsilons = concat_pred[:, 3]
        t_pred = (t_pred + 0.01) / 1.02

        #Perturbation
        y0_pert = y0_pred + epsilons * (t_true - t_pred)    
        
        #Treatment effect proxy with average treatment effect of current model fit
        global_pred = fn.predict(global_input)
        y0_pred_global = global_pred[:, 0].copy()
        y1_pred_global = global_pred[:, 1].copy()
        psi = (y1_pred_global - y0_pred_global).mean()
        
        #Orthogonal regularization
        orthogonal_regularization = tf.reduce_sum(tf.square(y_true - t_true * psi - y0_pert))
        
        # final
        loss = unreg_loss + ratio * orthogonal_regularization
        return loss

    return orthogonal_ATE_loss


def make_donut(input_dim, reg_l2):
    """
    Neural net predictive model. 
    :param input_dim:
    :param reg:
    :return:
    """

    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)

    # logistic regression
    t_predictions = Dense(units=1, activation='sigmoid')(inputs)

    # OUTCOME MODELS
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')

    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons, inputs])
    model = Model(inputs=inputs, outputs=concat_pred)

    return model

