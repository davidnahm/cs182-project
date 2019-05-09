import tensorflow as tf
import numpy as np

def mlp(input_placeholder, scope, out_size,
        hiddens = [128, 64], output_activation = None,
        activation = tf.tanh, reuse = False, flatten = True):
    with tf.variable_scope(scope, reuse = reuse):
        if flatten:
            x = tf.layers.flatten(input_placeholder)
        else:
            x = input_placeholder
        for hidden_size in hiddens:
            x = tf.layers.dense(x, hidden_size, activation = activation)
        x = tf.layers.dense(x, out_size, activation = output_activation)
    return x

def miniworld_preprocess(x, time = True):
    with tf.variable_scope("conv_preprocess"):
        original_shape = tf.shape(x)
        if time:
            new_shape = tf.concat([[original_shape[0] * original_shape[1]], original_shape[2:]], 0)
            x = tf.reshape(x, new_shape, name = "prepare_for_conv")
        x = tf.keras.layers.Conv2D(16, 5, 2, activation = tf.nn.relu)(x)
        x = tf.keras.layers.Conv2D(16, 5, 2, activation = tf.nn.relu)(x)

        # tensorflow doesn't seem to be able to infer shape by itself
        flattened_shape = tf.concat([tf.shape(x)[:1], [3264]], axis = 0)
        x = tf.reshape(x, flattened_shape, name = "flatten")
        if time:
            factor_shape = tf.concat([original_shape[:2], tf.shape(x)[1:]], axis = 0)
            x = tf.reshape(x, factor_shape, name = "reshape_to_time_and_batch")
        x = tf.layers.dense(x, 256, activation = tf.tanh)
    return x
