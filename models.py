import tensorflow as tf
import numpy as np

def mlp(input_placeholder, scope, out_size,
        hiddens = [128, 64],
        activation = tf.tanh, reuse = False):
    with tf.variable_scope(scope, reuse = reuse):
        x = tf.layers.flatten(input_placeholder)
        for hidden_size in hiddens:
            x = tf.layers.dense(x, hidden_size, activation = activation)
        x = tf.layers.dense(x, out_size)
    return x
