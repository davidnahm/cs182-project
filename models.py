import tensorflow as tf

# This very simple function is based on CS182 HW4's version,
# although it adds flattening.
# There are not too many ways to build this.
def mlp(input_placeholder, scope, out_size,
        n_layers = 1,
        hidden_size = 32,
        activation = tf.tanh):
    with tf.scope(scope):
        x = tf.flatten(input_placeholder)
        for _ in range(n_layers):
            x = tf.layers.dense(x, hidden_size, activation = activation)
        x = tf.layers.dense(x, out_size)
    return x

# def lstm(input_placeholder, scope, out_size,
#          hidden_size = 32):
#          lstm = tf.nn.rnn_cell.LSTMCell(hidden_size)
