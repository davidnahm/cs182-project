import tensorflow as tf

def mlp(input_placeholder, scope, out_size,
        hiddens = [128, 64],
        activation = tf.tanh):
    with tf.variable_scope(scope):
        x = tf.layers.flatten(input_placeholder)
        for hidden_size in hiddens:
            x = tf.layers.dense(x, hidden_size, activation = activation)
        x = tf.layers.dense(x, out_size)
        x = tf.nn.log_softmax(x)
    return x

# def lstm(input_placeholder, scope, out_size,
#          hidden_size = 32):
#          lstm = tf.nn.rnn_cell.LSTMCell(hidden_size)
