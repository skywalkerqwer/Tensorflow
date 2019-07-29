"""
添加层
"""
import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_func=None):
    # 定义 W
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 定义 b
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # Z = Wx + b
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # g(Z)
    if activation_func is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_func(Wx_plus_b)
    return outputs