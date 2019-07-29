import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mp


def add_layer(inputs, in_size, out_size, activation_func=None):
    """增加神经层"""
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


# 定义一个包含误差的 y= x^2 函数数据集
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.1, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 输入层：1  隐藏层设置为10  输出层：1
"""
l1: inputs = xs
    in_size = 输入端样本个数  因为只有一个x_data所以为1
    out_size = A[0] --> a[1]中有10个隐藏节点 所以为10
    act_func = 选择激活函数为ReLU
"""
l1 = add_layer(xs, 1, 10, activation_func=tf.nn.relu)

"""
输出层prediction
    inputs = 输入是前一层(隐藏层)
    in_size = 10
    out_size = 只有一个输出节点a 所以为1
"""
pred_y = add_layer(l1, 10, 1, activation_func=None)


loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - pred_y),reduction_indices=[1]))

# 调整学习率适中
train_step = tf.train.GradientDescentOptimizer(0.08).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    fig = mp.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)  # 输入
    mp.ion()
    mp.show()


    for i in range(100):  # 调整循环次数
        sess.run(train_step, feed_dict={xs:x_data,
                                        ys:y_data})

        # print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        try:
            ax.lines.remove(lines[0])  # 忽略第一次抹除的错误
        except:
            pass

        pred_value = sess.run(pred_y, feed_dict={xs:x_data})
        lines = mp.plot(x_data, pred_value, 'r-', lw=5)
        mp.pause(0.1)