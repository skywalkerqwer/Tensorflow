import tensorflow as tf
import numpy as np



def add_layer(inputs, in_size, out_size, activation_func=None):
    """增加神经层"""
    with tf.name_scope('layer'):
        with tf.name_scope('Weight'):
            # 定义 W
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),name='W')
        with tf.name_scope('biases'):
            # 定义 b
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Z'):
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
y_data = np.square(x_data) + 0.5 + noise

with tf.name_scope('Inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

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

with tf.name_scope('loss'):
# 选择交叉熵函数作为loss函数 1/m(-∑y(ln(a))
    loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(pred_y, 1e-10, 1.0)),  # 保证log(pred_y)值不能过小 会出现nan
                                        reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    # 调整学习率适中
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs/', sess.graph)
    sess.run(init)
    for i in range(500):
        sess.run(train_step, feed_dict={xs:x_data,ys:y_data})
        if i%25 == 0:
            result = sess.run(merged, feed_dict={xs:x_data,ys:y_data})
            writer.add_summary(result,i)