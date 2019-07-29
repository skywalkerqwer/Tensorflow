import tensorflow as tf
import numpy as np

# create data
x_data = np.random.randn(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# create tensorflow structure start
### 开始创建结构  ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))

# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(0.1)  # 学习率0.1
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()  # 初始化结构

### 创建结构完成 ###
sess = tf.Session()
sess.run(init)  # 初始化

for step in range(201):
    sess.run(train)  # 训练目标
    if step%20 == 0:
        print(step,sess.run(Weights),sess.run(biases))  # 每隔20步显示结果
