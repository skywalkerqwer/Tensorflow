# Tensorflow学习笔记大纲

# day01
- 01_firstDemo.py
  - 创建结构
  - 设置W,b
  - 运用梯度下降进行学习
  - init初始化变量
- 02_withSess.py
  - 两种创建Session的方法
- 03_variable.py
  - 创建更新变量
- 04_placeholder.py
  - 占位符的创建和使用
  - feed_dict传入对应的placeholder参数
- 05_addLayer.py
  - 定义对神经网络增加一层的方法
  - 添加激活函数
  - 本质就是构建g(Z), Z = Wx + b
- 06_build.py
  - 搭建一个神经网络
- 07_plotResult.py
  - 可视化显示拟合过程
- 08_tensorborad.py
  - 用with创建层
  - 用summary.FileWriter保存文件
  - 绘制loss函数曲线
  - 终端tensorboard --logdir='logs/' 指定文件目录