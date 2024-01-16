import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
# 定义变量
lr = [0.6, 0.2, 0.01] # 设置了三个学习率
epoch = 50 # 迭代50次
losses = []
for lrs in lr:
    wegt = tf.Variable(tf.constant(5, dtype=tf.float32))	# 定义了值为5，数据类型为float32
    for epoch in range(epoch):
        with tf.GradientTape() as tape:	        # 创建上下文管理器，用于记录网络训练的过程
            loss = tf.square(wegt + 1)	        # 计算损失函数的值，在tf.GradientTape上下文中，损失函数是loss = (wegt + 1)的平方
        grads = tape.gradient(loss, wegt)	    # 利用上下文管理器记录的信息，gradient()函数自动求微分，计算损失相对于权重 wegt 的梯度

        wegt.assign_sub(lrs * grads)	# 跟新权重：assign_sub()函数等效是自减=> wegt=wegt-lr * grads
        print("After %s epoch, w is %f, loss is %f" % (epoch, wegt.numpy(), loss))
        losses.append(loss.numpy())
    ax.plot(np.arange(0, epoch + 1, 1), losses, label="lr={}".format(lrs))
    losses.clear()

ax.set(xlabel='Epoch', ylabel='Loss', title='Gradient Descent')
ax.legend(loc='upper right')
ax.grid()
plt.show()