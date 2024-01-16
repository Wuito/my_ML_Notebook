import tensorflow as tf
from matplotlib import pyplot as plt

w = tf.Variable(tf.constant(5, dtype=tf.float32))

epoch = 400
LR_BASE = 0.4  # 最初学习率
LR_DECAY = 0.9  # 学习率衰减率
LR_STEP = 2  # 喂入多少轮BATCH_SIZE后，更新一次学习率
ls_list = []

epochs = epoch
for epoch in range(epoch):  # for epoch 定义顶层循环，表示对数据集循环epoch次，此例数据集数据仅有1个w,初始化时候constant赋值为5，循环100次迭代。
    #lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)    # 学习率指数衰减，每两轮衰减一次学习率
    if epoch < epochs*0.5:  # 分段处理
        lr = 0.4
    elif epoch < epochs*0.7:
        lr = 0.2
    elif epoch < epochs*0.9:
        lr = 0.1
    elif epoch < epochs:
        lr = 0.05

    ls_list.append(lr)
    with tf.GradientTape() as tape:  # with结构到grads框起了梯度的计算过程。
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)  # .gradient函数告知谁对谁求导，这里是loss函数对w求导

    w.assign_sub(lr * grads)  # .assign_sub 对变量做自减，更新参数 即：w -= lr*grads 即 w = w - lr*grads
    print("After %s epoch,w is %f,loss is %f,lr is %f" % (epoch, w.numpy(), loss, lr))

plt.title('Learning rate attenuation')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Lr')  # y轴变量名称
plt.plot(ls_list, label="$Learning Rate$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()