import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='E:\Code\myMachineLearning\Tensorflow2\Data\mnist.npz')

print('训练集样本的大小:', x_train.shape)
print('训练集标签的大小:', y_train.shape)
print('测试集样本的大小:', x_test.shape)
print('测试集标签的大小:', y_test.shape)

x_train, x_test = x_train/255.0, x_test/255.0   # 灰度图像数据归一化处理

fig = plt.figure(figsize=(20, 2))
plt.set_cmap('gray')
# 显示原始图片
for i in range(0, 12):
    ax = fig.add_subplot(1, 12, i + 1)
    ax.imshow(x_train[i])
fig.suptitle('Subset of Original Training Images', fontsize=20)
plt.show()

class mnistModel(Model):
    def __init__(self, *args, **kwargs):
        super(mnistModel, self).__init__(*args, **kwargs)

        self.flatten1=layers.Flatten()
        self.d1=layers.Dense(128, activation=tf.keras.activations.relu)
        self.d2=layers.Dense(10, activation=tf.keras.activations.softmax)

    def call(self, input):
        x = self.flatten1(input)
        x = self.d1(x)
        x = self.d2(x)
        return x

model = mnistModel()

# 定义保存和记录数据的回调器
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoint/mnist/mnist.ckpt",  # 保存模型权重参数
                                                 save_weights_only=True,
                                                 save_best_only=True)
# 设置TensorBoard输出的回调函数
tfbd_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/mnist/")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001 , beta_1=0.9, beta_2=0.999),     # 'adam'  tf.keras.optimizers.Adam(learning_rate=0.01 , beta_1=0.9, beta_2=0.999)
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # 使用交叉熵损失
                metrics=['sparse_categorical_accuracy']
                )
history = model.fit(x_train, y_train, batch_size=32, epochs=3,
                    validation_data = (x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback, tfbd_callback])  # 设置回调函数
model.summary()
model_save_path = './checkpoint/mnist/model'
os.makedirs(model_save_path, exist_ok=True)
model.save(model_save_path, save_format='tf')    # 保存模型为静态权重

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

