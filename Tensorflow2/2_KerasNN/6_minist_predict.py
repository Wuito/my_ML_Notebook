import cv2
import tensorflow as tf
import numpy as np

image_path = '../Data/MNIST/'
model_path = './checkpoint/mnist/model'

# 检查是否有可用的 GPU
physical_devices = tf.config.list_physical_devices()
print(physical_devices)

new_model = tf.keras.models.load_model(model_path)  # 从tf模型加载，无需重新实例化网络

preNum = int(input("place input how many jpg file while be test:"))
img_gen = input("place input Whether to use data augmentation:(Y or N)")
# 将用户输入的字符串转换为布尔值
img_gen = img_gen.lower() == 'y'
print(img_gen)
for i in range(preNum):
    imgNum = int(input("place input png name:"))
    img_path = image_path+str(imgNum)+'.png'
    print("read image:{}".format(img_path))

    img_ = cv2.imread(img_path)
    resized_img = cv2.resize(img_, (28, 28), interpolation=cv2.INTER_AREA)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    binary_image = np.where(gray_img < 200, 255, 0).astype(np.uint8)    # 反相和过滤
    # 准备图像数据，进行归一化和添加批次维度
    img_for_prediction = binary_image.astype(np.float32) / 255.0  # 归一化到 [0, 1]
    if img_gen:
        img_for_prediction = img_for_prediction.reshape(1, 28, 28, 1)  # 数据扩充要求输入的数据是四维
    elif not img_gen:
        img_for_prediction = np.expand_dims(img_for_prediction, axis=0)  # 添加批次维度
    else:
        print('data augmentation error')
        break
    result = new_model.predict(img_for_prediction)
    predNum = tf.argmax(result, axis=1)
    print("predict num is: ")
    tf.print(predNum)