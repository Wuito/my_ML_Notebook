# 深度学习小记

## 打开方式

程序依赖了一些python 的软件库，运行前请先安装对应版本的库，以下给出经过测试的版本。
如果没有大型网络的训练需求，或者电脑没有NVIDIA显卡可以不安装CUDA，tensorflow>2.0版本无需指定GPU，软件会自行识别
对Tensorflow,使用前请先确保以下软件环境：
```
cuda = 11.8	# CUDA也可以使用11.2版本
# 对应cuda的cudnn版本 cuda11.8对应cudnn为8.9
python=3.7
numpy==1.19.5
matplotlib== 3.5.3
notebook==6.4.12
scikit-learn==1.2.0
tensorflow==2.6.0
keras==2.6.0
```
matlab版本为R2020a及以上版本

## 内容结构

```
Tensorflow2: 使用tf2构建基础网络和相关代码
    1_FundamentalConcept : tf2的基本函数，构建深度学习的基本框架
    2_KerasNN： 使用tf2的高级API来构建网络结构
    3_CNN： 使用卷积网络实现的相关算法
    4_RNN： 使用循环神经网络实现的相关算法
    Data： 函数调用到的数据存放
        MNIST： 手写数字集数据
        CS2_38.csv： 消费用锂电池数据（多维-6300组数据）
        soc.csv： 动力锂电池soc数据（一维-2500组数据）
        SH600519.csv：股票历史数据

matlab：使用matlab实现部分基础的机器学习代码
    linear_xxx：线性回归，分别使用最小二乘法和迭代法计算

sk-learn：使用scikit-learn框架进行机器学习
    目前只存放了部分演示程序


```
