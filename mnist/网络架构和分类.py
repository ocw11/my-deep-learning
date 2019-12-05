from keras import models   #models即使用网络的模型，分序列模型（The Sequential model）和函数式模型
from keras import layers
from keras.datasets import mnist




#神经网络搭建
network=models.Sequential()  #序列模型

#这是一个两层的神经网络，均为全连接层，第一层神经元个数为512个，第二次神经元个数为10个
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

print(network.summary())  #共计407050个参数




#导入数据并进行预处理
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
train_images=train_images.reshape(60000,28*28)
