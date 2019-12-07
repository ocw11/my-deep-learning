#卷积神经网络

from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers



#数据集导入及预处理
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
train_images=train_images.reshape(60000,28,28,1)
test_images=test_images.reshape(10000,28,28,1)

train_images=train_images.astype('float32')/255
test_images=test_images.astype('float32')/255

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)


#网络搭建

CNN=models.Sequential()
CNN.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))  #卷积层，通道数32，卷积核大小3*3
CNN.add(layers.MaxPooling2D(2,2))   #池化层，树池2*2
CNN.add(layers.Conv2D(64,(3,3),activation='relu'))  #通道数64
CNN.add(layers.MaxPooling2D(2,2))
CNN.add(layers.Conv2D(64,(3,3),activation='relu'))  #通道数64
CNN.add(layers.Flatten())  #将多维数组降至一维数组
CNN.add(layers.Dense(64,activation='relu'))  #全连接层
CNN.add(layers.Dense(10,activation='softmax'))

CNN.summary()   #共计93322个参数

CNN.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])



#拟合训练集，对测试集进行预测
CNN.fit(train_images,train_labels,batch_size=64,epochs=5)
train_loss,train_acc=CNN.evaluate(train_images,train_labels)
test_loss,test_acc=CNN.evaluate(test_images,test_labels)


#输出测试集的训练效果
print("训练集误差：",train_loss)
print("训练集准确率:",train_acc)
print("测试集误差:",test_loss)
print("测试集准确率:",test_acc)