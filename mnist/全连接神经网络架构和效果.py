#全连接神经网络

from keras import models   #models即使用网络的模型，分序列模型（The Sequential model）和函数式模型
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical   #就是将类别向量转换为二进制（只有0和1）的矩阵类型表示



#导入数据并进行预处理
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

train_images=train_images.reshape(60000,28*28)  #修改图片尺寸
train_images=train_images.astype('float32')/255   #将颜色取值的范围缩小至0-1

test_images=test_images.reshape(10000,28*28)  
test_images=test_images.astype('float32')/255

'''
将数字转换为（0，1）矩阵是为了和输出层输出的矩阵相匹配
例如标签为7，转换为[0,0,0,0,0,0,0,1,0,0] 这正是这个样本的期望输出值
'''
#print(test_labels[0])
train_labels=to_categorical(train_labels) 
test_labels=to_categorical(test_labels)
#print(test_labels[0])

#神经网络搭建
network=models.Sequential()  #序列模型

#这是一个两层的神经网络，均为全连接层，第一层神经元个数为512个，第二次神经元个数为10个
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

#设置神经网络的优化器，损失函数（交叉熵），衡量指标（准确率）
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#network.summary()  #共计407050个参数


#拟合训练集，预测测试集
network.fit(train_images,train_labels,epochs=5,batch_size=128)
train_loss,train_acc=network.evaluate(train_images,train_labels)
test_loss,test_acc=network.evaluate(test_images,test_labels)


#输出测试集的训练效果
print("训练集误差：",train_loss)
print("训练集准确率:",train_acc)
print("测试集误差:",test_loss)
print("测试集准确率:",test_acc)