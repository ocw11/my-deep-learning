import os  #导入操作系统模块，用于操作文件
from keras.preprocessing.image import ImageDataGenerator  #用于对图像进行预处理
from keras import layers  #导入网络层模块
from keras import models  #导入网络模型模块
from keras import optimizers  #导入优化器
from matplotlib import pyplot as plt  #导入绘图模块
from keras.preprocessing import image #图像预处理模块



#导入数据
base_dir='E:/dogs-vs-cats/train/train_small'  #保存较小数据集的路径

train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
test_dir=os.path.join(base_dir,'test')

train_cats_dir=os.path.join(train_dir,'cats')
train_dogs_dir=os.path.join(train_dir,'dogs')

validation_cats_dir=os.path.join(validation_dir,'cats')
validation_dogs_dir=os.path.join(validation_dir,'dogs')

test_cats_dir=os.path.join(test_dir,'cats')
test_dogs_dir=os.path.join(test_dir,'dogs')





#建立卷积神经网络
model=models.Sequential()  #设定网络模型结构为Sequential，即为顺序结构，依次执行网络层


model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))  #卷积层1，卷积核32，激活函数为ReLU
model.add(layers.MaxPooling2D(2,2))   #池化层1

model.add(layers.Conv2D(64,(3,3),activation='relu'))     #卷积层2，通道数64，激活函数为ReLU
model.add(layers.MaxPooling2D(2,2))   #池化层2

model.add(layers.Conv2D(128,(3,3),activation='relu'))    #卷积层3，通道数128，激活函数为ReLU
model.add(layers.MaxPooling2D(2,2))  #池化层3

model.add(layers.Conv2D(128,(3,3),activation='relu'))     #卷积层4，通道数128，激活函数为ReLU
model.add(layers.MaxPooling2D(2,2))  #池化层4

model.add(layers.Flatten())  #扁平层，将多维数组压缩成1维

model.add(layers.Dropout(0.5))  #在网络中增加了Dropout层，随机使50%的神经元失活来避免过拟合

model.add(layers.Dense(512,activation='relu'))   #全连接层，神经元为512个，激活函数为ReLU
model.add(layers.Dense(1,activation='sigmoid'))  #全连接层，输出位1个，因为是二分问题，所以激活函数为sigmoid


model.summary()  



#优化网络，调节参数
model.compile(loss='binary_crossentropy',             #损失函数为交叉熵
              optimizer=optimizers.RMSprop(lr=1e-4),  #优化函数为RMSprop，学习率为0.0001
              metrics=['acc'])                        #衡量指标为准确率（acc）



#数据的预处理
train_datagen=ImageDataGenerator(
        rescale=1./255,   #将图像1/255进行缩小
        rotation_range=40,      #旋转角度（0-180）
        width_shift_range=0.2,  #水平方向平移
        height_shift_range=0.2, #垂直方向平移
        shear_range=0.2,        #随机错切变换角度
        zoom_range=0.2,         #随机缩放范围
        horizontal_flip=True,)   #随机将一半图像水平翻转 

test_datagen=ImageDataGenerator(rescale=1./255)  #注意：不能增加验证集的数据

train_generator=train_datagen.flow_from_directory(
        train_dir,               #目标路径
        target_size=(150,150),   #将图片大小调整为150*150
        batch_size=20,           #批次大小为20
        class_mode='binary')      #类型为二进制标签
            
validation_generator=train_datagen.flow_from_directory(
        validation_dir,               #目标路径
        target_size=(150,150),   #将图片大小调整为150*150
        batch_size=20,           #批次大小为20
        class_mode='binary')     #类型为二进制标签



#训练模型
history=model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50)


#保存模型为h5文件
model.save('low_variance_dogs_vs_cats.h5')



#绘制图像
train_acc=history.history['acc']             #准确率
val_acc=history.history['val_acc']     #验证集准确率
train_loss=history.history['loss']           #损失函数（误差值）
val_loss=history.history['val_loss']   #验证集损失函数（验证集误差值）


epochs=range(1,len(train_acc)+1)#定义批次数

#绘制准确率折线图
plt.plot(epochs,train_acc,'bo',label='Training acc')       #绘制准确率折线，蓝色圆点折线
plt.plot(epochs,val_acc,'b',label='Validation acc')  #绘制验证集准确率折线，蓝色折线
plt.title('Train and validation accuracy')          #设定标题
plt.legend()       #显示图例


plt.figure()


#绘制误差折线图
plt.plot(epochs,train_loss,'bo',label='Training train_loss')       #绘制误差折线，蓝色圆点折线
plt.plot(epochs,val_loss,'b',label='Validation train_loss')  #绘制验证集误差折线，蓝色折线
plt.title('Train and validation loss')          #设定标题
plt.legend()       #显示图例



plt.show()


print("训练集误差：",train_loss[-1])
print("训练集准确率：",train_acc[-1])
print("验证集误差：",val_loss[-1])
print("验证集准确率：",val_acc[-1])

