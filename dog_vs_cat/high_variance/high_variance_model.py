import os,shutil  #导入操作系统模块，用于操作文件
from keras.preprocessing.image import ImageDataGenerator  #用于对图像进行预处理
from keras import layers  #导入网络层模块
from keras import models  #导入网络模型模块
from keras import optimizers  #导入优化器
from matplotlib import pyplot as plt  #导入绘图模块



#导入数据
original_dataset_dir='E:/dogs-vs-cats/train/train'  #保存原始数据集的路径

base_dir='E:/dogs-vs-cats/train/train_small'  #保存较小数据集的路径
#os.mkdir(base_dir)   #创建保存较小数据集的文件夹

train_dir=os.path.join(base_dir,'train')
#os.mkdir(train_dir)  #创建训练集文件夹

validation_dir=os.path.join(base_dir,'validation')
#os.mkdir(validation_dir)  #创建验证集文件夹

test_dir=os.path.join(base_dir,'test')
#os.mkdir(test_dir)  #创建测试集文件夹


train_cats_dir=os.path.join(train_dir,'cats')
#os.mkdir(train_cats_dir)  #创建猫的训练集文件夹
train_dogs_dir=os.path.join(train_dir,'dogs')
#os.mkdir(train_dogs_dir)  #创建狗的训练集文件夹


validation_cats_dir=os.path.join(validation_dir,'cats')
#os.mkdir(validation_cats_dir)  #创建猫的验证集文件夹
validation_dogs_dir=os.path.join(validation_dir,'dogs')
#os.mkdir(validation_dogs_dir)  #创建猫的验证集文件夹


test_cats_dir=os.path.join(test_dir,'cats')
#os.mkdir(test_cats_dir)  #创建猫的测试集文件夹
test_dogs_dir=os.path.join(test_dir,'dogs')
#os.mkdir(test_dogs_dir)  #创建狗的测试集文件夹


#生成猫和狗的测试集，训练集和验证集
#猫
fnames=['cat.{}.jpg'.format(i) for i in range(1000)]  #选取1000个猫的图片
for fname in fnames:
    src=os.path.join(original_dataset_dir,fname)   #复制地址
    dst=os.path.join(train_cats_dir,fname)   #粘贴地址
    shutil.copyfile(src,dst)  #复制图片到猫训练集文件夹（train_cats_dir）

fnames=['cat.{}.jpg'.format(i) for i in range(1000,1500)]  #选取之后500个猫的图片
for fname in fnames:
    src=os.path.join(original_dataset_dir,fname)  #复制地址
    dst=os.path.join(validation_cats_dir,fname)  #粘贴地址
    shutil.copyfile(src,dst)  #复制图片到猫验证集文件夹（validation_cats_dir）

fnames=['cat.{}.jpg'.format(i) for i in range(1500,2000)]  #选取之后500个猫的图片
for fname in fnames:
    src=os.path.join(original_dataset_dir,fname)  #复制地址
    dst=os.path.join(test_cats_dir,fname)  #粘贴地址
    shutil.copyfile(src,dst)  #复制图片到猫测试集文件夹（test_cats_dir）
#狗
fnames=['dog.{}.jpg'.format(i) for i in range(1000)]  #选取1000个狗的图片
for fname in fnames:
    src=os.path.join(original_dataset_dir,fname)   #复制地址
    dst=os.path.join(train_dogs_dir,fname)   #粘贴地址
    shutil.copyfile(src,dst)  #复制图片到狗训练集文件夹（train_dogs_dir）

fnames=['dog.{}.jpg'.format(i) for i in range(1000,1500)]  #选取之后500个狗的图片
for fname in fnames:
    src=os.path.join(original_dataset_dir,fname)  #复制地址
    dst=os.path.join(validation_dogs_dir,fname)  #粘贴地址
    shutil.copyfile(src,dst)  #复制图片到狗验证集文件夹（validation_dogs_dir）

fnames=['dog.{}.jpg'.format(i) for i in range(1500,2000)]  #选取之后500个狗的图片
for fname in fnames:
    src=os.path.join(original_dataset_dir,fname)  #复制地址
    dst=os.path.join(test_dogs_dir,fname)  #粘贴地址
    shutil.copyfile(src,dst)  #复制图片到狗测试集文件夹（test_dogs_dir）




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

model.add(layers.Dense(512,activation='relu'))   #全连接层，神经元为512个，激活函数为ReLU
model.add(layers.Dense(1,activation='sigmoid'))  #全连接层，输出位1个，因为是二分问题，所以激活函数为sigmoid


model.summary()  



#优化网络，调节参数
model.compile(loss='binary_crossentropy',             #损失函数为交叉熵
              optimizer=optimizers.RMSprop(lr=1e-4),  #优化函数为RMSprop，学习率为0.0001
              metrics=['acc'])                        #衡量指标为准确率（acc）



#数据的预处理
train_datagen=ImageDataGenerator(rescale=1./255)  #将图像1/255进行缩小
test_datagen=ImageDataGenerator(rescale=1./255)   #将图像1/255进行缩小

train_generator=train_datagen.flow_from_directory(
        train_dir,               #目标路径
        target_size=(150,150),   #将图片大小调整为150*150
        batch_size=20,           #批次大小为20
        class_mode='binary')     #类型为二进制标签

validation_generator=train_datagen.flow_from_directory(
        validation_dir,               #目标路径
        target_size=(150,150),   #将图片大小调整为150*150
        batch_size=20,           #批次大小为20
        class_mode='binary')     #类型为二进制标签



#训练模型
history=model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)


#保存模型为h5文件
model.save('high_variance_dogs_vs_cats.h5')



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

