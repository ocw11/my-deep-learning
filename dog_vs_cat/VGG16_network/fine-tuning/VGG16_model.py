import os
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from matplotlib import pyplot as plt


#导入VGG16预训练网络
conv_base=VGG16(weights='imagenet',
                include_top=False,
                input_shape=(150,150,3))



#导入数据
base_dir='E:/dogs-vs-cats/train/train_small'
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
test_dir=os.path.join(base_dir,'test')


#训练集数据增强
train_datagen=ImageDataGenerator(rescale=1./255,
                                 rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

#验证集集数据不要增强
test_datagen=ImageDataGenerator(rescale=1./255)


#导入数据
train_generator=train_datagen.flow_from_directory(train_dir,
                                                  target_size=(150,150),
                                                  batch_size=20,
                                                  class_mode='binary')


validation_generator=test_datagen.flow_from_directory(validation_dir,
                                                  target_size=(150,150),
                                                  batch_size=20,
                                                  class_mode='binary')


#搭建网络
network=models.Sequential()
network.add(conv_base)
network.add(layers.Flatten())
network.add(layers.Dense(256,activation='relu'))
network.add(layers.Dense(1,activation='sigmoid'))

network.summary()







