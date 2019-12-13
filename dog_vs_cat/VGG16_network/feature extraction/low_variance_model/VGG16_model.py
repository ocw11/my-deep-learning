import os
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
network.add(layers.Dense(512,activation='relu'))
network.add(layers.Dense(1,activation='sigmoid'))

network.summary()

#将卷积基的权重设置为不会改变，即冻结（freeze）这些网络
conv_base.trainable=False


network.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                loss='binary_crossentropy',
                metrics=['acc'])

history=network.fit_generator(train_generator,
                    steps_per_epoch=100,
                    epochs=30,
                    validation_data=validation_generator,
                    validation_steps=50)

network.save('VGG16_model.h5')

#绘制图像
acc=history.history['acc']
loss=history.history['loss']

val_acc=history.history['val_acc']
val_loss=history.history['val_loss']

epochs=range(1,1+len(acc))

plt.plot(epochs,acc,'bo',label='Training_acc')
plt.plot(epochs,val_acc,'b',label='Validation_acc')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training_loss')
plt.plot(epochs,val_loss,'b',label='Validation')
plt.legend()

plt.show()



print("训练集误差：",loss[-1])
print("训练集准确率：",acc[-1])

print("验证集误差：",val_loss[-1])
print("验证集准确率：",val_acc[-1])
