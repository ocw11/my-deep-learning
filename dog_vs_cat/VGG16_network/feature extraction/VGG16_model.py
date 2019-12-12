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

conv_base.summary()


#导入数据
base_dir='E:/dogs-vs-cats/train/train_small'
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
test_dir=os.path.join(base_dir,'test')


datagen=ImageDataGenerator(rescale=1./255)  #缩小尺寸
batch_size=20


#特征提取
def extract_features(directory,sample_count):
    features=np.zeros(shape=(sample_count,4,4,512))   #提取出的特征
    labels=np.zeros(sample_count)                     #特征的标签
    generator=datagen.flow_from_directory(directory,
                                          target_size=(150,150),
                                          batch_size=batch_size,
                                          class_mode='binary')
    i=0
    for inputs_batch,labels_batch in generator:
        feature_batch=conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1)*batch_size]=feature_batch
        labels[i*batch_size:(i+1)*batch_size]=labels_batch
        i+=1
        if i*batch_size>=sample_count:
            break
    return features,labels

train_features,train_labels=extract_features(train_dir,2000)
validation_features,validation_labels=extract_features(validation_dir,2000)
test_features,test_labels=extract_features(test_dir,2000)


#改变形状（x，4，4，512）到（x，4*4*512）
train_features=np.reshape(train_features,(2000,4*4*512))
validation_features=np.reshape(validation_features,(2000,4*4*512))
test_features=np.reshape(test_features,(2000,4*4*512))


#搭建全连接层网络
network=models.Sequential()
network.add(layers.Dense(256,activation='relu',input_dim=4*4*512))
network.add(layers.Dropout(0.5))
network.add(layers.Dense(1,activation='sigmoid'))

network.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                loss='binary_crossentropy',
                metrics=['acc'])


history=network.fit(train_features,train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features,validation_labels))


#绘制图像
acc=history.history['acc']
loss=history.history['loss']
val_acc=history.history['val_acc']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training_acc')
plt.plot(epochs,val_acc,'b',label='Validation_acc')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training_loss')
plt.plot(epochs,val_loss,'b',label='Validaiton_loss')

plt.show


print("训练集误差：",loss[-1])
print("训练集准确率：",acc[-1])

print("验证集误差：",val_loss[-1])
print("验证集准确率：",val_acc[-1])

