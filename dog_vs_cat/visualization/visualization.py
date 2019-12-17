from keras.preprocessing import image
import numpy as np 
import matplotlib.pyplot as plt
from keras import models



#导入测试图片
img_path='E:/dogs-vs-cats/train/train_small/test/cats/cat.1700.jpg'
model_path='C:/Users/Administrator/Desktop/my-deep-learning/dog_vs_cat/low_variance/low_variance_dogs_vs_cats.h5'


#图片预处理
img=image.load_img(img_path,target_size=(150,150,3))
img_tensor=image.img_to_array(img)
img_tensor=np.expand_dims(img_tensor,axis=0)
img_tensor/=255.

#print(img_tensor.shape)
plt.imshow(img)


#导入模型
model=models.load_model(model_path)

layer_outputs=[layer.output for layer in model.layers[:8]]  #提取前8层输出
activation_model=models.Model(inputs=model.input,outputs=layer_outputs)


activations=activation_model.predict(img_tensor)
first_layer_activation=activations[0]
#print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0,:,:,3],cmap='viridis')  #颜色为翠绿色
plt.matshow(first_layer_activation[0,:,:,8],cmap='viridis')
