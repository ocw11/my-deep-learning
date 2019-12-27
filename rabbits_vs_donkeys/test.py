from keras import models
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np



model=models.load_model('C:\\Users\\Administrator\\Desktop\\rabbits_vs_donkeys.h5')


img_path = 'C:/Users/Administrator/Desktop/test/amiya.jpg'
img=image.load_img(img_path,target_size=(150,150))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


preds = model.predict(x)


if preds==1:
    print("这是一只兔子")
elif preds==0:
    print("这是一只驴")