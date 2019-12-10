import os  #导入操作系统模块，用于操作文件
from matplotlib import pyplot as plt  #导入绘图模块
from keras.preprocessing import image #图像预处理模块
from keras.preprocessing.image import ImageDataGenerator  #用于对图像进行预处理



base_dir='E:/dogs-vs-cats/train/train_small'  #保存较小数据集的路径
train_dir=os.path.join(base_dir,'train')
train_cats_dir=os.path.join(train_dir,'cats')

#将测试集中猫图片的路径提取出来
fnames=[os.path.join(train_cats_dir,fname) for
        fname in os.listdir(train_cats_dir)]
img_path=fnames[3]

img=image.load_img(img_path,target_size=(150,150))  #读取图片，并调整大小
x=image.img_to_array(img)   #将图片转换为数组
x=x.reshape((1,)+x.shape)   #将（150，150，3）的数组大小转换为（1，150，150，3）


#数据增强来防止过拟合
datagen=ImageDataGenerator(
        rotation_range=40,      #旋转角度（0-180）
        width_shift_range=0.2,  #水平方向平移
        height_shift_range=0.2, #垂直方向平移
        shear_range=0.2,        #随机错切变换角度
        zoom_range=0.2,         #随机缩放范围
        horizontal_flip=True,   #随机将一半图像水平翻转 
        fill_mode='nearest')    #填充像素选择最近的像素


#生成随机变换的图像
i=0  #计数器
for batch in datagen.flow(x,batch_size=1):
    plt.figure(i)
    imgplot=plt.imshow(image.array_to_img(batch[0]))
    i+=1
    if i%4==0:
        break

plt.show()
