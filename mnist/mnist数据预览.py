from keras.datasets import mnist  #datasets是keras中的一个模块，从https://s3.amazonaws.com/img-datasets/mnist.npz中下载一些经典的数据集
from matplotlib import pyplot as plt  #将mnist手写字体样本用matplotlib画出来，更加直观

#从datasets中导入mnist数据集
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

#观察一下训练集和测试集的大小，样本个数和标签

print("训练集大小：",train_images.shape)
print("训练集个数：",len(train_images))
print("训练集标签",train_labels)

print("测试集个数：",len(test_images))
print("测试集大小：",test_images.shape)
print("测试集标签",test_labels)


#试着抽样画出mnist的样本
#测试集
train_view=train_images[4]
plt.imshow(train_view,plt.cm.binary)  #cm的全称为：colormap，即色图，binary即二进制，0代表白色，1代表黑色，由浅到深
plt.title("The fifth sample in the training set")
plt.show()

#训练集
test_view=test_images[9]
plt.imshow(test_view,plt.cm.binary)
plt.title("The 10th sample in the test set")
plt.show()