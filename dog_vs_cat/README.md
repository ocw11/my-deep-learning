猫狗大战数据集是来自Kaggle上2013年末公开的计算机视觉竞赛：Dogs vs Cats
数据集下载地址：http://www.kaggle.com/c/dogs-vs-cats/data
2013年的猫狗分类Kaggle竞赛的优胜者使用的模型就是卷积神经网络，最佳结果达到了95%的精度。

数据集有25000个图片样本，猫狗各占一半，测试集有12500个样本，所有图像均为彩色JPEG图像，总大小为543MB（压缩后）
我们将使用其中的4000张图像来进行训练和预测，将这些图像分为train，validation，test三个文件夹，每个文件夹下又分dogs和cats两个文件夹。trian为训练集文件夹，其中每个类别各1000个样本，validation为验证集文件夹，其中每个类别各500个样本，test为测试机文件夹，每个类别各500个样本。

总计共4000张图像：2000张训练图像，1000张验证图像，1000张测试图像。


