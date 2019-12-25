用datasets.make_moons函数生成样本点1000个，参数设置为noise=0.1
用datasets.make_blobs函数生成样本点1000个, 参数设置为n_features=2, centers=[[1.2,1.2]], cluster_std=0.1,
二个函数的random_state都设置为5。
调整DBSCAN算法的参数正确识别出相应的类，代码中反映调整的过程。
