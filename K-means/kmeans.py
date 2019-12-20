import numpy as np
import matplotlib.pyplot as plt 

#导入数据
data=np.loadtxt("C:\\Users\\Administrator\\Desktop\\Python\\testSet.csv",delimiter=",")
#print(len(data))

#观察一下这个数据集的分布情况
plt.scatter(data[:,0],data[:,1])
plt.show()


#我们选用欧几里得距离，所以定义一个欧几里得距离的计算函数会很方便
def euclid(value,center):
    return abs(((value[0]-center[0])**2+(value[1]-center[1])**2)**0.5)



#%构建k-means算法

k=4    #从图中大致看出，簇为4
#首先初始化，随机选取k个样本作为初始均值向量init
np.random.seed(10)    
np.random.shuffle(data)
center=np.array(data[0:k])



max_time=10  #迭代10次
E=0.005   #质心距离和差异期望
last_distance=0







while True:
    for time in range(max_time):
        #创建四个簇（列表）
        cluster0=[]
        cluster1=[]
        cluster2=[]
        cluster3=[]
        
        total_distance=0   
        total_distance0=0
        total_distance1=0
        total_distance2=0
        total_distance3=0
        
        #将数据集的每个点分类到簇中，质心为center，距离为欧几里得距离
        for value in data:
            distance0=euclid(value,center[0])
            distance1=euclid(value,center[1])
            distance2=euclid(value,center[2])
            distance3=euclid(value,center[3])
            min_dis=min(distance0,distance1,distance2,distance3)
            #距离哪个质心最小，则归为哪个簇
            if min_dis==distance0:
                cluster0.append(value)
            elif min_dis==distance1:
                cluster1.append(value)
            elif min_dis==distance2:
                cluster2.append(value)
            elif min_dis==distance3:
                cluster3.append(value)

            
    
    #计算出当前每个簇的新的质心（均值向量）
    new_center0=sum(cluster0)/len(cluster0)
    new_center1=sum(cluster1)/len(cluster1)
    new_center2=sum(cluster2)/len(cluster2)
    new_center3=sum(cluster3)/len(cluster3)
    #print("time:",time)
    #print(new_center0)
    #print(new_center1)
    #print(new_center2)
    #print(new_center3)
    #print("--------")
    
    
    #计算出质心到簇各点的和
    for value in cluster0:
        total_distance0+=euclid(value,new_center0)
    for value in cluster1:
        total_distance1+=euclid(value,new_center1)
    for value in cluster2:
        total_distance2+=euclid(value,new_center2)
    for value in cluster3:
        total_distance3+=euclid(value,new_center3)
    total_distance=total_distance0+total_distance1+total_distance2+total_distance3
    delta_distance=abs(total_distance-last_distance)
    print('质心距离差值：',delta_distance)
    
    
    
    #这里我将list转换为array，因为list不好进行二维切片
    center[0]=np.array(new_center0)
    center[1]=np.array(new_center1)
    center[2]=np.array(new_center2)
    center[3]=np.array(new_center3)
    cluster0=np.array(cluster0)
    cluster1=np.array(cluster1)
    cluster2=np.array(cluster2)
    cluster3=np.array(cluster3)
    
    #画出质心点center在迭代过程中的变化情况
    plt.figure()
    plt.ylim([-8.2,8.2])
    plt.xlim([-8.5,8.5])
    plt.scatter(center[0,0],center[0,1],color='r')
    plt.scatter(center[1,0],center[1,1],color='b')
    plt.scatter(center[2,0],center[2,1],color='g')
    plt.scatter(center[3,0],center[3,1],color='y')
    plt.show()
    #plt.scatter(cluster0[:,0],cluster0[:,1],color='r')
    #plt.scatter(cluster1[:,0],cluster1[:,1],color='b')
    #plt.scatter(cluster2[:,0],cluster2[:,1],color='g')
    #plt.scatter(cluster3[:,0],cluster3[:,1],color='y')
    
    
    #当质心距离差值小于E时，跳出迭代
    if delta_distance<E:
        break
    last_distance=total_distance   #纪录上次迭代时的质心距离和
    
#我们画出最终分类之后的情况，不同簇用不同颜色区分开
plt.figure()
plt.title("K-means")
plt.scatter(cluster0[:,0],cluster0[:,1],color='r')
plt.scatter(cluster1[:,0],cluster1[:,1],color='b')
plt.scatter(cluster2[:,0],cluster2[:,1],color='g')
plt.scatter(cluster3[:,0],cluster3[:,1],color='y')

plt.show()



