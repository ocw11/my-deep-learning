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


#计算DBI
def DBI_index(cluster0,cluster1,cluster2,cluster3,center):
    x0=x1=x2=x3=0
    for value in cluster0:
        x0+=euclid(value,center[0,:])
    for value in cluster1:
        x1+=euclid(value,center[1,:])
    for value in cluster2:
        x2+=euclid(value,center[2,:])
    for value in cluster3:
        x3+=euclid(value,center[3,:])
    #计算簇内的质心距离均值Si
    Si0=x0/len(cluster0)
    Si1=x1/len(cluster1)
    Si2=x2/len(cluster2)
    Si3=x3/len(cluster3)
    #计算簇间距离M
    M01=euclid(center[0,:],center[1,:])
    M02=euclid(center[0,:],center[2,:])
    M03=euclid(center[0,:],center[3,:])
    M12=euclid(center[1,:],center[2,:])
    M13=euclid(center[1,:],center[3,:])
    M23=euclid(center[2,:],center[3,:])
    #计算相似度R
    R01=(Si0+Si1)/M01
    R02=(Si0+Si2)/M02
    R03=(Si0+Si3)/M03
    R12=(Si1+Si2)/M12
    R13=(Si1+Si3)/M13
    R23=(Si2+Si3)/M23
    #计算每个簇的最大相似度D
    D0=max(R01,R02,R03)
    D1=max(R01,R12,R13)
    D2=max(R02,R12,R23)
    D3=max(R03,R13,R23)
    #求平均数，这就是DBI
    DBI_index=(D0+D1+D2+D3)/4
    print("DBI指数为：",DBI_index)
    return DBI_index

#%构建k-means算法

k=4    #从图中大致看出，簇为4
#首先初始化，随机选取k个样本作为初始均值向量init
np.random.seed(14)    
np.random.shuffle(data)
center=np.array(data[0:k])


time=0
max_time=10  #迭代10次
E=0.005   #质心距离和差异期望
last_distance=0

time_list=[]
dis_list=[]
DBI_list=[]





while True:
    time+=1  #计数
    time_list.append(time)
    
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
    dis_list.append(delta_distance)  #纪录每次迭代的距离和
    
    
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
    #plt.show()
    DBI=DBI_index(cluster0,cluster1,cluster2,cluster3,center)
    DBI_list.append(DBI)

    #当质心距离差值小于E时，跳出迭代
    if delta_distance<E and time>=max_time:
        break
    last_distance=total_distance   #纪录上次迭代时的质心距离和




#画出距离和的曲线
plt.figure()
plt.plot(time_list,dis_list)
plt.title("centroid_distance_increment_plot")
plt.xlabel("time")
plt.ylabel("centroid distance increment")
plt.show()

plt.figure()
plt.plot(time_list,DBI_list)
plt.title("DBI_index_plot")
plt.xlabel("time")
plt.ylabel("DBI index")
plt.show()

#我们画出最终分类之后的情况，不同簇用不同颜色区分开
plt.figure()
plt.title("K-means")
plt.scatter(cluster0[:,0],cluster0[:,1],color='r')
plt.scatter(cluster1[:,0],cluster1[:,1],color='b')
plt.scatter(cluster2[:,0],cluster2[:,1],color='g')
plt.scatter(cluster3[:,0],cluster3[:,1],color='y')

plt.show()


