import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 计算欧拉距离
def calcDis(dataSet, centroids, k):
    clalist = []
    '''
    centroids = [[5, 4], [2, 1]]
    dataset = [[1, 1], [1, 2], [2, 1], [6, 4], [6, 3], [5, 4]]
    k = 2
    若data = [1,1]  np.tile(data, (k,1)) =[[1 1],[1 1]] diff =[[-4 -3],[-1 0]]
    squaredDiff = [[16 9] [1 0]]
    squaredDist = [25  1]
    distance = [5 1]
    clalist =
    [[5.         1.        ]
     [4.47213595 1.41421356]
     [4.24264069 0.        ]
     [1.         5.        ]
     [1.41421356 4.47213595]
     [0.         4.24264069]]
    '''
    for data in dataSet:
        # np.tile(data,res) 把data作为一个元素  构造shape为res的数组
        diff = np.tile(data, (k,
                              1)) - centroids  # 相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
        squaredDiff = diff ** 2  # 平方
        squaredDist = np.sum(squaredDiff, axis=1)  # 和  (axis=1表示行)
        distance = squaredDist ** 0.5  # 开根号
        clalist.append(distance)
    clalist = np.array(clalist)  # 返回一个每个点到质点的距离len(dateSet)*k的数组
    return clalist


# 计算质心
'''
print(clalist.shape) (6,2)
print(minDistIndices) [1 1 1 0 0 0]
print(newCentroids) [[5.66666667 3.66666667][1.33333333 1.33333333]]
print(changed) #[[ 0.66666667 -0.33333333][-0.66666667  0.33333333]]
'''
def classify(dataSet, centroids, k):
    # 计算样本到质心的距离
    clalist = calcDis(dataSet, centroids, k)
    # print(clalist.shape) (6,2)
    # 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)  # axis=1 表示求出每行的最小值的下标
    newCentroids = pd.DataFrame(dataSet).groupby(
        minDistIndices).mean()  # DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentroids = newCentroids.values

    # 计算变化量
    changed = newCentroids - centroids
    return changed, newCentroids


# 使用k-means分类
def kmeans(dataSet, k):
    # 随机取质心  随机取k个点
    centroids = random.sample(dataSet, k)

    # 更新质心 直到变化量全为0
    changed, newCentroids = classify(dataSet, centroids, k)
    while np.any(changed != 0):
        changed, newCentroids = classify(dataSet, newCentroids, k)

    centroids = sorted(newCentroids.tolist())  # tolist()将矩阵转换成列表 sorted()排序

    # 根据质心计算每个集群
    cluster = []
    clalist = calcDis(dataSet, centroids, k)  # 调用欧拉距离
    minDistIndices = np.argmin(clalist, axis=1)
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):  # enymerate()可同时遍历索引和遍历元素
        cluster[j].append(dataSet[i])

    return centroids, cluster


# 创建数据集
def createDataSet():
    return [[1, 1], [1, 2], [2, 1], [6, 4], [6, 3], [5, 4]]


if __name__ == '__main__':
    dataset = createDataSet()
    centroids, cluster = kmeans(dataset, 2)
    print('质心为：%s' % centroids)
    ('集群为：%s' % cluster)
    for i in range(len(dataset)):
        plt.scatter(dataset[i][0], dataset[i][1], marker='o', color='green', s=40, label='原始点')
        #  记号形状       颜色      点的大小      设置标签
        for j in range(len(centroids)):
            plt.scatter(centroids[j][0], centroids[j][1], marker='x', color='red', s=50, label='质心')
            plt.show()

