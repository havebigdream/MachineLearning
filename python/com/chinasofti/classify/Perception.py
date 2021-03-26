"""
         -*- coding:utf-8 -*-
        @Time : 2020/7/29 9:36
        @Author: 面向对象写Bug
        @File : Perception.py
        @conttent：单位阶跃函数实现的感知器
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------单位阶跃函数实现感知器----------
class Perception(object):

    def __init__(self, eta=0.01, n_iter=10):
        """
        :param eta: 学习率
        :param n_iter:  迭代次数
        """
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        :param X: 训练样本的特征信息组成的矩阵，一行代表一个样本信息
        :param y: label信息，一个样本肯定对应着一个label,这样才能用于训练

        --------------------其他相关解释------------------------
        :property w_: m+1维的一个np数组，其中第0位存储的是误差值，后面的m个数字用于和样本的m个属性做点积然后加上误差和target进行比较，然后
                    又由这个差值反作用于w_，对这个w_进行再次调节。初始值m+1维行零矩阵
        :property errors: 用于记录每次迭代有误差的样本数，不过这个误差是超过一定范围的误差errors += int(update != 0.0)，由这分析可以知道
                    这个列表最终元素个数应该等于迭代次数
        :return:
        """
        self.w_ = np.zeros(1 + X.shape[1])
        print(self.w_.shape)
        self.errors_ = []

        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[0] += update
                self.w_[1:] += update*xi
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self


    def net_input(self, X):
        """
        :param X: 输入的一个样本的特征信息，要求行矩阵
        :return: 计算 输出控制函数的 输入，需要后面的predict经过规则给出一个结果：其中这里的规则是单为阶跃函数，阈值比较
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        规则是单为阶跃函数，阈值比较,
        :param X:
        :return: 这个是最简单的，只有两种分类，所以结果就是1或-1，实际比这简单的多。
        """
        return np.where(self.net_input(X) >= 0, 1, -1)


# -----------基于鸢尾花数据训练感知器模型----------
#直接从UCI机器学习库中将鸢尾花数据集转化为DateFrame对象并加载到内存中，并使用tail方法显示数据的最后5行
df = pd.read_csv(r'F:\中软国际大数据工作\个人笔记\python\python_测试数据\data1.csv')
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)

'''
#-----------------------提取类标，数据可视化----------------------
    #从df中提取前100个类标，其中包含50个山鸢尾类和50个变色鸢尾类，定义规则：1代表变色鸢尾，-1代表山鸢尾，同时把产生的整数类标赋值给Numpy的向量y
    #提取数据的第一列和第三列赋值给属性矩阵X,然后使用二维散点图对这些数据进行可视化
    #将DataFrame提取出来的转化为数组直接在后面.values就好
    #山鸢尾（Iris-setosa) 、变色鸢尾（Iris-virginica)
'''

# 把特征矩阵x弄出来
x = df.iloc[:100,[0,2]].values
# 标签矩阵y
y = df.iloc[:100,4].values
y = np.where(y == 'Iris-setosa',-1,1)
# 画散点图看一看萼片长度（原表第一个特征列）和花瓣长度的关系（第三个特征列）的关系
plt.scatter(x[:50,0],x[:50,1],
           color='red',marker='o',label='setosa')
plt.scatter(x[50:100,0],x[50:100,1],
           color='blue',marker='x',label='virgincia')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

'''
#-----------------------------训练感知器------------------------
#我们需要绘制每次迭代的错误分类数量的折线图，已验证算法是否收敛并找到分开两种类型鸢尾花的决策边界
'''
ppn = Perception(eta=0.1,n_iter=10)
ppn.fit(x,y)
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassification')
plt.show()
print(ppn.w_)

# -----------用一个简单的函数实现对二维数据集决策边界的可视化-----------
from matplotlib.colors import ListedColormap

#--------用于二维数据集决策边界可视化的函数-------------------
def plot_decision_regions(X, y, classifier, resolution=0.02):
    """

    :param X: 属性矩阵
    :param y: label矩阵
    :param classifier:
    :param resolution:
    """
    # 建一个颜色、标记和cmap发生器
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    markers = ['s', 'x', 'o', '^', 'v']
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #画决策边界
    x1_min,x1_max = X[:,0].min() - 1 , X[:,0].max() + 1
    x2_min,x2_max = X[:,1].min() - 1 , X[:,1].max() + 1
    """
        这个meshgrid的参数应该可以很多,这里只有两个的原因在于我们只研究两个维度的特征
        这个meshgrid返回的xx1和xx2的分析：
        一般xx1和xx2的特征是：1，xx1的每行相同，xx2的每列相同，xx1存储的是描的点的在x1轴的坐标，...
                            2,xx1和xx2都是存储所有点的信息，维度是相同的
        
    """
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution)
                          ,np.arange(x2_min,x2_max,resolution))
    # ravel()对数组进行扁平化
    z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    # 这里应该是把前面得到的列向量转为行向量
    z = z.reshape(xx1.shape)
    # 调用matplotlib的contourf函数
    plt.contourf(xx1,xx2,z,alpha=0.4,cmap=cmap)
    plt.xlim(x1_min,x1_max)
    plt.ylim(x2_min,x2_max)

    # plot class sample
    # idx 应该是np.unique(y)后对每个y的索引
    #y==c1返回的是和y维度相同的一个boolean型矩阵，然后方便取值
    for idx,c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y==c1,0],y=X[y == c1,1],
                    alpha=0.8,c = cmap(idx),
                    marker=markers[idx],label = c1)

# 画出该数据集的决策边界可视化图
plot_decision_regions(x,y,classifier=ppn)
plt.xlabel('spal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()


