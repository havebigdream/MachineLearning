"""
 -*- coding:utf-8 -*-
@Time : 2020/7/30 0:07
@Author: 面向对象写Bug
@File : AdalineSGD.py 
@content: 大规模机器学习与随机梯度下降
"""
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

class AdalineSGD(object):

    def __init__(self,eta=0.01,n_iter=10
                 ,shuffle = True,random_state = None):
        """
        :param eta: 学习率，主要用于防止使用梯度时造成跳过最优的情况，实际中可以把学习率设置为随时间减少，更符合实际
        :param n_iter: 迭代次数
        :param shuffle: 主要用于是否打乱原始数据,shuffle的意思是洗牌
        :param random_state: 看是否有要求需要抽取样本进行小批量训练
        """
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)



    def fit(self,X,y):
        self._initial_weights(X.shape[1])
        self.cost_ = []

        for i in range(1,self.n_iter+1):
            # 洗牌
            if self.shuffle:
                X,y = self._shuffle(X,y)

            cost = []
            for xi,target in zip (X,y) :
                cost.append(self._update_weights(xi,target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self


    def partial_fit(self,X,y):
        if not self.w_initialized:
            self._initial_weights(X.shape[1])

        if y.ravel().shape[0]>1:
            for xi,target in zip(X,y):
                self._update_weights(xi,target)
        else:
            self._update_weights(X,y)
        return self

    def _initial_weights(self,m):
        self.w_initialized = True
        self.w_ = np.zeros(1 + m)

    def _shuffle(self,X,y):
        r = np.random.permutation(np.unique(y))
        return X[r],y[r]

    def _update_weights(self,xi,target):
        output = self.net_input(xi)
        error = target - output
        # 调节权重
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        # 计算代价
        cost = 0.5*error**2
        return cost


    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def activation(self,X):
        return self.net_input(X)

    def predict(self,X):
        return np.where(self.activation(X)>=0.0,1,-1)

# --------用于二维数据集决策边界可视化的函数-------------------
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

    # 画决策边界
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    """
        这个meshgrid的参数应该可以很多,这里只有两个的原因在于我们只研究两个维度的特征
        这个meshgrid返回的xx1和xx2的分析：
        一般xx1和xx2的特征是：1，xx1的每行相同，xx2的每列相同，xx1存储的是描的点的在x1轴的坐标，...
                            2,xx1和xx2都是存储所有点的信息，维度是相同的

    """
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution)
                           , np.arange(x2_min, x2_max, resolution))
    # ravel()对数组进行扁平化
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # 这里应该是把前面得到的列向量转为行向量
    z = z.reshape(xx1.shape)
    # 调用matplotlib的contourf函数
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    # plot class sample
    # idx 应该是np.unique(y)后对每个y的索引
    # y==c1返回的是和y维度相同的一个boolean型矩阵，然后方便取值
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y=X[y == c1, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=c1)

# --------------------获取测试数据------------------------
df = pd.read_csv(r'F:\中软国际大数据工作\个人笔记\python\python_测试数据\data1.csv')
'''
#-----------------------提取类标，数据可视化----------------------
    #从df中提取前100个类标，其中包含50个山鸢尾类和50个变色鸢尾类，定义规则：1代表变色鸢尾，-1代表山鸢尾，同时把产生的整数类标赋值给Numpy的向量y
    #提取数据的第一列和第三列赋值给属性矩阵X,然后使用二维散点图对这些数据进行可视化
    #将DataFrame提取出来的转化为数组直接在后面.values就好
    #山鸢尾（Iris-setosa) 、变色鸢尾（Iris-virginica)
'''

# 把特征矩阵x弄出来
X = df.iloc[:100,[0,2]].values
# 标签矩阵y
y = df.iloc[:100,4].values
y = np.where(y == 'Iris-setosa',-1,1)

#----------------------数据标准化----------------------
# 转化为符合(0,1)的正态分布
X_std = np.copy(X)
X_std[:,0] = (X_std[:,0] - X_std[:,0].mean()) / X_std[:,0].std()
X_std[:,1] = (X_std[:,1] - X_std[:,1].mean()) / X_std[:,1].std()

ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std,y)
plot_decision_regions(X_std,y,classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [Standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1,len(ada.cost_)+1),ada.cost_
         ,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()