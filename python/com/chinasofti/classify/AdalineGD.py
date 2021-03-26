"""
 -*- coding:utf-8 -*-
@Time : 2020/7/29 16:17
@Author: 面向对象写Bug
@File : AdalineGD.py 
@conttent : 使用python实现自适应线性神经元
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

# -------------------自适应神经元训练器----------------------------
class AdalineGD(object):

    def __init__(self,eta=0.01,n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self,X,y):
        """
        :param X: 假定m×n维矩阵（m个样本，n个维度）
        :param y: 假定m×1维矩阵（label矩阵）
        :content: 主要用于对数据进行训练，如果有特殊的要求，self.w_可以放在__init__中，然后按要求初始化
        self.cost_ : 用于描述代价，主要用于对此时训练结果的评价
        self.w_: 这里应该是 1×（n+1）维矩阵
        """
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []

        # 循环多次对数据进行训练
        for i in range(1,self.n_iter+1):
            #首先计算输出值和误差，output、errors应该是m×1维矩阵
            output = self.net_input(X)
            errors = y - output
            # 根据errors反向调节权重矩阵w_
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0]  += self.eta*errors.sum()

            # 计算代价函数的值，用于评价模型好坏
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self

    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def activation(self,X):
        return self.net_input(X)

    def predict(self,X):
        return np.where(self.activation(X) >=0 , 1 , -1)


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
# 画散点图看一看萼片长度（原表第一个特征列）和花瓣长度的关系（第三个特征列）的关系
plt.scatter(X[:50,0],X[:50,1],
           color='red',marker='o',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],
           color='blue',marker='x',label='virgincia')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

# ------------比较不同学习率情况下，代价函数和迭代次数的图像----------------
fig, ax = plt.subplots(1,2,figsize=(8,4))
# 学习率为0.01
ada1 = AdalineGD(eta=0.01,n_iter=10).fit(X,y)
ax[0].plot(range(1,ada1.n_iter+1),np.log(ada1.cost_)
           ,marker = 'o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(sum-squared-error)')
ax[0].set_title('Adaline - learning rate 0.01')

# 学习率为0.0001
ada2 = AdalineGD(eta=0.0001,n_iter=10).fit(X,y)
ax[1].plot(range(1,ada2.n_iter+1),ada2.cost_
           ,marker = 'o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('sum-squared-error')
ax[1].set_title('Adaline - learning rate 0.0001')
plt.show()

#----------------------数据标准化----------------------
# 转化为符合(0,1)的正态分布
X_std = np.copy(X)
X_std[:,0] = (X_std[:,0] - X_std[:,0].mean()) / X_std[:,0].std()
X_std[:,1] = (X_std[:,1] - X_std[:,1].mean()) / X_std[:,1].std()

# 进行标准化操作后，我们继续以学习速率为0.01再次对Adaline进行训练，看看它是否是收敛的
ada = AdalineGD(eta=0.01, n_iter=15)
ada.fit(X_std,y)
# 决策分界图
plot_decision_regions(X_std,y,classifier=ada)
plt.title('Adaline - Gradient descent')
plt.xlabel('sepal length [Standardized]')
plt.ylabel('petal length [Standardized]')
plt.legend(loc = 'upper left')
plt.show()
#  代价和迭代次数关系图
plt.plot(range(1,ada.n_iter+1),ada.cost_,
         marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()

