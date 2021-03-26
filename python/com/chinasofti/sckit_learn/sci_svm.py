"""
 -*- coding:utf-8 -*-
@Time : 2020/7/31 10:47
@Author: 面向对象写Bug
@File : sci_svm.py 
@content：使用核SVM算法对数据集进行分类
"""

# ---------------支持向量机的优势--------------
# 很容易使用核技巧来解决非线性可分可分问题
# 1，使用Numpy的logical_xor函数创建一个经过异或操作的数据集，
#   其中100个样本属于1，另外100个样本属于-1


# ---------------------可选择对测试数据增加高亮效果的决策边界函数-----------------------
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier
                          , test_idx = None, resolution=0.02):
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

    # plot all sample
    X_test, y_test = X[test_idx, :], y[test_idx]
    # idx 应该是np.unique(y)后对每个y的索引
    # y==c1返回的是和y维度相同的一个boolean型矩阵，然后方便取值
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y=X[y == c1, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=c1)

    # highlight test simple
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1]
                    ,facecolors='none',edgecolors='orange', alpha=1,linewidth=1,marker='o'
                    ,s=55,label='test set')
# 积累ListColormap的几个颜色
# 白青绿红蓝（'#FFFFFF','#9ff113','5fbb44','#e50b32'）

#----------数据集的生成
import numpy as np
import matplotlib.pyplot as plt
#from numpy import random

np.random.seed(0)   # 初始化generator
X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:,0] > 0,X_xor[:,1] > 0) #Compute the truth value of x1 XOR x2, element-wise.
y_xor = np.where(y_xor,1,-1)

plt.scatter(X_xor[y_xor==1,0],X_xor[y_xor==1,1]
            ,c='b',marker='x',label=1)
plt.scatter(X_xor[y_xor==-1,0],X_xor[y_xor==-1,1]
            ,c='r',marker='s',label=-1)
plt.ylim(-3,0)
plt.legend()
plt.show()

# ----------------训练一个核SVM来对“异或数据集”进行分类---------------
# 还是较好的对数据集进行了分类
# 参数gamma的设定值在这里设置为了0.1，可以理解为高斯球面的截止系数，如果
# 我们减少gamma的数值，将会增加受影响的训练样本范围，这将导致决策边界函数更加宽松
# gamma数据太大会造成过拟合
from sklearn.svm import SVC
# knernal参数选择rbf
svm = SVC(kernel='rbf',random_state=0,
          gamma=0.10,C=10.0)
svm.fit(X_xor,y_xor)
plot_decision_regions(X_xor,y_xor,classifier=svm)
plt.legend(loc='upper left')
plt.show()


