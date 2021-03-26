"""
 -*- coding:utf-8 -*-
@Time : 2020/8/2 20:07
@Author: 面向对象写Bug
@File : knn_demo.py 
@content：使用knn算法对数据进行分类
"""

# --------------数据处理、准备--------------
from sklearn import datasets
from sklearn.model_selection import train_test_split
# 数据
iris = datasets.load_iris()
y = iris.target
X = iris.data[:,[2,3]]

# -----------------------将数据分割为测试数据和训练数据------------------------
# 要求：将训练数据和测试数据按照7：3的比例划分,共150组数据
# from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y,
                 test_size=0.3,random_state= 0)

# -----------------------数据的标准化处理---------------------------------
from sklearn.preprocessing import StandardScaler
import numpy as np
# 实例化对象
sc = StandardScaler()
# Compute the mean and std to be used for later scaling.(计算均值和方差方便以后的缩放）
sc.fit(X,y)
# 对数据进行标准化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std =np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))

# ---------------------可选择对测试数据增加高亮效果的决策边界函数-----------------------
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
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

# --------------------构建knn模型----------------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5
                           ,metric='minkowski' # 闵可夫斯基距离
                           ,p=2
                           )
knn.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,y_combined
                      ,classifier=knn,test_idx=range(105,150))
plt.xlabel('petal length [Standardized]')
plt.ylabel('petal width [Standardized]')
plt.legend(loc = 'upper left')
plt.show()