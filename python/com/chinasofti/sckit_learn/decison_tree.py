"""
 -*- coding:utf-8 -*-
@Time : 2020/8/2 18:52
@Author: 面向对象写Bug
@File : decison_tree.py 
@content：利用决策树算法对数据进行分类
"""

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

# ----------------------比较三种不纯度衡量标准------------------
# 绘制的样本属于【0，1】情况下不纯度的图像
import matplotlib.pyplot as plt
import numpy as np

# 基尼系数
def gini(p):
    return p*(1-p)+(1-p)*(1-(1-p))
# 熵
def entropy(p):
    return -p*np.log2(p) - (1-p)*np.log2((1-p))
# 误分类率
def error(p):
    return 1 - np.max([p,1-p])

x = np.arange(0.0, 1.0, 0.01)
ent = [entropy(p) if p != 0 else None for p in x]      # 这他妈的是lambda写法吗？？？
sc_ent = [e*0.5 if e else None for e in ent]          #这种简化for循环的语句可以学习
err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111)
for i,lab,ls,c, in zip([ent,sc_ent,gini(x),err]
                       ,['Entropy','Entropy(scaled)','Gini Impurity','Misclassfication Error']
                       ,['-','-','--','-.']
                       ,['black','lightgray'
                         ,'red','green','cyan']):
    line = ax.plot(x,i,label=lab,linestyle=ls,lw=2,color=c)
ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.15)
          ,ncol=3,fancybox=True,shadow=False)
ax.axhline(y=0.5,linewidth=1,color='k',linestyle='--')
ax.axhline(y=1,linewidth=1,color='k',linestyle='--')
plt.ylim(0,1.1)
plt.xlabel('p(i=1)')
plt.ylabel('Impurity index')
plt.show()

# ---------------构建决策树------------------
# 决策树可以将特征空间进行矩形划分的方式来构建复杂的决策边界，必须注意，深度越大的决策树，决策边界越复杂，
# 因而深度过深容易过拟合，构建一个深度为3的决策树，特征缩放在决策算法中不是必须的，
# 处理的数据依然是鸢尾花数据
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


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy'
                              ,max_depth=3,random_state=0)
tree.fit(X_train,y_train)
X_combined = np.vstack((X_train,X_test))
y_combined = np.hstack((y_train,y_test))
plot_decision_regions(X_combined,y_combined
                      ,classifier=tree,test_idx=(range(105,150)))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.show()

# ----------------使用随机森林来进行数据分类-------------
# 与决策树相比，随机森林是集成的多个决策树结果进行投票，具有更高的鲁棒性
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy'
                                ,n_estimators=10
                                ,random_state=1
                                ,n_jobs=2)
forest.fit(X_train,y_train)
plot_decision_regions(X_combined,y_combined
                      ,classifier=forest,test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc = 'upper left')
plt.show()