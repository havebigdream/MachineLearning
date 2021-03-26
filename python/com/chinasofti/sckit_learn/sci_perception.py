"""
 -*- coding:utf-8 -*-
@Time : 2020/7/30 15:12
@Author: 面向对象写Bug
@File : sci_perception.py 
@content：使用scikit训练一个感知器模型
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# ---------------------可选择对测试数据增加高亮效果的决策边界函数-----------------------
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

# -------------------------提取数据---------------------------
# 提取鸢尾花数据的花瓣长度和花瓣宽度两个特征的值，并由此构建特征矩阵X，同时将所属类型的类标赋值给y
# 由于鸢尾花的数据比较出名，所里直接可以利用他们的api拿到，
# print(np.unique(y)，结果为0，1，2，他们已经将taget列的(Iris-Sentosa,Iris-Versicolor,Iris-Virginia)映射到（0，1，2）上了
# from sklearn import datasets

iris = datasets.load_iris()
y = iris.target
X = iris.data[:,[2,3]]

# -----------------------将数据分割为测试数据和训练数据------------------------
# 要求：将训练数据和测试数据按照7：3的比例划分,共150组数据
# from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y,
                 test_size=0.3,random_state= 0)

# -----------------------数据的标准化处理---------------------------------
# from sklearn.preprocessing import StandardScaler
# 实例化对象
sc = StandardScaler()
# Compute the mean and std to be used for later scaling.(计算均值和方差方便以后的缩放）
sc.fit(X,y)
# 对数据进行标准化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# ---------------------训练感知器（使用线性逻辑）---------------------------
# from sklearn.linear_model import Perceptron
# ---------random_state--------------
'''
random_state:洗牌时要使用的伪随机数生成器的种子

数据。如果是int，random_state是随机数使用的种子

生成器；如果是RandomState实例，random_state是随机数

生成器；如果没有，则随机数生成器为RandomState

实例使用者`np.随机`.
'''
ppn = Perceptron(eta0=0.1,n_iter_no_change=40,random_state=0)
# 训练数据
ppn.fit(X_train_std,y_train)

# 对测试数据集进行预测,返回数组类型
y_pred = ppn.predict(X_test_std)

# --------------------------评价模型结果----------------
# 1，----------------误判数量---------------
# Number of missclassification
Num = (y_pred != y_test).sum()
# print(type(y_pred))
print('Missclassified Sample:%d'%Num)

# 2，----------------使用metrics模块得出误判率或准确率------------------
#from sklearn.metrics import accuracy_score
print('Accuracy:%.2f'%(accuracy_score(y_test,y_pred)))



# ----------------------边界决策函数--------------------------------------
X_combined_std =np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))
plot_decision_regions(X=X_combined_std,y=y_combined
                      ,classifier=ppn,test_idx=range(105,150))
plt.xlabel('petal length [Standardized]')
plt.ylabel('sepal length [Standardized]')
plt.title('sklearn Perceptron')
plt.legend(loc='upper left')
plt.show()


#-------------------------使用scikit-learn训练logistic回归模型--------------------

from sklearn.linear_model import logistic
# 初始化模型
lr = logistic.LogisticRegression(C=1000.0,random_state=0)
# 训练数据
lr.fit(X_train_std,y_train)
# 画决策图
plot_decision_regions(X_combined_std,y_combined,
                      classifier=lr,test_idx=range(105,150))
plt.xlabel('length of petal [Standardized]')
plt.ylabel('width of petal [Standardized]')
plt.legend(loc = 'upper left')
plt.title('logistic Regression')
plt.show()

# 如何使用logis模型来判断样本属于哪个类别
a = lr.predict_proba(X_test_std)
print(a[0,:])
print(lr.score(X_combined_std,y_combined))

# --------logistic回归参数C,用于防止过拟合问题-------------
# 偏差-方差权衡（bias-variance tradeoff)就是通过正则化调整模型的复杂度，正则化是解决共线性问题的(特征间高度相关)的一个很有用的办法，
# 他可以过滤掉数据中的噪音,并最终防止过拟合。正则化的背后就是通过引入额外的信息（偏差）来对极端参数的权重做出惩罚，最为常见的正则化的形式
# 称为L2正则化，他有时也称作L2收缩 或者 权重缩减
# 具体的推倒需要结合代价函数于C的关系，大致的意思是减少C会增加正则化（额外信息）的影响，可能对权重惩罚的就更多
# 可视化C与params的关系

weights,params = [],[]
for c in np.arange(-5,5,dtype=float):
    lr = logistic.LogisticRegression(C=10**c,random_state=0)
    lr.fit(X_train_std,y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
plt.plot(params,weights[:,0]
         ,label='petal length')
plt.plot(params,weights[:,1]
         ,label='petal width'
         ,linestyle='--')
plt.xlabel('C')
plt.ylabel('weight coefficient')
plt.legend(loc = 'upper left')
# x轴坐标缩放
plt.xscale('log')
plt.show()









