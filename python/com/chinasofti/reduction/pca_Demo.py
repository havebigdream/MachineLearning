"""
 -*- coding:utf-8 -*-
@Time : 2020/8/12 1:18
@Author: 面向对象写Bug
@File : author.py 
@content：使用PCA技术进行降维
"""

#获取数据
import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
#df_wine = pd.read_csv(r'F:\MachineLearning\python\com\chinasofti\Data\wine.csv',header=None)
# 本地化
#df_wine.to_csv(r"F:\MachineLearning\python\com\chinasofti\Data\wine.csv",index=None,header=None)
#将数据划分为训练集和测试集--分别占数据的70%和30%，并使用单位方差来使其标准化
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X,y = df_wine.iloc[:,1:], df_wine.iloc[:,0]
X_train,X_test,y_train,y_test=train_test_split(X,y,
                 test_size=0.3, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

# 为了得到主成分，需要得到特征数据之间的协方差的协方差矩阵，他们的特征向量就是主成分
# 葡萄酒数据共有13个特征，得到的协方差矩阵应该是13*13的，手工计算特征值和特征向量有点
import numpy as np
cov_mat = np.cov(X_train_std.T) # 需要转置一下，不然就是搞得样本的协方差了
engen_vals, engen_vecs = np.linalg.eig(cov_mat)
print('\nEigevalues\n%s'% engen_vals)

# 特征值大的代表主成分的贡献大，所以我们需要将特征值降序排列，找到排在前k个特征值对应的特征向量即可
# 先绘制特征值的方差贡献率图像,单个特征值的贡献率是他的特征值与所有特质值的和的比值
import matplotlib.pyplot as plt
tot = sum(engen_vals)
var_exp = [(i / tot) for i in sorted(engen_vals,reverse=True)] # 默认返回升序序列，reverse=True实现降序
cum_var_exp = np.cumsum(var_exp)

# x_sup = range(1,14)   #
plt.bar(range(1,14),var_exp,align='center',
        label='individual explained variance')
plt.step(range(1,14),cum_var_exp,where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal Components')
plt.legend(loc='best')
plt.show()

# 按特征值的排序排列特征对
engen_pairs = [(np.abs(engen_vals[i]),engen_vecs[:,i])
               for i in range(len(engen_vals))]
engen_pairs.sort(reverse=True)

w = np.hstack((engen_pairs[0][1][:,np.newaxis],
               engen_pairs[1][1][:,np.newaxis])) # 后面的[:,newaxis]是将数据行向量映射到新系中了，变为两列
print('Matrix W:\n',w)

# 现在w变成了一个2X13维矩阵，从而生产了一个映射矩阵，这样X_train_std*w就可成功转化为样本量*2维矩阵
X_train_pca = X_train_std.dot(w)

# 可视化，简单描点，看看效果
colors = np.array(['!',
                   '#FF3333',  # red
                   '#0198E1',  # blue
                   '#BF5FFF',  # purple
                   '#FCD116',  # yellow
                   '#FF7216',  # orange
                   '#4DBD33',  # green
                   '#87421F'   # brown
                   ])
markers = ['s','x','o','^']
for l, c, m in zip(np.unique(y_train), colors[1:len(np.unique(y_train))+1], markers):
    plt.scatter(X_train_pca[y_train==l,0],
                X_train_pca[y_train==l,1],
                c=c,label=l,marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='best')
plt.show()

# 使用sci-learn进行PCA降维
# --------用于二维数据集决策边界可视化的函数-------------------
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    """

    :param X: 属性矩阵
    :param y: label矩阵
    :param classifier:
    :param resolution:
    """
    # 建一个颜色、标记和cmap发生器
    colors = np.array([
                       '#FF3333',  # red
                       '#0198E1',  # blue
                       '#BF5FFF',  # purple
                       '#FCD116',  # yellow
                       '#FF7216',  # orange
                       '#4DBD33',  # green
                       '#87421F'  # brown
                       ])
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
                    alpha=0.8, color=cmap(idx),
                    marker=markers[idx], label=c1)

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
lr = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std) # 这里就不是fit_transform了
lr.fit(X_train_pca,y_train)
plot_decision_regions(X_train_pca,y_train,classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()

# 测试数据分类情况
plot_decision_regions(X_test_pca,y_test,classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()



# 对比图,把训练数据和测试数据放在一起
def plot_decision_regions_2D(ax,X, y, classifier, resolution=0.02):
    """

    :param X: 属性矩阵
    :param y: label矩阵
    :param classifier:
    :param resolution:
    """
    # 建一个颜色、标记和cmap发生器
    colors = np.array([
                       '#FF3333',  # red
                       '#0198E1',  # blue
                       '#BF5FFF',  # purple
                       '#FCD116',  # yellow
                       '#FF7216',  # orange
                       '#4DBD33',  # green
                       '#87421F'  # brown
                       ])
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
    ax.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)

    for idx, c1 in enumerate(np.unique(y)):
        ax.scatter(x=X[y == c1, 0], y=X[y == c1, 1],
                    alpha=0.8, color=cmap(idx),
                    marker=markers[idx], label=c1)

_, (ax1, ax2) = plt.subplots(1, 2, sharex='row', sharey='row', figsize=(8, 4))
plt.subplots_adjust(bottom=.15)

plot_decision_regions_2D(ax1,X_train_pca,y_train,classifier=lr)
ax1.set_title('Train data status')
ax1.set_ylabel('PC 2')
ax1.set_xlabel('PC 1')


plot_decision_regions_2D(ax2,X_test_pca,y_test,classifier=lr)
ax2.set_ylabel('PC 2')
ax2.set_xlabel('PC 1')
ax2.set_title('Test Data Status')
plt.show()