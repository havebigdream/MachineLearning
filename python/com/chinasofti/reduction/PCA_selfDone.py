"""
 -*- coding:utf-8 -*-
@Time : 2020/8/12 1:18
@Author: 面向对象写Bug
@File : author.py 
@content：写一个用PCA降维的类
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pandas as pd

class PCA_selfDone():
    def __init__(self):
        self.n_components = 2
        self.classifier = LogisticRegression()
        self.pca = PCA(2)


    def fit(self,X,y):
        self.__preProcessing__(X,y)
        self._X_train_pca = self.pca.fit_transform(self._X_train_std)
        self._X_test_pca = self.pca.transform(self._X_test_std)
        self.classifier.fit(self._X_train_pca,self._y_train)
        self.__visualization__()


    def __visualization__(self):
        self.__attribute_ratio_visualization__()
        self.__scatter_plot__()
        _, (ax1, ax2) = plt.subplots(1, 2, sharex='row', sharey='row', figsize=(8, 4))
        plt.subplots_adjust(bottom=.15)
        self.__plot_decision_regions_2D__(ax1, self._X_train_pca, self._y_train)
        ax1.set_title('Train data status')
        ax1.set_ylabel('PC 2')
        ax1.set_xlabel('PC 1')

        self.__plot_decision_regions_2D__(ax2, self._X_test_pca, self._y_test)
        ax2.set_ylabel('PC 2')
        ax2.set_xlabel('PC 1')
        ax2.set_title('Test Data Status')
        plt.show()



    def __preProcessing__(self,X,y):
        '''

        Parameters
        ----------
        X : 未标准化的X
        y : 未标准化的y

        -------

        '''
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.3, random_state=0)
        # 标准化
        sc = StandardScaler()
        self._y_train = y_train
        self._y_test = y_test
        X_train_std = sc.fit_transform(X_train)
        X_test_std = sc.fit_transform(X_test)

        # 特征值和特征向量
        cov_mat = np.cov(X_train_std.T)  # 需要转置一下，不然就是搞得样本的协方差了
        engen_vals, engen_vecs = np.linalg.eig(cov_mat)
        self._X_train_std = X_train_std
        self._X_test_std = X_test_std
        self._engen_vals = engen_vals
        self._engen_vecs = engen_vecs

        # 将特征向量和特质值对按特征值顺序排列
        # engen_pairs = [(np.abs(self._engen_vals[i]), self._engen_vecs[:, i])
        #                for i in range(len(self._engen_vals))]
        # engen_pairs.sort(reverse=True)
        # if self.n_components > np.ndim(X,axis=1):
        #     raise IndexError("参数超过范围")
        # # 获取降维需要的转换矩阵
        # else:
        #     for i in range(1,self.n_components):
        #         w = np.hstack((engen_pairs[0][1][:, np.newaxis],
        #                 engen_pairs[i][1][:, np.newaxis]))  # 后面的[:,newaxis]是将数据行向量映射到新系中了
        # X_train_pca = X_test_std.dot(w)
        # X_test_pca = X_test_std.dot(w)
        # self._X_train_pca = X_train_pca
        # self._X_test_pca = X_test_pca
        return self

    def __attribute_ratio_visualization__(self):
        '''

        Returns 画出特征的累计贡献图
        -------

        '''
        tot = sum(self._engen_vals)
        var_exp = [(i / tot) for i in sorted(self._engen_vals, reverse=True)]  # 默认返回升序序列，reverse=True实现降序
        cum_var_exp = np.cumsum(var_exp)
        plt.bar(range(1,len(self._engen_vals)+1), var_exp, align='center',
                label='individual explained variance')
        plt.step(range(1, len(self._engen_vals)+1), cum_var_exp, where='mid',
                 label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal Components')
        plt.legend(loc='best')
        plt.show()
        pass

    def __scatter_plot__(self):
        # 只提供两维的逻辑
        colors = np.array(['!',
                           '#FF3333',  # red
                           '#0198E1',  # blue
                           '#BF5FFF',  # purple
                           '#FCD116',  # yellow
                           '#FF7216',  # orange
                           '#4DBD33',  # green
                           '#87421F'  # brown
                           ])
        markers = ['s', 'x', 'o', '^']
        for l, c, m in zip(np.unique(self._y_train), colors[1:len(np.unique(self._y_train)) + 1], markers[1:len(np.unique(self._y_train)) + 1]):
            plt.scatter(self._X_train_pca[self._y_train == l, 0],
                        self._X_train_pca[self._y_train == l, 1],
                        c=c, label=l, marker=m)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.legend(loc='best')
        plt.show()

    def __plot_decision_regions_2D__(self,ax,X, y, resolution=0.02):
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
        z = self.classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
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


df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
X,y = df_wine.iloc[:,1:], df_wine.iloc[:,0]
pca_instance = PCA_selfDone()
pca_instance.fit(X,y)