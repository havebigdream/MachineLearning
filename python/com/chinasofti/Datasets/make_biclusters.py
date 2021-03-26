"""
 -*- coding:utf-8 -*-
@Time : 2020/7/31 17:52
@Author: 面向对象写Bug
@File : make_biclusters.py 
@content：此示例演示了如何使用“光谱共聚”算法生成数据集并对其进行二聚化。
"""

'''
数据集是使用make_biclusters函数生成的，该函数创建一个较小值的矩阵并植入具有较大值的bicluster。
然后将行和列混洗，并传递给“光谱共聚”算法。重新排列经过改组的矩阵以使双簇相邻，这表明该算法找到双簇的准确度。
基本用法：sklearn.datasets.make_biclusters（shape，n_clusters，*，noise = 0.0，minval = 10，maxval = 100
                                        ，shuffle = True，random_state = None ）
'''

print(__doc__)

# Author: Kemal Eren <kemal@kemaleren.com>
# License: BSD 3 clause

import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import make_biclusters
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score

data, rows, columns = make_biclusters(
    shape=(300, 300), n_clusters=5, noise=5,
    shuffle=False, random_state=0)

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Original dataset")

# shuffle clusters
rng = np.random.RandomState(0)
row_idx = rng.permutation(data.shape[0])
col_idx = rng.permutation(data.shape[1])
data = data[row_idx][:, col_idx]

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Shuffled dataset")

model = SpectralCoclustering(n_clusters=5, random_state=0)
model.fit(data)
score = consensus_score(model.biclusters_,
                        (rows[:, row_idx], columns[:, col_idx]))

print("consensus score: {:.3f}".format(score))

fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")

plt.show()