"""
 -*- coding:utf-8 -*-
@Time : 2020/7/31 18:04
@Author: 面向对象写Bug
@File : RegressionDatset.py 
@content：用于回归的数据集
"""


'''
--------------------------------------基本用法-------------------------------
sklearn.datasets.make_regression（n_samples = 100，n_features = 100，
                                *，n_informative = 10，n_targets = 1，
                                    bias = 0.0，effective_rank = None
                                    ，tail_strength = 0.5，noise = 0.0
                                    ，shuffle = True
                                    ，coef = False，random_state = None ）
Ridge回归是此示例中使用的估计量。左图中的每种颜色代表系数向量的一个不同维度，并且显示为正则化参数的函数。右图显示了解决方案的精确度。
此示例说明了如何通过Ridge回归找到定义明确的解决方案，以及正则化如何影响系数及其值。右图显示了与估算器的系数差如何随正则化而变化。

在此示例中，因变量Y设置为输入特征的函数：y = X * w + c。从正态分布中随机采样系数向量w，而将偏置项c设置为常数。

随着alpha趋于零，通过Ridge回归发现的系数趋向于向随机采样的向量w稳定。对于较大的alpha（强正则化），系数较小（最终收敛于0），
从而导致更简单且有偏差的解决方案。这些依赖关系可以在左图上观察到。

右图显示了模型找到的系数与所选向量w之间的均方误差。较少的正则化模型检索精确系数（误差等于0），较强的正则化模型会增加误差。

请注意，在此示例中，数据是无噪声的，因此可以提取精确的系数。
'''

print(__doc__)

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

clf = Ridge()

X, y, w = make_regression(n_samples=10, n_features=10, coef=True,
                          random_state=1, bias=3.5)

coefs = []
errors = []

alphas = np.logspace(-6, 6, 200)

# Train the model with different regularisation strengths
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X, y)
    coefs.append(clf.coef_)
    errors.append(mean_squared_error(clf.coef_, w))

# Display results
plt.figure(figsize=(20, 6))

plt.subplot(121)
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')

plt.subplot(122)
ax = plt.gca()
ax.plot(alphas, errors)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('error')
plt.title('Coefficient error as a function of the regularization')
plt.axis('tight')

plt.show()