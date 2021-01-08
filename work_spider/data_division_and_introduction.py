"""
数据集的划分：
训练数据：用于训练，构建模型
测试数据：在模型检验时使用，用于评估模型是否有效
训练  测试
70   30
75   25
80   20
"""
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split

# li = load_iris()
# # print("特征值：\n", li.data)
# # print("目标值：\n", li.target)
# # print("描述信息：\n", li.DESCR)
# x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)
# print("训练集的特征值和目标值：", x_train, y_train)
# print("测试集的特征值和目标值：", x_test, y_test)

new_li = fetch_20newsgroups(subset='all')
print("", new_li.data)
print("", new_li.target)

"""
转换器 与 估计器（预估器）
转换器：
fit_transfer : 输入数据 直接转化
fit ： 输入数据
transfer ：转化数据
fit_transfer = fit + transfer 

估计器 estimator：
1、用于分类的估计器：
sklearn.neighbors k-近邻算法
sklearn.naive_bayes 贝叶斯
sklearn.linear_model.LogisticRegression 逻辑回归
sklearn.tree 决策树与随机森林
2、用于回归的估计器：
sklearn.linear_model.LinearRegression 线性回归
sklearn.linear_model.Ridge 岭回归
3、用于无监督学习的估计器
sklearn.cluster.KMeans 聚类

第一步：调用fit
第二步：输入测试集数据

"""
