"""
逻辑回归可以将线性问题转换为分类问题
逻辑回归 --分类（二分类）: logistic regression
sigmoid函数：
损失函数：
均方误差：损失函数只有一个最小值
对数自然损失：有多个局部最小值
            解决方式：1 多次随机初始化，获取多个最低点，通过比较得到最低点
                    2 调整学习率
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def logistic():
    columns = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size",
               "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size",
               "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
    data = pd.read_csv('./data_set/breast-cancer-wisconsin.data', header=None)
    data.columns = columns
    # 缺失值处理
    data = data.replace(to_replace='?', value=np.nan)
    data = data.dropna()
    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(data[columns[1:10]], data[columns[10]], test_size=0.25)
    # 标准化处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    # 逻辑回归预测
    lg = LogisticRegression(C=1.0)
    lg.fit(x_train, y_train)
    print(lg.coef_)
    y_predict = lg.predict(x_test)
    print("准确率", lg.score(x_test, y_test))
    print("召回率", classification_report(y_test, y_predict, labels=[2, 4], target_names=['良性',  '恶性']))

    return None


if __name__ == '__main__':
    logistic()
