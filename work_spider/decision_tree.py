"""
Decision tree:决策树
决策树的划分依据（其中之一）：信息增益 --当得知一个信息之后，使不确定性减少
ID3
信息增益 最大的准则
C4.5
信息增益比 最大的准则
CART
分类树: 基尼系数 最小的准则 在sklearn中可以选择划分的默认原则
优势：划分更加细致（从后面例子的树显示来理解）

Random forest:随机森林  --包含了多个决策树
Shannon's theorem:香农定理
"""
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
import pandas as pd


def decision_tree():
    titan = pd.read_csv("./data_set/titanic_train.csv")
    x = titan[['Pclass', 'Age', 'Sex']]
    y = titan['Survived']
    # 缺失值处理
    x["Age"].fillna(x["Age"].mean(), inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    dic = DictVectorizer(sparse=False)
    x_train = dic.fit_transform(x_train.to_dict(orient="records"))
    x_test = dic.transform(x_test.to_dict(orient="records"))
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(x_train, y_train)
    # 导出决策树结构
    exp = export_graphviz(dt, out_file="./data_set/decision_tree_map.dot",
                          feature_names=['年龄', 'Pclass=1st', 'Sex=female', 'Sex=male'])
    print("预测的准确率", dt.score(x_test, y_test))
    return None


if __name__ == '__main__':
    decision_tree()