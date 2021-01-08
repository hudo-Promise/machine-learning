"""
Random forest:随机森林  --包含了多个决策树
在n个样本中随机抽取一个样本 重复n次
随机在m个特征中选出
应对数据量大 特征多的数据 --通常使用最多
在当前所有算法中，具有极好的准确率
能够有效地运行在大数据集上，处理具有高维特征的输入样本，而且不需要降维
能够评估各个特征在分类问题上的重要性
"""
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
import pandas as pd


def random_forest():
    titan = pd.read_csv("./data_set/titanic_train.csv")
    x = titan[['Pclass', 'Age', 'Sex']]
    y = titan['Survived']
    # 缺失值处理
    x["Age"].fillna(x["Age"].mean(), inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    dic = DictVectorizer(sparse=False)
    x_train = dic.fit_transform(x_train.to_dict(orient="records"))
    x_test = dic.transform(x_test.to_dict(orient="records"))
    rf = RandomForestClassifier()
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200],
             "max_depth": [5, 8, 15, 25, 30]}
    gc = GridSearchCV(rf, param_grid=param, cv=2)
    gc.fit(x_train, y_train)
    print("准确率:  \n", gc.score(x_test, y_test))
    print("查看选择的参数模型: \n", gc.best_params_)
    return None


if __name__ == '__main__':
    random_forest()
