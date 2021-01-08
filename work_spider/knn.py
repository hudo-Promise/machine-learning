"""
比较样本特征间的距离，相似的样本特征之间的值应该是相近的
通过欧式距离求最近距离

交叉验证 将所有数据分为n等份
网格搜索： 调参数 （每组超参数采用交叉验证的方式进行验证）
"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def knncls():
    """
    预测每个人签到位置
    :return:
    """
    data = pd.read_csv('./data_set/train.csv')

    # 1.数据处理
    # 1.1 处理 x y 取值范围
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75")
    # 1.2 处理时间的数据
    time_value = pd.to_datetime(data["time"], unit='s')
    time_value = pd.DatetimeIndex(time_value)
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday
    data.drop(['time'], axis=1)
    # 1.3 将签到数量小于n个的位置删除
    place_count = data.groupby('place_id').count()
    # 1.4 reset_index()重置索引
    tf = place_count[place_count.row_id > 3].reset_index()
    data = data[data['place_id'].isin(tf.place_id)]

    # 2 获取特征值和目标值
    y = data['place_id']
    x = data.drop(['place_id'], axis=1)
    # 2.1 分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 3 标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.fit_transform(x_test)

    # 4 开始训练

    # knn = KNeighborsClassifier(n_neighbors=5)  # n_neighbors 即k值取值，需要通过调参获得最优参数
    # knn.fit(x_train, y_train)
    # y_predict = knn.predict(x_test)
    # print("预测测试集类别：", y_predict)
    # print("预测准确率："， knn.score(x_test, y_test))

    knn = KNeighborsClassifier(n_neighbors=5)
    params = {'n_neighbors': [3, 5, 10]}
    gc = GridSearchCV(knn, param_grid=params, cv=2)
    gc.fit(x_train, y_train)
    print("在测试集上准确率：", gc.score(x_test, y_test))
    print("在交叉验证当中最好的结果：", gc.best_score_)
    print("最好的模型是：", gc.best_estimator_)
    print("每次交叉验证的结果", gc.cv_results_)
    return None


if __name__ == "__main__":
    knncls()
