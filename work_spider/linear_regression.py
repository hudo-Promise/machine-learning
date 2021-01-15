"""
linear regression：线性回归
损失函数：--误差
最小二乘法
优化方法：梯度下降
GD：梯度下降(Gradient Descent)，原始的梯度下降法需要计算所有样本的值才能够得出梯度，
    计算量大，所以后面才有会一系列的改进。
SGD: 随机梯度下降(Stochastic gradient descent)是一个优化方法。
    它在一次迭代时只考虑一个训练样本。
SAG: 随机平均梯度法(Stochasitc Average Gradient)，由于收敛的速度太慢，
    有人提出SAG等基于梯度下降的算法

梯度下降	            正规方程
需要选择学习率	        不需要
需要迭代求解	        一次运算得出
特征数量较大可以使用	需要计算方程，时间复杂度高O(n3)

线性回归的损失函数-均方误差
线性回归的优化方法
正规方程
梯度下降
线性回归的性能衡量方法-均方误差
sklearn的SGDRegressor API 参数

过拟合(over fitting)：特征太多
欠拟合(under fitting)：特征太少
岭回归：Ridge regression
"""
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def linerReg():
    """
    线性方程预测房价
    :return:
    """
    lb = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.23)
    std_x = StandardScaler()
    std_y = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.fit_transform(x_test)
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))  # 0.19及以后版本需要传入二维数组
    y_test = std_y.fit_transform(y_test.reshape(-1, 1))
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print(lr.coef_)
    # y_predict = lr.predict(x_test)
    y_lr_predict = std_y.inverse_transform(lr.predict(x_test))
    print("测试集里面每个房子的预测价格", y_lr_predict)
    print("正规方程的均方误差为：", mean_squared_error(y_test, y_lr_predict))
    return None


def SGDReg():
    """
    (适合大量数据)
    梯度下降预测房价
    :return:
    """
    lb = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.23)
    std_x = StandardScaler()
    std_y = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.fit_transform(x_test)
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))  # 0.19及以后版本需要传入二维数组
    y_test = std_y.fit_transform(y_test.reshape(-1, 1))
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    print(sgd.coef_)
    # y_predict = lr.predict(x_test)
    y_lr_predict = std_y.inverse_transform(sgd.predict(x_test))
    print("测试集里面每个房子的预测价格", y_lr_predict)
    print("梯度下降的均方误差为：", mean_squared_error(y_test, y_lr_predict))
    return None


if __name__ == '__main__':
    linerReg()
    print("=" * 20)
    SGDReg()




