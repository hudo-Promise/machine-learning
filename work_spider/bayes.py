"""
找出样本属于每一个类别的概率，找出其中最大的概率
朴素贝叶斯（条件之间相互独立）

精准率 accurate
召回率 recall：预测结果为正例
混淆矩阵 confusion matrix
"""
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


def simple_bayes():
    # 加载数据
    news = fetch_20newsgroups(subset='all')
    # 数据集分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)
    # 特征抽取
    tf = TfidfVectorizer()
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)
    # 开始训练
    mlt = MultinomialNB(alpha=1.0)
    mlt.fit(x_train, y_train)
    y_predict = mlt.predict(x_test)
    score = mlt.score(x_test, y_test)

    print("预测文章类别：", y_predict)
    print("预测准确率：", score)
    print("精确率和召回率：\n", classification_report(y_test, y_predict, target_names=news.target_names))
    return None


if __name__ == '__main__':
    simple_bayes()  # 擅长文本分类
