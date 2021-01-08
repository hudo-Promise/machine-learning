import pandas as pd
from sklearn.decomposition import PCA
'''
主成分分析 降维
'''
# 1、获取数据集
# ·商品信息- products.csv：
# Fields：product_id, product_name, aisle_id, department_id
# ·订单与商品信息- order_products__prior.csv：
# Fields：order_id, product_id, add_to_cart_order, reordered
# ·用户的订单信息- orders.csv：
# Fields：order_id, user_id,eval_set, order_number,order_dow, order_hour_of_day, days_since_prior_order
# ·商品所属具体物品类别- aisles.csv：
# Fields：aisle_id, aisle
products = pd.read_csv("./instacart/products.csv")
order_products = pd.read_csv("./instacart/order_products__prior.csv")
orders = pd.read_csv("./instacart/orders.csv")
aisles = pd.read_csv("./instacart/aisles.csv")

# 2、合并表，将user_id和aisle放在一张表上
# 1）合并orders和order_products on=order_id tab1:order_id, product_id, user_id
tab1 = pd.merge(orders, order_products, on=["order_id", "order_id"])
# 2）合并tab1和products on=product_id tab2:aisle_id
tab2 = pd.merge(tab1, products, on=["product_id", "product_id"])
# 3）合并tab2和aisles on=aisle_id tab3:user_id, aisle
tab3 = pd.merge(tab2, aisles, on=["aisle_id", "aisle_id"])
# 3、交叉表处理，把user_id和aisle进行分组
table = pd.crosstab(tab3["user_id"], tab3["aisle"])

# 4、主成分分析的方法进行降维
# 1）实例化一个转换器类PCA
transfer = PCA(n_components=0.95)
# 2）fit_transform
data = transfer.fit_transform(table)
print(data.shape())

'''
1. 导入数据
2. 数据处理
3. 特征工程（特征处理 特征标准化（归一化）特征选择）
4. 算法选择 （模型 = 算法 + 数据  经过数据训练的算法）
'''