# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from pyspark import SparkContext
from pyspark.sql import SparkSession


# %%
spark = SparkSession.builder.master("local").appName("DataProcess").config("spark.executor.memory","3g").config("spark.executor.instances","5").getOrCreate()


# %%
#导入user_log数据
user_log = spark.read.csv(r"hdfs://node1:9000/user/root/exp4/data_format1/user_log_format1.csv", encoding='utf8', header=True, inferSchema=True)


# %%
user_log.orderBy("user_id").limit(10).show()


# %%
#导入训练集
df_train = spark.read.csv(r"hdfs://node1:9000/user/root/exp4/data_format1/train_format1.csv", encoding='utf8', header=True, inferSchema=True)


# %%
df_train.limit(10).show()


# %%
#导入用户信息
user_info = spark.read.csv(r"hdfs://node1:9000/user/root/exp4/data_format1/user_info_format1.csv", encoding='utf8', header=True, inferSchema=True)


# %%
user_info.limit(10).show()

# %% [markdown]
# 想要建立的特征
# 需要根据user_id，和merchant_id（seller_id）,从用户画像表以及用户日志表中提取特征，填写到df_train这个数据框中，从而训练评估模型 需要建立的特征如下：
# 
# 用户的年龄(age_range)  
# 用户的性别(gender)  
# 某用户在该商家日志的总条数(total_logs)  
# 用户浏览的商品的数目，就是浏览了多少个商品(unique_item_ids)  
# 浏览的商品的种类的数目，就是浏览了多少种商品(categories)  
# 用户浏览的天数(browse_days)  
# 用户单击的次数(one_clicks)  
# 用户添加购物车的次数(shopping_carts)  
# 用户购买的次数(purchase_times)  
# 用户收藏的次数(favourite_times)  

# %%
#age_range,gender特征添加
df_train = df_train.join(user_info,["user_id"],"left")


# %%
#total_logs(某用户在该商家日志的总条数)特征添加
total_logs_temp = user_log.groupby(["user_id","seller_id"]).count()
total_logs_temp.orderBy("user_id").limit(20).show()


# %%
total_logs_temp = total_logs_temp.withColumnRenamed("seller_id","merchant_id").withColumnRenamed("count","total_logs")
total_logs_temp.limit(1).show()


# %%
df_train = df_train.join(total_logs_temp,["user_id","merchant_id"],"left")
df_train.limit(10).show()


# %%
#unique_item_ids特征添加
unique_item_ids_temp = user_log.groupby(["user_id","seller_id","item_id"]).count()
unique_item_ids_temp = unique_item_ids_temp.selectExpr("user_id","seller_id","item_id")
unique_item_ids_temp.show()


# %%
unique_item_ids_temp = unique_item_ids_temp.groupBy(["user_id","seller_id"]).count()
unique_item_ids_temp = unique_item_ids_temp.withColumnRenamed("seller_id","merchant_id").withColumnRenamed("count","unique_item_ids")
unique_item_ids_temp.limit(10).show()


# %%
df_train = df_train.join(unique_item_ids_temp,["user_id","merchant_id"],"left")
df_train.limit(10).show()


# %%
#categories特征构建
categories_temp = user_log.groupby(["user_id", "seller_id", "cat_id"]).count()
#categories_temp.show()


# %%
categories_temp = categories_temp.selectExpr("user_id","seller_id","cat_id")
#categories_temp.show()


# %%
categories_temp = categories_temp.groupBy(["user_id","seller_id"]).count()
categories_temp = categories_temp.withColumnRenamed("seller_id","merchant_id").withColumnRenamed("count","categories")
#categories_temp.limit(10).show()


# %%
df_train = df_train.join(categories_temp,["user_id","merchant_id"],"left")
df_train.limit(10).show()


# %%
#browse_days特征构建
browse_days_temp = user_log.groupby(["user_id","seller_id","time_stamp"]).count()
#browse_days_temp.limit(10).show()


# %%
browse_days_temp = browse_days_temp.selectExpr("user_id","seller_id","time_stamp")
browse_days_temp = browse_days_temp.groupby(["user_id","seller_id"]).count()
browse_days_temp = browse_days_temp.withColumnRenamed("seller_id","merchant_id").withColumnRenamed("count","browse_days")
#browse_days_temp.limit(10).show()


# %%
df_train = df_train.join(browse_days_temp,["user_id","merchant_id"],"left")
df_train.limit(10).show()


# %%
#先将处理好的暂时写到文件中
df_train.write.options(header="true").csv("hdfs://node1:9000/user/root/exp4/procd_train_temp.csv", mode = 'overwrite')


# %%
#为了避免jvm崩掉，只能另起一个session
spark.stop()
spark = SparkSession        .builder        .master("local")        .appName("DataProcess2")        .config("spark.executor.memory","3g")        .config("spark.executor.instances","5")        .getOrCreate()


# %%
#导入user_log数据
user_log = spark.read.csv(r"hdfs://node1:9000/user/root/exp4/data_format1/user_log_format1.csv", encoding='utf8', header=True, inferSchema=True)


# %%
#one_clicks、shopping_carts、purchase_times、favourite_times特征构建
one_clicks_temp = user_log.groupby(["user_id","seller_id","action_type"]).count()
one_clicks_temp = one_clicks_temp.withColumnRenamed("seller_id","merchant_id").withColumnRenamed("count","times")
#one_clicks_temp.limit(10).show()


# %%
from pyspark.sql import functions
from pyspark.sql.types import *
def click_time(action,times):
    if action == 0:
        return 0
    else:
        return times
udf_click_time = functions.udf(click_time,IntegerType())
one_clicks_temp = one_clicks_temp.withColumn("one_clicks",udf_click_time("action_type","times"))
#one_clicks_temp.limit(10).show()


# %%
def shopping_click(action,times):
    if action == 1:
        return times
    else:
        return 0
udf_click_time = functions.udf(shopping_click,IntegerType())
one_clicks_temp = one_clicks_temp.withColumn("shopping_carts",udf_click_time("action_type","times"))
#one_clicks_temp.limit(10).show()


# %%
def purchase_click(action,times):
    if action == 2:
        return times
    else:
        return 0
udf_click_time = functions.udf(purchase_click,IntegerType())
one_clicks_temp = one_clicks_temp.withColumn("purchase_times",udf_click_time("action_type","times"))
#one_clicks_temp.limit(10).show()


# %%
def favor_click(action,times):
    if action == 3:
        return times
    else:
        return 0
udf_click_time = functions.udf(favor_click,IntegerType())
one_clicks_temp = one_clicks_temp.withColumn("favor_times",udf_click_time("action_type","times"))
#one_clicks_temp.limit(10).show()


# %%
four_features = one_clicks_temp.groupby(["user_id","merchant_id"]).sum()
#four_features.limit(10).show()


# %%
four_features = four_features.selectExpr("user_id","merchant_id","`sum(one_clicks)` as one_clicks",
"`sum(shopping_carts)` as shopping_carts","`sum(purchase_times)` as purchase_times","`sum(favor_times)` as favor_times")
four_features.limit(10).show()


# %%
#导入train_data数据
df_train = spark.read.csv(r"hdfs://node1:9000/user/root/exp4/procd_train_temp.csv", encoding='utf8', header=True, inferSchema=True)


# %%
df_train = df_train.join(four_features,["user_id","merchant_id"],"left")
df_train.limit(10).show()


# %%
df_train.write.options(header="true").csv("hdfs://node1:9000/user/root/exp4/procd_train_real.csv")
df_train.write.parquet("hdfs://node1:9000/user/root/exp4/procd_train_real.parquet")


# %%
#填充缺失值
#第一种策略是将后8个特征所有null值填充为0
df_train_filled = df_train.fillna(0)
df_train_filled.show()


# %%
#将数据转为合适的格式
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
#先转成RDD
df_train_rdd = df_train_filled.rdd
#改成(label,features)的格式
df_train_rdd = df_train_rdd.map(lambda line: LabeledPoint(line[2],Vectors.dense(line[3:])))


# %%
#保存为LibSVMFile格式，方便后面训练使用
from pyspark.mllib.util import MLUtils
MLUtils.saveAsLibSVMFile(df_train_rdd, "hdfs://node1:9000/user/root/exp4/procd_train_real")


# %%
#别忘了关掉session
spark.stop()


