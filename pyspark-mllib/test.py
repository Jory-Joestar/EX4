# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # 用训练好的模型对test_format1进行预测

# %%
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors


# %%
conf=SparkConf().setAppName("miniProject").setMaster("local").set("spark.executor.memory","3g").set("spark.executor.instances","2")
sc=SparkContext.getOrCreate(conf)


# %%
#导入测试集
spark = SparkSession.builder.master("local").appName("DataRead").getOrCreate()
test_data = spark.read.csv(r"hdfs://node1:9000/user/root/exp4/procd_test_real.csv", encoding='utf8', header=True, inferSchema=True) 
test_data = test_data.rdd


# %%
#将测试集的特征转为向量
test = test_data.map(lambda line: (line[0],line[1],line[2],Vectors.dense(line[3:])))

# %% [markdown]
# ## Logistic Regression

# %%
from pyspark.mllib.classification import LogisticRegressionModel
lr_model = LogisticRegressionModel.load(sc, "hdfs://node1:9000/user/root/exp4/models/LogisticRegressionModel")


# %%
lr_predictions = test.map(lambda line: (line[0],line[1],float(lr_model.predict(line[3]))))
lr_predictions.coalesce(1).toDF().write.options(header="true").csv("hdfs://node1:9000/user/root/exp4/predictions/lr_predictions.csv")

# %% [markdown]
# 日期:2020-12-20 14:08:52 排名: 无
# score:0.5015744
# %% [markdown]
# ## SVM

# %%
from pyspark.mllib.classification import SVMModel
svm_model = SVMModel.load(sc, "hdfs://node1:9000/user/root/exp4/models/SVMWithSGDModel")


# %%
svm_predictions = test.map(lambda line: (line[0],line[1],float(svm_model.predict(line[3]))))
svm_predictions.coalesce(1).toDF().write.options(header="true").csv("hdfs://node1:9000/user/root/exp4/predictions/svm_predictions.csv")

# %% [markdown]
# 日期:2020-12-20 14:18:59 排名: 无
# score:0.5156678
# %% [markdown]
# ## Gradient Boosted Trees

# %%
from pyspark.mllib.tree import GradientBoostedTreesModel
GBDT_model = GradientBoostedTreesModel.load(sc, "hdfs://node1:9000/user/root/exp4/models/myGradientBoostingClassificationModel")


# %%
predictions = GBDT_model.predict(test.map(lambda x: x[3]))
GBDT_predictions = test.map(lambda lp: (lp[0],lp[1])).zip(predictions).map(lambda lp:(lp[0][0],lp[0][1],lp[1]))
GBDT_predictions.coalesce(1).toDF().write.options(header="true").csv("hdfs://node1:9000/user/root/exp4/predictions/GBDT_predictions.csv")

# GBDT_predictions = test.map(lambda line: (line[0],line[1],float(GBDT_model.predict(line[3]))))
# GBDT_predictions.coalesce(1).toDF().write.options(header="true").csv("hdfs://node1:9000/user/root/exp4/predictions/GBDT_predictions.csv")

# %% [markdown]
# 日期:2020-12-20 14:51:00 排名: 无
# score:0.5000562
# %% [markdown]
# ## SVM with Normalized data

# %%
from pyspark.mllib.classification import SVMModel
svm_model2 = SVMModel.load(sc, "hdfs://node1:9000/user/root/exp4/models/NormalizedSVMWithSGDModel")


# %%
#将数据规范化
from pyspark.mllib.feature import Normalizer
features = test.map(lambda x: x[3])
normalizer = Normalizer()
normalized_test = test.map(lambda lp: (lp[0],lp[1])).zip(normalizer.transform(features)).map(lambda lp:(lp[0][0],lp[0][1],lp[1]))


# %%
print(normalized_test.take(10))


# %%
svm_predictions2 = normalized_test.map(lambda line: (line[0],line[1],float(svm_model2.predict(line[2]))))
svm_predictions2.coalesce(1).toDF().write.options(header="true").csv("hdfs://node1:9000/user/root/exp4/predictions/svm_predictions2.csv")


# %%

spark.stop()
sc.stop()

# %% [markdown]
# 日期:2020-12-20 15:06:01 排名: 无
# score:0.5000000

