# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # 用test_format1训练模型并保存

# %%
import pyspark
from pyspark import SparkContext
from pyspark import SparkConf


# %%
conf=SparkConf().setAppName("miniProject").setMaster("local").set("spark.executor.memory","3g")        .set("spark.executor.instances","2")
sc=SparkContext.getOrCreate(conf)


# %%
from pyspark.mllib.util import MLUtils


# %%
training = MLUtils.loadLibSVMFile(sc, "hdfs://node1:9000/user/root/exp4/procd_train_real")

# Split data into training (60%) and test (40%)
#training, test = data.randomSplit([0.6, 0.4], seed=11)
training.cache()

# %% [markdown]
# ## Logistic Regression

# %%
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
#from pyspark.mllib.evaluation import BinaryClassificationMetrics
# Logistic Regression
# Run training algorithm to build the model
model = LogisticRegressionWithLBFGS.train(training)


# %%
# Evaluating the model on training data
labelsAndPreds = training.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(training.count())
print("Training Error = " + str(trainErr))


# %%
model.save(sc, "hdfs://node1:9000/user/root/exp4/models/LogisticRegressionModel")
# sameModel = LogisticRegressionModel.load(sc, "hdfs://node1:9000/user/root/exp4/models/LogisticRegressionModel")

# %% [markdown]
# ## SVM

# %%
# SVM
# Build the model
from pyspark.mllib.classification import SVMWithSGD, SVMModel
model = SVMWithSGD.train(training, iterations=200)


# %%
# Evaluating the model on training data
labelsAndPreds = training.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(training.count())
print("Training Error = " + str(trainErr))


# %%
model.save(sc, "hdfs://node1:9000/user/root/exp4/models/SVMWithSGDModel")
# sameModel = SVMModel.load(sc, "hdfs://node1:9000/user/root/exp4/models/SVMWithSGDModel")

# %% [markdown]
# ## Decision Tree

# %%
# DecisionTree
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
model = DecisionTree.trainClassifier(training, numClasses=2, categoricalFeaturesInfo={},
                                         impurity='gini', maxDepth=5, maxBins=32)


# %%
# Evaluating the model on training data
predictions = model.predict(training.map(lambda x: x.features))
labelsAndPreds = training.map(lambda lp: lp.label).zip(predictions)
trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(training.count())
print("Training Error = " + str(trainErr))


# %%
print('Learned classification tree model:')
print(model.toDebugString())


# %%
# Save and load model
model.save(sc, "hdfs://node1:9000/user/root/exp4/models/myDecisionTreeClassificationModel")
# sameModel = DecisionTreeModel.load(sc, "hdfs://node1:9000/user/root/exp4/models/myDecisionTreeClassificationModel")

# %% [markdown]
# ## Naive Bayes

# %%
import shutil
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
# Train a naive Bayes model.
model = NaiveBayes.train(training, 1.0)


# %%
# Evaluating the model on training data
labelsAndPreds = training.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(training.count())
print("Training Error = " + str(trainErr))


# %%
# Save and load model
model.save(sc, "hdfs://node1:9000/user/root/exp4/models/NaiveBayesModel")
# sameModel = NaiveBayesModel.load(sc, "hdfs://node1:9000/user/root/exp4/models/NaiveBayesModel")

# %% [markdown]
# ## Random Forest

# %%
from pyspark.mllib.tree import RandomForest, RandomForestModel
# Train a RandomForest model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
#  Note: Use larger numTrees in practice.
#  Setting featureSubsetStrategy="auto" lets the algorithm choose.
model = RandomForest.trainClassifier(training, numClasses=2, categoricalFeaturesInfo={},
                                        numTrees=3, featureSubsetStrategy="auto",
                                        impurity='gini', maxDepth=4, maxBins=32)


# %%
# Evaluating the model on training data
predictions = model.predict(training.map(lambda x: x.features))
labelsAndPreds = training.map(lambda lp: lp.label).zip(predictions)
trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(training.count())
print("Training Error = " + str(trainErr))


# %%
print('Learned classification forest model:')
print(model.toDebugString())


# %%
# Save and load model
model.save(sc, "hdfs://node1:9000/user/root/exp4/models/myRandomForestClassificationModel")
# sameModel = RandomForestModel.load(sc, "hdfs://node1:9000/user/root/exp4/models/myRandomForestClassificationModel")

# %% [markdown]
# ## Gradient Boosted Trees

# %%
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
# Train a GradientBoostedTrees model.
#  Notes: (a) Empty categoricalFeaturesInfo indicates all features are continuous.
#         (b) Use more iterations in practice.
model = GradientBoostedTrees.trainClassifier(training,
                                                categoricalFeaturesInfo={}, numIterations=30)


# %%
# Evaluating the model on training data
predictions = model.predict(training.map(lambda x: x.features))
labelsAndPreds = training.map(lambda lp: lp.label).zip(predictions)
trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(training.count())
print("Training Error = " + str(trainErr))


# %%
print('Learned classification GBT model:')
print(model.toDebugString())


# %%
# Save and load model
model.save(sc, "hdfs://node1:9000/user/root/exp4/models/myGradientBoostingClassificationModel")
# sameModel = GradientBoostedTreesModel.load(sc,
#                                             "hdfs://node1:9000/user/root/exp4/models/myGradientBoostingClassificationModel")

# %% [markdown]
# # Don't forget to stop the SparkContext!

# %%
sc.stop()


