# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pyspark
from pyspark import SparkContext
from pyspark import SparkConf


# %%
conf=SparkConf().setAppName("miniProject").setMaster("local").set("spark.executor.memory","3g").set("spark.executor.instances","2")
sc=SparkContext.getOrCreate(conf)


# %%
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.util import MLUtils


# %%
data = MLUtils.loadLibSVMFile(sc, "hdfs://node1:9000/user/root/exp4/procd_train_real")

# Split data into training (60%) and test (40%)
training, test = data.randomSplit([0.6, 0.4], seed=11)
training.cache()

# %% [markdown]
# ## Logistic Regression

# %%
# Logistic Regression
# Run training algorithm to build the model
model = LogisticRegressionWithLBFGS.train(training)


# %%
# Compute raw scores on the test set
predictionAndLabels = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))

# Instantiate metrics object
metrics = BinaryClassificationMetrics(predictionAndLabels)

# Area under precision-recall curve
print("Area under PR = %s" % metrics.areaUnderPR)

# Area under ROC curve
print("Area under ROC = %s" % metrics.areaUnderROC)


# %%
testErr = predictionAndLabels.filter(lambda lp: lp[0] != lp[1]).count() / float(test.count())
print("Test Error = " + str(testErr))

# %% [markdown]
# ## SVM

# %%
# SVM
# Build the model
from pyspark.mllib.classification import SVMWithSGD, SVMModel
svm_model = SVMWithSGD.train(training, iterations=100)


# %%
# Evaluating the model on training data
labelsAndPreds = training.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(training.count())
print("Training Error = " + str(trainErr))


# %%
# Evaluating the model on test data
labelsAndPreds = test.map(lambda p: (p.label, svm_model.predict(p.features)))
testErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(test.count())
print("Test Error = " + str(testErr))


# %%
# model.save(sc, "target/tmp/pythonSVMWithSGDModel")
# sameModel = SVMModel.load(sc, "target/tmp/pythonSVMWithSGDModel")

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
# Evaluate model on test instances and compute test error
predictions = model.predict(test.map(lambda x: x.features))
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(test.count())
print('Test Error = ' + str(testErr))
print('Learned classification tree model:')
print(model.toDebugString())


# %%
# Save and load model
# model.save(sc, "target/tmp/myDecisionTreeClassificationModel")
# sameModel = DecisionTreeModel.load(sc, "target/tmp/myDecisionTreeClassificationModel")

# %% [markdown]
# ## Naive Bayes

# %%
import shutil
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
# Train a naive Bayes model.
model = NaiveBayes.train(training, 1.0)


# %%
# Make prediction and test accuracy.
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
print('model accuracy {}'.format(accuracy))


# %%
# Save and load model
# output_dir = 'target/tmp/myNaiveBayesModel'
# shutil.rmtree(output_dir, ignore_errors=True)
# model.save(sc, output_dir)
# sameModel = NaiveBayesModel.load(sc, output_dir)
# predictionAndLabel = test.map(lambda p: (sameModel.predict(p.features), p.label))
# accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
# print('sameModel accuracy {}'.format(accuracy))

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
# Evaluate model on test instances and compute test error
predictions = model.predict(test.map(lambda x: x.features))
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(test.count())
print('Test Error = ' + str(testErr))
print('Learned classification forest model:')
print(model.toDebugString())


# %%
# Save and load model
# model.save(sc, "target/tmp/myRandomForestClassificationModel")
# sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel")

# %% [markdown]
# ## Gradient Boosted Trees

# %%
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
# Train a GradientBoostedTrees model.
#  Notes: (a) Empty categoricalFeaturesInfo indicates all features are continuous.
#         (b) Use more iterations in practice.
model = GradientBoostedTrees.trainClassifier(training,
                                                categoricalFeaturesInfo={}, numIterations=3)


# %%
# Evaluate model on test instances and compute test error
predictions = model.predict(test.map(lambda x: x.features))
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(test.count())
print('Test Error = ' + str(testErr))
print('Learned classification GBT model:')
print(model.toDebugString())


# %%
# Save and load model
# model.save(sc, "target/tmp/myGradientBoostingClassificationModel")
# sameModel = GradientBoostedTreesModel.load(sc,
#                                             "target/tmp/myGradientBoostingClassificationModel")

# %% [markdown]
# # Don't forget to stop the SparkContext!

# %%
sc.stop()


