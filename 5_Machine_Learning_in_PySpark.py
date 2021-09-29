from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, Bucketizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
import numpy as np

##
# Create SparkSession object (ex. spark://13.59.151.161:7077)
spark = SparkSession.builder. \
    master('local[*]'). \
    appName('test'). \
    getOrCreate()

## Load data
flights = spark.read.csv('D:/Users/k_chi/PycharmProjects/pySpark_Notes/datasets/chapter5/flights.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')
schema = StructType([
    StructField("id", IntegerType()),
    StructField("text", StringType()),
    StructField("label", IntegerType())
])
sms = spark.read.csv('D:/Users/k_chi/PycharmProjects/pySpark_Notes/datasets/chapter5/sms.csv',
                     sep=';',
                     header=False,
                     schema=schema, )

## ################
# Data Wrangling ##
# #################
# ML Data Preparation
flights = flights.drop('flight')
flights = flights.dropna()

flights = flights.withColumn('km', round(flights.mile * 1.60934, 0)).drop('mile')
flights = flights.withColumn('label', (flights.delay >= 15).cast('integer'))

idx1 = StringIndexer(inputCol='carrier', outputCol='carrier_idx')
idx2 = StringIndexer(inputCol='org', outputCol='org_idx')
onehot = OneHotEncoder(inputCols=['org_idx', 'carrier_idx'], outputCols=['org_dummy', 'carrier_dummy'])
assembler1 = VectorAssembler(
    inputCols=['mon', 'dom', 'dow', 'carrier_idx', 'org_idx', 'km', 'depart', 'duration'],
    outputCol='features')
assembler2 = VectorAssembler(
    inputCols=['mon', 'dom', 'dow', 'carrier_dummy', 'org_dummy', 'km', 'depart', 'duration'],
    outputCol='features')

evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC')

flights_train, flights_test = flights.randomSplit([.8, .2], seed=17)

## ##################
# Classification   ##
# ###################
# Decision Tree
dt = DecisionTreeClassifier(seed=17)
estimator = Pipeline(stages=[idx1, idx2, assembler1, dt])
params = ParamGridBuilder().build()
cv = CrossValidator(estimator=estimator, estimatorParamMaps=params, evaluator=evaluator, numFolds=5)
dt_cv = cv.fit(flights_train)
dt_predictions = dt_cv.transform(flights_test)

##
# Logistic Regression
lr = LogisticRegression()
estimator = Pipeline(stages=[idx1, idx2, onehot, assembler2, lr])
# Try hard Mode: addGrid(lr.regParam, np.arange(0, .1, .01)).\
params = ParamGridBuilder(). \
    addGrid(lr.regParam, [0, 1]). \
    addGrid(lr.elasticNetParam, [0, 1]).build()
cv = CrossValidator(estimator=estimator, estimatorParamMaps=params, evaluator=evaluator, numFolds=5)
lr_cv = cv.fit(flights_train)
lr_predictions = lr_cv.transform(flights_test)

##
# Decision Tree with depart time buckets
buckets = Bucketizer(splits=[0, 3, 6, 9, 12, 15, 18, 21, 24], inputCol='depart', outputCol='depart_bucket')
assembler3 = VectorAssembler(
    inputCols=['mon', 'dom', 'dow', 'carrier_idx', 'org_idx', 'km', 'depart_bucket', 'duration'],
    outputCol='features')
# For regression models we need to onehot too!
estimator = Pipeline(stages=[idx1, idx2, buckets, assembler3, dt])
params = ParamGridBuilder().build()
cv = CrossValidator(estimator=estimator, estimatorParamMaps=params, evaluator=evaluator, numFolds=5)
dt2_cv = cv.fit(flights_train)
dt2_predictions = dt2_cv.transform(flights_test)

##
# Gradient-Boosted Trees
gbt = GBTClassifier(seed=17)
estimator = Pipeline(stages=[idx1, idx2, assembler1, gbt])
params = ParamGridBuilder().build()
cv = CrossValidator(estimator=estimator, estimatorParamMaps=params, evaluator=evaluator, numFolds=5)
gbt_cv = cv.fit(flights_train)
gbt_predictions = gbt_cv.transform(flights_test)

##
# Random Forest
rf = RandomForestClassifier(seed=17)
estimator = Pipeline(stages=[idx1, idx2, assembler1, rf])
params = ParamGridBuilder().build()
cv = CrossValidator(estimator=estimator, estimatorParamMaps=params, evaluator=evaluator, numFolds=5)
rf_cv = cv.fit(flights_train)
rf_predictions = rf_cv.transform(flights_test)


##
# Evaluation
def evaluate_overall(pred):
    TN = pred.filter('prediction = 0 AND label = prediction').count()
    TP = pred.filter('prediction = 1 AND label = prediction').count()
    FN = pred.filter('prediction = 0 AND label = 1').count()
    FP = pred.filter('prediction = 1 AND label = 0').count()

    # print("Accuracy:  ", MulticlassClassificationEvaluator().evaluate(pred, {
    #     MulticlassClassificationEvaluator().metricName: "accuracy"}))
    print("Accuracy:            ", (TN + TP) / (TN + TP + FN + FP))
    print("Precision (Label 1): ", TP / (TP + FP))
    print("Precision (Label 0): ", TN / (TN + FN))
    print("Recall (Label 1):    ", TP / (TP + FN))
    print("Recall (Label 0):    ", TN / (TN + FP))
    print("Weighted Precision:  ", MulticlassClassificationEvaluator().evaluate(pred, {
        MulticlassClassificationEvaluator().metricName: "weightedPrecision"}))
    print("Weighted Recall:     ", MulticlassClassificationEvaluator().evaluate(pred, {
        MulticlassClassificationEvaluator().metricName: "weightedRecall"}))
    print("F1:                  ", MulticlassClassificationEvaluator().evaluate(pred, {
        MulticlassClassificationEvaluator().metricName: "f1"}))
    print("AUC:                 ", BinaryClassificationEvaluator().evaluate(pred, {
        BinaryClassificationEvaluator().metricName: "areaUnderROC"}))


print("~~~~ Decision Tree ~~~~")
evaluate_overall(dt_predictions)
print("~~~~ Logistic Regression ~~~~")
evaluate_overall(lr_predictions)
print("~~~~ Decision Tree with buckets ~~~~")
evaluate_overall(dt2_predictions)
print("~~~~ Gradient-Boosted Trees ~~~~")
evaluate_overall(gbt_predictions)
print("~~~~ Random Forest ~~~~")
evaluate_overall(rf_predictions)

##
# Also NLP (SMS data) is covered but currently not in my interest
