##
import numpy as np
from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.recommendation import Rating, ALS
from pyspark.mllib.regression import LabeledPoint

##
sc = SparkContext(master="local", appName="Spark Notes")

## #########################
# Python lambda functions ##
# ###########################
test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(list(map(lambda x: x * x, test_list)))
print(list(filter(lambda x: (x % 2 == 0), test_list)))
del test_list

## ################
# Creating RDDs ##
# #################
# RDDs from Parallelized collections
RDD = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# RDDs from External Datasets (Files in HDFS, Objects in Amazon S3 bucket, lines in a text file) AND Partitions
fileRDD = sc.textFile("C:/Users/k_chi/PycharmProjects/pySpark_Notes/datasets/ratings.csv")

## #######################################################
# Basic RDD Transformations: Map, Filter,Flatmap, Union ##
# ########################################################
print(RDD.map(lambda x: x * x * x).collect())
print((RDD.filter(lambda x: x % 2 == 0).collect()))
print(sc.parallelize(["hello world", "how are you"]).flatMap(lambda x: x.split(" ")).collect())
print(sc.parallelize(["hello world", "how are you"]).union(sc.parallelize(["I am fine", "Thank You!"])).collect())

## #########################################################
# Basic RDD Actions: collect(), take(N), first(), count() ##
# ##########################################################
print(RDD.collect())
print(RDD.take(2))
print(RDD.first())
print(RDD.count())
print(RDD.reduce(lambda x, y: x + y))
print(RDD.sum())

## #####################################
# Creating pair RDDs (we need tuples) ##
# ######################################
RDD = sc.parallelize([('Sam', 23), ('Mary', 34), ('Sam', 25)])
RDD = sc.parallelize(['Sam 23', 'Mary 23', 'Peter 25']).map(lambda s: (s.split(' ')[0], s.split(' ')[1]))

## ##############################
# Transformations on pair RDDs ##
# ###############################
RDD = sc.parallelize([(1, 2), (3, 4), (3, 6), (4, 5)])
print(RDD.reduceByKey(lambda x, y: x + y).collect())
print(RDD.sortByKey(ascending=False).collect())
print(RDD.join(RDD).collect())
total = RDD.countByKey()
for k, v in total.items():
    print("key", k, "has", v, "counts")
grouped = RDD.groupByKey()
for key, list_val in grouped.collect():
    print(key, list(list_val))
print(sc.parallelize([(1, 2), (3, 4)]).collectAsMap())

## ############################################
# Machine Learning - Collaborative Filtering ##
# #############################################
# Theory: https://www.youtube.com/watch?v=h9gpufJFF-0&t=30s&ab_channel=ArtificialIntelligence-AllinOne

# Transformations - Data preparation
ratings = fileRDD.map(lambda r: r.split(','))
ratings = ratings.map(
    lambda line: Rating(int(line[0]), int(line[1]), float(line[2])))    # dataset final form list of lists
training_data, test_data = ratings.randomSplit([0.8, 0.2])              # splitting the data
test_data = test_data.map(lambda p: (p[0], p[1]))                       # removing the rate column from test data

# Fitting the ALS model on the training data
model = ALS.train(training_data, rank=10, iterations=10)

# Predict
predictions = model.predictAll(test_data)

# Model evaluation using MSE
# Prepare ratings data
rates = ratings.map(lambda r: ((r[0], r[1]), r[2]))

# Prepare predictions data
preds = predictions.map(lambda r: ((r[0], r[1]), r[2]))

# Join the ratings data with predictions data
rates_and_preds = rates.join(preds)

# Calculate MSE: Average value of the square of (actual rating - predicted rating)
MSE = rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error of the model for the test data = {:.2f}".format(MSE))

## ###################################################
# Machine Learning - Logistic Regression with LBFGS ##
# ####################################################
# Load the datasets into RDDs
spam_rdd = sc.textFile("C:/Users/k_chi/PycharmProjects/pySpark_Notes/datasets/spam.csv")
non_spam_rdd = sc.textFile("C:/Users/k_chi/PycharmProjects/pySpark_Notes/datasets/non_spam.csv")

# Split the email messages into words
spam_words = spam_rdd.flatMap(lambda email: email.split(' '))
non_spam_words = non_spam_rdd.flatMap(lambda email: email.split(' '))

# Create a HashingTf instance with 200 features
tf = HashingTF(numFeatures=200)

# Map each word to one feature
spam_features = tf.transform(spam_words)
non_spam_features = tf.transform(non_spam_words)

# Label the features: 1 for spam, 0 for non-spam
spam_samples = spam_features.map(lambda features:LabeledPoint(1, features))
non_spam_samples = non_spam_features.map(lambda features:LabeledPoint(0, features))

# Combine the two datasets
samples = spam_samples.join(non_spam_samples)
##

