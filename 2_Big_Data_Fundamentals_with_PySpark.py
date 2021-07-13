##
import numpy as np
from pyspark import SparkContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.recommendation import Rating, ALS
from pyspark.mllib.regression import LabeledPoint
import pandas as pd
from pyspark.sql import SparkSession

##
sc = SparkContext(master="local", appName="Spark Notes")
spark = SparkSession.builder.getOrCreate()

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
# - Basic implementation with centered cosine similarity
# - Spark implementation with Alternating Least Squares (ALS)

# Transformations - Data preparation
ratings = fileRDD.map(lambda r: r.split(','))
ratings = ratings.map(
    lambda line: Rating(int(line[0]), int(line[1]), float(line[2])))  # dataset final form list of lists
training_data, test_data = ratings.randomSplit([0.8, 0.2])  # splitting the data
test_data = test_data.map(lambda p: (p[0], p[1]))  # removing the rate column from test data

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
MSE = rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
print("Mean Squared Error of the model for the test data = {:.2f}".format(MSE))

## ###################################################
# Machine Learning - Logistic Regression with LBFGS ##
# ####################################################
# Advantages of LBFGS over Gradient Descent:
# 1. No need to manually pick α,
# 2. Often faster than Gradient Descent
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
spam_samples = spam_features.map(lambda features: LabeledPoint(1, features))
non_spam_samples = non_spam_features.map(lambda features: LabeledPoint(0, features))

# Combine the two datasets
samples = spam_samples.join(non_spam_samples)

# Split the data into training and testing
train_samples, test_samples = samples.randomSplit([0.8, 0.2])

# Train the model
model = LogisticRegressionWithLBFGS.train(train_samples)

# Create a prediction label from the test data
predictions = model.predict(test_samples.map(lambda x: x.features))

# Combine original labels with the predicted labels
labels_and_preds = test_samples.map(lambda x: x.label).zip(predictions)

# Check the accuracy of the model on the test data
accuracy = labels_and_preds.filter(lambda x: x[0] == x[1]).count() / float(test_samples.count())
print("Model accuracy : {:.2f}".format(accuracy))

## ############################################
# Machine Learning - Clustering with K Means ##
# #############################################
# Load the dataset into an RDD
clusterRDD = sc.textFile("C:/Users/k_chi/PycharmProjects/pySpark_Notes/datasets/5000_points.csv")

# Split the RDD based on tab
rdd_split = clusterRDD.map(lambda x: x.split('\t'))

# Transform the split RDD by creating a list of integers
rdd_split_int = rdd_split.map(lambda x: [int(x[0]), int(x[1])])

# Train the model with clusters from 13 to 16 and compute WSSSE
for clst in range(13, 17):
    model = KMeans.train(rdd_split_int, clst, seed=1)
    WSSSE = rdd_split_int.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("The cluster {} has Within Set Sum of Squared Error {}".format(clst, WSSSE))

# Train the model again with the best k
model = KMeans.train(rdd_split_int, k=15, seed=1)

# Get cluster centers
cluster_centers = model.clusterCenters

# Convert rdd_split_int RDD into Spark DataFrame and then to Pandas DataFrame
rdd_split_int_df_pandas = spark.createDataFrame(rdd_split_int, schema=["col1", "col2"]).toPandas()

# Convert cluster_centers to a pandas DataFrame
cluster_centers_pandas = pd.DataFrame(cluster_centers, columns=["col1", "col2"])

# Create an overlaid scatter plot of clusters and centroids
plt.scatter(rdd_split_int_df_pandas["col1"], rdd_split_int_df_pandas["col2"])
plt.scatter(cluster_centers_pandas["col1"], cluster_centers_pandas["col2"], color="red", marker="x")
plt.show()
##

