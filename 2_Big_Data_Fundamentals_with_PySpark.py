##
from pyspark import SparkContext

##
sc = SparkContext(master="local", appName="Spark Notes")

## #########################
# Python lambda functions ##
############################
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
# fileRDD_part = sc.textFile("C:/Users/k_chi/PycharmProjects/pySpark_Notes/datasets/ratings.csv", minPartitions=5)
# print("Number of partitions in fileRDD_part is", fileRDD_part.getNumPartitions())

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
print(RDD.groupByKey().collect())
print(RDD.join(RDD).collect())
##

