from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

##
sc = SparkContext(master="local", appName="Spark Demo")

##
spark = SparkSession.builder.getOrCreate()
df = spark.sql('''select 'spark' as hello ''')

##

