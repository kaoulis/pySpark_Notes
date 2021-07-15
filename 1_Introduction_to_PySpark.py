# Big Data concepts and Terminology
# Clustered computing:      Collection of resources of multiple machines
# Parallel computing:       Simultaneous computation
# Distributed computing:    Collection of nodes (networked computers) that run in parallel
# Batch processing:         Breaking the job into small pieces and running them on individual machines
# Real-time processing:     Immediate processing of data
import numpy as np
## Spark imports
# SparkContext() class constructor creates a connection to a cluster.
from pyspark import SparkContext
# SparkConf() object can hold the attributes of the cluster you are connecting to.
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
import pyspark.ml.evaluation as evals
import pyspark.ml.tuning as tune

##
# Examining The SparkContext and creating a SparkSession
sc = SparkContext(master="local", appName="Spark Notes")
spark = SparkSession.builder.getOrCreate()
print(sc)
print(sc.version)
print(sc.pythonVer)
print(sc.master)

##
# Read a data source into Spark DataFrame
airports = spark.read.csv(
    path="D:/Users/k_chi/PycharmProjects/pySpark_Notes/datasets/chapter1/airports.csv",
    header="true",
    inferSchema=True  # It understands the dtypes!
)
flights = spark.read.csv(
    path="D:/Users/k_chi/PycharmProjects/pySpark_Notes/datasets/chapter1/flights.csv",
    header="true",
    inferSchema=True
)
planes = spark.read.csv(
    path="D:/Users/k_chi/PycharmProjects/pySpark_Notes/datasets/chapter1/planes.csv",
    header="true",
    inferSchema=True
)
print(airports.show())

##
# Pandafy a Spark DataFrame OR Sparkfy a Pandas DataFrame.
pd_airports = airports.toPandas()
airports = spark.createDataFrame(pd_airports)
print(pd_airports)
print(airports.show())

##
# The Spark DataFrames are stored locally, not in the SparkSession catalog.
# That is, you can't access the data in other contexts.
# Let's register the DataFrames in the catalog as temporary tables(can only be accessed from the specific SparkSession).
print(spark.catalog.listTables())
airports.createOrReplaceTempView("airports_tmp")
flights.createOrReplaceTempView("flights_tmp")
planes.createOrReplaceTempView("planes_tmp")
print(spark.catalog.listTables())

##
# Spark + SQL = BFFE
print(spark.sql("SELECT origin, dest, COUNT(*) as N FROM flights_tmp GROUP BY origin, dest").show())

##
# Creating columns
flights = spark.table("flights_tmp")
flights = flights.withColumn("duration_hours", flights.air_time / 60)

##
# Filtering, Selecting, Aggregating
flights.filter("distance > 1000").show()
flights.filter(flights.distance > 1000).show()
flights.where(flights.distance > 1000).show()
flights.where(flights['distance'] > 1000).show()

flights.select("tailnum", "origin", "dest").show()
flights.select(flights.origin, flights.dest, flights.carrier).show()
flights.select(["tailnum", "origin", "dest"]).show()

# Find the shortest flight from PDX in terms of distance
flights.filter(flights.origin == "PDX").groupBy().min("distance").show()

# Number of flights each plane made
flights.groupBy("tailnum").count().show()

##
# Joining
airports = airports.withColumnRenamed("faa", "dest")
flights_with_airports = flights.join(airports, on="dest", how="leftouter")

##
# Machine Learning Pipelines
# At the core of the pyspark.ml module are the Transformer (.transform()) and Estimator (.fit()) classes.
flights = spark.table("flights_tmp")
planes = planes.withColumnRenamed("year", "plane_year")
model_data = flights.join(planes, on="tailnum", how="leftouter")

# Spark only handles numeric data. So we cast the columns to integers (or doubles).
model_data = model_data.withColumn("arr_delay", model_data.arr_delay.cast("integer"))
model_data = model_data.withColumn("air_time", model_data.air_time.cast("integer"))
model_data = model_data.withColumn("month", model_data.month.cast("integer"))
model_data = model_data.withColumn("plane_year", model_data.plane_year.cast("integer"))

# Create the column plane_age
model_data = model_data.withColumn("plane_age", model_data.year - model_data.plane_year)

# Create is_late label column
model_data = model_data.withColumn("label", (model_data.arr_delay > 0).cast("integer"))

# Remove missing values
print((model_data.count(), len(model_data.columns)))
model_data = model_data.filter(
    "arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")
# model_data.na.drop()
# model_data = model_data.dropDuplicates()
print((model_data.count(), len(model_data.columns)))

##
# Encoding categorical features as one-hot vector. STEPS:
# 1. Create a StringIndexer
# 2. # Create a OneHotEncoder
carr_indexer = StringIndexer(inputCol="carrier", outputCol="carrier_index")
carr_encoder = OneHotEncoder(inputCol="carrier_index", outputCol="carrier_fact")

dest_indexer = StringIndexer(inputCol="dest", outputCol="dest_index")
dest_encoder = OneHotEncoder(inputCol="dest_index", outputCol="dest_fact")

# Combine all of the columns containing our features into a single column (Spark ML way).
vec_assembler = VectorAssembler(inputCols=["month", "air_time", "carrier_fact", "dest_fact", "plane_age"], outputCol="features")

# Finally, create the pipeline, fit and transform !!!
flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])
piped_data = flights_pipe.fit(model_data).transform(model_data)

##
# Splitting the data train and test
training, test = piped_data.randomSplit([.6, .4])

# Estimator (Classifier)
lr = LogisticRegression()

# Create the parameter grid
grid = tune.ParamGridBuilder()\
    .addGrid(lr.regParam, np.arange(0, .1, .01))\
    .addGrid(lr.elasticNetParam, [0, 1]).build()
# Evaluation Metric
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")

# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr,
                         estimatorParamMaps=grid,
                         evaluator=evaluator)

##
model = lr.fit(training)
# Use the model to predict the test set
test_results = model.transform(test)
# Evaluate the predictions
print(evaluator.evaluate(test_results))
##

