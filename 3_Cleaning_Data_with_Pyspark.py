import time

from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F

##
spark = SparkSession.builder.getOrCreate()
##
voter_df = spark.read.csv('C:/Users/k_chi/PycharmProjects/pySpark_Notes/datasets/chapter3/DallasCouncilVoters.csv',
                          header=True)
## ###############
# Spark Schemas ##
# ################
# - Can filter garbage data during import
# - Improves read performance
people_schema = StructType([
    # Define a StructField for each field
    StructField('name', StringType(), False),
    StructField('age', IntegerType(), False),
    StructField('city', StringType(), False)
])
##
# VM VS CONTAINERS!!!???
## ########################
# Split - getItem - Size ##
# #########################
voter_df = voter_df.dropna(subset='VOTER_NAME')
voter_df = voter_df.withColumn('splits', F.split(voter_df.VOTER_NAME, '\s'))
voter_df = voter_df.withColumn('first_name', voter_df.splits.getItem(0))
voter_df = voter_df.withColumn('last_name', voter_df.splits.getItem(F.size(voter_df.splits) - 1))

## ##########################
# If Else ~ When Otherwise ##
# ###########################
voter_df = voter_df.withColumn('random_val',
                               F.when(voter_df.TITLE == 'Councilmember', F.rand())
                               .when(voter_df.TITLE == 'Mayor', 2)
                               .otherwise(0))
voter_df = voter_df.drop('random_val')


## #############################
# User Defined Function - UDF ##
# ##############################
def getFirstAndMiddle(names):
    # Return a space separated string of names
    return ' '.join(names[:-1])


# Define the method as a UDF
udfFirstAndMiddle = F.udf(getFirstAndMiddle, StringType())

# Create a new column using your UDF
voter_df = voter_df.withColumn('first_and_middle_name', udfFirstAndMiddle(voter_df.splits))

## ####################
# Adding an ID Field ##
# #####################
voter_df = voter_df.withColumn('ROW_ID', F.monotonically_increasing_id())

## #####################
# Caching a DataFrame ##
# ######################
voter_df = voter_df.cache()

# Count the unique rows, noting how long the operation takes
start_time = time.time()
print("Counting %d rows took %f seconds" % (voter_df.count(), time.time() - start_time))

# Count the rows again, noting the variance in time of a cached DataFrame
start_time = time.time()
print("Counting %d rows again took %f seconds" % (voter_df.count(), time.time() - start_time))

# Remove voter_df from the cache
voter_df.unpersist()

## ####################
# Import Performance ##
# #####################
# - More objects better than larger ones ~ex. airport_df = spark.read.csv('airports-*.txt.gz')
# - Spark performs better if objects are of similar size
# - A well-defined schema will drastically improve import performance + provides validation
# - Use parquet format ~ It is column wise ~ Structured defined schema

## ###############################################################################################
# Optimization by limiting shuffling ~ moving data around to various workers to complete a task ##
# ################################################################################################
# - Limit use of .repartition(num_partitions)
# - Use .coalesce(num_partitions) instead
# - Use .broadcast() ~ Provides a copy of an object to each worker ~ Drastically speed up .join() operations
#   ~ Broadcast the smaller DataFrame

