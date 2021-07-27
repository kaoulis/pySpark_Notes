from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
import seaborn as sns

##
spark = SparkSession.builder.getOrCreate()

## ###########################
# Exploratory Data Analysis ##
# ############################
df = spark.read.csv('D:/Users/k_chi/PycharmProjects/pySpark_Notes/datasets/chapter4/real_estates.csv',
                    header=True,
                    inferSchema=True)
df.show()
print('Shape: ', df.count(), ',', len(df.columns))
print(df.dtypes)
df.describe(['SALESCLOSEPRICE']).show()
# Sample before converting to Pandas because of BIG DATA!
pandas_df = df.select('SALESCLOSEPRICE').sample(False, 0.5, 42).toPandas()
sns.displot(pandas_df, log_scale=True)
plt.show()
print(df.agg({'LISTPRICE': 'skewness'}).collect()[0][0])
pandas_df = df.select('LIVINGAREA', 'SALESCLOSEPRICE').sample(False, 0.5, 42).toPandas()
sns.lmplot(x='LIVINGAREA', y='SALESCLOSEPRICE', data=pandas_df)
plt.show()

## ################
# Data Wrangling ##
# #################
# Dropping columns
df = df.drop(*['No.', 'UNITNUMBER', 'CLASS', 'STREETNUMBERNUMERIC'])
# Dropping records (based on column values)
df = df.where(
    ~df['ASSUMABLEMORTGAGE'].isin(['Yes w/ Qualifying', 'Yes w/No Qualifying']) | df['ASSUMABLEMORTGAGE'].isNull())
# Dropping records (based on column outliers - 3σ rule)
df = df.withColumn('log_SalesClosePrice', log('SALESCLOSEPRICE'))  # First transform with log because 3σ works with nearly normal data
mean_val = df.agg({'log_SalesClosePrice': 'mean'}).collect()[0][0]
stddev_val = df.agg({'log_SalesClosePrice': 'stddev'}).collect()[0][0]
low_bound = mean_val - (3 * stddev_val)
hi_bound = mean_val + (3 * stddev_val)
df = df.filter((df['log_SalesClosePrice'] < hi_bound) & (df['log_SalesClosePrice'] > low_bound))
# Drop any records with NULL values
df = df.dropna()
# Drop records if both LISTPRICE and SALESCLOSEPRICE are NULL
df = df.dropna(how='all', subset=['LISTPRICE', 'SALESCLOSEPRICE'])
# Drop records where at least two columns have NULL values
df = df.dropna(thresh=2)
# Drop duplicate records
df.dropDuplicates()
# Drop records based on duplicate column values
df.dropDuplicates(['streetaddress'])
##
