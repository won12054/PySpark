
'''
1. Load the data for January 9th and 10th
'''
df_jungyu = spark.read.option('inferSchema', True).option('header', True).csv(['/home/centos/data/retail-data/by-day/2011-01-09.csv', '/home/centos/data/retail-data/by-day/2011-01-10.csv'])

'''
3. Count the number of records & Print inferred schema
'''
df_jungyu.count()
df_jungyu.printSchema()

'''
4. Show all the transactions that are related to the purchase of stock id that starts with  "227"   
    with the type of product “ALARM CLOCK” mentioned as part of the description or a unit price greater than 5.
5. Store the results into a new dataframe name it df2_jungyu.
'''
from pyspark.sql.functions import col

df2_jungyu = df_jungyu.filter(
    (col("StockCode").startswith("227")) & 
    ((col("Description").contains("ALARM CLOCK")) | (col("UnitPrice") > 5))
)

df2_jungyu.show(truncate=False)

'''
6. Show the sum of the quantities ordered 
    and the minimum quantity order 
    and the maximum quantity order for the transactions you extracted in point 4 above.
'''
from pyspark.sql.functions import sum, min, max

df_6 = df2_jungyu.agg(
    sum("Quantity").alias("sum_quantity"),
    min("Quantity").alias("min_quantity"),
    max("Quantity").alias("max_quantity")
)

df_6.show()

df_6.rdd.getNumPartitions()

'''
7- Show all the transactions mentioned in point 4 above that have originated form outside the United Kingdom.
'''
df_7 = df2_jungyu.filter(col("Country") != "United Kingdom")
df_7.show(truncate=False)
