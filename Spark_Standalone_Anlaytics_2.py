
# 1 
df_jungyu = spark.read.option("inferSchema", True).option("header", True).csv("/home/centos/data/retail-data/by-day/2011-08-18.csv")

# 2
df_jungyu.printSchema()

# 3 
df_jungyu.cache()
# 4 
df_jungyu.count()


# 5 
df_jungyu.withColumn("DiscountedPrice", df_jungyu["UnitPrice"] * 0.9)
df_jungyu.show(10)

# 6
df_jungyu.select("Country").distinct().count()

# 7
df_6 = df_jungyu.filter((df_jungyu["Description"].contains("CHRISTMAS")) & (df_jungyu["Quantity"] >= 10))
df_6.count()
# 8.
df_7 = df_jungyu.groupBy("Country").agg(
    {
        "Quantity": "sum",
        "UnitPrice": "avg",
        "InvoiceNo": "count"
    }
).withColumnRenamed("sum(Quantity)", "Total Quantity") \
 .withColumnRenamed("avg(UnitPrice)", "Average UnitPrice") \
 .withColumnRenamed("count(InvoiceNo)", "Count of InvoiceNo")

df_7.count()

df_7.show(5)    

