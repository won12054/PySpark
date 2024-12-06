'''
1. Download the wine dataset wine.csv accompanied with this assignment and move it to a folder on your virtual machine.
'''
# Done

'''
2. Load the wine dataset into a data frame named wine_x1
'''
file_path = "/home/centos/data/wine.csv"
wine_x1 = spark.read.format('csv').option('header', 'true').option('inferSchema', 'true').option('sep', ';').load(file_path)
wine_x1.show()

'''
3. Using spark high level api functions (i.e. not pandas), carry out some initial investigation and record the results in your analysis, at minimum provide the following:
a. Printout the names of columns
b. Printout the types of each column
c. Printout the basic statistics mean, median, the four quartiles  
d. Printout the  minimum, maximum value for each column
e. Generate and printout a table showing the number of missing values for each column. 
(Hint: use isnan, when, count, col)
'''
from pyspark.sql import functions as F

wine_x1.columns

wine_x1.dtypes

mean_values_dict = {col: wine_x1.select(F.mean(col).alias("mean")).collect()[0]["mean"] for col in wine_x1.columns}

for col, mean in mean_values_dict.items():
    print(f"Column: {col}")
    print(f"  Mean: {mean}")

quartiles = {col: wine_x1.approxQuantile(col, [0.0, 0.25, 0.5, 0.75, 1.0], 0.0) for col in wine_x1.columns}
for col, q in quartiles.items():
    print(f"Column: {col}")
    print(f"  Min: {q[0]}")
    print(f"  Q1 (25%): {q[1]}")
    print(f"  Median (50%): {q[2]}")
    print(f"  Q3 (75%): {q[3]}")
    print(f"  Max: {q[4]}")


for col in wine_x1.columns:
    missing_count = wine_x1.select(F.count(F.when(F.isnan(col) | F.col(col).isNull(), col)).alias("missing")).collect()[0]["missing"]
    print(f"Column: {col}")
    print(f"  Missing Values: {missing_count}")

'''
4. Show all the  distinct values in the “quality” column. 
'''
wine_x1.select('quality').distinct().orderBy('quality').show()

'''
5. Show the mean of the various chemical compositions 
across samples for the different groups of the wine quality.
'''
mean_by_quality = wine_x1.groupBy("quality").agg(
    *[F.mean(col).alias(f"{col}_mean") for col in wine_x1.columns if col != "quality"]
)

mean_by_quality.orderBy("quality").show()

'''
6. Re-load the wine dataset into a data frame named wine_x 
as you load add a new column named feature_x of vector type 
that contains four columns as follows:
"citric acid", "volatile acidity", "chlorides", "sulphates"
Spread the data frame across 3 RDD partitions. (Hint: use coalesce)
'''
from pyspark.ml.feature import VectorAssembler

wine_x = spark.read.format('csv').option('header', 'true').option('inferSchema', 'true').option('sep', ';').load(file_path)

feature_columns = ['citric acid', 'volatile acidity', 'chlorides', 'sulphates']
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="feature_x")

wine_x = vector_assembler.transform(wine_x)

wine_x = wine_x.coalesce(3)
wine_x.select('feature_x').show()

'''
7. Cache the dataframe.
'''
wine_x.cache()

'''
8. Define a estimator that uses K-means clustering 
to cluster all the wine instances into 6 clusters 
using the new feature_x vector column you added in step #6.
'''
from pyspark.ml.clustering import KMeans

kmeans = KMeans(featuresCol='feature_x', k=6, seed=42)

kmeans_model = kmeans.fit(wine_x)

wine_clusters = kmeans_model.transform(wine_x)

wine_clusters.select('feature_x', 'prediction').show()

'''
9. Print the cluster sizes and the cluster centroids, 
record the results in your analysis report and write some conclusions.
'''
for idx, center in enumerate(kmeans_model.clusterCenters()):
    print(f'Cluster {idx}: {center}')

cluster_sizes = wine_clusters.groupBy('prediction').count()
cluster_sizes.orderBy('prediction').show()

'''
10. Repeat steps 8&9 but set the number of k to 4.
'''
kmeans_k4 = KMeans(featuresCol='feature_x', k=4, seed=42)

kmeans_model_k4 = kmeans_k4.fit(wine_x)

wine_clusters_k4 = kmeans_model_k4.transform(wine_x)

wine_clusters_k4.select('feature_x', 'prediction').show()

print("\nCluster Centroids (k=4):")
for idx, center in enumerate(kmeans_model_k4.clusterCenters()):
    print(f'Cluster {idx}: {center}')

cluster_sizes_k4 = wine_clusters_k4.groupBy('prediction').count()
print("\nCluster Sizes (k=4):")
cluster_sizes_k4.orderBy('prediction').show()
