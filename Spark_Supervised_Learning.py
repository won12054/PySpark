'''
1. Research and investigate the LIBSVM format, 
in your analysis report define the format 
and show an example with explanation.
'''
# Refer to the report

'''
2. Load the data stored in the file “sample_libsvm_data.txt” 
from the data available on the VMware image under the directory /home/centos/data/ into a dataframe 
and name it df_x where x is your firstname. 
Use infer schema, notice that you need to use the format is LIBSVM when you create the dataframe.
'''

file_path = "/home/centos/data/sample_libsvm_data.txt"
df_jungyu = spark.read.format('libsvm').option('inferSchema', 'true').load(file_path)
df_jungyu.show(5, truncate=False)

'''
3. Carry out some basic investigation: count the number of records, 
count the number of columns print the inferred schema 
and explain what each column contains and record the results in your analysis report.
'''

df_jungyu.count()
len(df_jungyu.columns)
df_jungyu.printSchema()

# Refer to the report

'''
4. Use the StringIndexer to index labels, in other words you will add metadata to the label column. 
Name the output column "indexedLabel_x” where x is your first name. 
Store the result in a variable named labelIndexer_x where x is your first name.  
'''
from pyspark.ml.feature import StringIndexer

labelIndexer_jungyu = StringIndexer(inputCol='label', outputCol='indexedLabel_jungyu')

df_jungyu_with_label = labelIndexer_jungyu.fit(df_jungyu).transform(df_jungyu)

df_jungyu_with_label.show(5)


'''
5. Use the VectorIndexer to automatically identify categorical features, and index them. 
Set the maxCategories to 4. 
Name the output column "indexedFeatures _x" where x is your first name. 
Store the result in a variable named featureIndexer _x where x is your first name.   

VectorIndexer encountered invalid value 88.0 on feature index 100.
'''
from pyspark.ml.feature import VectorIndexer

featureIndexer_jungyu = VectorIndexer(
    inputCol='features',
    outputCol='indexedFeatures_jungyu',
    maxCategories=4,
    handleInvalid='keep'
)

featureIndexerModel_jungyu = featureIndexer_jungyu.fit(df_jungyu)
df_jungyu_with_features = featureIndexerModel_jungyu.transform(df_jungyu)
df_jungyu_with_features.show(5) 

'''
6.	Printout the following:
a.	Name of input column
b.	Name of output column
c.	# of features
d.	Map of categories
Also note the results in your written response.
'''
featureIndexerModel_jungyu.getInputCol()
featureIndexerModel_jungyu.getOutputCol()
featureIndexerModel_jungyu.numFeatures
category_maps = featureIndexerModel_jungyu.categoryMaps
for feature_idx, mapping in category_maps.items():
    print(f"  Feature {feature_idx}: {mapping}")


# Refer to the report

'''
7. Split your original data into 65% for training and 35% for testing 
and store the training data into a datafrmae named training_x and testing_x respectively 
where x is your firstname.
'''
training_jungyu, testing_jungyu = df_jungyu.randomSplit([0.65, 0.35], seed=42)
training_jungyu.count()
testing_jungyu.count()

'''
8. Create an estimator object that contains a decision tree classifier 
Make sure to set the correct input and output columns you created during the transformation steps 4 & 5 above. 
Name the estimator DT_x where x is your firstname. 
'''
from pyspark.ml.classification import DecisionTreeClassifier

DT_jungyu = DecisionTreeClassifier(
    featuresCol='indexedFeatures_jungyu',
    labelCol='indexedLabel_jungyu'
)

'''
9. Create a pipeline object with three stages 
The first two are the transformers you defined in steps 4 & 5 
and the third is the decision tree estimator you defined in step 8. 
Name the pipeline object pipeline_x where x is your firstname.
'''
from pyspark.ml import Pipeline

pipeline_jungyu = Pipeline(stages=[
    labelIndexer_jungyu,
    featureIndexer_jungyu,
    DT_jungyu
])

pipeline_jungyu.getStages()

'''
10. Fit the training data to the pipeline. 
Store the results into an object named model_x, where x is your first name.
'''
model_jungyu = pipeline_jungyu.fit(training_jungyu)

'''
11. Using the model_x predict the testing data. 
Store the results into a dataframe named predictions_x 
where x is your firstname.
'''
predictions_jungyu = model_jungyu.transform(testing_jungyu)

'''
12.	Print the schema of the predictions and note the results into your analysis report.
'''
predictions_jungyu.printSchema()
# Refer to the report
'''
13.	Print the accuracy of your model and the test error and note the results in your analysis report.
'''
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
    labelCol='indexedLabel_jungyu',
    predictionCol='prediction',
    metricName='accuracy'
)

accuracy = evaluator.evaluate(predictions_jungyu)
test_error = 1.0 - accuracy

accuracy
test_error

# Refer to the report
'''
14.	Show the first 10 predictions with the actual labels and features take a screenshot and add it to your analysis report.
'''
predictions_jungyu.select("label", "features", "prediction").show(10)
predictions_jungyu.show(10)

# Refer to the report