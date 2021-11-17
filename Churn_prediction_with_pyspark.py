import warnings
import findspark
import pandas as pd
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import when, count, col

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', 10)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

findspark.init(r"C:\spark")

spark = SparkSession.builder.appName("churn prediction App").getOrCreate()

sc = spark.sparkContext

spark_df = spark.read.csv("datasets/churn2.csv", header=True, inferSchema=True)
type(spark_df)

############################
# Exploratory Data Analysis and data prep
############################


# Gözlem ve değişken sayısı
print("Shape: ", (spark_df.count(), len(spark_df.columns)))

# Değişken tipleri
spark_df.printSchema()
spark_df.dtypes

# Head
spark_df.show(3, truncate=True)

# row number silindi
spark_df = spark_df.drop("RowNumber", "Surname")

# Değişken isimlerinin küçültülmesi
spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns])
spark_df.show(5)

# Eşsiz sınıflar
for col in spark_df.columns:
    spark_df.groupby(col).count().show()

# Feature Interaction
bucketizer = Bucketizer(splits=[0, 30, 40, 50, 60, 92], inputCol="age", outputCol="new_age")
spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)

spark_df = spark_df.withColumn('new_numofproducts_tenure', spark_df.numofproducts * spark_df.tenure)
spark_df = spark_df.withColumn('new_numofproducts_tenure_active',
                               spark_df.isactivemember * spark_df.tenure * spark_df.numofproducts)
# spark_df = spark_df.withColumn('new_gender_credit_card', spark_df.gender + str(spark_df.hascrcard))
spark_df = spark_df.withColumn('new_tenure_age', spark_df.tenure + spark_df.new_age)
spark_df.show(5)

# Tüm numerik değişkenlerin seçimi ve özet istatistikleri
num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string' and
            spark_df.select(col[0]).distinct().count() > 11 and
            col[0] != "customerid"]

spark_df.select(num_cols).describe().toPandas().transpose()

# Tüm kategorik değişkenlerin seçimi ve özeti
cat_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string' or
            spark_df.select(col[0]).distinct().count() <= 11]

for col in cat_cols:
    spark_df.groupby(col).count().show()

# target değişkene göre num_cols ve cat_cols frekansları
for col in cat_cols:
    spark_df.groupby("exited").agg({col: ["count","avg"]}).show()

for col in num_cols:
    spark_df.groupby("exited").agg({col: "avg"}).show()
# Kategorik değişken sınıf istatistikleri
spark_df.groupby().count().show()

# missing value kontrol
spark_df.select([count(when(col(c).isNull(), c)).alias(c)
                 for c in spark_df.columns]).toPandas().T
############################
# Label Encoding
############################

spark_df.show(5)
indexer = StringIndexer(inputCol="segment", outputCol="segment_label")
indexer.fit(spark_df).transform(spark_df).show(5)

temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("segment_label", temp_sdf["segment_label"].cast("integer"))
spark_df.show(5)

spark_df = spark_df.drop('segment')

############################
# One Hot Encoding
############################
droplist = ["hascrcard", "isactivemember", "exited"]

for i in droplist:
    cat_cols.remove(i)
inputs = cat_cols

outputs = []
for i in cat_cols:
    outputs.append(i + "ohe")

stringIndexer = StringIndexer(inputCols=inputs, outputCols=outputs)

spark_df.show(5)

############################
# TARGET'ın Tanımlanması
############################

stringIndexer = StringIndexer(inputCol='exited', outputCol='label')
temp_sdf = stringIndexer.fit(spark_df).transform(spark_df)

spark_df = temp_sdf.withColumn("label", temp_sdf["label"].cast("integer"))
spark_df.show(5)

############################
# Feature'ların Tanımlanması
############################

cols = spark_df.columns

# Vectorize independent variables.
va = VectorAssembler(inputCols=cols, outputCol="features")
va_df = va.transform(spark_df)
va_df.show()

# Final sdf
final_df = va_df.select("features", "label")
final_df.show(5)

# Split the dataset into test and train sets.
train_df, test_df = final_df.randomSplit([0.7, 0.3], seed=17)
train_df.show(10)
test_df.show(10)

print("Training Dataset Count: " + str(train_df.count()))
print("Test Dataset Count: " + str(test_df.count()))

##################################################
# Modeling
##################################################
############################
# Gradient Boosted Tree Classifier
############################

gbm = GBTClassifier(maxIter=100, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df)
y_pred = gbm_model.transform(test_df)
y_pred.show(5)

y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()

############################
# Model Tuning
############################

evaluator = BinaryClassificationEvaluator()

gbm_params = (ParamGridBuilder()
              .addGrid(gbm.maxDepth, [2, 4, 6])
              .addGrid(gbm.maxBins, [20, 30])
              .addGrid(gbm.maxIter, [10, 20])
              .build())

cv = CrossValidator(estimator=gbm,
                    estimatorParamMaps=gbm_params,
                    evaluator=evaluator,
                    numFolds=5)

cv_model = cv.fit(train_df)

y_pred = cv_model.transform(test_df)
ac = y_pred.select("label", "prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()
