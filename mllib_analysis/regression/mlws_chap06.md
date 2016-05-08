#// mlws_chap06(Spark机器学习,第六章)

##// 0、封装包和代码环境
export SPARK_HOME=/Users/erichan/garden/spark-1.5.1-bin-hadoop2.6  
export PYTHONPATH=${SPARK_HOME}/python/:${SPARK_HOME}/python/lib/py4j-0.8.2.1-src.zip  

cd $SPARK_HOME  
IPYTHON=1 IPYTHON_OPTS="--pylab" ./bin/pyspark --driver-memory 4G --executor-memory 4G --driver-cores 2  

```scala
package mllib_book.mlws.chap06_regression

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.tree import DecisionTree
import numpy as np

object Lr01 extends App{  
  val conf = new SparkConf().setAppName("Spark_Lr").setMaster("local")  
  val sc = new SparkContext(conf)  
```  
  
##// 1、数据导入和特征抽取   
```scala
// first remove the headers by using the 'sed' command: 
// 数据集下载地址：http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset.  
// 论文下载地址：http://link.springer.com/article/10.1007%2Fs13748-013-0040-3.
// sed 1d hour.csv > hour_noheader.csv
path = "/PATH/hour_noheader.csv"
raw_data = sc.textFile(path)
num_data = raw_data.count()
records = raw_data.map(lambda x: x.split(","))

first = records.first()
print first
print num_data
//[u'1', u'2011-01-01', u'1', u'0', u'1', u'0', u'0', u'6', u'0', u'1', u'0.24',
//u'0.2879', u'0.81', u'0', u'3', u'13', u'16']
//17379

// 1.1、转换为二元向量
# cache the dataset to speed up subsequent operations
records.cache()
# function to get the categorical feature mapping for a given variable column
def get_mapping(rdd, idx):
    return rdd.map(lambda fields: fields[idx]).distinct().zipWithIndex().collectAsMap()

# we want to extract the feature mappings for columns 2 - 9
# try it out on column 2 first
print "Mapping of first categorical feasture column: %s" % get_mapping(records, 2)
//Mapping of first categorical feasture column: {u'1': 0, u'3': 1, u'2': 2, u'4': 3}

mappings = [get_mapping(records, i) for i in range(2,10)]
cat_len = sum(map(len, mappings))
num_len = len(records.first()[11:15])
total_len = num_len + cat_len

print "Feature vector length for categorical features: %d" % cat_len
print "Feature vector length for numerical features: %d" % num_len
print "Total feature vector length: %d" % total_len
//Feature vector length for categorical features: 57
//Feature vector length for numerical features: 4
//Total feature vector length: 61

// 1.2、创建线性模型特征向量
// # 提取特征
def extract_features(record):
    cat_vec = np.zeros(cat_len)
    i = 0
    step = 0
    for field in record[2:9]:
        m = mappings[i]
        idx = m[field]
        cat_vec[idx + step] = 1
        i = i + 1
        step = step + len(m)
    num_vec = np.array([float(field) for field in record[10:14]])
    return np.concatenate((cat_vec, num_vec))

// # 提取标签
def extract_label(record):
     return float(record[-1])

data = http://www.cnblogs.com/tychyg/p/records.map(lambda r: LabeledPoint(extract_label(r), extract_features(r)))

first_point = data.first()
print"Raw data: " + str(first[2:])
print "Label: " + str(first_point.label)
print "Linear Model feature vector:\n" + str(first_point.features)
print "Linear Model feature vector length: " + str(len(first_point.features))
//Raw data: [u'1', u'0', u'1', u'0', u'0', u'6', u'0', u'1', u'0.24', u'0.2879', u'0.81', u'0', u'3', u'13', u'16']
//Label: 16.0
//Linear Model feature vector:
//[1.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.24,0.2879,0.81,0.0]
//Linear Model feature vector length: 61

// 1.3、创建决策树模型特征向量
def extract_features_dt(record):
    return np.array(map(float, record[2:14]))

data_dt = records.map(lambda r: LabeledPoint(extract_label(r), extract_features_dt(r)))

first_point_dt = data_dt.first()
print "Decision Tree feature vector: " + str(first_point_dt.features)
print "Decision Tree feature vector length: " + str(len(first_point_dt.features))
//Decision Tree feature vector: [1.0,0.0,1.0,0.0,0.0,6.0,0.0,1.0,0.24,0.2879,0.81,0.0]
//Decision Tree feature vector length: 12
```  

##// 2、训练回归模型  
```scala
// 2.1、帮助
help(LinearRegressionWithSGD.train)
help(DecisionTree.trainRegressor)

// 2.2、训练线性模型并测试预测效果
linear_model = LinearRegressionWithSGD.train(data, iterations=10, step=0.1, intercept=False)
true_vs_predicted = data.map(lambda p: (p.label, linear_model.predict(p.features)))
print "Linear Model predictions: " + str(true_vs_predicted.take(5))
//Linear Model predictions: [(16.0, 117.89250386724845), (40.0, 116.2249612319211), (32.0, 116.02369145779234), (13.0, 115.67088016754433), (1.0, 115.56315650834317)]

// 2.3、训练决策树模型并测试预测效果
dt_model = DecisionTree.trainRegressor(data_dt, {})
preds = dt_model.predict(data_dt.map(lambda p: p.features))
actual = data.map(lambda p: p.label)
true_vs_predicted_dt = actual.zip(preds)

print "Decision Tree predictions: " + str(true_vs_predicted_dt.take(5))
print "Decision Tree depth: " + str(dt_model.depth())
print "Decision Tree number of nodes: " + str(dt_model.numNodes())
//Decision Tree predictions: [(16.0, 54.913223140495866), (40.0, 54.913223140495866), (32.0, 53.171052631578945), (13.0, 14.284023668639053), (1.0, 14.284023668639053)]
//Decision Tree depth: 5
//Decision Tree number of nodes: 63
```  

##// 3、评估回归模型性能
```scala
// 评估回归模型的方法：
// 均方误差(MSE, Mean Sequared Error)
// 均方根误差(RMSE, Root Mean Squared Error)
// 平均绝对误差(MAE, Mean Absolute Error)
// R-平方系数(R-squared coefficient)
// 均方根对数误差(RMSLE)

// 3.1、均方误差&均方根误差
def squared_error(actual, pred):
    return (pred - actual)**2

mse = true_vs_predicted.map(lambda (t, p): squared_error(t, p)).mean()
mse_dt = true_vs_predicted_dt.map(lambda (t, p): squared_error(t, p)).mean()

cat_features = dict([(i - 2, len(get_mapping(records, i)) + 1) for i in range(2,10)])

# train the model again
dt_model_2 = DecisionTree.trainRegressor(data_dt, categoricalFeaturesInfo=cat_features)
preds_2 = dt_model_2.predict(data_dt.map(lambda p: p.features))
actual_2 = data.map(lambda p: p.label)
true_vs_predicted_dt_2 = actual_2.zip(preds_2)

# compute performance metrics for decision tree model
mse_dt_2 = true_vs_predicted_dt_2.map(lambda (t, p): squared_error(t, p)).mean()

print "Linear Model - Mean Squared Error: %2.4f" % mse
print "Decision Tree - Mean Squared Error: %2.4f" % mse_dt
print "Categorical feature size mapping %s" % cat_features
print "Decision Tree [Categorical feature]- Mean Squared Error: %2.4f" % mse_dt_2
//Linear Model - Mean Squared Error: 30679.4539
//Decision Tree - Mean Squared Error: 11560.7978
//Decision Tree [Categorical feature]- Mean Squared Error: 7912.5642

//3.2、平均绝对误差
def abs_error(actual, pred):
    return np.abs(pred - actual)

mae = true_vs_predicted.map(lambda (t, p): abs_error(t, p)).mean()
mae_dt = true_vs_predicted_dt.map(lambda (t, p): abs_error(t, p)).mean()
mae_dt_2 = true_vs_predicted_dt_2.map(lambda (t, p): abs_error(t, p)).mean()

print "Linear Model - Mean Absolute Error: %2.4f" % mae
print "Decision Tree - Mean Absolute Error: %2.4f" % mae_dt
print "Decision Tree [Categorical feature]- Mean Absolute Error: %2.4f" % mae_dt_2
//Linear Model - Mean Absolute Error: 130.6429
//Decision Tree - Mean Absolute Error: 71.0969
//Decision Tree [Categorical feature]- Mean Absolute Error: 59.4409

// 3.3、均方根对数误差
def squared_log_error(pred, actual):
    return (np.log(pred + 1) - np.log(actual + 1))**2

rmsle = np.sqrt(true_vs_predicted.map(lambda (t, p): squared_log_error(t, p)).mean())
rmsle_dt = np.sqrt(true_vs_predicted_dt.map(lambda (t, p): squared_log_error(t, p)).mean())
rmsle_dt_2 = np.sqrt(true_vs_predicted_dt_2.map(lambda (t, p): squared_log_error(t, p)).mean())

print "Linear Model - Root Mean Squared Log Error: %2.4f" % rmsle
print "Decision Tree - Root Mean Squared Log Error: %2.4f" % rmsle_dt
print "Decision Tree [Categorical feature]- Root Mean Squared Log Error: %2.4f" % rmsle_dt_2
//Linear Model - Root Mean Squared Log Error: 1.4653
//Decision Tree - Root Mean Squared Log Error: 0.6259
//Decision Tree [Categorical feature]- Root Mean Squared Log Error: 0.6192
```
##// 4、评估分类模型性能
```scala
// 4.1、accuracy
// compute accuracy for logistic regression
val lrTotalCorrect = data.map { point =>
  if (lrModel.predict(point.features) == point.label) 1 else 0
}.sum
// lrTotalCorrect: Double = 3806.0

// accuracy is the number of correctly classified examples (same as true label)
// divided by the total number of examples
val lrAccuracy = lrTotalCorrect / numData
// lrAccuracy: Double = 0.5146720757268425

// compute accuracy for the other models
val svmTotalCorrect = data.map { point =>
  if (svmModel.predict(point.features) == point.label) 1 else 0
}.sum
val nbTotalCorrect = nbData.map { point =>
  if (nbModel.predict(point.features) == point.label) 1 else 0
}.sum
// decision tree threshold needs to be specified
val dtTotalCorrect = data.map { point =>
  val score = dtModel.predict(point.features)
  val predicted = if (score > 0.5) 1 else 0 
  if (predicted == point.label) 1 else 0
}.sum
val svmAccuracy = svmTotalCorrect / numData
// svmAccuracy: Double = 0.5146720757268425
val nbAccuracy = nbTotalCorrect / numData
// nbAccuracy: Double = 0.5803921568627451
val dtAccuracy = dtTotalCorrect / numData
// dtAccuracy: Double = 0.6482758620689655

// ### 4.2 ROC和AUC
// compute area under PR and ROC curves for each model
// generate binary classification metrics
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
val metrics = Seq(lrModel, svmModel).map { model => 
  val scoreAndLabels = data.map { point =>
    (model.predict(point.features), point.label)
  }
  val metrics = new BinaryClassificationMetrics(scoreAndLabels)
  (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
}
// again, we need to use the special nbData for the naive Bayes metrics 
val nbMetrics = Seq(nbModel).map{ model =>
  val scoreAndLabels = nbData.map { point =>
    val score = model.predict(point.features)
    (if (score > 0.5) 1.0 else 0.0, point.label)
  }	
  val metrics = new BinaryClassificationMetrics(scoreAndLabels)
  (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
}
// here we need to compute for decision tree separately since it does 
// not implement the ClassificationModel interface
val dtMetrics = Seq(dtModel).map{ model =>
  val scoreAndLabels = data.map { point =>
    val score = model.predict(point.features)
    (if (score > 0.5) 1.0 else 0.0, point.label)
  }	
  val metrics = new BinaryClassificationMetrics(scoreAndLabels)
  (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
}
val allMetrics = metrics ++ nbMetrics ++ dtMetrics
allMetrics.foreach{ case (m, pr, roc) => 
  println(f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%") 
}
/*
LogisticRegressionModel, Area under PR: 75.6759%, Area under ROC: 50.1418%
SVMModel, Area under PR: 75.6759%, Area under ROC: 50.1418%
NaiveBayesModel, Area under PR: 68.0851%, Area under ROC: 58.3559%
DecisionTreeModel, Area under PR: 74.3081%, Area under ROC: 64.8837%
*/
```

##// 5、模型参数调优
```scala
// 5.1、连续数值型特征的转换与提取
// standardizing the numerical features
import org.apache.spark.mllib.linalg.distributed.RowMatrix
val vectors = data.map(lp => lp.features)
val matrix = new RowMatrix(vectors)
val matrixSummary = matrix.computeColumnSummaryStatistics()

println(matrixSummary.mean)
// [0.41225805299526636,2.761823191986623,0.46823047328614004, ...
println(matrixSummary.min)
// [0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,0.045564223,-1.0, ...
println(matrixSummary.max)
// [0.999426,363.0,1.0,1.0,0.980392157,0.980392157,21.0,0.25,0.0,0.444444444, ...
println(matrixSummary.variance)
// [0.1097424416755897,74.30082476809638,0.04126316989120246, ...
println(matrixSummary.numNonzeros)
// [5053.0,7354.0,7172.0,6821.0,6160.0,5128.0,7350.0,1257.0,0.0,7362.0, ...

// scale the input features using MLlib's StandardScaler
import org.apache.spark.mllib.feature.StandardScaler
val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
val scaledData = data.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
// compare the raw features with the scaled features
println(data.first.features)
// [0.789131,2.055555556,0.676470588,0.205882353,
println(scaledData.first.features)
// [1.1376439023494747,-0.08193556218743517,1.025134766284205,-0.0558631837375738,
println((0.789131 - 0.41225805299526636)/math.sqrt(0.1097424416755897))
 // 1.137647336497682

// train a logistic regression model on the scaled data, and compute metrics
val lrModelScaled = LogisticRegressionWithSGD.train(scaledData, numIterations)
val lrTotalCorrectScaled = scaledData.map { point =>
  if (lrModelScaled.predict(point.features) == point.label) 1 else 0
}.sum
val lrAccuracyScaled = lrTotalCorrectScaled / numData
// lrAccuracyScaled: Double = 0.6204192021636241
val lrPredictionsVsTrue = scaledData.map { point => 
	(lrModelScaled.predict(point.features), point.label) 
}
val lrMetricsScaled = new BinaryClassificationMetrics(lrPredictionsVsTrue)
val lrPr = lrMetricsScaled.areaUnderPR
val lrRoc = lrMetricsScaled.areaUnderROC
println(f"${lrModelScaled.getClass.getSimpleName}\nAccuracy: ${lrAccuracyScaled * 100}%2.4f%%\nArea under PR: ${lrPr * 100.0}%2.4f%%\nArea under ROC: ${lrRoc * 100.0}%2.4f%%") 
/*
LogisticRegressionModel
Accuracy: 62.0419%
Area under PR: 72.7254%
Area under ROC: 61.9663%
*/

// 5.2、类别型特征的转换与提取
// Investigate the impact of adding in the 'category' feature
val categories = records.map(r => r(3)).distinct.collect.zipWithIndex.toMap
// categories: scala.collection.immutable.Map[String,Int] = Map("weather" -> 0, "sports" -> 6, 
//	"unknown" -> 4, "computer_internet" -> 12, "?" -> 11, "culture_politics" -> 3, "religion" -> 8,
// "recreation" -> 2, "arts_entertainment" -> 9, "health" -> 5, "law_crime" -> 10, "gaming" -> 13, 
// "business" -> 1, "science_technology" -> 7)
val numCategories = categories.size
// numCategories: Int = 14
val dataCategories = records.map { r =>
	val trimmed = r.map(_.replaceAll("\"", ""))
	val label = trimmed(r.size - 1).toInt
	val categoryIdx = categories(r(3))
	val categoryFeatures = Array.ofDim[Double](numCategories)
	categoryFeatures(categoryIdx) = 1.0
	val otherFeatures = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
	val features = categoryFeatures ++ otherFeatures
	LabeledPoint(label, Vectors.dense(features))
}
println(dataCategories.first)
// LabeledPoint(0.0, [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.789131,2.055555556,
//	0.676470588,0.205882353,0.047058824,0.023529412,0.443783175,0.0,0.0,0.09077381,0.0,0.245831182,
// 0.003883495,1.0,1.0,24.0,0.0,5424.0,170.0,8.0,0.152941176,0.079129575])


// standardize the feature vectors
val scalerCats = new StandardScaler(withMean = true, withStd = true).fit(dataCategories.map(lp => lp.features))
val scaledDataCats = dataCategories.map(lp => LabeledPoint(lp.label, scalerCats.transform(lp.features)))
println(dataCategories.first.features)
// [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.789131,2.055555556,0.676470588,0.205882353,
// 0.047058824,0.023529412,0.443783175,0.0,0.0,0.09077381,0.0,0.245831182,0.003883495,1.0,1.0,24.0,0.0,
// 5424.0,170.0,8.0,0.152941176,0.079129575]
println(scaledDataCats.first.features)
/*
[-0.023261105535492967,2.720728254208072,-0.4464200056407091,-0.2205258360869135,-0.028492999745483565,
-0.2709979963915644,-0.23272692307249684,-0.20165301179556835,-0.09914890962355712,-0.381812077600508,
-0.06487656833429316,-0.6807513271391559,-0.2041811690290381,-0.10189368073492189,1.1376439023494747,
-0.08193556218743517,1.0251347662842047,-0.0558631837375738,-0.4688883677664047,-0.35430044806743044
,-0.3175351615705111,0.3384496941616097,0.0,0.8288021759842215,-0.14726792180045598,0.22963544844991393,
-0.14162589530918376,0.7902364255801262,0.7171932152231301,-0.29799680188379124,-0.20346153667348232,
-0.03296720969318916,-0.0487811294839849,0.9400696843533806,-0.10869789547344721,-0.2788172632659348]
*/

// train model on scaled data and evaluate metrics
val lrModelScaledCats = LogisticRegressionWithSGD.train(scaledDataCats, numIterations)
val lrTotalCorrectScaledCats = scaledDataCats.map { point =>
  if (lrModelScaledCats.predict(point.features) == point.label) 1 else 0
}.sum
val lrAccuracyScaledCats = lrTotalCorrectScaledCats / numData
val lrPredictionsVsTrueCats = scaledDataCats.map { point => 
	(lrModelScaledCats.predict(point.features), point.label) 
}
val lrMetricsScaledCats = new BinaryClassificationMetrics(lrPredictionsVsTrueCats)
val lrPrCats = lrMetricsScaledCats.areaUnderPR
val lrRocCats = lrMetricsScaledCats.areaUnderROC
println(f"${lrModelScaledCats.getClass.getSimpleName}\nAccuracy: ${lrAccuracyScaledCats * 100}%2.4f%%\nArea under PR: ${lrPrCats * 100.0}%2.4f%%\nArea under ROC: ${lrRocCats * 100.0}%2.4f%%") 
/*
LogisticRegressionModel
Accuracy: 66.5720%
Area under PR: 75.7964%
Area under ROC: 66.5483%
*/

// 5.3、使用正确的数据格式
// 使用1-of-k编码的类型特征构建数据集
// train naive Bayes model with only categorical data
val dataNB = records.map { r =>
	val trimmed = r.map(_.replaceAll("\"", ""))
	val label = trimmed(r.size - 1).toInt
	val categoryIdx = categories(r(3))
	val categoryFeatures = Array.ofDim[Double](numCategories)
	categoryFeatures(categoryIdx) = 1.0
	LabeledPoint(label, Vectors.dense(categoryFeatures))
}
val nbModelCats = NaiveBayes.train(dataNB)
val nbTotalCorrectCats = dataNB.map { point =>
  if (nbModelCats.predict(point.features) == point.label) 1 else 0
}.sum
val nbAccuracyCats = nbTotalCorrectCats / numData
val nbPredictionsVsTrueCats = dataNB.map { point => 
	(nbModelCats.predict(point.features), point.label) 
}
val nbMetricsCats = new BinaryClassificationMetrics(nbPredictionsVsTrueCats)
val nbPrCats = nbMetricsCats.areaUnderPR
val nbRocCats = nbMetricsCats.areaUnderROC
println(f"${nbModelCats.getClass.getSimpleName}\nAccuracy: ${nbAccuracyCats * 100}%2.4f%%\nArea under PR: ${nbPrCats * 100.0}%2.4f%%\nArea under ROC: ${nbRocCats * 100.0}%2.4f%%") 
/*
NaiveBayesModel
Accuracy: 60.9601%
Area under PR: 74.0522%
Area under ROC: 60.5138%
*/

// 5.4、模型参数调优 主要是模型的参数
// investigate the impact of model parameters on performance
// create a training function
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.optimization.Updater
import org.apache.spark.mllib.optimization.SimpleUpdater
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.mllib.classification.ClassificationModel

// 5.4.1、逻辑斯特回归模型
// helper function to train a logistic regresson model
def trainWithParams(input: RDD[LabeledPoint], regParam: Double, numIterations: Int, updater: Updater, stepSize: Double) = {
	val lr = new LogisticRegressionWithSGD
	lr.optimizer.setNumIterations(numIterations).setUpdater(updater).setRegParam(regParam).setStepSize(stepSize)
	lr.run(input)
}
// helper function to create AUC metric
def createMetrics(label: String, data: RDD[LabeledPoint], model: ClassificationModel) = {
	val scoreAndLabels = data.map { point =>
  		(model.predict(point.features), point.label)
	}
	val metrics = new BinaryClassificationMetrics(scoreAndLabels)
	(label, metrics.areaUnderROC)
}

// cache the data to increase speed of multiple runs agains the dataset
scaledDataCats.cache
// num iterations
val iterResults = Seq(1, 5, 10, 50).map { param =>
	val model = trainWithParams(scaledDataCats, 0.0, param, new SimpleUpdater, 1.0)
	createMetrics(s"$param iterations", scaledDataCats, model)
}
iterResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
/*
1 iterations, AUC = 64.97%
5 iterations, AUC = 66.62%
10 iterations, AUC = 66.55%
50 iterations, AUC = 66.81%
*/

// step size
val stepResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
	val model = trainWithParams(scaledDataCats, 0.0, numIterations, new SimpleUpdater, param)
	createMetrics(s"$param step size", scaledDataCats, model)
}
stepResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
/*
0.001 step size, AUC = 64.95%
0.01 step size, AUC = 65.00%
0.1 step size, AUC = 65.52%
1.0 step size, AUC = 66.55%
10.0 step size, AUC = 61.92%
*/

// regularization
val regResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
	val model = trainWithParams(scaledDataCats, param, numIterations, new SquaredL2Updater, 1.0)
	createMetrics(s"$param L2 regularization parameter", scaledDataCats, model)
}
regResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
/*
0.001 L2 regularization parameter, AUC = 66.55%
0.01 L2 regularization parameter, AUC = 66.55%
0.1 L2 regularization parameter, AUC = 66.63%
1.0 L2 regularization parameter, AUC = 66.04%
10.0 L2 regularization parameter, AUC = 35.33%
*/

// 5.4.2、决策树模型
// investigate decision tree
import org.apache.spark.mllib.tree.impurity.Impurity
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.tree.impurity.Gini
def trainDTWithParams(input: RDD[LabeledPoint], maxDepth: Int, impurity: Impurity) = {
	DecisionTree.train(input, Algo.Classification, impurity, maxDepth)
}
 
// investigate tree depth impact for Entropy impurity
val dtResultsEntropy = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
	val model = trainDTWithParams(data, param, Entropy)
	val scoreAndLabels = data.map { point =>
		val score = model.predict(point.features)
  		(if (score > 0.5) 1.0 else 0.0, point.label)
	}
	val metrics = new BinaryClassificationMetrics(scoreAndLabels)
	(s"$param tree depth", metrics.areaUnderROC)
}
dtResultsEntropy.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
/*
1 tree depth, AUC = 59.33%
2 tree depth, AUC = 61.68%
3 tree depth, AUC = 62.61%
4 tree depth, AUC = 63.63%
5 tree depth, AUC = 64.88%
10 tree depth, AUC = 76.26%
20 tree depth, AUC = 98.45%
*/

// investigate tree depth impact for Gini impurity
val dtResultsGini = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
	val model = trainDTWithParams(data, param, Gini)
	val scoreAndLabels = data.map { point =>
		val score = model.predict(point.features)
  		(if (score > 0.5) 1.0 else 0.0, point.label)
	}
	val metrics = new BinaryClassificationMetrics(scoreAndLabels)
	(s"$param tree depth", metrics.areaUnderROC)
}
dtResultsGini.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
/*
1 tree depth, AUC = 59.33%
2 tree depth, AUC = 61.68%
3 tree depth, AUC = 62.61%
4 tree depth, AUC = 63.63%
5 tree depth, AUC = 64.89%
10 tree depth, AUC = 78.37%
20 tree depth, AUC = 98.87%
*/

// 5.4.3、朴素贝叶斯模型
// investigate Naive Bayes parameters
def trainNBWithParams(input: RDD[LabeledPoint], lambda: Double) = {
	val nb = new NaiveBayes
	nb.setLambda(lambda)
	nb.run(input)
}
val nbResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
	val model = trainNBWithParams(dataNB, param)
	val scoreAndLabels = dataNB.map { point =>
  		(model.predict(point.features), point.label)
	}
	val metrics = new BinaryClassificationMetrics(scoreAndLabels)
	(s"$param lambda", metrics.areaUnderROC)
}
nbResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
/*
0.001 lambda, AUC = 60.51%
0.01 lambda, AUC = 60.51%
0.1 lambda, AUC = 60.51%
1.0 lambda, AUC = 60.51%
10.0 lambda, AUC = 60.51%
*/

// 5.4.4、交叉验证
// illustrate cross-validation
// create a 60% / 40% train/test data split
val trainTestSplit = scaledDataCats.randomSplit(Array(0.6, 0.4), 123)
val train = trainTestSplit(0)
val test = trainTestSplit(1)
// now we train our model using the 'train' dataset, and compute predictions on unseen 'test' data
// in addition, we will evaluate the differing performance of regularization on training and test datasets
val regResultsTest = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
	val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, 1.0)
	createMetrics(s"$param L2 regularization parameter", test, model)
}
regResultsTest.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.6f%%") }
/*
0.0 L2 regularization parameter, AUC = 66.480874%
0.001 L2 regularization parameter, AUC = 66.480874%
0.0025 L2 regularization parameter, AUC = 66.515027%
0.005 L2 regularization parameter, AUC = 66.515027%
0.01 L2 regularization parameter, AUC = 66.549180%
*/

// training set results
val regResultsTrain = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
	val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, 1.0)
	createMetrics(s"$param L2 regularization parameter", train, model)
}
regResultsTrain.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.6f%%") }
/*
0.0 L2 regularization parameter, AUC = 66.260311%
0.001 L2 regularization parameter, AUC = 66.260311%
0.0025 L2 regularization parameter, AUC = 66.260311%
0.005 L2 regularization parameter, AUC = 66.238294%
0.01 L2 regularization parameter, AUC = 66.238294%
*/
```
