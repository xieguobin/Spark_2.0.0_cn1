# mlws_chap05

## 0、封装包和代码环境
```scala
package mllib_book.mlws.chap05_classification

import org.apache.spark.{SparkConf,SparkContext}  
import org.apache.spark.mllib.feature._  
import org.apache.spark.mllib.regression.LabeledPoint  
import org.apache.spark.mllib.linalg.Vectors  
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics  
import org.apache.spark.mllib.linalg.distributed.RowMatrix  
import org.apache.spark.rdd.RDD  
import org.apache.spark.mllib.optimization.  
import org.apache.spark.mllib.classification._  
import org.apache.spark.mllib.evaluation._   
import org.apache.spark.mllib.tree.DecisionTree  
import org.apache.spark.mllib.tree.configuration.Algo  
import org.apache.spark.mllib.tree.impurity._  

object Lr01 extends App{  
  val conf = new SparkConf().setAppName("Spark_Lr").setMaster("local")  
  val sc = new SparkContext(conf)  
```  
  
## 1、数据导入和特征抽取   
```scala
// kaggle2.blob.core.windows.net/competitions-data/kaggle/3526/train.tsv  
// sed 1d train.tsv > train_noheader.tsv  
// load raw data
val rawData = sc.textFile("C:/spark_in_data/train_noheader.tsv")  
val records = rawData.map(line => line.split("\t"))  
records.first  
// Array[String] = Array("http://www.bloomberg.com/news/2010-12-23/ibm-predicts-holographic-calls-air-breathing-batteries-by-2015.html", "4042", ...  

val data = records.map { r =>  
  val trimmed = r.map(_.replaceAll("\"", ""))  
  val label = trimmed(r.size - 1).toInt  
  val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)  
  LabeledPoint(label, Vectors.dense(features))  
}  
data.cache  
val numData = data.count  
// numData: Long = 7395  

// note that some of our data contains negative feature vaues. For naive Bayes we convert these to zeros
val nbData = records.map { r =>
  val trimmed = r.map(_.replaceAll("\"", ""))
  val label = trimmed(r.size - 1).toInt
  val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble).map(d => if (d < 0) 0.0 else d)
  LabeledPoint(label, Vectors.dense(features))
}
```  

// ## 2、训练分类模型  
```scala
val numIterations = 10
val maxTreeDepth = 5
val lrModel = LogisticRegressionWithSGD.train(data, numIterations)                   //逻辑斯特回归
val svmModel = SVMWithSGD.train(data, numIterations)                                 //支持向量机
// note we use nbData here for the NaiveBayes model training
val nbModel = NaiveBayes.train(nbData)                                               //朴素贝叶斯
val dtModel = DecisionTree.train(data, Algo.Classification, Entropy, maxTreeDepth)   //决策树
```  

## 3、使用分类模型进行预测
```scala
val dataPoint = data.first
// dataPoint: org.apache.spark.mllib.regression.LabeledPoint = LabeledPoint(0.0, [0.789131,2.055555556,0.676470588, ...
val prediction = lrModel.predict(dataPoint.features)
// prediction: Double = 1.0
val trueLabel = dataPoint.label
// trueLabel: Double = 0.0
val predictions = lrModel.predict(data.map(lp => lp.features))
predictions.take(5)
// res1: Array[Double] = Array(1.0, 1.0, 1.0, 1.0, 1.0)
```
## 4、评估分类模型性能
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




