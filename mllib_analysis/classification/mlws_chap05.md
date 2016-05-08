# mlws_chap05

## 封装包和代码环境
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
  
  
## 1、数据导入和特征抽取  
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



