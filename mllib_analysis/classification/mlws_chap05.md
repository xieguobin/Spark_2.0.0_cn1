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
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity._
