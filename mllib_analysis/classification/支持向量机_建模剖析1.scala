//支持向量机_建模剖析1

//一、数学理论


//二、建模实例
package org.apache.spark.mllib_analysis.classification

import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

object Lr01 extends App{
  val conf = new SparkConf().setAppName("Spark_Lr").setMaster("local")
  val sc = new SparkContext(conf)
  
//加载训练数据、切分数据
val data = MLUtils.loadLibSVMFile(sc, "C:/my_install/spark/data/mllib/sample_libsvm_data.txt")
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)

//训练模型
val numIterations = 100
val model = SVMWithSGD.train(training, numIterations)
// Clear the default threshold.
model.clearThreshold()

//模型测试
val scoreAndLabels = test.map { point =>
  val score = model.predict(point.features)
  (score, point.label)
}

//模型评估
val metrics = new BinaryClassificationMetrics(scoreAndLabels)
val auROC = metrics.areaUnderROC()
println("Area under ROC = " + auROC)

}

//三、源码调用解析
//和逻辑回归一样，训练过程均使用GeneralizedLinearModel中的run训练，只是训练使用的Gradient和Updater不同。
//在线性支持向量机中，使用HingeGradient计算梯度，使用SquaredL2Updater进行更新。 它的实现过程分为四个步骤。
//参考逻辑斯特回归了解这几步的详情。下面描述HingeGradient和SquaredL2Updater的实现。
//1、HingeGradient
class HingeGradient extends Gradient {
  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val dotProduct = dot(data, weights)
    // 我们的损失函数是 max(0, 1 - (2y - 1) (f_w(x)))
    // 所以梯度是 -(2y - 1)*x
    val labelScaled = 2 * label - 1.0
    if (1.0 > labelScaled * dotProduct) {
      val gradient = data.copy
      scal(-labelScaled, gradient)
      (gradient, 1.0 - labelScaled * dotProduct)
    } else {
      (Vectors.sparse(weights.size, Array.empty, Array.empty), 0.0)
    }
  }

  override def compute(
      data: Vector,
      label: Double,
      weights: Vector,
      cumGradient: Vector): Double = {
    val dotProduct = dot(data, weights)
    // 我们的损失函数是 max(0, 1 - (2y - 1) (f_w(x)))
    // 所以梯度是 -(2y - 1)*x
    val labelScaled = 2 * label - 1.0
    if (1.0 > labelScaled * dotProduct) {
      //cumGradient -= labelScaled * data
      axpy(-labelScaled, data, cumGradient)
      //损失值
      1.0 - labelScaled * dotProduct
    } else {
      0.0
    }
  }
}

//2、SquaredL2Updater
class SquaredL2Updater extends Updater {
  override def compute(
      weightsOld: Vector,
      gradient: Vector,
      stepSize: Double,
      iter: Int,
      regParam: Double): (Vector, Double) = {
    // w' = w - thisIterStepSize * (gradient + regParam * w)
    // w' = (1 - thisIterStepSize * regParam) * w - thisIterStepSize * gradient
    //表示步长，即负梯度方向的大小
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    //正则化，brzWeights每行数据均乘以(1.0 - thisIterStepSize * regParam)
    brzWeights :*= (1.0 - thisIterStepSize * regParam)
    //y += x * a，即brzWeights -= gradient * thisInterStepSize
    brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
    //正则化||w||_2
    val norm = brzNorm(brzWeights, 2.0)
    (Vectors.fromBreeze(brzWeights), 0.5 * regParam * norm * norm)
  }
}

//该函数的实现规则是：
//w' = w - thisIterStepSize * (gradient + regParam * w)
//w' = (1 - thisIterStepSize * regParam) * w - thisIterStepSize * gradient
//这里thisIterStepSize表示参数沿负梯度方向改变的速率，它随着迭代次数的增多而减小。

//3、模型打分的方法
override protected def predictPoint(
      dataMatrix: Vector,
      weightMatrix: Vector,
      intercept: Double) = {
    //w^Tx
    val margin = weightMatrix.toBreeze.dot(dataMatrix.toBreeze) + intercept
    threshold match {
      case Some(t) => if (margin > t) 1.0 else 0.0
      case None => margin
    }
  }
  
  
