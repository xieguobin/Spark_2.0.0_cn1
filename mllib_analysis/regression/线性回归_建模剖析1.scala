//线性回归_建模剖析1

//一、数学理论


//二、建模实例
package org.apache.spark.mllib_analysis.regression

import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors

object Lr01 extends App{
  val conf = new SparkConf().setAppName("Spark_Lr").setMaster("local")
  val sc = new SparkContext(conf)
  
//获取数据
val data = sc.textFile("C:/my_install/spark/data/mllib/ridge-data/lpsa.data")
val parsedData = data.map { line =>
  val parts = line.split(',')
  LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
}.cache()

//模型训练
val numIterations = 100
val stepSize = 0.00000001
val model = LinearRegressionWithSGD.train(parsedData, numIterations, stepSize)

//模型评价
val valuesAndPreds = parsedData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
println("training Mean Squared Error = " + MSE)

//模型保存与加载
//model.save(sc, "myModelPath")
//val sameModel = LinearRegressionModel.load(sc, "myModelPath")

}

//三、源码调用解析
//1、LeastSquaresGradient
//训练过程均使用GeneralizedLinearModel中的run训练，只是训练使用的Gradient和Updater不同。在一般的线性回归中，
//使用LeastSquaresGradient计算梯度，使用SimpleUpdater进行更新。 它的实现过程分为4步。普通线性回归的损失函数是最
//小二乘损失。
class LeastSquaresGradient extends Gradient {
  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    //diff = xw-y
    val diff = dot(data, weights) - label
    val loss = diff * diff / 2.0
    val gradient = data.copy
    //gradient = diff * gradient
    scal(diff, gradient)
    (gradient, loss)
  }
  override def compute(
      data: Vector,
      label: Double,
      weights: Vector,
      cumGradient: Vector): Double = {
    //diff = xw-y
    val diff = dot(data, weights) - label
    //计算梯度
    //cumGradient += diff * data
    axpy(diff, data, cumGradient)
    diff * diff / 2.0
  }
}

//2、普通线性回归SimpleUpdater
//普通线性回归的不适用正则化方法，所以它用SimpleUpdater实现Updater。
class SimpleUpdater extends Updater {
  override def compute(
      weightsOld: Vector,
      gradient: Vector,
      stepSize: Double,
      iter: Int,
      regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    //计算 y += x * a，即 brzWeights -= thisIterStepSize * gradient.toBreeze
    //梯度下降方向
    brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
    (Vectors.fromBreeze(brzWeights), 0)
  }
}
  //这里thisIterStepSize表示参数沿负梯度方向改变的速率，它随着迭代次数的增多而减小。
  
//3、lasso回归L1Updater
//lasso回归和普通线性回归不同的地方是，它使用L1正则化方法。即使用L1Updater。
class L1Updater extends Updater {
  override def compute(
      weightsOld: Vector,
      gradient: Vector,
      stepSize: Double,
      iter: Int,
      regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    //计算 y += x * a，即 brzWeights -= thisIterStepSize * gradient.toBreeze
    //梯度下降方向
    val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
    // Apply proximal operator (soft thresholding)
    val shrinkageVal = regParam * thisIterStepSize
    var i = 0
    val len = brzWeights.length
    while (i < len) {
      val wi = brzWeights(i)
      brzWeights(i) = signum(wi) * max(0.0, abs(wi) - shrinkageVal)
      i += 1
    }
    (Vectors.fromBreeze(brzWeights), brzNorm(brzWeights, 1.0) * regParam)
  }
}
//这个类解决L1范式正则化问题。这里thisIterStepSize表示参数沿负梯度方向改变的速率，它随着迭代次数的增多而减小。
//该实现没有使用线性模型中介绍的子梯度方法，而是使用了邻近算子（proximal operator）来解决，该方法的结果拥有更好的稀疏性。 
//L1范式的邻近算子是软阈值（soft-thresholding）函数。
//当w > shrinkageVal时，权重组件等于w-shrinkageVal
//当w < -shrinkageVal时，权重组件等于w+shrinkageVal
//当-shrinkageVal < w < shrinkageVal时，权重组件等于0
//signum函数是子梯度函数，当w<0时，返回-1，当w>0时，返回1，当w=0时，返回0。

//4、ridge回归SquaredL2Updater
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

