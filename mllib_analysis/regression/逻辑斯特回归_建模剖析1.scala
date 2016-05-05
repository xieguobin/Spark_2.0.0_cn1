//逻辑斯特回归_建模剖析1

//一、数学理论


//二、建模实例
package org.apache.spark.mllib_analysis.regression

import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils

object Lr01 extends App{
  val conf = new SparkConf().setAppName("Spark_Lr").setMaster("local")
  val sc = new SparkContext(conf)

//加载训练数据
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

//切分数据，training (60%) and test (40%).
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)

//训练模型
val model = new LogisticRegressionWithLBFGS()
  .setNumClasses(10)
  .run(training)
  
//模型测试
val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

//模型评估
val metrics = new MulticlassMetrics(predictionAndLabels)
val precision = metrics.precision
println("Precision = " + precision)

//保存和加载模型
model.save(sc, "myModelPath")
val sameModel = LogisticRegressionModel.load(sc, "myModelPath")

}

//三、源码调用解析
//(一)、建立线性回归
//1、线性回归伴生对象 object LinearRegressionWithSGD
     //详见https://github.com/xieguobin/Spark_2.0.0_cn1/blob/master/mllib/regression/LinearRegression.scala 第124行
//2、线性回归类 class LinearRegressionWithSGD
     //详见https://github.com/xieguobin/Spark_2.0.0_cn1/blob/master/mllib/regression/LinearRegression.scala 第90行  
     
//(二)、模型训练run方法
     //详见https://github.com/xieguobin/Spark_2.0.0_cn1/blob/master/mllib/regression/GeneralizedLinearAlgorithm.scala 第243行开始
//1、训练模型
//逻辑斯特回归中，分别使用了梯度下降法和L-BFGS实现逻辑回归参数的计算。这两个算法的实现在最优化部分进行介绍，这里介绍公共的部分。
//LogisticRegressionWithLBFGS和LogisticRegressionWithSGD的入口函数均是GeneralizedLinearAlgorithm.run，下面详细分析该方法。
def run(input: RDD[LabeledPoint]): M = {
    if (numFeatures < 0) {
      //计算特征数
      numFeatures = input.map(_.features.size).first()
    }
    val initialWeights = {
          if (numOfLinearPredictor == 1) {
            Vectors.zeros(numFeatures)
          } else if (addIntercept) {
            Vectors.zeros((numFeatures + 1) * numOfLinearPredictor)
          } else {
            Vectors.zeros(numFeatures * numOfLinearPredictor)
          }
    }
    run(input, initialWeights)
}
//上面的代码初始化权重向量，向量的值均初始化为0。需要注意的是，addIntercept表示是否添加截距(Intercept，指函数图形与坐标的交点到原点的距离)，默认是不添加的。numOfLinearPredictor表示二元逻辑回归模型的个数。 我们重点看run(input, initialWeights)的实现。它的实现分四步。

//2、根据提供的参数缩放特征并添加截距
val scaler = if (useFeatureScaling) {
      new StandardScaler(withStd = true, withMean = false).fit(input.map(_.features))
    } else {
      null
    }
val data =
      if (addIntercept) {
        if (useFeatureScaling) {
          input.map(lp => (lp.label, appendBias(scaler.transform(lp.features)))).cache()
        } else {
          input.map(lp => (lp.label, appendBias(lp.features))).cache()
        }
      } else {
        if (useFeatureScaling) {
          input.map(lp => (lp.label, scaler.transform(lp.features))).cache()
        } else {
          input.map(lp => (lp.label, lp.features))
        }
      }
val initialWeightsWithIntercept = if (addIntercept && numOfLinearPredictor == 1) {
      appendBias(initialWeights)
    } else {
      /** If `numOfLinearPredictor > 1`, initialWeights already contains intercepts. */
      initialWeights
    }
//在最优化过程中，收敛速度依赖于训练数据集的条件数(condition number)，缩放变量经常可以启发式地减少这些条件数，提高收敛速度。不减少条件数，一些混合有不同范围列的数据集可能不能收敛。 在这里使用StandardScaler将数据集的特征进行缩放。详细信息请看StandardScaler。appendBias方法很简单，就是在每个向量后面加一个值为1的项。

def appendBias(vector: Vector): Vector = {
    vector match {
      case dv: DenseVector =>
        val inputValues = dv.values
        val inputLength = inputValues.length
        val outputValues = Array.ofDim[Double](inputLength + 1)
        System.arraycopy(inputValues, 0, outputValues, 0, inputLength)
        outputValues(inputLength) = 1.0
        Vectors.dense(outputValues)
      case sv: SparseVector =>
        val inputValues = sv.values
        val inputIndices = sv.indices
        val inputValuesLength = inputValues.length
        val dim = sv.size
        val outputValues = Array.ofDim[Double](inputValuesLength + 1)
        val outputIndices = Array.ofDim[Int](inputValuesLength + 1)
        System.arraycopy(inputValues, 0, outputValues, 0, inputValuesLength)
        System.arraycopy(inputIndices, 0, outputIndices, 0, inputValuesLength)
        outputValues(inputValuesLength) = 1.0
        outputIndices(inputValuesLength) = dim
        Vectors.sparse(dim + 1, outputIndices, outputValues)
      case _ => throw new IllegalArgumentException(s"Do not support vector type ${vector.getClass}")
    }

//3、使用最优化算法计算最终的权重值
val weightsWithIntercept = optimizer.optimize(data, initialWeightsWithIntercept)
//梯度下降算法和L-BFGS两种算法来计算最终的权重值，查看梯度下降法和L-BFGS了解详细实现。 这两种算法均使用Gradient的实现类计算梯度，使用Updater的实现类更新参数。在LogisticRegressionWithSGD和LogisticRegressionWithLBFGS中，它们均使用LogisticGradient实现类计算梯度，使用SquaredL2Updater实现类更新参数。

//在LinearRegressionWithSGD类中调用GradientDescent.scala
private val gradient = new LogisticGradient()
private val updater = new SquaredL2Updater()
override val optimizer = new GradientDescent(gradient, updater)
    .setStepSize(stepSize)
    .setNumIterations(numIterations)
    .setRegParam(regParam)
    .setMiniBatchFraction(miniBatchFraction)
//在LBFGS中
override val optimizer = new LBFGS(new LogisticGradient, new SquaredL2Updater)

//(三)、最优化计算方法 
//1、GradientDescent.scala
//下面将详细介绍LogisticGradient的实现和SquaredL2Updater的实现。
//1.1、LogisticGradient
//LogisticGradient中使用compute方法计算梯度。计算分为两种情况，即二元逻辑回归的情况和多元逻辑回归的情况。虽然多元逻辑回归也可以实现二元分类，但是为了效率，compute方法仍然实现了一个二元逻辑回归的版本。
val margin = -1.0 * dot(data, weights)
val multiplier = (1.0 / (1.0 + math.exp(margin))) - label
//y += a * x，即cumGradient += multiplier * data
axpy(multiplier, data, cumGradient)
if (label > 0) {
    // The following is equivalent to log(1 + exp(margin)) but more numerically stable.
    MLUtils.log1pExp(margin)
} else {
    MLUtils.log1pExp(margin) - margin
}
//这里的multiplier就是上文的公式(2)。axpy方法用于计算梯度，这里表示的意思是h(x) * x。下面是多元逻辑回归的实现方法。

//权重
val weightsArray = weights match {
    case dv: DenseVector => dv.values
    case _ =>
            throw new IllegalArgumentException
}
//梯度
val cumGradientArray = cumGradient match {
    case dv: DenseVector => dv.values
    case _ =>
        throw new IllegalArgumentException
}
// 计算所有类别中最大的margin
var marginY = 0.0
var maxMargin = Double.NegativeInfinity
var maxMarginIndex = 0
val margins = Array.tabulate(numClasses - 1) { i =>
    var margin = 0.0
    data.foreachActive { (index, value) =>
        if (value != 0.0) margin += value * weightsArray((i * dataSize) + index)
    }
    if (i == label.toInt - 1) marginY = margin
    if (margin > maxMargin) {
            maxMargin = margin
            maxMarginIndex = i
    }
    margin
}
//计算sum，保证每个margin都小于0，避免出现算术溢出的情况
val sum = {
     var temp = 0.0
     if (maxMargin > 0) {
         for (i <- 0 until numClasses - 1) {
              margins(i) -= maxMargin
              if (i == maxMarginIndex) {
                temp += math.exp(-maxMargin)
              } else {
                temp += math.exp(margins(i))
              }
         }
     } else {
         for (i <- 0 until numClasses - 1) {
              temp += math.exp(margins(i))
         }
     }
     temp
}
//计算multiplier并计算梯度
for (i <- 0 until numClasses - 1) {
     val multiplier = math.exp(margins(i)) / (sum + 1.0) - {
          if (label != 0.0 && label == i + 1) 1.0 else 0.0
     }
     data.foreachActive { (index, value) =>
         if (value != 0.0) cumGradientArray(i * dataSize + index) += multiplier * value
     }
}
//计算损失函数,
val loss = if (label > 0.0) math.log1p(sum) - marginY else math.log1p(sum)
if (maxMargin > 0) {
     loss + maxMargin
} else {
     loss
}

//1.2、SquaredL2Updater
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

//1.3、对最终的权重值进行后处理

val intercept = if (addIntercept && numOfLinearPredictor == 1) {
      weightsWithIntercept(weightsWithIntercept.size - 1)
    } else {
      0.0
    }
var weights = if (addIntercept && numOfLinearPredictor == 1) {
      Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1))
    } else {
      weightsWithIntercept
    }
//该段代码获得了intercept以及最终的权重值。由于intercept和权重是在收缩的空间进行训练的，所以我们需要再把它们转换到元素的空间。数学显示，如果我们仅仅执行标准化而没有减去均值，即withStd = true, withMean = false， 那么intercept的值并不会发送改变。所以下面的代码仅仅处理权重向量。

if (useFeatureScaling) {
      if (numOfLinearPredictor == 1) {
        weights = scaler.transform(weights)
      } else {
        var i = 0
        val n = weights.size / numOfLinearPredictor
        val weightsArray = weights.toArray
        while (i < numOfLinearPredictor) {
          //排除intercept
          val start = i * n
          val end = (i + 1) * n - { if (addIntercept) 1 else 0 }
          val partialWeightsArray = scaler.transform(
            Vectors.dense(weightsArray.slice(start, end))).toArray
          System.arraycopy(partialWeightsArray, 0, weightsArray, start, partialWeightsArray.size)
          i += 1
        }
        weights = Vectors.dense(weightsArray)
      }
    }
//1.4、创建模型
createModel(weights, intercept)

//(四)、模型预测
//训练完模型之后，我们就可以通过训练的模型计算得到测试数据的分类信息。predictPoint用来预测分类信息。它针对二分类和多分类，分别进行处理。
//1、二分类的情况
val margin = dot(weightMatrix, dataMatrix) + intercept
val score = 1.0 / (1.0 + math.exp(-margin))
threshold match {
    case Some(t) => if (score > t) 1.0 else 0.0
    case None => score
}
//我们可以看到1.0 / (1.0 + math.exp(-margin))就是上文提到的逻辑函数即sigmoid函数。

//2、多分类情况
var bestClass = 0
var maxMargin = 0.0
val withBias = dataMatrix.size + 1 == dataWithBiasSize
(0 until numClasses - 1).foreach { i =>
        var margin = 0.0
        dataMatrix.foreachActive { (index, value) =>
          if (value != 0.0) margin += value * weightsArray((i * dataWithBiasSize) + index)
        }
        // Intercept is required to be added into margin.
        if (withBias) {
          margin += weightsArray((i * dataWithBiasSize) + dataMatrix.size)
        }
        if (margin > maxMargin) {
          maxMargin = margin
          bestClass = i + 1
        }
}
bestClass.toDouble
//该段代码计算并找到最大的margin。如果maxMargin为负，那么第一类是该数据的类别。

