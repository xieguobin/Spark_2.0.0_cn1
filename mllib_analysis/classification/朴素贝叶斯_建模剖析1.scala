//朴素贝叶斯_建模剖析1

//一、数学理论


//二、建模实例
package org.apache.spark.mllib_analysis.classification

import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object Lr01 extends App{
  val conf = new SparkConf().setAppName("Spark_Lr").setMaster("local")
  val sc = new SparkContext(conf)
  
//加载训练数据、切分数据
val data = sc.textFile("C:/my_install/spark/data/mllib/sample_naive_bayes_data.txt")
val parsedData = data.map { line =>
  val parts = line.split(',')
  LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
}
val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0)
val test = splits(1)

//训练模型
val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

//模型测试
val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))

//模型评估
val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

//保存和加载模型
model.save(sc, "myModelPath")
val sameModel = SVMModel.load(sc, "myModelPath")

}

//三、源码调用解析
//模型训练过程需获取概率p(C)和p(F|C)。训练过程分为两步：第一步是聚合计算每个标签对应的term的频率，第二步是迭代计算p(C)和p(F|C)。
//1、计算每个标签对应的term的频率
val aggregated = data.map(p => (p.label, p.features)).combineByKey[(Long, DenseVector)](
      createCombiner = (v: Vector) => {
        if (modelType == Bernoulli) {
          requireZeroOneBernoulliValues(v)
        } else {
          requireNonnegativeValues(v)
        }
        (1L, v.copy.toDense)
      },
      mergeValue = (c: (Long, DenseVector), v: Vector) => {
        requireNonnegativeValues(v)
        //c._2 = v*1 + c._2
        BLAS.axpy(1.0, v, c._2)
        (c._1 + 1L, c._2)
      },
      mergeCombiners = (c1: (Long, DenseVector), c2: (Long, DenseVector)) => {
        BLAS.axpy(1.0, c2._2, c1._2)
        (c1._1 + c2._1, c1._2)
      }
    //根据标签进行排序
    ).collect().sortBy(_._1)
    //combineByKey函数中，createCombiner的作用是将原RDD中的Vector类型转换为(long,Vector)类型。

//如果modelType为multinomial，那么v中包含的值必须大于等于0；如果modelType为Bernoulli，那么v中包含的值只能为0或者1。
//值>=0
val requireNonnegativeValues: Vector => Unit = (v: Vector) => {
      val values = v match {
        case sv: SparseVector => sv.values
        case dv: DenseVector => dv.values
      }
      if (!values.forall(_ >= 0.0)) {
        throw new SparkException(s"Naive Bayes requires nonnegative feature values but found $v.")
      }
}
//值为0或者1
val requireZeroOneBernoulliValues: Vector => Unit = (v: Vector) => {
      val values = v match {
        case sv: SparseVector => sv.values
        case dv: DenseVector => dv.values
      }
      if (!values.forall(v => v == 0.0 || v == 1.0)) {
        throw new SparkException(
          s"Bernoulli naive Bayes requires 0 or 1 feature values but found $v.")
      }
}
//mergeValue函数的作用是将新来的Vector累加到已有向量中，并更新词率。
//mergeCombiners则是合并不同分区的(long,Vector)数据。 通过这个函数，找到了每个标签对应的词频，并得到标签对应的所有文档的累加向量。

2、迭代计算p(C)和p(F|C)
//标签数
val numLabels = aggregated.length
//文档数
var numDocuments = 0L
aggregated.foreach { case (_, (n, _)) =>
  numDocuments += n
}
//特征维数
val numFeatures = aggregated.head match { case (_, (_, v)) => v.size }
val labels = new Array[Double](numLabels)
//表示logP(C)
val pi = new Array[Double](numLabels)
//表示logP(F|C)
val theta = Array.fill(numLabels)(new Array[Double](numFeatures))
val piLogDenom = math.log(numDocuments + numLabels * lambda)
var i = 0
aggregated.foreach { case (label, (n, sumTermFreqs)) =>
      labels(i) = label
      //训练步骤第5步
      pi(i) = math.log(n + lambda) - piLogDenom
      val thetaLogDenom = modelType match {
        case Multinomial => math.log(sumTermFreqs.values.sum + numFeatures * lambda)
        case Bernoulli => math.log(n + 2.0 * lambda)
        case _ =>
          // This should never happen.
          throw new UnknownError(s"Invalid modelType: $modelType.")
      }
      //训练步骤第6步
      var j = 0
      while (j < numFeatures) {
        theta(i)(j) = math.log(sumTermFreqs(j) + lambda) - thetaLogDenom
        j += 1
      }
      i += 1
    }
  //这段代码计算上文提到的p(C)和p(F|C)。这里的lambda表示平滑因子，一般情况下，将它设置为1。
  //代码中，p(c_i)=log (n+lambda)/(numDocs+numLabels*lambda)，这对应上文训练过程的第5步prior(c)=N_c/N。
  //根据modelType类型的不同，p(F|C)的实现则不同：
  //当modelType为Multinomial时，P(F|C)=T_ct/sum(T_ct)，这里sum(T_ct)=sumTermFreqs.values.sum + numFeatures * lambda。这对应训练过程的第10步。 
  //当modelType为Bernoulli时，P(F|C)=(N_ct+lambda)/(N_c+2*lambda)。这对应训练算法的第8步骤。
  //需注意:上述代码中的所有计算都是取对数计算的。

//3、预测数据
override def predict(testData: Vector): Double = {
    modelType match {
      case Multinomial =>
        labels(multinomialCalculation(testData).argmax)
      case Bernoulli =>
        labels(bernoulliCalculation(testData).argmax)
    }
}

//预测也是根据modelType的不同作不同的处理。
//当modelType为Multinomial时，调用multinomialCalculation函数。
private def multinomialCalculation(testData: Vector) = {
    val prob = thetaMatrix.multiply(testData)
    BLAS.axpy(1.0, piVector, prob)
    prob
  }
//这里的thetaMatrix和piVector即上文中训练得到的P(F|C)和P(C)，根据P(C|F)=P(F|C)*P(C)即可以得到预测数据归属于某类别的概率。 
//注意，这些概率都是基于对数结果计算的。

//当modelType为Bernoulli时，实现代码略有不同。
private def bernoulliCalculation(testData: Vector) = {
    testData.foreachActive((_, value) =>
      if (value != 0.0 && value != 1.0) {
        throw new SparkException(
          s"Bernoulli naive Bayes requires 0 or 1 feature values but found $testData.")
      }
    )
    val prob = thetaMinusNegTheta.get.multiply(testData)
    BLAS.axpy(1.0, piVector, prob)
    BLAS.axpy(1.0, negThetaSum.get, prob)
    prob
  }
//当词在训练数据中出现与否处理的过程不同。见伯努利模型测试过程。这里用矩阵和向量的操作来实现这个过程。

 private val (thetaMinusNegTheta, negThetaSum) = modelType match {
    case Multinomial => (None, None)
    case Bernoulli =>
      val negTheta = thetaMatrix.map(value => math.log(1.0 - math.exp(value)))
      val ones = new DenseVector(Array.fill(thetaMatrix.numCols){1.0})
      val thetaMinusNegTheta = thetaMatrix.map { value =>
        value - math.log(1.0 - math.exp(value))
      }
      (Option(thetaMinusNegTheta), Option(negTheta.multiply(ones)))
    case _ =>
      // This should never happen.
      throw new UnknownError(s"Invalid modelType: $modelType.")
  }
 //这里math.exp(value)将对数概率恢复成真实的概率。
