#// mlws_chap10(Spark机器学习,第十章)

##// 0、封装包和代码环境  
// 准备环境 build.sbt  
依赖Spark MLlib和Spark Streaming  
name := "scala-spark-streaming-app"  
version := "1.0"  
scalaVersion := "2.11.7"  

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.5.1"  
libraryDependencies += "org.apache.spark" %% "spark-streaming" % "1.5.1"  
使用国内镜像仓库  
~/.sbt/repositories  

[repositories]  
local  
osc: http://maven.oschina.net/content/groups/public/  
typesafe: http://repo.typesafe.com/typesafe/ivy-releases/, [organization]/[module]/(scala_[scalaVersion]/)(sbt_[sbtVersion]/)[revision]/[type]s/[artifact](-[classifier]).[ext], bootOnly  
sonatype-oss-releases  
maven-central  
sonatype-oss-snapshots  

##// 1、生产消息  
```scala
object StreamingProducer {

  def main(args: Array[String]) {

    val random = new Random()

    // Maximum number of events per second
    val MaxEvents = 6

    // Read the list of possible names
    val namesResource = this.getClass.getResourceAsStream("/names.csv")
    val names = scala.io.Source.fromInputStream(namesResource)
      .getLines()
      .toList
      .head
      .split(",")
      .toSeq

    // Generate a sequence of possible products
    val products = Seq(
      "iPhone Cover" -> 9.99,
      "Headphones" -> 5.49,
      "Samsung Galaxy Cover" -> 8.95,
      "iPad Cover" -> 7.49
    )

    /** Generate a number of random product events */
    def generateProductEvents(n: Int) = {
      (1 to n).map { i =>
        val (product, price) = products(random.nextInt(products.size))
        val user = random.shuffle(names).head
        (user, product, price)
      }
    }

    // create a network producer
    val listener = new ServerSocket(9999)
    println("Listening on port: 9999")

    while (true) {
      val socket = listener.accept()
      new Thread() {
        override def run = {
          println("Got client connected from: " + socket.getInetAddress)
          val out = new PrintWriter(socket.getOutputStream(), true)

          while (true) {
            Thread.sleep(1000)
            val num = random.nextInt(MaxEvents)
            val productEvents = generateProductEvents(num)
            productEvents.foreach{ event =>
              out.write(event.productIterator.mkString(","))
              out.write("\n")
            }
            out.flush()
            println(s"Created $num events...")
          }
          socket.close()
        }
      }.start()
    }
  }
}
// sbt run
//
//Multiple main classes detected, select one to run:
//
// [1] MonitoringStreamingModel
// [2] SimpleStreamingApp
// [3] SimpleStreamingModel
// [4] StreamingAnalyticsApp
// [5] StreamingModelProducer
// [6] StreamingProducer
// [7] StreamingStateApp
//
//Enter number: 6
```

##// 2、打印消息
```scala
object SimpleStreamingApp {
  def main(args: Array[String]) {
    val ssc = new StreamingContext("local[2]", "First Streaming App", Seconds(10))
    val stream = ssc.socketTextStream("localhost", 9999)
    // here we simply print out the first few elements of each batch
    stream.print()
    ssc.start()
    ssc.awaitTermination()
  }
}
// sbt run

//Enter number: 2
```

##// 3、流式分析
```scala
object StreamingAnalyticsApp {
  def main(args: Array[String]) {
    val ssc = new StreamingContext("local[2]", "First Streaming App", Seconds(10))
    val stream = ssc.socketTextStream("localhost", 9999)

    // create stream of events from raw text elements
    val events = stream.map { record =>
      val event = record.split(",")
      (event(0), event(1), event(2))
    }

    /*
      We compute and print out stats for each batch.
      Since each batch is an RDD, we call forEeachRDD on the DStream, and apply the usual RDD functions
      we used in Chapter 1.
     */
    events.foreachRDD { (rdd, time) =>
      val numPurchases = rdd.count()
      val uniqueUsers = rdd.map { case (user, _, _) => user }.distinct().count()
      val totalRevenue = rdd.map { case (_, _, price) => price.toDouble }.sum()
      val productsByPopularity = rdd
        .map { case (user, product, price) => (product, 1) }
        .reduceByKey(_ + _)
        .collect()
        .sortBy(-_._2)
      val mostPopular = productsByPopularity(0)

      val formatter = new SimpleDateFormat
      val dateStr = formatter.format(new Date(time.milliseconds))
      println(s"== Batch start time: $dateStr ==")
      println("Total purchases: " + numPurchases)
      println("Unique users: " + uniqueUsers)
      println("Total revenue: " + totalRevenue)
      println("Most popular product: %s with %d purchases".format(mostPopular._1, mostPopular._2))
    }

    // start the context
    ssc.start()
    ssc.awaitTermination()
  }
}
// sbt run

//Enter number: 4
```

##// 4、有状态的流计算
```scala
object StreamingStateApp {

  import org.apache.spark.streaming.StreamingContext._

  def updateState(prices: Seq[(String, Double)], currentTotal: Option[(Int, Double)]) = {
    val currentRevenue = prices.map(_._2).sum
    val currentNumberPurchases = prices.size
    val state = currentTotal.getOrElse((0, 0.0))
    Some((currentNumberPurchases + state._1, currentRevenue + state._2))
  }

  def main(args: Array[String]) {
    val ssc = new StreamingContext("local[2]", "First Streaming App", Seconds(10))
    // for stateful operations, we need to set a checkpoint location
    ssc.checkpoint("/tmp/sparkstreaming/")
    val stream = ssc.socketTextStream("localhost", 9999)
    // create stream of events from raw text elements
    val events = stream.map { record =>
      val event = record.split(",")
      (event(0), event(1), event(2).toDouble)
    }
    val users = events.map { case (user, product, price) => (user, (product, price)) }
    val revenuePerUser = users.updateStateByKey(updateState)
    revenuePerUser.print()
    // start the context
    ssc.start()
    ssc.awaitTermination()
  }
}
// sbt run

//Enter number: 7
```

##// 5、线性流回归
// 线性回归StreamingLinearRegressionWithSGD
```scala
// trainOn
// predictOn
// 5.1、流数据生成器
object StreamingModelProducer {

  import breeze.linalg._

  def main(args: Array[String]) {
    // Maximum number of events per second
    val MaxEvents = 100
    val NumFeatures = 100
    val random = new Random()

    /** Function to generate a normally distributed dense vector */
    def generateRandomArray(n: Int) = Array.tabulate(n)(_ => random.nextGaussian())

    // Generate a fixed random model weight vector
    val w = new DenseVector(generateRandomArray(NumFeatures))
    val intercept = random.nextGaussian() * 10

    /** Generate a number of random product events */
    def generateNoisyData(n: Int) = {
      (1 to n).map { i =>
        val x = new DenseVector(generateRandomArray(NumFeatures))
        val y: Double = w.dot(x)
        val noisy = y + intercept //+ 0.1 * random.nextGaussian()
        (noisy, x)
      }
    }

    // create a network producer
    val listener = new ServerSocket(9999)
    println("Listening on port: 9999")

    while (true) {
      val socket = listener.accept()
      new Thread() {
        override def run = {
          println("Got client connected from: " + socket.getInetAddress)
          val out = new PrintWriter(socket.getOutputStream(), true)

          while (true) {
            Thread.sleep(1000)
            val num = random.nextInt(MaxEvents)
            val data = generateNoisyData(num)
            data.foreach { case (y, x) =>
              val xStr = x.data.mkString(",")
              val eventStr = s"$y\t$xStr"
              out.write(eventStr)
              out.write("\n")
            }
            out.flush()
            println(s"Created $num events...")
          }
          socket.close()
        }
      }.start()
    }
  }
}
// sbt run

//Enter number: 5

// 5.2、流回归模型
object SimpleStreamingModel {
  def main(args: Array[String]) {
    val ssc = new StreamingContext("local[2]", "First Streaming App", Seconds(10))
    val stream = ssc.socketTextStream("localhost", 9999)

    val NumFeatures = 100
    val zeroVector = DenseVector.zeros[Double](NumFeatures)
    val model = new StreamingLinearRegressionWithSGD()
      .setInitialWeights(Vectors.dense(zeroVector.data))
      .setNumIterations(1)
      .setStepSize(0.01)

    // create a stream of labeled points
    val labeledStream: DStream[LabeledPoint] = stream.map { event =>
      val split = event.split("\t")
      val y = split(0).toDouble
      val features: Array[Double] = split(1).split(",").map(_.toDouble)
      LabeledPoint(label = y, features = Vectors.dense(features))
    }

    // train and test model on the stream, and print predictions for illustrative purposes
    model.trainOn(labeledStream)
    //model.predictOn(labeledStream).print()

    ssc.start()
    ssc.awaitTermination()
  }
}
// sbt run

//Enter number: 5

// 5.3、流K均值
// K-均值聚类：StreamingKMeans
```

```scala
##// 6、模型评估
object MonitoringStreamingModel {
  def main(args: Array[String]) {
    val ssc = new StreamingContext("local[2]", "First Streaming App", Seconds(10))
    val stream = ssc.socketTextStream("localhost", 9999)

    val NumFeatures = 100
    val zeroVector = DenseVector.zeros[Double](NumFeatures)
    val model1 = new StreamingLinearRegressionWithSGD()
      .setInitialWeights(Vectors.dense(zeroVector.data))
      .setNumIterations(1)
      .setStepSize(0.01)

    val model2 = new StreamingLinearRegressionWithSGD()
      .setInitialWeights(Vectors.dense(zeroVector.data))
      .setNumIterations(1)
      .setStepSize(1.0)

    // create a stream of labeled points
    val labeledStream = stream.map { event =>
      val split = event.split("\t")
      val y = split(0).toDouble
      val features = split(1).split(",").map(_.toDouble)
      LabeledPoint(label = y, features = Vectors.dense(features))
    }

    // train both models on the same stream
    model1.trainOn(labeledStream)
    model2.trainOn(labeledStream)

    // use transform to create a stream with model error rates
    val predsAndTrue = labeledStream.transform { rdd =>
      val latest1 = model1.latestModel()
      val latest2 = model2.latestModel()
      rdd.map { point =>
        val pred1 = latest1.predict(point.features)
        val pred2 = latest2.predict(point.features)
        (pred1 - point.label, pred2 - point.label)
      }
    }

    // print out the MSE and RMSE metrics for each model per batch
    predsAndTrue.foreachRDD { (rdd, time) =>
      val mse1 = rdd.map { case (err1, err2) => err1 * err1 }.mean()
      val rmse1 = math.sqrt(mse1)
      val mse2 = rdd.map { case (err1, err2) => err2 * err2 }.mean()
      val rmse2 = math.sqrt(mse2)
      println(
        s"""
           |-------------------------------------------
           |Time: $time
           |-------------------------------------------
         """.stripMargin)
      println(s"MSE current batch: Model 1: $mse1; Model 2: $mse2")
      println(s"RMSE current batch: Model 1: $rmse1; Model 2: $rmse2")
      println("...\n")
    }
    ssc.start()
    ssc.awaitTermination()
  }
}
// sbt run

//Enter number: 1
```
