#// mlws_chap04(Spark机器学习,第四章)

##// 0、封装包和代码环境  
// 准备环境  
jblas  
https://gcc.gnu.org/wiki/GFortranBinaries#MacOS  
org.jblas:jblas:1.2.4-SNAPSHOT  
git clone https://github.com/mikiobraun/jblas.git  
cd jblas  
mvn install  
//运行环境  
cd /Users/erichan/Garden/spark-1.5.1-bin-cdh4  
bin/spark-shell --name my_mlib --packages org.jblas:jblas:1.2.4-SNAPSHOT --driver-memory 4G --executor-memory 4G --driver-cores 2  

```scala
package mllib_book.mlws.chap04_recommendation

import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating

object chap04 extends App{  
  val conf = new SparkConf().setAppName("Spark_chap04").setMaster("local")  
  val sc = new SparkContext(conf)  
```  
  
##// 1、特征抽取   
```scala
val PATH = "/Users/erichan/sourcecode/book/Spark机器学习"
val rawData = http://www.cnblogs.com/tychyg/p/sc.textFile(PATH+"/ml-100k/u.data")
rawData.first()
//res1: String = 196 242 3 881250949
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
val rawRatings=rawData.map(_.split("\t").take(3))
val ratings = rawRatings.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
ratings.first()
//res2: org.apache.spark.mllib.recommendation.Rating = Rating(196,242,3.0)
```  

##// 2、训练回归模型  
```scala
val model = ALS.train(ratings, 50, 10, 0.01) //rank=50, iterations=10, lambda=0.01
model.userFeatures.count
//res3: Long = 943
model.productFeatures.count
//res4: Long = 1682
```  

##// 3、使用模型
```scala
// 3.1、用户推荐
// 3.1.1、用户推荐
val predictedRating = model.predict(789, 123)

val userId = 789
val K = 10
val topKRecs = model.recommendProducts(userId, K)
println(topKRecs.mkString("\n"))
//Rating(789,176,5.732688958436494)
//Rating(789,201,5.682340265545152)
//Rating(789,182,5.5902224300291214)
//Rating(789,183,5.5877871075408585)
//Rating(789,96,5.4425266495153455)
//Rating(789,76,5.39730369058763)
//Rating(789,195,5.356822356978749)
//Rating(789,589,5.1464233861748925)
//Rating(789,134,5.109287533257644)
//Rating(789,518,5.106161562126567)

// 3.1.2、校验推荐
val movies = sc.textFile(PATH+"/ml-100k/u.item")
val titles = movies.map(line => line.split("\\|").take(2)).map(array => (array(0).toInt, array(1))).collectAsMap()
titles(123)
val moviesForUser = ratings.keyBy(_.user).lookup(789)
println(moviesForUser.size)
//33

moviesForUser.sortBy(-_.rating).take(10).map(rating => (titles(rating.product), rating.rating)).foreach(println)
//(Godfather, The (1972),5.0)
//(Trainspotting (1996),5.0)
//(Dead Man Walking (1995),5.0)
//(Star Wars (1977),5.0)
//(Swingers (1996),5.0)
//(Leaving Las Vegas (1995),5.0)
//(Bound (1996),5.0)
//(Fargo (1996),5.0)
//(Last Supper, The (1995),5.0)
//(Private Parts (1997),4.0)
topKRecs.map(rating => (titles(rating.product), rating.rating)).foreach(println)
//(Aliens (1986),5.732688958436494)
//(Evil Dead II (1987),5.682340265545152)
//(GoodFellas (1990),5.5902224300291214)
//(Alien (1979),5.5877871075408585)
//(Terminator 2: Judgment Day (1991),5.4425266495153455)
//(Carlito's Way (1993),5.39730369058763)
//(Terminator, The (1984),5.356822356978749)
//(Wild Bunch, The (1969),5.1464233861748925)
//(Citizen Kane (1941),5.109287533257644)
//(Miller's Crossing (1990),5.106161562126567)

// 3.2、物品推荐
// 3.2.1、物品推荐
import org.jblas.DoubleMatrix
val aMatrix = new DoubleMatrix(Array(1.0, 2.0, 3.0))
def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double = {
    vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
}
val itemId = 567
val itemFactor = model.productFeatures.lookup(itemId).head
val itemVector = new DoubleMatrix(itemFactor)
cosineSimilarity(itemVector, itemVector)
res10: Double = 1.0
val sims = model.productFeatures.map{ case (id, factor) =>
    val factorVector = new DoubleMatrix(factor)
    val sim = cosineSimilarity(factorVector, itemVector)
    (id, sim)
}
val sortedSims = sims.top(K)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
println(sortedSims.mkString("\n"))
//(567,1.0)
//(413,0.7309050775072655)
//(895,0.6992030886048359)
//(853,0.6960095521899471)
//(219,0.6806270119940826)
//(302,0.6757242121714326)
//(257,0.6721490667554395)
//(160,0.6672080746572076)
//(563,0.6621573120106216)
//(1019,0.6591520069387037)

// 3.2.2、校验推荐
println(titles(itemId))
//Wes Craven's New Nightmare (1994)

val sortedSims2 = sims.top(K + 1)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
sortedSims2.slice(1, 11).map{ case (id, sim) => (titles(id), sim) }.mkString("\n")
//res13: String =
//(Tales from the Crypt Presents: Bordello of Blood (1996),0.7309050775072655)
//(Scream 2 (1997),0.6992030886048359)
//(Braindead (1992),0.6960095521899471)
//(Nightmare on Elm Street, A (1984),0.6806270119940826)
//(L.A. Confidential (1997),0.6757242121714326)
//(Men in Black (1997),0.6721490667554395)
//(Glengarry Glen Ross (1992),0.6672080746572076)
//(Stephen King's The Langoliers (1995),0.6621573120106216)
//(Die xue shuang xiong (Killer, The) (1989),0.6591520069387037)
//(Evil Dead II (1987),0.655134288821937)
```
##// 4、模型效果评估
```scala
// 4.1、均方差（Mean Squared Error，MSE）
val actualRating = moviesForUser.take(1)(0)
val predictedRating = model.predict(789, actualRating.product)
val squaredError = math.pow(predictedRating - actualRating.rating, 2.0)
val usersProducts = ratings.map{ case Rating(user, product, rating)  => (user, product)}
val predictions = model.predict(usersProducts).map{
    case Rating(user, product, rating) => ((user, product), rating)
}
val ratingsAndPredictions = ratings.map{
    case Rating(user, product, rating) => ((user, product), rating)
}.join(predictions)
val MSE = ratingsAndPredictions.map{
    case ((user, product), (actual, predicted)) =>  math.pow((actual - predicted), 2)
}.reduce(_ + _) / ratingsAndPredictions.count
println("Mean Squared Error = " + MSE)
//Mean Squared Error = 0.08527363423596633
val RMSE = math.sqrt(MSE)
println("Root Mean Squared Error = " + RMSE)
//Root Mean Squared Error = 0.2920164965134099

// 4.2、K值平均准确率（MAPK）
def avgPrecisionK(actual: Seq[Int], predicted: Seq[Int], k: Int): Double = {
  val predK = predicted.take(k)
  var score = 0.0
  var numHits = 0.0
  for ((p, i) <- predK.zipWithIndex) {
    if (actual.contains(p)) {
      numHits += 1.0
      score += numHits / (i.toDouble + 1.0)
    }
  }
  if (actual.isEmpty) {
    1.0
  } else {
    score / scala.math.min(actual.size, k).toDouble
  }
}
val actualMovies = moviesForUser.map(_.product)
val predictedMovies = topKRecs.map(_.product)
val apk10 = avgPrecisionK(actualMovies, predictedMovies, 10)
val itemFactors = model.productFeatures.map { case (id, factor) => factor }.collect()
val itemMatrix = new DoubleMatrix(itemFactors)
println(itemMatrix.rows, itemMatrix.columns)
//(1682,50)
val imBroadcast = sc.broadcast(itemMatrix)
val allRecs = model.userFeatures.map{ case (userId, array) =>
  val userVector = new DoubleMatrix(array)
  val scores = imBroadcast.value.mmul(userVector)
  val sortedWithId = scores.data.zipWithIndex.sortBy(-_._1)
  val recommendedIds = sortedWithId.map(_._2 + 1).toSeq
  (userId, recommendedIds)
}
val userMovies = ratings.map{ case Rating(user, product, rating) => (user, product) }.groupBy(_._1)
val K = 10
val MAPK = allRecs.join(userMovies).map{ case (userId, (predicted, actualWithIds)) =>
  val actual = actualWithIds.map(_._2).toSeq
  avgPrecisionK(actual, predicted, K)
}.reduce(_ + _) / allRecs.count
println("Mean Average Precision at K = " + MAPK)
//Mean Average Precision at K = 0.030001472840815356

//4.3、MLib内置评估函数·RMSE和MSE
import org.apache.spark.mllib.evaluation.RegressionMetrics
val predictedAndTrue = ratingsAndPredictions.map { case ((user, product), (actual, predicted)) => (actual, predicted) }
val regressionMetrics = new RegressionMetrics(predictedAndTrue)
println("Mean Squared Error = " + regressionMetrics.meanSquaredError)
//Mean Squared Error = 0.08527363423596633
println("Root Mean Squared Error = " + regressionMetrics.rootMeanSquaredError)
//Root Mean Squared Error = 0.2920164965134099

// 4.4、MLib内置评估函数·MAP（平均准确率）
import org.apache.spark.mllib.evaluation.RankingMetrics
val predictedAndTrueForRanking = allRecs.join(userMovies).map{ case (userId, (predicted, actualWithIds)) =>
  val actual = actualWithIds.map(_._2)
  (predicted.toArray, actual.toArray)
}
val rankingMetrics = new RankingMetrics(predictedAndTrueForRanking)
println("Mean Average Precision = " + rankingMetrics.meanAveragePrecision)
//Mean Average Precision = 0.07208991526855565
val MAPK2000 = allRecs.join(userMovies).map{ case (userId, (predicted, actualWithIds)) =>
  val actual = actualWithIds.map(_._2).toSeq
  avgPrecisionK(actual, predicted, 2000)
}.reduce(_ + _) / allRecs.count
println("Mean Average Precision = " + MAPK2000)
//Mean Average Precision = 0.07208991526855561
```
