#// mlws_chap07(Spark机器学习,第七章)

##// 0、封装包和代码环境  
// 准备环境  
cd $SPARK_HOME  
bin/spark-shell --name my_mlib --packages org.jblas:jblas:1.2.4 --driver-memory 4G --executor-memory 4G --driver-cores 2  

```scala
package mllib_book.mlws.chap07_clustering

import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.clustering.KMeans
import breeze.linalg._
import breeze.numerics.pow 

object chap07 extends App{  
  val conf = new SparkConf().setAppName("Spark_chap07").setMaster("local")  
  val sc = new SparkContext(conf)  
```  
  
##// 1、特征抽取   
```scala
val PATH = "/Users/erichan/sourcecode/book/Spark机器学习"
val movies = sc.textFile(PATH+"/ml-100k/u.item")

println(movies.first)
//1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
// 提取标签
val genres = sc.textFile(PATH+"/ml-100k/u.genre")
genres.take(5).foreach(println)
//unknown|0
//Action|1
//Adventure|2
//Animation|3
//Children's|4
val genreMap = genres.filter(!_.isEmpty).map(line => line.split("\\|")).map(array => (array(1), array(0))).collectAsMap
println(genreMap)
//Map(2 -> Adventure, 5 -> Comedy, 12 -> Musical, 15 -> Sci-Fi, 8 -> Drama, 18 -> Western, 7 -> Documentary, 17 -> War, 1 -> Action, 4 -> Children's, 11 -> Horror, 14 -> Romance, 6 -> Crime, 0 -> unknown, 9 -> Fantasy, 16 -> Thriller, 3 -> Animation, 10 -> Film-Noir, 13 -> Mystery)

val titlesAndGenres = movies.map(_.split("\\|")).map { array =>
    val genres = array.toSeq.slice(5, array.size)
    val genresAssigned = genres.zipWithIndex.filter { case (g, idx) =>
        g == "1"
    }.map { case (g, idx) =>
        genreMap(idx.toString)
    }
    (array(0).toInt, (array(1), genresAssigned))
}
println(titlesAndGenres.first)
//(1,(Toy Story (1995),ArrayBuffer(Animation, Children's, Comedy)))
```  

##// 2、训练模型  
```scala
val rawData = sc.textFile(PATH+"/ml-100k/u.data")
val rawRatings = rawData.map(_.split("\t").take(3))
val ratings = rawRatings.map{ case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
ratings.cache
val alsModel = ALS.train(ratings, 50, 10, 0.1)

val movieFactors = alsModel.productFeatures.map { case (id, factor) => (id, Vectors.dense(factor)) }
val movieVectors = movieFactors.map(_._2)
val userFactors = alsModel.userFeatures.map { case (id, factor) => (id, Vectors.dense(factor)) }
val userVectors = userFactors.map(_._2)
// 归一化
val movieMatrix = new RowMatrix(movieVectors)
val movieMatrixSummary = movieMatrix.computeColumnSummaryStatistics()
val userMatrix = new RowMatrix(userVectors)
val userMatrixSummary = userMatrix.computeColumnSummaryStatistics()

println("Movie factors mean: " + movieMatrixSummary.mean)
println("Movie factors variance: " + movieMatrixSummary.variance)
println("User factors mean: " + userMatrixSummary.mean)
println("User factors variance: " + userMatrixSummary.variance)
//Movie factors mean: [-0.2955253575969453,-0.2894017158566661,-0.2319822953560126,0.002917648331681182,0.16553261386128745,-0.21992534888550966,-0.03380127825698873,-0.20603088790834398,-0.15619861138444532,-0.028497688228493936,0.16963530616257805,0.14388067884376599,-0.0017092576491059591,-0.09626837303920982,-0.06064127207772349,-0.06045518556672421,0.18751923345914,0.2399624456229244,0.26532560070303446,0.05541910564428427,-0.015674971004015527,0.011168436718639107,-0.04741294377492476,0.11574693735017375,0.1289987655696671,0.44134038441588025,-0.5900688729554584,-0.03768358034266212,0.008887881298347921,0.20425041421871237,-0.20022602485759528,0.2697605004694663,0.10361325058554109,0.210277185021123,-0.22259636797095098,0.1174637780755839,-0.13688720440722232,0.03767713022869551,-0.0558405163043045,-0.12431407617904076,-0.046046222769634326,-0.20808223343120555,-0.3272035383525689,-0.2069514616509938,-0.0754149005227642,0.0856900404902959,0.06164062157888312,-0.06518672356795488,-0.32742867325628294,0.20285276122166002]
//Movie factors variance: [0.04665858952268178,0.0348699710379137,0.03579479789217705,0.029901233287017822,0.03584001448747631,0.030266373892482327,0.03305718879524182,0.02686945635500392,0.025320493042287756,0.024438753389466123,0.02575390293411112,0.028972744965903668,0.02827972104601086,0.033613911928577246,0.033662480558315735,0.02833746789842838,0.02994391891556463,0.04394221701123749,0.03435422169469965,0.03561654218653061,0.02682492748548697,0.029604704664741063,0.024673702738648908,0.030823597518982247,0.028442111141151614,0.03613743157595084,0.05449475590178096,0.042012621165520236,0.028896894307921802,0.033241681676696215,0.02633619851965672,0.035711481477364235,0.025481764774248593,0.028764828375131987,0.0272530758482775,0.029673397451581197,0.03148963131813852,0.03622387708462999,0.02816170323774573,0.033017372289155716,0.028641152942670445,0.02904189221086495,0.030234076747492195,0.04509970117296679,0.029449883713724593,0.02756067270740738,0.04139144468263727,0.030245838006703486,0.03131689738936245,0.03378427027186054]
//User factors mean: [-0.49867551877683824,-0.44459915531291566,-0.36051481893169574,0.017151848798776233,0.3213603826583396,-0.3196901675619378,-0.07224328943358119,-0.29041744434669287,-0.22727507102332345,-0.03178720415880569,0.25862894293461186,0.22888402894019788,0.012327199821030293,-0.14990885838046697,-0.1281515413333295,-0.09431829455241085,0.2679196025618735,0.38335691552119355,0.34604572069945905,0.11174974992685119,-0.02180706957147866,0.005610012764397019,-0.08491397316018835,0.20231176194774866,0.17161396689284497,0.6398163397864598,-0.8673987425745228,-0.10283010351171737,0.028330477167842844,0.30443187793692406,-0.301912604145753,0.4138735923728453,0.2256847560401456,0.285848070636566,-0.2605794171061,0.22449121780469036,-0.23998269543836812,0.036814175516996,-0.0679476059994798,-0.14427917258340417,-0.0833994179810923,-0.3582564875155623,-0.4564982359022274,-0.358039104184582,-0.12317214145750788,0.15037235678650748,0.06053431528961892,-0.06831269426575506,-0.5051800522709825,0.3151860279443428]
//User factors variance: [0.04443021235777847,0.03057049227554263,0.03697914530468325,0.037294292401115044,0.035107560835514376,0.03456306589890253,0.03582898189508532,0.028049150877918694,0.032265557628909904,0.033678972590911474,0.03107771568048393,0.03456737756860466,0.035184013404102404,0.04264936219513472,0.04120372326054623,0.03364277736735525,0.040292531435941865,0.04006060147670186,0.03950365342886879,0.04560154697337463,0.030231562691714647,0.041120732342916626,0.03118953330313852,0.03508187607535198,0.03228272297984499,0.03603017959168009,0.04917534366846078,0.059425007832722164,0.03161224197770566,0.04211986001194535,0.02891350391303218,0.05259534335774597,0.03483271651803892,0.040489027307905476,0.03125884956067426,0.0379774604293261,0.035875980098136084,0.043509576391072786,0.03338290356822281,0.03675372599031079,0.03379511912889908,0.02951817116168268,0.0430380317818896,0.04214608566562065,0.03376833767379957,0.0314188022932176,0.048481326691437995,0.03724671278315033,0.034103714500646594,0.046064657833824844]
//训练
val numClusters = 5
val numIterations = 10
val numRuns = 3
val movieClusterModel = KMeans.train(movieVectors, numClusters, numIterations, numRuns)
val movieClusterModelConverged = KMeans.train(movieVectors, numClusters, 100)
val userClusterModel = KMeans.train(userVectors, numClusters, numIterations, numRuns)
```  

##// 3、模型预测
```scala
val movie1 = movieVectors.first
val movieCluster = movieClusterModel.predict(movie1)
println(movieCluster)
val predictions = movieClusterModel.predict(movieVectors)
println(predictions.take(10).mkString(","))
def computeDistance(v1: DenseVector[Double], v2: DenseVector[Double]): Double = pow(v1 - v2, 2).sum
val titlesWithFactors = titlesAndGenres.join(movieFactors)
val moviesAssigned = titlesWithFactors.map { case (id, ((title, genres), vector)) =>
    val pred = movieClusterModel.predict(vector)
    val clusterCentre = movieClusterModel.clusterCenters(pred)
    val dist = computeDistance(DenseVector(clusterCentre.toArray), DenseVector(vector.toArray))
    (id, title, genres.mkString(" "), pred, dist)
}
val clusterAssignments = moviesAssigned.groupBy { case (id, title, genres, cluster, dist) => cluster }.collectAsMap

for ( (k, v) <- clusterAssignments.toSeq.sortBy(_._1)) {
    println(s"Cluster $k:")
    val m = v.toSeq.sortBy(_._5)
    println(m.take(20).map { case (_, title, genres, _, d) => (title, genres, d) }.mkString("\n"))
    println("=====\n")
}
```
##// 4、模型性能评估
```scala
// 内部评价指标
// &emsp;&emsp;   WCSS
// &emsp;&emsp;   Davies-Bouldin指数
// &emsp;&emsp;   Dunn指数
// &emsp;&emsp;   轮廓系数(silhouetee coefficient)
// 外不评价指标
// &emsp;&emsp;   Rand measure
// &emsp;&emsp;   F-measure
// &emsp;&emsp;  雅卡尔系数(Jaccard index)
// MLlib的computeCost
val movieCost = movieClusterModel.computeCost(movieVectors)
val userCost = userClusterModel.computeCost(userVectors)
println("WCSS for movies: " + movieCost)
println("WCSS for users: " + userCost)
```

```scala
##// 5、模型调优
// 通过交叉验证选择K
val trainTestSplitMovies = movieVectors.randomSplit(Array(0.6, 0.4), 123)
val trainMovies = trainTestSplitMovies(0)
val testMovies = trainTestSplitMovies(1)
val costsMovies = Seq(2, 3, 4, 5, 10, 20).map { k => (k, KMeans.train(trainMovies, numIterations, k, numRuns).computeCost(testMovies)) }
println("Movie clustering cross-validation:")
costsMovies.foreach { case (k, cost) => println(f"WCSS for K=$k id $cost%2.2f") }
//WCSS for K=2 id 870.36
//WCSS for K=3 id 858.28
//WCSS for K=4 id 847.40
//WCSS for K=5 id 840.71
//WCSS for K=10 id 842.58
//WCSS for K=20 id 843.24
val trainTestSplitUsers = userVectors.randomSplit(Array(0.6, 0.4), 123)
val trainUsers = trainTestSplitUsers(0)
val testUsers = trainTestSplitUsers(1)
val costsUsers = Seq(2, 3, 4, 5, 10, 20).map { k => (k, KMeans.train(trainUsers, numIterations, k, numRuns).computeCost(testUsers)) }
println("User clustering cross-validation:")
costsUsers.foreach { case (k, cost) => println(f"WCSS for K=$k id $cost%2.2f") }
//WCSS for K=2 id 573.50
//WCSS for K=3 id 580.33
//WCSS for K=4 id 574.84
//WCSS for K=5 id 575.61
//WCSS for K=10 id 586.05
//WCSS for K=20 id 577.01
```

