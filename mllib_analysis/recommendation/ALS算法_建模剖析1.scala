使用Spark MLlib给豆瓣用户推荐电影
推荐算法就是利用用户的一些行为，通过一些数学算法，推测出用户可能喜欢的东西。
随着电子商务规模的不断扩大，商品数量和种类不断增长，用户对于检索和推荐提出了更高的要求。由于不同用户在兴趣爱好、关注领域、个人经历等方面的不同，以满足不同用户的不同推荐需求为目的、不同人可以获得不同推荐为重要特征的个性化推荐系统应运而生。

推荐系统成为一个相对独立的研究方向一般被认为始自1994年明尼苏达大学GroupLens研究组推出的GroupLens系统。该系统有两大重要贡献：一是首次提出了基于协同过滤(Collaborative Filtering)来完成推荐任务的思想，二是为推荐问题建立了一个形式化的模型。基于该模型的协同过滤推荐引领了之后推荐系统在今后十几年的发展方向。

目前，推荐算法已经已经被广泛集成到了很多商业应用系统中，比较著名的有Netflix在线视频推荐系统、Amazon网络购物商城等。实际上，大多数的电子商务平台尤其是网络购物平台，都不同程度地集成了推荐算法，如淘宝、京东商城等。Amazon发布的数据显示，亚马逊网络书城的推荐算法为亚马逊每年贡献近三十个百分点的创收。

常用的推荐算法

基于人口统计学的推荐(Demographic-Based Recommendation):该方法所基于的基本假设是“一个用户有可能会喜欢与其相似的用户所喜欢的物品”。当我们需要对一个User进行个性化推荐时，利用User Profile计算其它用户与其之间的相似度，然后挑选出与其最相似的前K个用户，之后利用这些用户的购买和打分信息进行推荐。
基于内容的推荐(Content-Based Recommendation):Content-Based方法所基于的基本假设是“一个用户可能会喜欢和他曾经喜欢过的物品相似的物品”。
基于协同过滤的推荐(Collaborative Filtering-Based Recommendation)是指收集用户过去的行为以获得其对产品的显式或隐式信息，即根据用户对
物品或者信息的偏好，发现物品或者内容本身的相关性、或用户的相关性，然后再基于这些关联性进行推荐。基于协同过滤的推荐可以分基于用户的推荐（User-based Recommendation），基于物品的推荐（Item-based Recommendation），基于模型的推荐（Model-based Recommendation）等子类。
以上内容copy自参考文档1

ALS算法

ALS是alternating least squares的缩写 , 意为交替最小二乘法。该方法常用于基于矩阵分解的推荐系统中。例如：将用户(user)对商品(item)的评分矩阵分解为两个矩阵：一个是用户对商品隐含特征的偏好矩阵，另一个是商品所包含的隐含特征的矩阵。在这个矩阵分解的过程中，评分缺失项得到了填充，也就是说我们可以基于这个填充的评分来给用户最商品推荐了。
由于评分数据中有大量的缺失项，传统的矩阵分解SVD（奇异值分解）不方便处理这个问题，而ALS能够很好的解决这个问题。对于R(m×n)的矩阵，ALS旨在找到两个低维矩阵X(m×k)和矩阵Y(n×k)，来近似逼近R(m×n)，即：R~=XYR~=XY ，其中 ，X∈Rm×dX∈Rm×d，Y∈Rd×nY∈Rd×n，d 表示降维后的维度，一般 d<<r，r表示矩阵 R 的秩，r<<min(m,n)r<<min(m,n)。

为了找到低维矩阵X,Y最大程度地逼近矩分矩阵R，最小化下面的平方误差损失函数。
L(X,Y)=∑u,i(rui−xTuyi)2
L(X,Y)=∑u,i(rui−xuTyi)2
为防止过拟合给公式 (1) 加上正则项，公式改下为：
L(X,Y)=∑u,i(rui−xTuyi)2+λ(|xu|2+　|yi|2)......(2)
L(X,Y)=∑u,i(rui−xuTyi)2+λ(|xu|2+　|yi|2)......(2)
其中xu∈Rd，yi∈Rdxu∈Rd，yi∈Rd，1⩽u⩽m1⩽u⩽m，1⩽i⩽n1⩽i⩽n，λλ是正则项的系数。
MLlib 的实现算法中有以下一些参数：


numBlocks

用于并行化计算的分块个数 (-1为自动分配)

rank

模型中隐藏因子的个数，也就是上面的r

iterations

迭代的次数，推荐值：10-20

lambda

惩罚函数的因数，是ALS的正则化参数，推荐值：0.01

implicitPrefs

决定了是用显性反馈ALS的版本还是用适用隐性反馈数据集的版本

alpha

是一个针对于隐性反馈 ALS 版本的参数，这个参数决定了偏好行为强度的基准

隐性反馈 vs 显性反馈
基于矩阵分解的协同过滤的标准方法一般将用户商品矩阵中的元素作为用户对商品的显性偏好。 在许多的现实生活中的很多场景中，我们常常只能接触到隐性的反馈（例如游览，点击，购买，喜欢，分享等等）在 MLlib 中所用到的处理这种数据的方法来源于文献： Collaborative Filtering for Implicit Feedback Datasets。 本质上，这个方法将数据作为二元偏好值和偏好强度的一个结合，而不是对评分矩阵直接进行建模。因此，评价就不是与用户对商品的显性评分而是和所观察到的用户偏好强度关联了起来。然后，这个模型将尝试找到隐语义因子来预估一个用户对一个商品的偏好。

以上的介绍带着浓重的学术气息，需要阅读更多的背景知识才能了解这些算法的奥秘。Spark MLlib为我们提供了很好的协同算法的封装。 当前MLlib支持基于模型的协同过滤算法，其中user和product对应上面的user和item，user和product之间有一些隐藏因子。MLlib使用ALS(alternating least squares)来学习/得到这些潜在因子。

下面我们就以实现一个豆瓣电影推荐系统为例看看如何使用Spark实现此类推荐系统。以此类推，你也可以尝试实现豆瓣图书，豆瓣音乐，京东电器商品推荐系统。

豆瓣数据集

一般学习Spark MLlib ALS会使用movielens数据集。这个数据集保存了用户对电影的评分。
但是这个数据集对于国内用户来说有点不接地气，事实上国内有一些网站可以提供这样的数据集，比如豆瓣，它的人气还是挺高的。
但是豆瓣并没有提供这样一个公开的数据集，所以我用抓取了一些数据做测试。
测试数据地址：https://github.com/smallnest/douban-recommender/tree/master/data


数据集分为两个文件：

hot_movies.csv: 这个文件包含了热门电影的列表，一种166个热门电影。格式为 <movieID>,<评分>,<电影名>，如
1
2
3
4
5
6
20645098,8.2,小王子
26259677,8.3,垫底辣妹
11808948,7.2,海绵宝宝
26253733,6.4,突然变异
25856265,6.7,烈日迷踪
26274810,6.6,侦探：为了原点
user_movies.csv: 这个文件包含用户对热门电影的评价，格式为<userID>:<movieID>:<评分>
1
2
3
4
5
6
7
adamwzw,20645098,4
baka_mono,20645098,3
iRayc,20645098,2
blueandgreen,20645098,3
130992805,20645098,4
134629166,20645098,5
wangymm,20645098,3
可以看到，用户名并不完全是整数类型的，但是MLlib ALS算法要求user,product都是整型的，所以我们在编程的时候需要处理一下。
有些用户只填写了评价，并没有打分，文件中将这样的数据记为-1。在ALS算法中，把它转换成3.0，也就是及格60分。虽然可能和用户的实际情况不相符，但是为了简化运算，我在这里做了简化处理。
用户的评分收集了大约100万条，实际用户大约22万。这个矩阵还是相当的稀疏。

注意这个数据集完全基于豆瓣公开的网页，不涉及任何个人的隐私。

模型实现

本系统使用Scala实现。
首先读入这两个文件，得到相应的弹性分布数据集RDD (第7行和第8行)。

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
object DoubanRecommender {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("DoubanRecommender"))
    //val base = "/opt/douban/"
    val base = if (args.length > 0) args(0) else "/opt/douban/"
    //获取RDD
    val rawUserMoviesData = sc.textFile(base + "user_movies.csv")
    val rawHotMoviesData = sc.textFile(base + "hot_movies.csv")
    //准备数据
    preparation(rawUserMoviesData, rawHotMoviesData)
    println("准备完数据")
    model(sc, rawUserMoviesData, rawHotMoviesData)
  }
  ......
}
第10行调用preparation方法，这个方法主要用来检查分析数据，得到数据集的一些基本的统计信息，还没有到协同算法那一步。

1
2
3
4
5
6
7
8
9
10
def preparation( rawUserMoviesData: RDD[String],
                 rawHotMoviesData: RDD[String]) = {
  val userIDStats = rawUserMoviesData.map(_.split(',')(0).trim).distinct().zipWithUniqueId().map(_._2.toDouble).stats()
  val itemIDStats = rawUserMoviesData.map(_.split(',')(1).trim.toDouble).distinct().stats()
  println(userIDStats)
  println(itemIDStats)
  val moviesAndName = buildMovies(rawHotMoviesData)
  val (movieID, movieName) = moviesAndName.head
  println(movieID + " -> " + movieName)
}
第5行和第6行打印RDD的statCounter的值，主要是最大值，最小值等。
第9行输出热门电影的第一个值。
输出结果如下：

1
2
3
(count: 223239, mean: 111620.188663, stdev: 64445.607152, max: 223966.000000, min: 0.000000)
(count: 165, mean: 20734733.139394, stdev: 8241677.225813, max: 26599083.000000, min: 1866473.000000)
6866928 -> 进击的巨人真人版：前篇
方法buildMovies读取rawHotMoviesData，因为rawHotMoviesData的每一行是一条类似20645098,8.2,小王子的字符串，需要按照,分割，得到第一个值和第三个值：

1
2
3
4
5
6
7
8
9
def buildMovies(rawHotMoviesData: RDD[String]): Map[Int, String] =
  rawHotMoviesData.flatMap { line =>
    val tokens = line.split(',')
    if (tokens(0).isEmpty) {
      None
    } else {
      Some((tokens(0).toInt, tokens(2)))
    }
  }.collectAsMap()
我们使用这个Map可以根据电影的ID得到电影实际的名字。

下面就重点看看如何使用算法建立模型的：

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
def model(sc: SparkContext,
            rawUserMoviesData: RDD[String],
            rawHotMoviesData: RDD[String]): Unit = {
    val moviesAndName = buildMovies(rawHotMoviesData)
    val bMoviesAndName = sc.broadcast(moviesAndName)
    val data = buildRatings(rawUserMoviesData)
    val userIdToInt: RDD[(String, Long)] =
      data.map(_.userID).distinct().zipWithUniqueId()
    val reverseUserIDMapping: RDD[(Long, String)] =
      userIdToInt map { case (l, r) => (r, l) }
    val userIDMap: Map[String, Int] =   userIdToInt.collectAsMap().map { case (n, l) => (n, l.toInt) }
    val bUserIDMap = sc.broadcast(userIDMap)
    val ratings: RDD[Rating] = data.map { r => Rating(bUserIDMap.value.get(r.userID).get, r.movieID, r.rating)}.cache()
    //使用协同过滤算法建模
    //val model = ALS.trainImplicit(ratings, 10, 10, 0.01, 1.0)
    val model = ALS.train(ratings, 50, 10, 0.0001)
    ratings.unpersist()
    println("输出第一个userFeature")
    println(model.userFeatures.mapValues(_.mkString(", ")).first())
    for (userID <- Array(100,1001,10001,100001,110000)) {
      checkRecommenderResult(userID, rawUserMoviesData, bMoviesAndName, reverseUserIDMapping, model)
    }
    unpersist(model)
  }
第4行到第12行是准备辅助数据，第13行准备好ALS算法所需的数据RDD[Rating]。
第16行设置一些参数训练数据。这些参数可以根据下一节的评估算法挑选一个较好的参数集合作为最终的模型参数。
第21行是挑选几个用户，查看这些用户看过的电影，以及这个模型推荐给他们的电影。

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
def checkRecommenderResult(userID: Int, rawUserMoviesData: RDD[String], bMoviesAndName: Broadcast[Map[Int, String]], reverseUserIDMapping: RDD[(Long, String)], model: MatrixFactorizationModel): Unit = {
    val userName = reverseUserIDMapping.lookup(userID).head
    val recommendations = model.recommendProducts(userID, 5)
    //给此用户的推荐的电影ID集合
    val recommendedMovieIDs = recommendations.map(_.product).toSet
    //得到用户点播的电影ID集合
    val rawMoviesForUser = rawUserMoviesData.map(_.split(',')).
      filter { case Array(user, _, _) => user.trim == userName }
    val existingUserMovieIDs = rawMoviesForUser.map { case Array(_, movieID, _) => movieID.toInt }.
      collect().toSet
    println("用户" + userName + "点播过的电影名")
    //点播的电影名
    bMoviesAndName.value.filter { case (id, name) => existingUserMovieIDs.contains(id) }.values.foreach(println)
    println("推荐给用户" + userName + "的电影名")
    //推荐的电影名
    bMoviesAndName.value.filter { case (id, name) => recommendedMovieIDs.contains(id) }.values.foreach(println)
  }
比如用户yimiao曾经点评过以下的电影：


然后这个模型为他推荐

基本都属于喜剧动作，爱情类的，看起来还不错。

评价

当然，我们不能凭着自己的感觉评价模型的好坏，尽管我们直觉告诉我们，这个结果看不错。我们需要量化的指标来评价模型的优劣。
我们可以通过计算均方差（Mean Squared Error, MSE）来衡量模型的好坏。数理统计中均方误差是指参数估计值与参数真值之差平方的期望值，记为MSE。MSE是衡量“平均误差”的一种较方便的方法，MSE可以评价数据的变化程度，MSE的值越小，说明预测模型描述实验数据具有更好的精确度。
我们可以调整rank，numIterations，lambda，alpha这些参数，不断优化结果，使均方差变小。比如：iterations越多，lambda较小，均方差会较小，推荐结果较优。

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
def evaluate( sc: SparkContext,
                rawUserMoviesData: RDD[String],
                rawHotMoviesData: RDD[String]): Unit = {
    val moviesAndName = buildMovies(rawHotMoviesData)
    val bMoviesAndName = sc.broadcast(moviesAndName)
    val data = buildRatings(rawUserMoviesData)
    val userIdToInt: RDD[(String, Long)] =
      data.map(_.userID).distinct().zipWithUniqueId()
    val userIDMap: Map[String, Int] =
      userIdToInt.collectAsMap().map { case (n, l) => (n, l.toInt) }
    val bUserIDMap = sc.broadcast(userIDMap)
    val ratings: RDD[Rating] = data.map { r =>
      Rating(bUserIDMap.value.get(r.userID).get, r.movieID, r.rating)
    }.cache()
	val numIterations = 10
    for (rank   <- Array(10,  50);
         lambda <- Array(1.0, 0.01,0.0001)) {
      val model = ALS.train(ratings, rank, numIterations, lambda)
      // Evaluate the model on rating data
      val usersMovies = ratings.map { case Rating(user, movie, rate) =>
        (user, movie)
      }
      val predictions =
        model.predict(usersMovies).map { case Rating(user, movie, rate) =>
          ((user, movie), rate)
        }
      val ratesAndPreds = ratings.map { case Rating(user, movie, rate) =>
        ((user, movie), rate)
      }.join(predictions)
      val MSE = ratesAndPreds.map { case ((user, movie), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
      }.mean()
      println(s"(rank:$rank, lambda: $lambda, Explicit ) Mean Squared Error = " + MSE)
    }
    for (rank   <- Array(10,  50);
         lambda <- Array(1.0, 0.01,0.0001);
         alpha  <- Array(1.0, 40.0)) {
      val model = ALS.trainImplicit(ratings, rank, numIterations, lambda, alpha)
      // Evaluate the model on rating data
      val usersMovies = ratings.map { case Rating(user, movie, rate) =>
        (user, movie)
      }
      val predictions =
        model.predict(usersMovies).map { case Rating(user, movie, rate) =>
          ((user, movie), rate)
        }
      val ratesAndPreds = ratings.map { case Rating(user, movie, rate) =>
        ((user, movie), rate)
      }.join(predictions)
      val MSE = ratesAndPreds.map { case ((user, movie), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
      }.mean()
      println(s"(rank:$rank, lambda: $lambda,alpha:$alpha ,implicit  ) Mean Squared Error = " + MSE)
    }
  }
第16行到第35行评估显性反馈的参数的结果，第36行到第56行评估隐性反馈的参数的结果。
评估的结果如下：

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
(rank:10, lambda: 1.0, Explicit ) Mean Squared Error = 1.5592024394027315                                                                                
(rank:10, lambda: 0.01, Explicit ) Mean Squared Error = 0.1597401855959523                                                                                
(rank:10, lambda: 1.0E-4, Explicit ) Mean Squared Error = 0.12000266211936791                                                                                
(rank:50, lambda: 1.0, Explicit ) Mean Squared Error = 1.559198310777233                                                                                
(rank:50, lambda: 0.01, Explicit ) Mean Squared Error = 0.015537276558121003                                                                                
(rank:50, lambda: 1.0E-4, Explicit ) Mean Squared Error = 0.0029577581713741545                                                                                
(rank:10, lambda: 1.0,alpha:1.0 ,implicit  ) Mean Squared Error = 10.352420717999916                                                                                
(rank:10, lambda: 1.0,alpha:40.0 ,implicit  ) Mean Squared Error = 7.37758192206552                                                                                
(rank:10, lambda: 0.01,alpha:1.0 ,implicit  ) Mean Squared Error = 9.138333638388543                                                                                
(rank:10, lambda: 0.01,alpha:40.0 ,implicit  ) Mean Squared Error = 7.288950103420938                                                                                
(rank:10, lambda: 1.0E-4,alpha:1.0 ,implicit  ) Mean Squared Error = 9.090678049662575                                                                                
(rank:10, lambda: 1.0E-4,alpha:40.0 ,implicit  ) Mean Squared Error = 7.20726197573743                                                                               
(rank:50, lambda: 1.0,alpha:1.0 ,implicit  ) Mean Squared Error = 9.920570381082038                                                                                
(rank:50, lambda: 1.0,alpha:40.0 ,implicit  ) Mean Squared Error = 7.202627234339378                                                                                
(rank:50, lambda: 0.01,alpha:1.0 ,implicit  ) Mean Squared Error = 7.756830091892575                                                                                
(rank:50, lambda: 0.01,alpha:40.0 ,implicit  ) Mean Squared Error = 7.054065456899226                                                               
(rank:50, lambda: 1.0E-4,alpha:1.0 ,implicit  ) Mean Squared Error = 7.599617817478698                                                                                
(rank:50, lambda: 1.0E-4,alpha:40.0 ,implicit  ) Mean Squared Error = 7.0397787030727645
可以看到rank为50, lambda为0.0001的显性反馈时的MSE最小。我们就已这组参数作为我们的推荐模型。

模型应用

既然我们已经得到了一个很好的推荐模型，下一步就是使用它为所有的用户生成推荐集合。

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
def recommend(sc: SparkContext,
                rawUserMoviesData: RDD[String],
                rawHotMoviesData: RDD[String],
                base:String): Unit = {
    val moviesAndName = buildMovies(rawHotMoviesData)
    val bMoviesAndName = sc.broadcast(moviesAndName)
    val data = buildRatings(rawUserMoviesData)
    val userIdToInt: RDD[(String, Long)] =
      data.map(_.userID).distinct().zipWithUniqueId()
    val reverseUserIDMapping: RDD[(Long, String)] =
      userIdToInt map { case (l, r) => (r, l) }
    val userIDMap: Map[String, Int] =
      userIdToInt.collectAsMap().map { case (n, l) => (n, l.toInt) }
    val bUserIDMap = sc.broadcast(userIDMap)
    val bReverseUserIDMap = sc.broadcast(reverseUserIDMapping.collectAsMap())
    val ratings: RDD[Rating] = data.map { r =>
      Rating(bUserIDMap.value.get(r.userID).get, r.movieID, r.rating)
    }.cache()
    //使用协同过滤算法建模
    //val model = ALS.trainImplicit(ratings, 10, 10, 0.01, 1.0)
    val model = ALS.train(ratings, 50, 10, 0.0001)
    ratings.unpersist()
    //model.save(sc, base+"model")
    //val sameModel = MatrixFactorizationModel.load(sc, base + "model")
    val allRecommendations = model.recommendProductsForUsers(5) map {
      case (userID, recommendations) => {
        var recommendationStr = ""
        for (r <- recommendations) {
          recommendationStr += r.product + ":" + bMoviesAndName.value.getOrElse(r.product, "") + ","
        }
        if (recommendationStr.endsWith(","))
          recommendationStr = recommendationStr.substring(0,recommendationStr.length-1)
        (bReverseUserIDMap.value.get(userID).get,recommendationStr)
      }
    }
    allRecommendations.saveAsTextFile(base + "result.csv")
    unpersist(model)
  }
这里将推荐结果写入到文件中，更实际的情况是把它写入到HDFS中，或者将这个RDD写入到关系型数据库中如Mysql, Postgresql,或者NoSQL数据库中，如MongoDB, cassandra等。 这样我们就可以提供接口为指定的用户提供推荐的电影。

查看本例生成的推荐结果，下面是其中的一个片段，第一个字段是用户名，后面是五个推荐的电影(电影ID:电影名字)

1
2
3
4
5
6
7
8
9
10
11
12
13
(god8knows,25986688:流浪者年代记,26582787:斗地主,24405378:王牌特工：特工学院,22556810:猛龙特囧,25868191:极道大战争)
(60648596,25853129:瑞奇和闪电,26582787:斗地主,3445457:无境之兽,3608742:冲出康普顿,26297388:这时对那时错)
(120501579,25856265:烈日迷踪,3608742:冲出康普顿,26275494:橘色,26297388:这时对那时错,25868191:极道大战争)
(xrzsdan,24405378:王牌特工：特工学院,26599083:妈妈的朋友,10440076:最后的女巫猎人,25868191:极道大战争,25986688:流浪者年代记)
(HoldonBoxer,10604554:躲藏,26297388:这时对那时错,26265099:白河夜船,26275494:橘色,3608742:冲出康普顿)
(46896492,1972724:斯坦福监狱实验,26356488:1944,25717176:新宿天鹅,26582787:斗地主,25919385:长寿商会)
(blankscreen,24405378:王牌特工：特工学院,26599083:妈妈的朋友,25955372:1980年代的爱情,25853129:瑞奇和闪电,25856265:烈日迷踪)
(linyiqing,3608742:冲出康普顿,25868191:极道大战争,26275494:橘色,25955372:1980年代的爱情,26582787:斗地主)
(1477412,25889465:抢劫,25727048:福尔摩斯先生,26252196:卫生间的圣母像,26303865:维多利亚,26276359:酷毙了)
(130875640,24405378:王牌特工：特工学院,25856265:烈日迷踪,25986688:流浪者年代记,25868191:极道大战争,25898213:军犬麦克斯)
(49996306,25919385:长寿商会,26582787:斗地主,26285777:有客到,25830802:对风说爱你,25821461:旅程终点)
(fanshuren,10604554:躲藏,26582787:斗地主,25856265:烈日迷踪,25843352:如此美好,26275494:橘色)
(sweetxyy,26582787:斗地主,25868191:极道大战争,3608742:冲出康普顿,25859495:思悼,22556810:猛龙特囧)
综述

通过前面的介绍，我们可以了解如何使用Spark MLlib的ALS算法为22万豆瓣用户实现一个可用的推荐系统，如何加载数据集和输出数据结果，以及如何对模型进行有效的评估。
你可以使用本文的算法实现其它的推荐系统，如图书，文章，商品等。

代码下载: github

参考文档

Advanced Analytics with Spark
http://yongfeng.me/attach/rs-survey-zhang.pdf
https://github.com/ceys/jdml/wiki/ALS
http://spark.apache.org/docs/latest/mllib-collaborative-filtering.html
https://www.codementor.io/spark/tutorial/building-a-recommender-with-apache-spark-python-example-app-part1
http://blog.javachen.com/2015/04/17/spark-mllib-collaborative-filtering.html
http://www.zhihu.com/question/31509438
