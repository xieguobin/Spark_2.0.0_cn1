#// mlws_chap09(Spark机器学习,第九章)

##// 0、封装包和代码环境  
// 准备环境  
tar xfvz 20news-bydate.tar.gz  

export SPARK_HOME=/Users/erichan/Garden/spark-1.5.1-bin-hadoop2.6  
cd $SPARK_HOME  
bin/spark-shell --name my_mlib --packages org.jblas:jblas:1.2.4-SNAPSHOT --driver-memory 4G --executor-memory 4G --driver-cores 2  

```scala
package mllib_book.mlws.chap09_textprocess

object chap09 extends App{  
  val conf = new SparkConf().setAppName("Spark_chap09").setMaster("local")  
  val sc = new SparkContext(conf)  
```  
  
##// 1、特征抽取   
```scala
val PATH = "/Users/erichan/sourcecode/book/Spark机器学习/20news-bydate"
val path = PATH+"/20news-bydate-train/*"
val rdd = sc.wholeTextFiles(path)
println(rdd.count)
//11314
// 查看新闻组主题
val newsgroups = rdd.map { case (file, text) => file.split("/").takeRight(2).head }
val countByGroup = newsgroups.map(n => (n, 1)).reduceByKey(_ + _).collect.sortBy(-_._2).mkString("\n")
println(countByGroup)
//(rec.sport.hockey,600)
//(soc.religion.christian,599)
//(rec.motorcycles,598)
//(rec.sport.baseball,597)
//(sci.crypt,595)
//(rec.autos,594)
//(sci.med,594)
//(comp.windows.x,593)
//(sci.space,593)
//(sci.electronics,591)
//(comp.os.ms-windows.misc,591)
//(comp.sys.ibm.pc.hardware,590)
//(misc.forsale,585)
//(comp.graphics,584)
//(comp.sys.mac.hardware,578)
//(talk.politics.mideast,564)
//(talk.politics.guns,546)
//(alt.atheism,480)
//(talk.politics.misc,465)
//(talk.religion.misc,377)
```  

##// 2、建模  
```scala
// 2.1、分词
val text = rdd.map { case (file, text) => text }
val whiteSpaceSplit = text.flatMap(t => t.split(" ").map(_.toLowerCase))
println(whiteSpaceSplit.distinct.count)
println(whiteSpaceSplit.sample(true, 0.3, 42).take(100).mkString(","))
//402978
//from:,mathew,mathew,faq:,faq:,atheist,resources
//summary:,music,--,fiction,,mantis,consultants,,uk.
//supersedes:,290
//archive-name:,1.0
//,,,,,,,,,,,,,,,,,,,organizations
//,organizations
//,,,,,,,,,,,,,,,,stickers,and,and,the,from,from,in,to:,to:,ffrf,,256-8900
//evolution,designs
//evolution,a,stick,cars,,written
//inside.,fish,us.
//write,evolution,,,,,,,bay,can,get,get,,to,the
//price,is,of,the,the,so,on.,and,foote.,,atheist,pp.,0-910309-26-4,,,atrocities,,foote:,aap.,,the

// 2.2、改进分词
val nonWordSplit = text.flatMap(t => t.split("""\W+""").map(_.toLowerCase))
println(nonWordSplit.distinct.count)
println(nonWordSplit.distinct.sample(true, 0.3, 42).take(100).mkString(","))
val regex = """[^0-9]*""".r
val filterNumbers = nonWordSplit.filter(token => regex.pattern.matcher(token).matches)
println(filterNumbers.distinct.count)
println(filterNumbers.distinct.sample(true, 0.3, 42).take(100).mkString(","))

// 2.3、移除停用词
val tokenCounts = filterNumbers.map(t => (t, 1)).reduceByKey(_ + _)
val oreringDesc = Ordering.by[(String, Int), Int](_._2)
//println(tokenCounts.top(20)(oreringDesc).mkString("\n"))

val stopwords = Set(
    "the","a","an","of","or","in","for","by","on","but", "is", "not", "with", "as", "was", "if",
    "they", "are", "this", "and", "it", "have", "from", "at", "my", "be", "that", "to"
)
val tokenCountsFilteredStopwords = tokenCounts.filter { case (k, v) => !stopwords.contains(k) }
//println(tokenCountsFilteredStopwords.top(20)(oreringDesc).mkString("\n"))

val tokenCountsFilteredSize = tokenCountsFilteredStopwords.filter { case (k, v) => k.size >= 2 }
println(tokenCountsFilteredSize.top(20)(oreringDesc).mkString("\n"))

// 2.4、移除低频词
val oreringAsc = Ordering.by[(String, Int), Int](-_._2)
//println(tokenCountsFilteredSize.top(20)(oreringAsc).mkString("\n"))

val rareTokens = tokenCounts.filter{ case (k, v) => v < 2 }.map { case (k, v) => k }.collect.toSet
val tokenCountsFilteredAll = tokenCountsFilteredSize.filter { case (k, v) => !rareTokens.contains(k) }
println(tokenCountsFilteredAll.top(20)(oreringAsc).mkString("\n"))

def tokenize(line: String): Seq[String] = {
    line.split("""\W+""")
        .map(_.toLowerCase)
        .filter(token => regex.pattern.matcher(token).matches)
        .filterNot(token => stopwords.contains(token))
        .filterNot(token => rareTokens.contains(token))
        .filter(token => token.size >= 2)
        .toSeq
}

//println(text.flatMap(doc => tokenize(doc)).distinct.count)
val tokens = text.map(doc => tokenize(doc))
println(tokens.first.take(20))

// 2.5、提取词干
// 标准NLP方法
// 搜索引擎
//   NLTK
//   OpenNLP
//   Lucene
```  

##// 3、训练模型
```scala
// 3.1、HashingTF 特征哈希
import org.apache.spark.mllib.linalg.{ SparseVector => SV }
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF
// set the dimensionality of TF-IDF vectors to 2^18
val dim = math.pow(2, 18).toInt
val hashingTF = new HashingTF(dim)
val tf = hashingTF.transform(tokens)
tf.cache
val v = tf.first.asInstanceOf[SV]
println(v.size)
println(v.values.size)
println(v.values.take(10).toSeq)
println(v.indices.take(10).toSeq)
//262144
//706
//WrappedArray(1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0)
//WrappedArray(313, 713, 871, 1202, 1203, 1209, 1795, 1862, 3115, 3166)
fit & transform
val idf = new IDF().fit(tf)
val tfidf = idf.transform(tf)
val v2 = tfidf.first.asInstanceOf[SV]
println(v2.values.size)
println(v2.values.take(10).toSeq)
println(v2.indices.take(10).toSeq)
//706
//WrappedArray(2.3869085659322193, 4.670445463955571, 6.561295835827856, 4.597686109673142, 8.932700215224111, 5.750365619611528, 2.1871123786150006, 5.520408782213984, 3.4312512246662714, 1.7430324343790569)
//WrappedArray(313, 713, 871, 1202, 1203, 1209, 1795, 1862, 3115, 3166) 

// 3.2、分析权重
val minMaxVals = tfidf.map { v =>
    val sv = v.asInstanceOf[SV]
    (sv.values.min, sv.values.max)
}
val globalMinMax = minMaxVals.reduce { case ((min1, max1), (min2, max2)) =>
    (math.min(min1, min2), math.max(max1, max2))
}
println(globalMinMax)
//globalMinMax: (Double, Double) = (0.0,66155.39470409753)
// 常用词
val common = sc.parallelize(Seq(Seq("you", "do", "we")))
val tfCommon = hashingTF.transform(common)
val tfidfCommon = idf.transform(tfCommon)
val commonVector = tfidfCommon.first.asInstanceOf[SV]
println(commonVector.values.toSeq)
//WrappedArray(0.9965359935704624, 1.3348773448236835, 0.5457486182039175)
// 不常出现的单词
val uncommon = sc.parallelize(Seq(Seq("telescope", "legislation", "investment")))
val tfUncommon = hashingTF.transform(uncommon)
val tfidfUncommon = idf.transform(tfUncommon)
val uncommonVector = tfidfUncommon.first.asInstanceOf[SV]
println(uncommonVector.values.toSeq)
//WrappedArray(5.3265513728351666, 5.308532867332488, 5.483736956357579)
```

##// 4、使用模型
```scala
// 4.1、余弦相似度
import breeze.linalg._

val hockeyText = rdd.filter { case (file, text) => file.contains("hockey") }
val hockeyTF = hockeyText.mapValues(doc => hashingTF.transform(tokenize(doc)))
val hockeyTfIdf = idf.transform(hockeyTF.map(_._2))

val hockey1 = hockeyTfIdf.sample(true, 0.1, 42).first.asInstanceOf[SV]
val breeze1 = new SparseVector(hockey1.indices, hockey1.values, hockey1.size)

val hockey2 = hockeyTfIdf.sample(true, 0.1, 43).first.asInstanceOf[SV]
val breeze2 = new SparseVector(hockey2.indices, hockey2.values, hockey2.size)
val cosineSim = breeze1.dot(breeze2) / (norm(breeze1) * norm(breeze2))
println(cosineSim)
//cosineSim: Double = 0.060250114361164626
val graphicsText = rdd.filter { case (file, text) => file.contains("comp.graphics") }
val graphicsTF = graphicsText.mapValues(doc => hashingTF.transform(tokenize(doc)))
val graphicsTfIdf = idf.transform(graphicsTF.map(_._2))
val graphics = graphicsTfIdf.sample(true, 0.1, 42).first.asInstanceOf[SV]
val breezeGraphics = new SparseVector(graphics.indices, graphics.values, graphics.size)
val cosineSim2 = breeze1.dot(breezeGraphics) / (norm(breeze1) * norm(breezeGraphics))
println(cosineSim2)
//cosineSim2: Double = 0.004664850323792852
val baseballText = rdd.filter { case (file, text) => file.contains("baseball") }
val baseballTF = baseballText.mapValues(doc => hashingTF.transform(tokenize(doc)))
val baseballTfIdf = idf.transform(baseballTF.map(_._2))
val baseball = baseballTfIdf.sample(true, 0.1, 42).first.asInstanceOf[SV]
val breezeBaseball = new SparseVector(baseball.indices, baseball.values, baseball.size)
val cosineSim3 = breeze1.dot(breezeBaseball) / (norm(breeze1) * norm(breezeBaseball))
println(cosineSim3)
//0.05047395039466008

// 4.2 学习单词与主题的映射关系
// 多分类映射
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.MulticlassMetrics

val newsgroupsMap = newsgroups.distinct.collect().zipWithIndex.toMap
val zipped = newsgroups.zip(tfidf)
val train = zipped.map { case (topic, vector) => LabeledPoint(newsgroupsMap(topic), vector) }
train.cache
// 朴素贝叶斯训练
val model = NaiveBayes.train(train, lambda = 0.1)
// 加载测试数据集
val testPath = PATH+"/20news-bydate-test/*"
val testRDD = sc.wholeTextFiles(testPath)
val testLabels = testRDD.map { case (file, text) =>
    val topic = file.split("/").takeRight(2).head
    newsgroupsMap(topic)
}
val testTf = testRDD.map { case (file, text) => hashingTF.transform(tokenize(text)) }
val testTfIdf = idf.transform(testTf)
val zippedTest = testLabels.zip(testTfIdf)
val test = zippedTest.map { case (topic, vector) => LabeledPoint(topic, vector) }
// 计算准确度和多分类加权F-指标
val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
println(accuracy)
//0.7915560276155071
val metrics = new MulticlassMetrics(predictionAndLabel)
println(metrics.weightedFMeasure)
//0.7810675969031116
```

##// 5、评估模型
```scala
val rawTokens = rdd.map { case (file, text) => text.split(" ") }
val rawTF = rawTokens.map(doc => hashingTF.transform(doc))
val rawTrain = newsgroups.zip(rawTF).map { case (topic, vector) => LabeledPoint(newsgroupsMap(topic), vector) }
val rawModel = NaiveBayes.train(rawTrain, lambda = 0.1)
val rawTestTF = testRDD.map { case (file, text) => hashingTF.transform(text.split(" ")) }
val rawZippedTest = testLabels.zip(rawTestTF)
val rawTest = rawZippedTest.map { case (topic, vector) => LabeledPoint(topic, vector) }
val rawPredictionAndLabel = rawTest.map(p => (rawModel.predict(p.features), p.label))
val rawAccuracy = 1.0 * rawPredictionAndLabel.filter(x => x._1 == x._2).count() / rawTest.count()
println(rawAccuracy)
//0.7648698884758365
val rawMetrics = new MulticlassMetrics(rawPredictionAndLabel)
println(rawMetrics.weightedFMeasure)
//0.7653320418573546
```

##// 6、Word2Vec模型
// Word2Vec模型（分布向量表示）：把每个单词表示成一个向量，MLlib中使用skip-gram模型
```scala
// 6.1 训练
import org.apache.spark.mllib.feature.Word2Vec
val word2vec = new Word2Vec()
word2vec.setSeed(42) // we do this to generate the same results each time
val word2vecModel = word2vec.fit(tokens)

// 6.2 使用
// 最相似的20个单词
word2vecModel.findSynonyms("hockey", 20).foreach(println)
//(sport,1.4818968962277133)
//(ecac,1.467546566194254)
//(hispanic,1.4166835301985194)
//(glens,1.4061103042432825)
//(woofers,1.3810090447028116)
//(tournament,1.3148823031671586)
//(champs,1.3133863003013941)
//(boxscores,1.307735040384543)
//(aargh,1.274986851270267)
//(ahl,1.265165428167253)
//(playoff,1.2645991118770572)
//(ncaa,1.2383382015648046)
//(pool,1.2261154635870224)
//(champion,1.2119919989539134)
//(filinuk,1.2062208620660915)
//(olympic,1.2026738930160243)
//(motorcycles,1.2008032355579679)
//(yankees,1.1989755767973371)
//(calder,1.194001886835493)
//(homeruns,1.1800625883573932)

word2vecModel.findSynonyms("legislation", 20).foreach(println)
//(accommodates,0.9918184454068688)
//(briefed,0.9256758135452989)
//(amended,0.9105987267173344)
//(telephony,0.8679173760123956)
//(pitted,0.8609974033962533)
//(aclu,0.8605885863332372)
//(licensee,0.8493930472487975)
//(agency,0.836706135804648)
//(policies,0.8337986602365566)
//(senate,0.8327312936220903)
//(businesses,0.8291191155630467)
//(permit,0.8266658804181389)
//(cpsr,0.8231228090944367)
//(cooperation,0.8195562469006543)
//(surveillance,0.8134342524628756)
//(congress,0.8132899468772855)
//(restricted,0.8115013134507126)
//(procure,0.8096839595766356)
//(inquiry,0.8086297702914405)
//(industry,0.8077900093754752)
```
