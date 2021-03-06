Classification and regression - spark.ml

In spark.ml, we implement popular linear methods such as logistic regression and linear least squares with L1L1 or L2L2 regularization. Refer to the linear methods in mllib for details about implementation and tuning. We also include a DataFrame API for Elastic net, a hybrid of L1L1 and L2L2 regularization proposed in Zou et al, Regularization and variable selection via the elastic net. Mathematically, it is defined as a convex combination of the L1L1 and the L2L2 regularization terms:
α(λ∥w∥1)+(1−α)(λ2∥w∥22),α∈[0,1],λ≥0
α(λ∥w∥1)+(1−α)(λ2∥w∥22),α∈[0,1],λ≥0
By setting αα properly, elastic net contains both L1L1 and L2L2 regularization as special cases. For example, if a linear regression model is trained with the elastic net parameter αα set to 11, it is equivalent to a Lasso model. On the other hand, if αα is set to 00, the trained model reduces to a ridge regression model. We implement Pipelines API for both linear regression and logistic regression with elastic net regularization.

Classification
Logistic regression
Logistic regression is a popular method to predict a binary response. It is a special case of Generalized Linear models that predicts the probability of the outcome. For more background and more details about the implementation, refer to the documentation of the logistic regression in spark.mllib.

The current implementation of logistic regression in spark.ml only supports binary classes. Support for multiclass regression will be added in the future.
Example

The following example shows how to train a logistic regression model with elastic net regularization. elasticNetParam corresponds to αα and regParam corresponds to λλ.

Scala
Java
Python
import org.apache.spark.ml.classification.LogisticRegression

// Load training data
val training = sqlCtx.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)

// Fit the model
val lrModel = lr.fit(training)

// Print the coefficients and intercept for logistic regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
Find full example code at "examples/src/main/scala/org/apache/spark/examples/ml/LogisticRegressionWithElasticNetExample.scala" in the Spark repo.
The spark.ml implementation of logistic regression also supports extracting a summary of the model over the training set. Note that the predictions and metrics which are stored as DataFrame in BinaryLogisticRegressionSummary are annotated @transient and hence only available on the driver.

Scala
Java
Python
LogisticRegressionTrainingSummary provides a summary for a LogisticRegressionModel. Currently, only binary classification is supported and the summary must be explicitly cast to BinaryLogisticRegressionTrainingSummary. This will likely change when multiclass classification is supported.

Continuing the earlier example:

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}

// Extract the summary from the returned LogisticRegressionModel instance trained in the earlier
// example
val trainingSummary = lrModel.summary

// Obtain the objective per iteration.
val objectiveHistory = trainingSummary.objectiveHistory
objectiveHistory.foreach(loss => println(loss))

// Obtain the metrics useful to judge performance on test data.
// We cast the summary to a BinaryLogisticRegressionSummary since the problem is a
// binary classification problem.
val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]

// Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
val roc = binarySummary.roc
roc.show()
println(binarySummary.areaUnderROC)

// Set the model threshold to maximize F-Measure
val fMeasure = binarySummary.fMeasureByThreshold
val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure)
  .select("threshold").head().getDouble(0)
lrModel.setThreshold(bestThreshold)
Find full example code at "examples/src/main/scala/org/apache/spark/examples/ml/LogisticRegressionSummaryExample.scala" in the Spark repo.
Decision tree classifier
Decision trees are a popular family of classification and regression methods. More information about the spark.ml implementation can be found further in the section on decision trees.

Example

The following examples load a dataset in LibSVM format, split it into training and test sets, train on the first dataset, and then evaluate on the held-out test set. We use two feature transformers to prepare the data; these help index categories for the label and categorical features, adding metadata to the DataFrame which the Decision Tree algorithm can recognize.

Scala
Java
Python
More details on parameters can be found in the Scala API documentation.

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorIndexer}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Load the data stored in LIBSVM format as a DataFrame.
val data = sqlContext.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)
// Automatically identify categorical features, and index them.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4) // features with > 4 distinct values are treated as continuous
  .fit(data)

// Split the data into training and test sets (30% held out for testing)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a DecisionTree model.
val dt = new DecisionTreeClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// Chain indexers and tree in a Pipeline
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// Train model.  This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("precision")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println("Learned classification tree model:\n" + treeModel.toDebugString)
Find full example code at "examples/src/main/scala/org/apache/spark/examples/ml/DecisionTreeClassificationExample.scala" in the Spark repo.
Random forest classifier
Random forests are a popular family of classification and regression methods. More information about the spark.ml implementation can be found further in the section on random forests.

Example

The following examples load a dataset in LibSVM format, split it into training and test sets, train on the first dataset, and then evaluate on the held-out test set. We use two feature transformers to prepare the data; these help index categories for the label and categorical features, adding metadata to the DataFrame which the tree-based algorithms can recognize.

Scala
Java
Python
Refer to the Scala API docs for more details.

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Load and parse the data file, converting it to a DataFrame.
val data = sqlContext.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)
// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4)
  .fit(data)

// Split the data into training and test sets (30% held out for testing)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a RandomForest model.
val rf = new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setNumTrees(10)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// Chain indexers and forest in a Pipeline
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// Train model.  This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("precision")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println("Learned classification forest model:\n" + rfModel.toDebugString)
Find full example code at "examples/src/main/scala/org/apache/spark/examples/ml/RandomForestClassifierExample.scala" in the Spark repo.
Gradient-boosted tree classifier
Gradient-boosted trees (GBTs) are a popular classification and regression method using ensembles of decision trees. More information about the spark.ml implementation can be found further in the section on GBTs.

Example

The following examples load a dataset in LibSVM format, split it into training and test sets, train on the first dataset, and then evaluate on the held-out test set. We use two feature transformers to prepare the data; these help index categories for the label and categorical features, adding metadata to the DataFrame which the tree-based algorithms can recognize.

Scala
Java
Python
Refer to the Scala API docs for more details.

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Load and parse the data file, converting it to a DataFrame.
val data = sqlContext.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)
// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4)
  .fit(data)

// Split the data into training and test sets (30% held out for testing)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a GBT model.
val gbt = new GBTClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setMaxIter(10)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// Chain indexers and GBT in a Pipeline
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

// Train model.  This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("precision")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println("Learned classification GBT model:\n" + gbtModel.toDebugString)
Find full example code at "examples/src/main/scala/org/apache/spark/examples/ml/GradientBoostedTreeClassifierExample.scala" in the Spark repo.
Multilayer perceptron classifier
Multilayer perceptron classifier (MLPC) is a classifier based on the feedforward artificial neural network. MLPC consists of multiple layers of nodes. Each layer is fully connected to the next layer in the network. Nodes in the input layer represent the input data. All other nodes maps inputs to the outputs by performing linear combination of the inputs with the node’s weights ww and bias bb and applying an activation function. It can be written in matrix form for MLPC with K+1K+1 layers as follows:
y(x)=fK(...f2(wT2f1(wT1x+b1)+b2)...+bK)
y(x)=fK(...f2(w2Tf1(w1Tx+b1)+b2)...+bK)
Nodes in intermediate layers use sigmoid (logistic) function:
f(zi)=11+e−zi
f(zi)=11+e−zi
Nodes in the output layer use softmax function:
f(zi)=ezi∑Nk=1ezk
f(zi)=ezi∑k=1Nezk
The number of nodes NN in the output layer corresponds to the number of classes.

MLPC employes backpropagation for learning the model. We use logistic loss function for optimization and L-BFGS as optimization routine.

Example

Scala
Java
Python
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Load the data stored in LIBSVM format as a DataFrame.
val data = sqlContext.read.format("libsvm")
  .load("data/mllib/sample_multiclass_classification_data.txt")
// Split the data into train and test
val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)
// specify layers for the neural network:
// input layer of size 4 (features), two intermediate of size 5 and 4
// and output of size 3 (classes)
val layers = Array[Int](4, 5, 4, 3)
// create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setBlockSize(128)
  .setSeed(1234L)
  .setMaxIter(100)
// train the model
val model = trainer.fit(train)
// compute precision on the test set
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("precision")
println("Precision:" + evaluator.evaluate(predictionAndLabels))
Find full example code at "examples/src/main/scala/org/apache/spark/examples/ml/MultilayerPerceptronClassifierExample.scala" in the Spark repo.
One-vs-Rest classifier (a.k.a. One-vs-All)
OneVsRest is an example of a machine learning reduction for performing multiclass classification given a base classifier that can perform binary classification efficiently. It is also known as “One-vs-All.”

OneVsRest is implemented as an Estimator. For the base classifier it takes instances of Classifier and creates a binary classification problem for each of the k classes. The classifier for class i is trained to predict whether the label is i or not, distinguishing class i from all other classes.

Predictions are done by evaluating each binary classifier and the index of the most confident classifier is output as label.

Example

The example below demonstrates how to load the Iris dataset, parse it as a DataFrame and perform multiclass classification using OneVsRest. The test error is calculated to measure the algorithm accuracy.

Scala
Java
Refer to the Scala API docs for more details.

import org.apache.spark.examples.mllib.AbstractParams
import org.apache.spark.ml.classification.{OneVsRest, LogisticRegression}
import org.apache.spark.ml.util.MetadataUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.DataFrame

val inputData = sqlContext.read.format("libsvm").load(params.input)
// compute the train/test split: if testInput is not provided use part of input.
val data = params.testInput match {
  case Some(t) => {
    // compute the number of features in the training set.
    val numFeatures = inputData.first().getAs[Vector](1).size
    val testData = sqlContext.read.option("numFeatures", numFeatures.toString)
      .format("libsvm").load(t)
    Array[DataFrame](inputData, testData)
  }
  case None => {
    val f = params.fracTest
    inputData.randomSplit(Array(1 - f, f), seed = 12345)
  }
}
val Array(train, test) = data.map(_.cache())

// instantiate the base classifier
val classifier = new LogisticRegression()
  .setMaxIter(params.maxIter)
  .setTol(params.tol)
  .setFitIntercept(params.fitIntercept)

// Set regParam, elasticNetParam if specified in params
params.regParam.foreach(classifier.setRegParam)
params.elasticNetParam.foreach(classifier.setElasticNetParam)

// instantiate the One Vs Rest Classifier.

val ovr = new OneVsRest()
ovr.setClassifier(classifier)

// train the multiclass model.
val (trainingDuration, ovrModel) = time(ovr.fit(train))

// score the model on test data.
val (predictionDuration, predictions) = time(ovrModel.transform(test))

// evaluate the model
val predictionsAndLabels = predictions.select("prediction", "label")
  .map(row => (row.getDouble(0), row.getDouble(1)))

val metrics = new MulticlassMetrics(predictionsAndLabels)

val confusionMatrix = metrics.confusionMatrix

// compute the false positive rate per label
val predictionColSchema = predictions.schema("prediction")
val numClasses = MetadataUtils.getNumClasses(predictionColSchema).get
val fprs = Range(0, numClasses).map(p => (p, metrics.falsePositiveRate(p.toDouble)))

println(s" Training Time ${trainingDuration} sec\n")

println(s" Prediction Time ${predictionDuration} sec\n")

println(s" Confusion Matrix\n ${confusionMatrix.toString}\n")

println("label\tfpr")

println(fprs.map {case (label, fpr) => label + "\t" + fpr}.mkString("\n"))
Find full example code at "examples/src/main/scala/org/apache/spark/examples/ml/OneVsRestExample.scala" in the Spark repo.
Regression
Linear regression
The interface for working with linear regression models and model summaries is similar to the logistic regression case.

Example

The following example demonstrates training an elastic net regularized linear regression model and extracting model summary statistics.

Scala
Java
Python
import org.apache.spark.ml.regression.LinearRegression

// Load training data
val training = sqlCtx.read.format("libsvm")
  .load("data/mllib/sample_linear_regression_data.txt")

val lr = new LinearRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)

// Fit the model
val lrModel = lr.fit(training)

// Print the coefficients and intercept for linear regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

// Summarize the model over the training set and print out some metrics
val trainingSummary = lrModel.summary
println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")
Find full example code at "examples/src/main/scala/org/apache/spark/examples/ml/LinearRegressionWithElasticNetExample.scala" in the Spark repo.
Decision tree regression
Decision trees are a popular family of classification and regression methods. More information about the spark.ml implementation can be found further in the section on decision trees.

Example

The following examples load a dataset in LibSVM format, split it into training and test sets, train on the first dataset, and then evaluate on the held-out test set. We use a feature transformer to index categorical features, adding metadata to the DataFrame which the Decision Tree algorithm can recognize.

Scala
Java
Python
More details on parameters can be found in the Scala API documentation.

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.evaluation.RegressionEvaluator

// Load the data stored in LIBSVM format as a DataFrame.
val data = sqlContext.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Automatically identify categorical features, and index them.
// Here, we treat features with > 4 distinct values as continuous.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4)
  .fit(data)

// Split the data into training and test sets (30% held out for testing)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a DecisionTree model.
val dt = new DecisionTreeRegressor()
  .setLabelCol("label")
  .setFeaturesCol("indexedFeatures")

// Chain indexer and tree in a Pipeline
val pipeline = new Pipeline()
  .setStages(Array(featureIndexer, dt))

// Train model.  This also runs the indexer.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

// Select (prediction, true label) and compute test error
val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)

val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
println("Learned regression tree model:\n" + treeModel.toDebugString)
Find full example code at "examples/src/main/scala/org/apache/spark/examples/ml/DecisionTreeRegressionExample.scala" in the Spark repo.
Random forest regression
Random forests are a popular family of classification and regression methods. More information about the spark.ml implementation can be found further in the section on random forests.

Example

The following examples load a dataset in LibSVM format, split it into training and test sets, train on the first dataset, and then evaluate on the held-out test set. We use a feature transformer to index categorical features, adding metadata to the DataFrame which the tree-based algorithms can recognize.

Scala
Java
Python
Refer to the Scala API docs for more details.

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}

// Load and parse the data file, converting it to a DataFrame.
val data = sqlContext.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4)
  .fit(data)

// Split the data into training and test sets (30% held out for testing)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a RandomForest model.
val rf = new RandomForestRegressor()
  .setLabelCol("label")
  .setFeaturesCol("indexedFeatures")

// Chain indexer and forest in a Pipeline
val pipeline = new Pipeline()
  .setStages(Array(featureIndexer, rf))

// Train model.  This also runs the indexer.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

// Select (prediction, true label) and compute test error
val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)

val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
println("Learned regression forest model:\n" + rfModel.toDebugString)
Find full example code at "examples/src/main/scala/org/apache/spark/examples/ml/RandomForestRegressorExample.scala" in the Spark repo.
Gradient-boosted tree regression
Gradient-boosted trees (GBTs) are a popular regression method using ensembles of decision trees. More information about the spark.ml implementation can be found further in the section on GBTs.

Example

Note: For this example dataset, GBTRegressor actually only needs 1 iteration, but that will not be true in general.

Scala
Java
Python
Refer to the Scala API docs for more details.

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}

// Load and parse the data file, converting it to a DataFrame.
val data = sqlContext.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4)
  .fit(data)

// Split the data into training and test sets (30% held out for testing)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a GBT model.
val gbt = new GBTRegressor()
  .setLabelCol("label")
  .setFeaturesCol("indexedFeatures")
  .setMaxIter(10)

// Chain indexer and GBT in a Pipeline
val pipeline = new Pipeline()
  .setStages(Array(featureIndexer, gbt))

// Train model.  This also runs the indexer.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

// Select (prediction, true label) and compute test error
val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)

val gbtModel = model.stages(1).asInstanceOf[GBTRegressionModel]
println("Learned regression GBT model:\n" + gbtModel.toDebugString)
Find full example code at "examples/src/main/scala/org/apache/spark/examples/ml/GradientBoostedTreeRegressorExample.scala" in the Spark repo.
Survival regression
In spark.ml, we implement the Accelerated failure time (AFT) model which is a parametric survival regression model for censored data. It describes a model for the log of survival time, so it’s often called log-linear model for survival analysis. Different from Proportional hazards model designed for the same purpose, the AFT model is more easily to parallelize because each instance contribute to the objective function independently.

Given the values of the covariates x‘x‘, for random lifetime titi of subjects i = 1, …, n, with possible right-censoring, the likelihood function under the AFT model is given as:
L(β,σ)=∏i=1n[1σf0(logti−x′βσ)]δiS0(logti−x′βσ)1−δi
L(β,σ)=∏i=1n[1σf0(log⁡ti−x′βσ)]δiS0(log⁡ti−x′βσ)1−δi
Where δiδi is the indicator of the event has occurred i.e. uncensored or not. Using ϵi=logti−x‘βσϵi=log⁡ti−x‘βσ, the log-likelihood function assumes the form:
ι(β,σ)=∑i=1n[−δilogσ+δilogf0(ϵi)+(1−δi)logS0(ϵi)]
ι(β,σ)=∑i=1n[−δilog⁡σ+δilog⁡f0(ϵi)+(1−δi)log⁡S0(ϵi)]
Where S0(ϵi)S0(ϵi) is the baseline survivor function, and f0(ϵi)f0(ϵi) is corresponding density function.

The most commonly used AFT model is based on the Weibull distribution of the survival time. The Weibull distribution for lifetime corresponding to extreme value distribution for log of the lifetime, and the S0(ϵ)S0(ϵ) function is:
S0(ϵi)=exp(−eϵi)
S0(ϵi)=exp⁡(−eϵi)
the f0(ϵi)f0(ϵi) function is:
f0(ϵi)=eϵiexp(−eϵi)
f0(ϵi)=eϵiexp⁡(−eϵi)
The log-likelihood function for AFT model with Weibull distribution of lifetime is:
ι(β,σ)=−∑i=1n[δilogσ−δiϵi+eϵi]
ι(β,σ)=−∑i=1n[δilog⁡σ−δiϵi+eϵi]
Due to minimizing the negative log-likelihood equivalent to maximum a posteriori probability, the loss function we use to optimize is −ι(β,σ)−ι(β,σ). The gradient functions for ββ and logσlog⁡σ respectively are:
∂(−ι)∂β=∑1=1n[δi−eϵi]xiσ
∂(−ι)∂β=∑1=1n[δi−eϵi]xiσ
∂(−ι)∂(logσ)=∑i=1n[δi+(δi−eϵi)ϵi]
∂(−ι)∂(log⁡σ)=∑i=1n[δi+(δi−eϵi)ϵi]

The AFT model can be formulated as a convex optimization problem, i.e. the task of finding a minimizer of a convex function −ι(β,σ)−ι(β,σ) that depends coefficients vector ββ and the log of scale parameter logσlog⁡σ. The optimization algorithm underlying the implementation is L-BFGS. The implementation matches the result from R’s survival function survreg

Example

Scala
Java
Python
import org.apache.spark.ml.regression.AFTSurvivalRegression
import org.apache.spark.mllib.linalg.Vectors

val training = sqlContext.createDataFrame(Seq(
  (1.218, 1.0, Vectors.dense(1.560, -0.605)),
  (2.949, 0.0, Vectors.dense(0.346, 2.158)),
  (3.627, 0.0, Vectors.dense(1.380, 0.231)),
  (0.273, 1.0, Vectors.dense(0.520, 1.151)),
  (4.199, 0.0, Vectors.dense(0.795, -0.226))
)).toDF("label", "censor", "features")
val quantileProbabilities = Array(0.3, 0.6)
val aft = new AFTSurvivalRegression()
  .setQuantileProbabilities(quantileProbabilities)
  .setQuantilesCol("quantiles")

val model = aft.fit(training)

// Print the coefficients, intercept and scale parameter for AFT survival regression
println(s"Coefficients: ${model.coefficients} Intercept: " +
  s"${model.intercept} Scale: ${model.scale}")
model.transform(training).show(false)
Find full example code at "examples/src/main/scala/org/apache/spark/examples/ml/AFTSurvivalRegressionExample.scala" in the Spark repo.
Decision trees
Decision trees and their ensembles are popular methods for the machine learning tasks of classification and regression. Decision trees are widely used since they are easy to interpret, handle categorical features, extend to the multiclass classification setting, do not require feature scaling, and are able to capture non-linearities and feature interactions. Tree ensemble algorithms such as random forests and boosting are among the top performers for classification and regression tasks.

The spark.ml implementation supports decision trees for binary and multiclass classification and for regression, using both continuous and categorical features. The implementation partitions data by rows, allowing distributed training with millions or even billions of instances.

Users can find more information about the decision tree algorithm in the MLlib Decision Tree guide. The main differences between this API and the original MLlib Decision Tree API are:

support for ML Pipelines
separation of Decision Trees for classification vs. regression
use of DataFrame metadata to distinguish continuous and categorical features
The Pipelines API for Decision Trees offers a bit more functionality than the original API. In particular, for classification, users can get the predicted probability of each class (a.k.a. class conditional probabilities).

Ensembles of trees (Random Forests and Gradient-Boosted Trees) are described below in the Tree ensembles section.

Inputs and Outputs
We list the input and output (prediction) column types here. All output columns are optional; to exclude an output column, set its corresponding Param to an empty string.

Input Columns
Param name	Type(s)	Default	Description
labelCol	Double	"label"	Label to predict
featuresCol	Vector	"features"	Feature vector
Output Columns
Param name	Type(s)	Default	Description	Notes
predictionCol	Double	"prediction"	Predicted label	
rawPredictionCol	Vector	"rawPrediction"	Vector of length # classes, with the counts of training instance labels at the tree node which makes the prediction	Classification only
probabilityCol	Vector	"probability"	Vector of length # classes equal to rawPrediction normalized to a multinomial distribution	Classification only
Tree Ensembles
The DataFrame API supports two major tree ensemble algorithms: Random Forests and Gradient-Boosted Trees (GBTs). Both use spark.ml decision trees as their base models.

Users can find more information about ensemble algorithms in the MLlib Ensemble guide.
In this section, we demonstrate the DataFrame API for ensembles.

The main differences between this API and the original MLlib ensembles API are:

support for DataFrames and ML Pipelines
separation of classification vs. regression
use of DataFrame metadata to distinguish continuous and categorical features
more functionality for random forests: estimates of feature importance, as well as the predicted probability of each class (a.k.a. class conditional probabilities) for classification.
Random Forests
Random forests are ensembles of decision trees. Random forests combine many decision trees in order to reduce the risk of overfitting. The spark.ml implementation supports random forests for binary and multiclass classification and for regression, using both continuous and categorical features.

For more information on the algorithm itself, please see the spark.mllib documentation on random forests.

Inputs and Outputs
We list the input and output (prediction) column types here. All output columns are optional; to exclude an output column, set its corresponding Param to an empty string.

Input Columns

Param name	Type(s)	Default	Description
labelCol	Double	"label"	Label to predict
featuresCol	Vector	"features"	Feature vector
Output Columns (Predictions)

Param name	Type(s)	Default	Description	Notes
predictionCol	Double	"prediction"	Predicted label	
rawPredictionCol	Vector	"rawPrediction"	Vector of length # classes, with the counts of training instance labels at the tree node which makes the prediction	Classification only
probabilityCol	Vector	"probability"	Vector of length # classes equal to rawPrediction normalized to a multinomial distribution	Classification only
Gradient-Boosted Trees (GBTs)
Gradient-Boosted Trees (GBTs) are ensembles of decision trees. GBTs iteratively train decision trees in order to minimize a loss function. The spark.ml implementation supports GBTs for binary classification and for regression, using both continuous and categorical features.

For more information on the algorithm itself, please see the spark.mllib documentation on GBTs.

Inputs and Outputs
We list the input and output (prediction) column types here. All output columns are optional; to exclude an output column, set its corresponding Param to an empty string.

Input Columns

Param name	Type(s)	Default	Description
labelCol	Double	"label"	Label to predict
featuresCol	Vector	"features"	Feature vector
Note that GBTClassifier currently only supports binary labels.

Output Columns (Predictions)

Param name	Type(s)	Default	Description	Notes
predictionCol	Double	"prediction"	Predicted label	
In the future, GBTClassifier will also output columns for rawPrediction and probability, just as RandomForestClassifier does.
