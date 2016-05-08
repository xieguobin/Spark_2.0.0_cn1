#// mlws_chap06(Spark机器学习,第六章)

##// 0、封装包和代码环境
//export SPARK_HOME=/Users/erichan/garden/spark-1.5.1-bin-hadoop2.6  
//export PYTHONPATH=${SPARK_HOME}/python/:${SPARK_HOME}/python/lib/py4j-0.8.2.1-src.zip  

//cd $SPARK_HOME  
//IPYTHON=1 IPYTHON_OPTS="--pylab" ./bin/pyspark --driver-memory 4G --executor-memory 4G --driver-cores 2  

```scala
package mllib_book.mlws.chap06_regression

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.tree import DecisionTree
import numpy as np

object Lr01 extends App{  
  val conf = new SparkConf().setAppName("Spark_Lr").setMaster("local")  
  val sc = new SparkContext(conf)  
```  
  
##// 1、数据导入和特征抽取   
```scala
// first remove the headers by using the 'sed' command: 
// 数据集下载地址：http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset.  
// 论文下载地址：http://link.springer.com/article/10.1007%2Fs13748-013-0040-3.
// sed 1d hour.csv > hour_noheader.csv
path = "/PATH/hour_noheader.csv"
raw_data = sc.textFile(path)
num_data = raw_data.count()
records = raw_data.map(lambda x: x.split(","))

first = records.first()
print first
print num_data
//[u'1', u'2011-01-01', u'1', u'0', u'1', u'0', u'0', u'6', u'0', u'1', u'0.24',
//u'0.2879', u'0.81', u'0', u'3', u'13', u'16']
//17379

// 1.1、转换为二元向量
# cache the dataset to speed up subsequent operations
records.cache()
# function to get the categorical feature mapping for a given variable column
def get_mapping(rdd, idx):
    return rdd.map(lambda fields: fields[idx]).distinct().zipWithIndex().collectAsMap()

# we want to extract the feature mappings for columns 2 - 9
# try it out on column 2 first
print "Mapping of first categorical feasture column: %s" % get_mapping(records, 2)
//Mapping of first categorical feasture column: {u'1': 0, u'3': 1, u'2': 2, u'4': 3}

mappings = [get_mapping(records, i) for i in range(2,10)]
cat_len = sum(map(len, mappings))
num_len = len(records.first()[11:15])
total_len = num_len + cat_len

print "Feature vector length for categorical features: %d" % cat_len
print "Feature vector length for numerical features: %d" % num_len
print "Total feature vector length: %d" % total_len
//Feature vector length for categorical features: 57
//Feature vector length for numerical features: 4
//Total feature vector length: 61

// 1.2、创建线性模型特征向量
// # 提取特征
def extract_features(record):
    cat_vec = np.zeros(cat_len)
    i = 0
    step = 0
    for field in record[2:9]:
        m = mappings[i]
        idx = m[field]
        cat_vec[idx + step] = 1
        i = i + 1
        step = step + len(m)
    num_vec = np.array([float(field) for field in record[10:14]])
    return np.concatenate((cat_vec, num_vec))

// # 提取标签
def extract_label(record):
     return float(record[-1])

data = http://www.cnblogs.com/tychyg/p/records.map(lambda r: LabeledPoint(extract_label(r), extract_features(r)))

first_point = data.first()
print"Raw data: " + str(first[2:])
print "Label: " + str(first_point.label)
print "Linear Model feature vector:\n" + str(first_point.features)
print "Linear Model feature vector length: " + str(len(first_point.features))
//Raw data: [u'1', u'0', u'1', u'0', u'0', u'6', u'0', u'1', u'0.24', u'0.2879', u'0.81', u'0', u'3', u'13', u'16']
//Label: 16.0
//Linear Model feature vector:
//[1.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.24,0.2879,0.81,0.0]
//Linear Model feature vector length: 61

// 1.3、创建决策树模型特征向量
def extract_features_dt(record):
    return np.array(map(float, record[2:14]))

data_dt = records.map(lambda r: LabeledPoint(extract_label(r), extract_features_dt(r)))

first_point_dt = data_dt.first()
print "Decision Tree feature vector: " + str(first_point_dt.features)
print "Decision Tree feature vector length: " + str(len(first_point_dt.features))
//Decision Tree feature vector: [1.0,0.0,1.0,0.0,0.0,6.0,0.0,1.0,0.24,0.2879,0.81,0.0]
//Decision Tree feature vector length: 12
```  

##// 2、训练回归模型  
```scala
// 2.1、帮助
help(LinearRegressionWithSGD.train)
help(DecisionTree.trainRegressor)

// 2.2、训练线性模型并测试预测效果
linear_model = LinearRegressionWithSGD.train(data, iterations=10, step=0.1, intercept=False)
true_vs_predicted = data.map(lambda p: (p.label, linear_model.predict(p.features)))
print "Linear Model predictions: " + str(true_vs_predicted.take(5))
//Linear Model predictions: [(16.0, 117.89250386724845), (40.0, 116.2249612319211), (32.0, 116.02369145779234), (13.0, 115.67088016754433), (1.0, 115.56315650834317)]

// 2.3、训练决策树模型并测试预测效果
dt_model = DecisionTree.trainRegressor(data_dt, {})
preds = dt_model.predict(data_dt.map(lambda p: p.features))
actual = data.map(lambda p: p.label)
true_vs_predicted_dt = actual.zip(preds)

print "Decision Tree predictions: " + str(true_vs_predicted_dt.take(5))
print "Decision Tree depth: " + str(dt_model.depth())
print "Decision Tree number of nodes: " + str(dt_model.numNodes())
//Decision Tree predictions: [(16.0, 54.913223140495866), (40.0, 54.913223140495866), (32.0, 53.171052631578945), (13.0, 14.284023668639053), (1.0, 14.284023668639053)]
//Decision Tree depth: 5
//Decision Tree number of nodes: 63
```  

##// 3、评估回归模型性能
```scala
// 评估回归模型的方法：
// 均方误差(MSE, Mean Sequared Error)
// 均方根误差(RMSE, Root Mean Squared Error)
// 平均绝对误差(MAE, Mean Absolute Error)
// R-平方系数(R-squared coefficient)
// 均方根对数误差(RMSLE)

// 3.1、均方误差&均方根误差
def squared_error(actual, pred):
    return (pred - actual)**2

mse = true_vs_predicted.map(lambda (t, p): squared_error(t, p)).mean()
mse_dt = true_vs_predicted_dt.map(lambda (t, p): squared_error(t, p)).mean()

cat_features = dict([(i - 2, len(get_mapping(records, i)) + 1) for i in range(2,10)])

# train the model again
dt_model_2 = DecisionTree.trainRegressor(data_dt, categoricalFeaturesInfo=cat_features)
preds_2 = dt_model_2.predict(data_dt.map(lambda p: p.features))
actual_2 = data.map(lambda p: p.label)
true_vs_predicted_dt_2 = actual_2.zip(preds_2)

# compute performance metrics for decision tree model
mse_dt_2 = true_vs_predicted_dt_2.map(lambda (t, p): squared_error(t, p)).mean()

print "Linear Model - Mean Squared Error: %2.4f" % mse
print "Decision Tree - Mean Squared Error: %2.4f" % mse_dt
print "Categorical feature size mapping %s" % cat_features
print "Decision Tree [Categorical feature]- Mean Squared Error: %2.4f" % mse_dt_2
//Linear Model - Mean Squared Error: 30679.4539
//Decision Tree - Mean Squared Error: 11560.7978
//Decision Tree [Categorical feature]- Mean Squared Error: 7912.5642

//3.2、平均绝对误差
def abs_error(actual, pred):
    return np.abs(pred - actual)

mae = true_vs_predicted.map(lambda (t, p): abs_error(t, p)).mean()
mae_dt = true_vs_predicted_dt.map(lambda (t, p): abs_error(t, p)).mean()
mae_dt_2 = true_vs_predicted_dt_2.map(lambda (t, p): abs_error(t, p)).mean()

print "Linear Model - Mean Absolute Error: %2.4f" % mae
print "Decision Tree - Mean Absolute Error: %2.4f" % mae_dt
print "Decision Tree [Categorical feature]- Mean Absolute Error: %2.4f" % mae_dt_2
//Linear Model - Mean Absolute Error: 130.6429
//Decision Tree - Mean Absolute Error: 71.0969
//Decision Tree [Categorical feature]- Mean Absolute Error: 59.4409

// 3.3、均方根对数误差
def squared_log_error(pred, actual):
    return (np.log(pred + 1) - np.log(actual + 1))**2

rmsle = np.sqrt(true_vs_predicted.map(lambda (t, p): squared_log_error(t, p)).mean())
rmsle_dt = np.sqrt(true_vs_predicted_dt.map(lambda (t, p): squared_log_error(t, p)).mean())
rmsle_dt_2 = np.sqrt(true_vs_predicted_dt_2.map(lambda (t, p): squared_log_error(t, p)).mean())

print "Linear Model - Root Mean Squared Log Error: %2.4f" % rmsle
print "Decision Tree - Root Mean Squared Log Error: %2.4f" % rmsle_dt
print "Decision Tree [Categorical feature]- Root Mean Squared Log Error: %2.4f" % rmsle_dt_2
//Linear Model - Root Mean Squared Log Error: 1.4653
//Decision Tree - Root Mean Squared Log Error: 0.6259
//Decision Tree [Categorical feature]- Root Mean Squared Log Error: 0.6192
```
##// 4、模型参数调优
```scala
// 原变量分布 Distributon of Raw Target
targets = records.map(lambda r: float(r[-1])).collect()

hist(targets, bins=40, color='lightblue', normed=True)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(16, 10)
``` 

<div  align="center"><img src="https://github.com/xieguobin/Spark_2.0.0_cn1/blob/master/figures/chap06_4.0.png" width = "500" height = "250" alt="4.0" align="center" /></div><br>  
//因为**不符合正态分布**，所以**对数变换**（用目标值的对数代替原始数值）或者平方根     

```scala
// 4.1、对数变换
log_targets = records.map(lambda r: np.log(float(r[-1]))).collect()

hist(log_targets, bins=40, color='lightblue', normed=True)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(16, 10)
```

<div  align="center"><img src="https://github.com/xieguobin/Spark_2.0.0_cn1/blob/master/figures/chap06_4.1.png" width = "500" height = "250" alt="4.1" align="center" /></div><br> 

```scala
// 4.2、平方根变换
sqrt_targets = records.map(lambda r: np.sqrt(float(r[-1]))).collect()

hist(sqrt_targets, bins=40, color='lightblue', normed=True)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(16, 10)
```
<div  align="center"><img src="https://github.com/xieguobin/Spark_2.0.0_cn1/blob/master/figures/chap06_4.2.png" width = "500" height = "250" alt="4.2" align="center" /></div><br> 

```scala
// 4.3、对数变换的影响
data_log = data.map(lambda lp: LabeledPoint(np.log(lp.label), lp.features))
model_log = LinearRegressionWithSGD.train(data_log, iterations=10, step=0.1)
true_vs_predicted_log = data_log.map(lambda p: (np.exp(p.label), np.exp(model_log.predict(p.features))))

data_dt_log = data_dt.map(lambda lp: LabeledPoint(np.log(lp.label), lp.features))
dt_model_log = DecisionTree.trainRegressor(data_dt_log, {})
preds_log = dt_model_log.predict(data_dt_log.map(lambda p: p.features))
actual_log = data_dt_log.map(lambda p: p.label)
true_vs_predicted_dt_log = actual_log.zip(preds_log).map(lambda (t, p): (np.exp(t), np.exp(p)))

mse_log = true_vs_predicted_log.map(lambda (t, p): squared_error(t, p)).mean()
mae_log = true_vs_predicted_log.map(lambda (t, p): abs_error(t, p)).mean()
rmsle_log = np.sqrt(true_vs_predicted_log.map(lambda (t, p): squared_log_error(t, p)).mean())

mse_log_dt = true_vs_predicted_dt_log.map(lambda (t, p): squared_error(t, p)).mean()
mae_log_dt = true_vs_predicted_dt_log.map(lambda (t, p): abs_error(t, p)).mean()
rmsle_log_dt = np.sqrt(true_vs_predicted_dt_log.map(lambda (t, p): squared_log_error(t, p)).mean())

print "Mean Squared Error: %2.4f" % mse_log
print "Mean Absolute Error: %2.4f" % mae_log
print "Root Mean Squared Log Error: %2.4f" % rmsle_log
print "Non log-transformed predictions:\n" + str(true_vs_predicted.take(3))
print "Log-transformed predictions:\n" + str(true_vs_predicted_log.take(3))
print "Mean Squared Error: %2.4f" % mse_log_dt
print "Mean Absolute Error: %2.4f" % mae_log_dt
print "Root Mean Squared Log Error: %2.4f" % rmsle_log_dt
print "Non log-transformed predictions:\n" + str(true_vs_predicted_dt.take(3))
print "Log-transformed predictions:\n" + str(true_vs_predicted_dt_log.take(3))
//Mean Squared Error: 50685.5559
//Mean Absolute Error: 155.2955
//Root Mean Squared Log Error: 1.5411
//Non log-transformed predictions:
//[(16.0, 117.89250386724845), (40.0, 116.2249612319211), (32.0, 116.02369145779234)]
//Log-transformed predictions:
//[(15.999999999999998, 28.080291845456237), (40.0, 26.959480191001784), (32.0, 26.654725629458031)]
//Mean Squared Error: 14781.5760
//Mean Absolute Error: 76.4131
//Root Mean Squared Log Error: 0.6406
//Non log-transformed predictions:
//[(16.0, 54.913223140495866), (40.0, 54.913223140495866), (32.0, 53.171052631578945)]
//Log-transformed predictions:
//[(15.999999999999998, 37.530779787154522), (40.0, 37.530779787154522), (32.0, 7.2797070993907287)]

// 4.4、为交叉验证创建训练集和测试集
// create training and testing sets for linear model
data_with_idx = data.zipWithIndex().map(lambda (k, v): (v, k))
test = data_with_idx.sample(False, 0.2, 42)
train = data_with_idx.subtractByKey(test)

train_data = train.map(lambda (idx, p): p)
test_data = test.map(lambda (idx, p) : p)

// create training and testing sets for decision tree
data_with_idx_dt = data_dt.zipWithIndex().map(lambda (k, v): (v, k))
test_dt = data_with_idx_dt.sample(False, 0.2, 42)
train_dt = data_with_idx_dt.subtractByKey(test_dt)

train_data_dt = train_dt.map(lambda (idx, p): p)
test_data_dt = test_dt.map(lambda (idx, p) : p)

train_size = train_data.count()
test_size = test_data.count()
print"Training data size: %d" % train_size
print "Test data size: %d" % test_size
print "Total data size: %d " % num_data
print "Train + Test size : %d" % (train_size + test_size)
//Training data size: 13934
//Test data size: 3445
//Total data size: 17379
//Train + Test size : 17379

// 4.5、线性模型调优
// 4.5.1、评估函数
def evaluate(train, test, iterations, step, regParam, regType, intercept):
    model = LinearRegressionWithSGD.train(train, iterations, step, regParam=regParam, regType=regType, intercept=intercept)
    tp = test.map(lambda p: (p.label, model.predict(p.features)))
    rmsle = np.sqrt(tp.map(lambda (t, p): squared_log_error(t, p)).mean())
    return rmsle
// 4.5.2、迭代次数
params = [1, 5, 10, 20, 50, 100]
metrics = [evaluate(train_data, test_data, param, 0.01, 0.0, 'l2', False) for param in params]
print params
print metrics
[1, 5, 10, 20, 50, 100]
[2.8779465130028199, 2.0390187660391499, 1.7761565324837874, 1.5828778102209105, 1.4382263191764473, 1.4050638054019446]
plot(params, metrics)
fig = matplotlib.pyplot.gcf()
pyplot.xscale('log')
```

//迭代次数与RMSLE关系图

<div  align="center"><img src="https://github.com/xieguobin/Spark_2.0.0_cn1/blob/master/figures/chap06_5.2.png" width = "700" height = "150" alt="4.5.2" align="center" /></div><br> 

```scala
// 4.5.3、步长
params = [0.01, 0.025, 0.05, 0.1, 1.0]
metrics = [evaluate(train_data, test_data, 10, param, 0.0, 'l2', False) for param in params]
print params
print metrics
[0.01, 0.025, 0.05, 0.1, 1.0]
[1.7761565324837874, 1.4379348243997032, 1.4189071944747715, 1.5027293911925559, nan]
plot(params, metrics)
fig = matplotlib.pyplot.gcf()
pyplot.xscale('log')
```

//步长对预测结果的影响
<div  align="center"><img src="https://github.com/xieguobin/Spark_2.0.0_cn1/blob/master/figures/chap06_5.3.png" width = "700" height = "200" alt="4.5.3" align="center" /></div><br> 

```scala
// 4.5.4、L2正则化
params = [0.0, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0]
metrics = [evaluate(train_data, test_data, 10, 0.1, param, 'l2', False) for param in params]
print params
print metrics
plot(params, metrics)
fig = matplotlib.pyplot.gcf()
pyplot.xscale('log')
//[0.0, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0]
//[1.5027293911925559, 1.5020646031965639, 1.4961903335175231, 1.4479313176192781, 1.4113329999970989, 1.5379824584440471, //1.8279564444985839]
```

<div  align="center"><img src="https://github.com/xieguobin/Spark_2.0.0_cn1/blob/master/figures/chap06_5.4.png" width = "500" height = "400" alt="1.1" align="center" /></div><br> 

```scala
// 4.5.5、L1正则化
params = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
metrics = [evaluate(train_data, test_data, 10, 0.1, param, 'l1', False) for param in params]
print params
print metrics
plot(params, metrics)
fig = matplotlib.pyplot.gcf()
pyplot.xscale('log')
//[0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
//[1.5027293911925559, 1.5026938950690176, 1.5023761634555699, 1.499412856617814, 1.4713669769550108, 1.7596682962964318, //4.7551250073268614]
```

<div  align="center"><img src="https://github.com/xieguobin/Spark_2.0.0_cn1/blob/master/figures/chap06_5.5.png" width = "500" height = "400" alt="1.1" align="center" /></div><br> 

```scala
model_l1 = LinearRegressionWithSGD.train(train_data, 10, 0.1, regParam=1.0, regType='l1', intercept=False)
model_l1_10 = LinearRegressionWithSGD.train(train_data, 10, 0.1, regParam=10.0, regType='l1', intercept=False)
model_l1_100 = LinearRegressionWithSGD.train(train_data, 10, 0.1, regParam=100.0, regType='l1', intercept=False)
print "L1 (1.0) number of zero weights: " + str(sum(model_l1.weights.array == 0))
print "L1 (10.0) number of zeros weights: " + str(sum(model_l1_10.weights.array == 0))
print "L1 (100.0) number of zeros weights: " + str(sum(model_l1_100.weights.array == 0))
//L1 (1.0) number of zero weights: 4
//L1 (10.0) number of zeros weights: 33
//L1 (100.0) number of zeros weights: 58

// 4.5.6、截距
// Intercept
params = [False, True]
metrics = [evaluate(train_data, test_data, 10, 0.1, 1.0, 'l2', param) for param in params]
print params
print metrics
bar(params, metrics, color='lightblue')
fig = matplotlib.pyplot.gcf()
//[False, True]
//[1.4479313176192781, 1.4798261513419801]
```

<div  align="center"><img src="https://github.com/xieguobin/Spark_2.0.0_cn1/blob/master/figures/chap06_5.6.png" width = "500" height = "250" alt="1.1" align="center" /></div><br> 

```scala
// 4.6、决策树调优
// 4.6.1、评估函数
def evaluate_dt(train, test, maxDepth, maxBins):
    model = DecisionTree.trainRegressor(train, {}, impurity='variance', maxDepth=maxDepth, maxBins=maxBins)
    preds = model.predict(test.map(lambda p: p.features))
    actual = test.map(lambda p: p.label)
    tp = actual.zip(preds)
    rmsle = np.sqrt(tp.map(lambda (t, p): squared_log_error(t, p)).mean())
    return rmsle
// 4.6.2、树深度
params = [1, 2, 3, 4, 5, 10, 20]
metrics = [evaluate_dt(train_data_dt, test_data_dt, param, 32) for param in params]
print params
print metrics
plot(params, metrics)
fig = matplotlib.pyplot.gcf()
[1, 2, 3, 4, 5, 10, 20]
[1.0280339660196287, 0.92686672078778276, 0.81807794023407532, 0.74060228537329209, 0.63583503599563096, 0.4276659008415965, 0.45481197001756291]
```

<div  align="center"><img src="https://github.com/xieguobin/Spark_2.0.0_cn1/blob/master/figures/chap06_6.2.png" width = "500" height = "350" alt="1.1" align="center" /></div><br> 

```scala
// 4.6.3、最大划分数
params = [2, 4, 8, 16, 32, 64, 100]
metrics = [evaluate_dt(train_data_dt, test_data_dt, 5, param) for param in params]
print params
print metrics
plot(params, metrics)
fig = matplotlib.pyplot.gcf()
//[2, 4, 8, 16, 32, 64, 100]
//[1.3076555360778914, 0.81721457107308615, 0.75651792347650992, 0.63786761731722474, 0.63583503599563096, 0.63583503599563096, //0.63583503599563096]
```

<div  align="center"><img src="https://github.com/xieguobin/Spark_2.0.0_cn1/blob/master/figures/chap06_6.3.png" width = "500" height = "350" alt="1.1" align="center" /></div><br> 

