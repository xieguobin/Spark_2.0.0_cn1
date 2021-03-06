#// mlws_chap03(Spark机器学习,第三章)

##// 0、封装包和代码环境
// anaconda  
nano ~/.zshrc  
export PATH=$PATH:/anaconda/bin  
source ~/.zshrc  
echo $HOME  
echo $PATH  
// ipython  
conda update conda && conda update ipython ipython-notebook ipython-qtconsole  
conda install scipy  
//PYTHONPATH  
export SPARK_HOME=/Users/erichan/garden/spark-1.5.1-bin-hadoop2.6  
export PYTHONPATH=${SPARK_HOME}/python/:${SPARK_HOME}/python/lib/py4j-0.8.2.1-src.zip  
//运行环境  
cd $SPARK_HOME  
IPYTHON=1 IPYTHON_OPTS="--pylab" ./bin/pyspark  

```scala
package mllib_book.mlws.chap03_dataprepare

object chap03 extends App{  
  val conf = new SparkConf().setAppName("Spark_chap03").setMaster("local")  
  val sc = new SparkContext(conf)  
```  
  
##// 1、数据导入  
```scala
PATH = "/Users/erichan/sourcecode/book/Spark机器学习"
user_data = http://www.cnblogs.com/tychyg/p/sc.textFile("%s/ml-100k/u.user" % PATH)
user_fields = user_data.map(lambda line: line.split("|"))
movie_data = http://www.cnblogs.com/tychyg/p/sc.textFile("%s/ml-100k/u.item" % PATH)
movie_fields = movie_data.map(lambda lines: lines.split("|"))
rating_data_raw = sc.textFile("%s/ml-100k/u.data" % PATH)
rating_data = http://www.cnblogs.com/tychyg/p/rating_data_raw.map(lambda line: line.split("\t"))
num_movies = movie_data.count()

print num_movies
//1682

user_data.first()
//u'1|24|M|technician|85711'

movie_data.first()
//u'1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0'

rating_data_raw.first()
//u'196\t242\t3\t881250949'
```  

##// 2、探索数据
```scala
// 2.1、按列统计
user_fields = user_data.map(lambda line: line.split("|"))
num_users = user_fields.map(lambda fields: fields[0]).count()
num_genders = user_fields.map(lambda fields: fields[2]).distinct().count()
num_occupations = user_fields.map(lambda fields: fields[3]).distinct().count()
num_zipcodes = user_fields.map(lambda fields: fields[4]).distinct().count()

ratings = rating_data.map(lambda fields: int(fields[2]))
num_ratings = ratings.count()
max_rating = ratings.reduce(lambda x, y: max(x, y))
min_rating = ratings.reduce(lambda x, y: min(x, y))
mean_rating = ratings.reduce(lambda x, y: x + y) / float(num_ratings)
median_rating = np.median(ratings.collect())
ratings_per_user = num_ratings / num_users
ratings_per_movie = num_ratings / num_movies
print "Users: %d, genders: %d, occupations: %d, ZIP codes: %d" % (num_users, num_genders, num_occupations, num_zipcodes)
//Users: 943, genders: 2, occupations: 21, ZIP codes: 795
print "Min rating: %d" % min_rating
//Min rating: 1
print "Max rating: %d" % max_rating
//Max rating: 5
print "Average rating: %2.2f" % mean_rating
//Average rating: 3.53
print "Median rating: %d" % median_rating
//Median rating: 4
print "Average # of ratings per user: %2.2f" % ratings_per_user
//Average # of ratings per user: 106.00
print "Average # of ratings per movie: %2.2f" % ratings_per_movie
//Average # of ratings per movie: 59.00
ratings.stats()
//(count: 100000, mean: 3.52986, stdev: 1.12566797076, max: 5, min: 1)

// 2.2、使用matplotlib的hist函数绘制直方图
ages = user_fields.map(lambda x: int(x[1])).collect()
hist(ages, bins=20, color='lightblue', normed=True)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(16, 10)

count_by_rating = ratings.countByValue()
x_axis = np.array(count_by_rating.keys())
y_axis = np.array([float(c) for c in count_by_rating.values()])
# we normalize the y-axis here to percentages
y_axis_normed = y_axis / y_axis.sum()

pos = np.arange(len(x_axis))
width = 1.0

ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(x_axis)

plt.bar(pos, y_axis_normed, width, color='lightblue')
plt.xticks(rotation=30)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(16, 10)

count_by_occupation = user_fields.map(lambda fields: (fields[3], 1)).reduceByKey(lambda x, y: x + y).collect()
x_axis1 = np.array([c[0] for c in count_by_occupation])
y_axis1 = np.array([c[1] for c in count_by_occupation])
x_axis = x_axis1[np.argsort(y_axis1)]
y_axis = y_axis1[np.argsort(y_axis1)]

pos = np.arange(len(x_axis))
width = 1.0

ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(x_axis)

plt.bar(pos, y_axis, width, color='lightblue')
plt.xticks(rotation=30)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(16, 10)

//2.3、使用countByValue函数统计
count_by_occupation2 = user_fields.map(lambda fields: fields[3]).countByValue()
print "Map-reduce approach:"
print dict(count_by_occupation2)
//{u'administrator': 79, u'retired': 14, u'lawyer': 12, u'healthcare': 16, u'marketing': 26, u'executive': 32, u'scientist': 31, u'student': 196, u'technician': 27, u'librarian': 51, u'programmer': 66, u'salesman': 12, u'homemaker': 7, u'engineer': 67, u'none': 9, u'doctor': 7, u'writer': 45, u'entertainment': 18, u'other': 105, u'educator': 95, u'artist': 28}

print ""
print "countByValue approach:"
print dict(count_by_occupation)
//{u'administrator': 79, u'writer': 45, u'retired': 14, u'lawyer': 12, u'doctor': 7, u'marketing': 26, u'executive': 32, u'none': 9, u'entertainment': 18, u'healthcare': 16, u'scientist': 31, u'student': 196, u'educator': 95, u'technician': 27, u'librarian': 51, u'programmer': 66, u'artist': 28, u'salesman': 12, u'other': 105, u'homemaker': 7, u'engineer': 67}

//2.4、使用filter转换
def convert_year(x):
    try:
        return int(x[-4:])
    except:
        return 1900

years = movie_fields.map(lambda fields: fields[2]).map(lambda x: convert_year(x))
years_filtered = years.filter(lambda x: x != 1900)
movie_ages = years_filtered.map(lambda yr: 1998-yr).countByValue()
values = movie_ages.values()
bins = movie_ages.keys()
hist(values, bins=bins, color='lightblue', normed=True)
//(array([ 0. , 0.07575758, 0.09090909, 0.09090909, 0.18181818,
//0.18181818, 0.04545455, 0.07575758, 0.07575758, 0.03030303,
//0. , 0.01515152, 0.01515152, 0.03030303, 0. ,
//0.03030303, 0. , 0. , 0. , 0. ,
//0. , 0. , 0.01515152, 0. , 0.01515152,
//0. , 0. , 0. , 0. , 0. ,
//0. , 0. , 0. , 0. , 0. ,
//0. , 0. , 0.01515152, 0. , 0. ,
//0. , 0. , 0. , 0. , 0. ,
//0. , 0. , 0. , 0. , 0. ,
//0. , 0. , 0. , 0. , 0. ,
//0. , 0. , 0. , 0. , 0. ,
//0. , 0. , 0. , 0. , 0. ,
//0.01515152, 0. , 0. , 0. , 0. ]),
//array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
//17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
//34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
//51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
//68, 72, 76]),
//)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(16,10)

//2.5、使用groupByKey分组
// to compute the distribution of ratings per user, we first group the ratings by user id
user_ratings_grouped = rating_data.map(lambda fields: (int(fields[0]), int(fields[2]))).groupByKey()
// then, for each key (user id), we find the size of the set of ratings, which gives us the # ratings for that user
user_ratings_byuser = user_ratings_grouped.map(lambda (k, v): (k, len(v)))
user_ratings_byuser.take(5)
[(2, 62), (4, 24), (6, 211), (8, 59), (10, 184)]
user_ratings_byuser_local = user_ratings_byuser.map(lambda (k, v): v).collect()
hist(user_ratings_byuser_local, bins=200, color='lightblue', normed=True)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(16,10)
```  

##// 3、处理转换
```scala
// 3.1、填充缺失
years_pre_processed = movie_fields.map(lambda fields: fields[2]).map(lambda x: convert_year(x)).filter(lambda yr: yr != 1900).collect()
years_pre_processed_array = np.array(years_pre_processed)
# first we compute the mean and median year of release, without the 'bad' data point
mean_year = np.mean(years_pre_processed_array[years_pre_processed_array!=1900])
median_year = np.median(years_pre_processed_array[years_pre_processed_array!=1900])
idx_bad_data = http://www.cnblogs.com/tychyg/p/np.where(years_pre_processed_array==1900)[0]
years_pre_processed_array[idx_bad_data] = median_year
print"Mean year of release: %d" % mean_year
//Mean year of release: 1989
print "Median year of release: %d" % median_year
//Median year of release: 1995
print "Index of '1900' after assigning median: %s" % np.where(years_pre_processed_array == 1900)[0]
//Index of '1900' after assigning median: []
```
##// 4、提取特征
```scala
// 4.1、类别特征（norminal变量/ordinal变量）
all_occupations = user_fields.map(lambda fields: fields[3]).distinct().collect()
all_occupations.sort()
// create a new dictionary to hold the occupations, and assign the "1-of-k" indexes
idx = 0
all_occupations_dict = {}
for o in all_occupations:
    all_occupations_dict[o] = idx
    idx +=1

// try a few examples to see what "1-of-k" encoding is assigned
print "Encoding of 'doctor': %d" % all_occupations_dict['doctor']
print "Encoding of 'programmer': %d" % all_occupations_dict['programmer']
//Encoding of 'doctor': 2
//Encoding of 'programmer': 14

//numpy的zeros函数
K = len(all_occupations_dict)
binary_x = np.zeros(K)
k_programmer = all_occupations_dict['programmer']
binary_x[k_programmer] = 1
print "Binary feature vector: %s" % binary_x
print "Length of binary vector: %d" % K
//Binary feature vector: [ 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.
//0. 0.] Length of binary vector: 21

// 4.2、派生特征
// 时间戳转换为类别特征
def extract_datetime(ts):
    import datetime
    return datetime.datetime.fromtimestamp(ts)

def assign_tod(hr):
    times_of_day = {
        'morning' : range(7, 12),
        'lunch' : range(12, 15),
        'afternoon' : range(15, 18),
        'evening' : range(18, 23),
        'night' : {23,24,0,1,2,3,4,5,6,7}
    }
    for k, v in times_of_day.iteritems():
        if hr in v:
            return k

timestamps = rating_data.map(lambda fields: int(fields[3]))
hour_of_day = timestamps.map(lambda ts: extract_datetime(ts).hour)
// now apply the "time of day" function to the "hour of day" RDD
time_of_day = hour_of_day.map(lambda hr: assign_tod(hr))
timestamps.take(5)
//[881250949, 891717742, 878887116, 880606923, 886397596]
hour_of_day.take(5)
//[23, 3, 15, 13, 13]
time_of_day.take(5)
//['night', 'night', 'afternoon', 'lunch', 'lunch']

// 4.3、文本特征
def extract_title(raw):
    import re
    grps = re.search("\((\w+)\)", raw)
    if grps:
        return raw[:grps.start()].strip()
    else:
        return raw

raw_titles = movie_fields.map(lambda fields: fields[1])
for raw_title in raw_titles.take(5):
    print extract_title(raw_title)

//Toy Story
//GoldenEye
//Four Rooms
//Get Shorty
//Copycat

movie_titles = raw_titles.map(lambda m: extract_title(m))
// next we tokenize the titles into terms. We'll use simple whitespace tokenization
title_terms = movie_titles.map(lambda t: t.split(" "))
print title_terms.take(5)
//[[u'Toy', u'Story'], [u'GoldenEye'], [u'Four', u'Rooms'], [u'Get', u'Shorty'], [u'Copycat']]

flatMap
all_terms = title_terms.flatMap(lambda x: x).distinct().collect()
// create a new dictionary to hold the terms, and assign the "1-of-k" indexes
idx = 0
all_terms_dict = {}
for term in all_terms:
    all_terms_dict[term] = idx
    idx +=1

num_terms = len(all_terms_dict)
print "Total number of terms: %d" % num_terms
//Total number of terms: 2645
print "Index of term 'Dead': %d" % all_terms_dict['Dead']
//Index of term 'Dead': 147
print "Index of term 'Rooms': %d" % all_terms_dict['Rooms']
//Index of term 'Rooms': 1963
zipWithIndex
all_terms_dict2 = title_terms.flatMap(lambda x: x).distinct().zipWithIndex().collectAsMap()
print "Index of term 'Dead': %d" % all_terms_dict2['Dead']
print "Index of term 'Rooms': %d" % all_terms_dict2['Rooms']
//Index of term 'Dead': 147
//Index of term 'Rooms': 1963
//创建稀疏向量/广播变量
scipy depends $PYTHONPATH
def create_vector(terms, term_dict):
    from scipy import sparse as sp
    x = sp.csc_matrix((1, num_terms))
    for t in terms:
        if t in term_dict:
            idx = term_dict[t]
            x[0, idx] = 1
    return x

all_terms_bcast = sc.broadcast(all_terms_dict)
term_vectors = title_terms.map(lambda terms: create_vector(terms, all_terms_bcast.value))
term_vectors.take(5)
//[<1x2645 sparse matrix of type ''
//with 1 stored elements in Compressed Sparse Column format>,
//<1x2645 sparse matrix of type ''
//with 1 stored elements in Compressed Sparse Column format>,
//<1x2645 sparse matrix of type ''
//with 1 stored elements in Compressed Sparse Column format>,
//<1x2645 sparse matrix of type ''
//with 1 stored elements in Compressed Sparse Column format>,
//<1x2645 sparse matrix of type ''
//with 1 stored elements in Compressed Sparse Column format>]

// 4.4、正则化特征
np.random.seed(42)
x = np.random.randn(10)
norm_x_2 = np.linalg.norm(x)
normalized_x = x / norm_x_2
print "x:\n%s" % x
print "2-Norm of x: %2.4f" % norm_x_2
print "Normalized x:\n%s" % normalized_x
print "2-Norm of normalized_x: %2.4f" % np.linalg.norm(normalized_x)
//x:
//[ 0.49671415 -0.1382643 0.64768854 1.52302986 -0.23415337 -0.23413696
//1.57921282 0.76743473 -0.46947439 0.54256004]
//2-Norm of x: 2.5908
//Normalized x:
//[ 0.19172213 -0.05336737 0.24999534 0.58786029 -0.09037871 -0.09037237
//0.60954584 0.29621508 -0.1812081 0.20941776]
//2-Norm of normalized_x: 1.0000

from pyspark.mllib.feature import Normalizer
normalizer = Normalizer()
vector = sc.parallelize([x])
normalized_x_mllib = normalizer.transform(vector).first().toArray()

print "x:\n%s" % x
print "2-Norm of x: %2.4f" % norm_x_2
print "Normalized x MLlib:\n%s" % normalized_x_mllib
print "2-Norm of normalized_x_mllib: %2.4f" % np.linalg.norm(normalized_x_mllib)
//x:
//[ 0.49671415 -0.1382643 0.64768854 1.52302986 -0.23415337 -0.23413696
//1.57921282 0.76743473 -0.46947439 0.54256004]
//2-Norm of x: 2.5908
//Normalized x MLlib:
//[ 0.19172213 -0.05336737 0.24999534 0.58786029 -0.09037871 -0.09037237
//0.60954584 0.29621508 -0.1812.20941776]
//2-Norm of normalized_x_mllib: 1.0000
```
