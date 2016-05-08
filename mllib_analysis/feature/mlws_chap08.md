#// mlws_chap08(Spark机器学习,第八章)

##// 0、封装包和代码环境  
// 准备环境  
export SPARK_HOME=/Users/erichan/Garden/spark-1.5.1-bin-hadoop2.6  
cd $SPARK_HOME  
bin/spark-shell --name my_mlib --packages org.jblas:jblas:1.2.4-SNAPSHOT --driver-memory 4G --executor-memory 4G --driver-cores 2  

```scala
package mllib_book.mlws.chap07_clustering

object chap07 extends App{  
  val conf = new SparkConf().setAppName("Spark_chap07").setMaster("local")  
  val sc = new SparkContext(conf)  
```  
##// 1、特征抽取   
```scala
// 1.1、载入脸部数据
val PATH = "/Users/erichan/sourcecode/book/Spark机器学习"
val path = PATH+"/lfw/*"
val rdd = sc.wholeTextFiles(path)
val files = rdd.map { case (fileName, content) => fileName.replace("file:", "") }
println(files.count)
//1054

// 1.2、可视化脸部数据(python)
// ipython -pylab
PATH = "/Users/erichan/sourcecode/book/Spark机器学习"
path = PATH+"/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
ae = imread(path)
imshow(ae)
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;              // Aaron_Eckhart_0001

tmpPath = "/tmp/aeGray.jpg"
aeGary = imread(tmpPath)
imshow(aeGary, cmap=plt.cm.gray)
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;              // Aaron_Eckhart_0001_gray

// 1.3、提取脸部图片作为向量
// 1.3.1、载入图片
import java.awt.image.BufferedImage
def loadImageFromFile(path: String): BufferedImage = {
    import javax.imageio.ImageIO
    import java.io.File
    ImageIO.read(new File(path))
}

val aePath = PATH+"/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
val aeImage = loadImageFromFile(aePath)
//1.3.2、转换灰度、改变尺寸
def processImage(image: BufferedImage, width: Int, height: Int): BufferedImage = {
    val bwImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
    val g = bwImage.getGraphics()
    g.drawImage(image, 0, 0, width, height, null)
    g.dispose()
    bwImage
}

val grayImage = processImage(aeImage, 100, 100)

import javax.imageio.ImageIO
import java.io.File
ImageIO.write(grayImage, "jpg", new File("/tmp/aeGray.jpg"))
                                                   //aeGray

// 1.3.3 提取特征向量
def getPixelsFromImage(image: BufferedImage): Array[Double] = {
    val width = image.getWidth
    val height = image.getHeight
    val pixels = Array.ofDim[Double](width * height)
    image.getData.getPixels(0, 0, width, height, pixels)
    // pixels.map(p => p / 255.0)       // optionally scale to [0, 1] domain
}

// put all the functions together
def extractPixels(path: String, width: Int, height: Int): Array[Double] = {
    val raw = loadImageFromFile(path)
    val processed = processImage(raw, width, height)
    getPixelsFromImage(processed)
}

val pixels = files.map(f => extractPixels(f, 50, 50))
println(pixels.take(10).map(_.take(10).mkString("", ",", ", ...")).mkString("\n"))
//1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0, ...
//247.0,173.0,159.0,144.0,139.0,155.0,32.0,7.0,4.0,5.0, ...
//253.0,254.0,253.0,253.0,253.0,253.0,253.0,253.0,253.0,253.0, ...
//242.0,242.0,246.0,239.0,238.0,239.0,225.0,165.0,140.0,167.0, ...
//47.0,221.0,205.0,46.0,41.0,154.0,127.0,214.0,232.0,232.0, ...
//0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, ...
//75.0,76.0,72.0,72.0,72.0,74.0,71.0,78.0,54.0,26.0, ...
//25.0,27.0,24.0,22.0,26.0,27.0,19.0,16.0,22.0,25.0, ...
//240.0,240.0,240.0,240.0,240.0,240.0,240.0,240.0,240.0,240.0, ...
//0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, ...
import org.apache.spark.mllib.linalg.Vectors
val vectors = pixels.map(p => Vectors.dense(p))
vectors.setName("image-vectors")
vectors.cache
//1

// 1.4、正则化
import org.apache.spark.mllib.feature.StandardScaler
val scaler = new StandardScaler(withMean = true, withStd = false).fit(vectors)

val scaledVectors = vectors.map(v => scaler.transform(v))
```


##// 2、训练降维模型
```scala
// 2.1、前k个主成分
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
val matrix = new RowMatrix(scaledVectors)
val K = 10
val pc = matrix.computePrincipalComponents(K)
val rows = pc.numRows
val cols = pc.numCols
println(rows, cols)
//(2500,10)

// 2.2 可视化特征脸
import breeze.linalg.DenseMatrix
val pcBreeze = new DenseMatrix(rows, cols, pc.toArray)
import breeze.linalg.csvwrite
import java.io.File
csvwrite(new File("/tmp/pc.csv"), pcBreeze)
pc = np.loadtxt("/tmp/pc.csv", delimiter=",")
print(pc.shape)
def plot_gallery(images, h, w, n_row=2, n_col=5):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[:, i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title("Eigenface %d" % (i + 1), size=12)
        plt.xticks(())
        plt.yticks(())

plot_gallery(pc, 50, 50)
//8_3
```
// 3、使用降维模型
```scala
// 3.1、PCA投影（图像矩阵x主成分矩阵）
val projected = matrix.multiply(pc)
println(projected.numRows, projected.numCols)
println(projected.rows.take(5).mkString("\n"))    

// 3.2、PCA与SVD
val svd = matrix.computeSVD(10, computeU = true)
println(s"U dimension: (${svd.U.numRows}, ${svd.U.numCols})")
println(s"S dimension: (${svd.s.size}, )")
println(s"V dimension: (${svd.V.numRows}, ${svd.V.numCols})")
U dimension: (1054, 10)
S dimension: (10, )
V dimension: (2500, 10)
def approxEqual(array1: Array[Double], array2: Array[Double], tolerance: Double = 1e-6): Boolean = {
    // note we ignore sign of the principal component / singular vector elements
    val bools = array1.zip(array2).map { case (v1, v2) => if (math.abs(math.abs(v1) - math.abs(v2)) > 1e-6) false else true }
    bools.fold(true)(_ & _)
}
println(approxEqual(Array(1.0, 2.0, 3.0), Array(1.0, 2.0, 3.0)))
println(approxEqual(Array(1.0, 2.0, 3.0), Array(3.0, 2.0, 1.0)))
println(approxEqual(svd.V.toArray, pc.toArray))
//true
//false
//true

// compare projections
val breezeS = breeze.linalg.DenseVector(svd.s.toArray)
val projectedSVD = svd.U.rows.map { v =>
    val breezeV = breeze.linalg.DenseVector(v.toArray)
    val multV = breezeV :* breezeS
    Vectors.dense(multV.data)
}
projected.rows.zip(projectedSVD).map { case (v1, v2) => approxEqual(v1.toArray, v2.toArray) }.filter(b => true).count
```

// 4、评价降维模型
```scala
// 4.1、评估SVD的k值
val sValues = (1 to 5).map { i => matrix.computeSVD(i, computeU = false).s }
val svd300 = matrix.computeSVD(300, computeU = false)
val sMatrix = new DenseMatrix(1, 300, svd300.s.toArray)
csvwrite(new File("/tmp/s.csv"), sMatrix)
s = np.loadtxt("/tmp/s.csv", delimiter=",")
print(s.shape)
plot(s)
//8_4

plot(cumsum(s))
plt.yscale('log')
//8_5
```
