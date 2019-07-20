package cn.edu.swpu.csc
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.DecisionTreeRegressor
/**
 * @author ${user.name}
 */
object App {
  def main(args: Array[String]): Unit = {
//    val conf = new SparkConf().setAppName("WordCount").setMaster("local")
    val sess = SparkSession.builder().appName("App").master("local[2]").config("spark.testing.memory", "2147480000").getOrCreate();
    val sc = sess.sparkContext;
    import sess.implicits._
    case class gas(features: org.apache.spark.ml.linalg.Vector, label: String)//气体类
    val data1 = sc.textFile("file:///usr/2003.txt")
      .map(_.split(","))
      .filter(p => p(2).toDouble!=0&&p(6).toDouble!=0&&p(7).toDouble!=0&&p(9).toDouble!=0&&p(10).toDouble!=0&&p(12).toDouble!=0)
      .map(p => gas(Vectors.dense(p(2).toDouble,p(6).toDouble,p(7).toDouble,p(9).toDouble,p(12).toDouble),p(10).toString()))
      .toDF()//创建DataFrame
    val data2 = sc.textFile("file:///usr/2004.txt")
      .map(_.split(","))
      .filter(p => p(2).toDouble!=0&&p(6).toDouble!=0&&p(7).toDouble!=0&&p(9).toDouble!=0&&p(10).toDouble!=0&&p(12).toDouble!=0)
      .map(p => gas(Vectors.dense(p(2).toDouble,p(6).toDouble,p(7).toDouble,p(9).toDouble,p(12).toDouble),p(10).toString()))
      .toDF()//创建DataFrame
    val data3 = sc.textFile("file:///usr/2005.txt")
      .map(_.split(","))
      .filter(p => p(2).toDouble!=0&&p(6).toDouble!=0&&p(7).toDouble!=0&&p(9).toDouble!=0&&p(10).toDouble!=0&&p(12).toDouble!=0)
      .map(p => gas(Vectors.dense(p(2).toDouble,p(6).toDouble,p(7).toDouble,p(9).toDouble,p(12).toDouble),p(10).toString()))
      .toDF()//创建DataFrame
    val data = data1.union(data2).union(data3)
    data.createOrReplaceTempView("gas")//创建临时视图
    val df = sess.sql("select * from gas")//查询并返回创建的DataFrame
    df.map(t => t(1)+":"+t(0)).collect().foreach(println)//在屏幕输出DataFrame
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(df)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(df)
    //设置一个labelConverter，把预测的类别重新转化成字符型的
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
    val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2))
    //训练决策树模型
    val dtRegressor = new DecisionTreeRegressor().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
      .setMaxBins(100) // 离散化"连续特征"的最大划分数
      .setMaxDepth(10) // 树的最大深度
      .setMinInfoGain(0.1) //一个节点分裂的最小信息增益，值为[0,1]
      .setMinInstancesPerNode(1000) //每个节点包含的最小样本数
    val pipelineRegressor = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dtRegressor, labelConverter))
    //训练决策树模型
    val modelRegressor = pipelineRegressor.fit(trainingData)
    //进行预测
    val predictionsRegressor = modelRegressor.transform(testData)
    //查看部分预测结果
    predictionsRegressor.select("predictedLabel", "label", "features").show(20)
    //保存模型
    modelRegressor.write.overwrite().save("file:///d:/docment/air/model/")
    //评估决策树分类模型
    val evaluatorRegressor = new RegressionEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("rmse")
    val rmse = evaluatorRegressor.evaluate(predictionsRegressor)
  }
}
