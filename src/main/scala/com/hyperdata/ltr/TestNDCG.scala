package com.hyperdata.ltr

import breeze.linalg.split
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.{FeatureType, Strategy}
import org.apache.spark.mllib.tree.impurity.Variance
import org.apache.spark.mllib.tree.model.{Predict, Node, DecisionTreeModel, GradientBoostedTreesModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext._

import scala.beans.BeanInfo

@BeanInfo
case class IndexItem(qid: String, rank: Int, x: Vector, y: Int)

case class RankList(qid: String, ys: Array[Int]) {
  override def toString: String = {
    "%s: %s".format(qid, ys.mkString(","))
  }
}

case class LambdaRank(xy: LabeledPoint, w: Double)

case class NDCG(k: Int) {
  def score(ranklist: RankList): Double = {
    var sum = 0d
    var count = 0
    for(i <- 0 until ranklist.ys.length) {
      if(ranklist.ys(i) != 0) {
        sum += NDCG.discount(i)
        count += 1
      }
    }
    if(count > 0) sum / NDCG.idealDCG(count)
    else 0
  }
}
object NDCG {
  val maxSize = 1000

  val discount = {
    val LOG2 = math.log(2)
    val arr = Array.ofDim[Double](maxSize)
    for(i <- 0 until arr.length) arr(i) = LOG2 / math.log(2 + i)
    arr
  }

  val idealDCG = {
    val arr = Array.ofDim[Double](maxSize + 1)
    arr(0) = 0
    for(i <- 1 until arr.length) {
      arr(i) = arr(i-1) + discount(i-1)
    }
    arr
  }
}

trait RankModel extends Serializable {
  def predict(x: Vector): Double
  def ensemble(model: DecisionTreeModel, weight: Double): RankModel
}

case class TreeModel(model: DecisionTreeModel, weight: Double)
class MART(trees: Seq[TreeModel]) extends RankModel {
  override def predict(x: Vector): Double = {
    trees.foldLeft(0d) { case (sum, TreeModel(m, w)) =>
      sum + w * m.predict(x)
    }
  }

  override def ensemble(model: DecisionTreeModel, weight: Double): RankModel = {
    new MART(trees ++ Seq(TreeModel(model, weight)))
  }
}

case object ZeroRankModel extends RankModel {
  override def predict(x: Vector): Double = 0

  override def ensemble(model: DecisionTreeModel, weight: Double): RankModel = {
    new MART(Seq(TreeModel(model, weight)))
  }
}

object NodeUtil {
  def predictNode(node: Node, features: Vector): Node = {
    if (node.isLeaf) {
      node
    } else{
      if (node.split.get.featureType == FeatureType.Continuous) {
        if (features(node.split.get.feature) <= node.split.get.threshold) {
          predictNode(node.leftNode.get, features)
        } else {
          predictNode(node.rightNode.get, features)
        }
      } else {
        if (node.split.get.categories.contains(features(node.split.get.feature))) {
          predictNode(node.leftNode.get, features)
        } else {
          predictNode(node.rightNode.get, features)
        }
      }
    }
  }

  def updateOutput(node: Node, node2output: Map[Int, Double]): Unit = {
    if(node.isLeaf) node.predict = new Predict(node2output(node.id), node.predict.prob)
    else {
      updateOutput(node.leftNode.get, node2output)
      updateOutput(node.rightNode.get, node2output)
    }
  }
}

object TestNDCG {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ndcg").setMaster("local[3]")
    val sc = new SparkContext(conf)

    val treeStrategy = new Strategy(Regression, Variance, 1, 0)
    val gamma = 0.1 // learning rate
    val numIterations = 2

    val dataset = loadQueryDoc(sc, "dataset/example.dat").groupBy(_.qid).mapValues(_.toArray)
    dataset.persist(StorageLevel.MEMORY_AND_DISK_2)
    println("total: " + ndcg(dataset, NDCG(20)))

    var model: RankModel = ZeroRankModel
    for(ite <- 0 until numIterations) {
      val lrData = buildLambda(dataset, model).persist(StorageLevel.MEMORY_AND_DISK_2)
      val mi = new DecisionTree(treeStrategy).run(lrData.map(_.xy))
      val node2output = lrData.map { lr =>
        val node = NodeUtil.predictNode(mi.topNode, lr.xy.features)
//        println(node.id + " => y:" + lr.xy.label + ", w:" + lr.w)
        (node.id, (lr.xy.label, lr.w))
      }.reduceByKey { case ((y1, w1), (y2, w2)) => (y1 + y2, w1 + w2)}
        .map { case (nodeid, (sy, sw)) => (nodeid, sy / sw)}
        .collectAsMap.toMap
      println("node2output: " + node2output)
      NodeUtil.updateOutput(mi.topNode, node2output)
      println("model" + ite + ": " + mi.toDebugString)
      model = model.ensemble(mi, gamma)
      lrData.unpersist(true)
      println("ndcg " + ite + " => " + ndcg(dataset, model, NDCG(20)))
    }
  }

  def buildLambda(input: RDD[(String, Array[IndexItem])], model: RankModel): RDD[LambdaRank] = {
    input.flatMap { case (qid, items) =>
      val scoreys = items.toArray.map { item =>
//        println(item.qid + "/" + item.rank + " => " + model.predict(item.x))
        (item.x, model.predict(item.x), item.y)
      }
      val count = scoreys.map(_._3).sum
      val idealGCD = NDCG.idealDCG(count)
      val pseudoResponses = Array.ofDim[Double](scoreys.length)
      val weights = Array.ofDim[Double](scoreys.length)
      for(i <- 0 until pseudoResponses.length) {
        val (_, si, yi) = scoreys(i)
        for(j <- 0 until pseudoResponses.length if i != j) {
          val (_, sj, yj) = scoreys(j)
          if(yi > yj) {
            val deltaNDCG = math.abs((yi - yj) * NDCG.discount(i) + (yj - yi) * NDCG.discount(j)) / idealGCD
            val rho = 1.0 / (1 + math.exp(si - sj))
            val lambda = rho * deltaNDCG
            pseudoResponses(i) += lambda
            pseudoResponses(j) -= lambda
            val delta = rho * (1.0 - rho) * deltaNDCG
            weights(i) += delta
            weights(j) += delta
          }
        }
      }
      for(i <- 0 until scoreys.length) yield {
        val (x, s, y) = scoreys(i)
        LambdaRank(LabeledPoint(pseudoResponses(i), x), weights(i))
      }
    }
  }

  def ndcg(input: RDD[(String, Array[IndexItem])], ndcg: NDCG): Double = {
    val (total, count) = input
      .map { case (qid, items) =>
        (qid, RankList(qid, items.sortBy(_.rank).map(_.y)))
      }
      .map { case (qid, ranklist) =>
//        println(qid + " => " + ndcg.score(ranklist))
        (ndcg.score(ranklist), 1)
      }
      .reduce { case ((t1, c1), (t2, c2)) => (t1 + t2, c1 + c2)}
    total / count
  }

  def ndcg(input: RDD[(String, Array[IndexItem])], model: RankModel, ndcg: NDCG): Double = {
    val (total, count) = input
      .map { case (qid, items) =>
      (qid, RankList(qid, items.sortBy(item => -model.predict(item.x)).map(_.y)))
    }
      .map { case (qid, ranklist) =>
      (ndcg.score(ranklist), 1)
    }
      .reduce { case ((t1, c1), (t2, c2)) => (t1 + t2, c1 + c2)}
    total / count
  }

  def loadQueryDoc(sc: SparkContext, path: String): RDD[IndexItem] = {
    loadQueryDoc(sc, path, -1)
  }

  def loadQueryDoc(
      sc: SparkContext,
      path: String,
      numFeatures: Int): RDD[IndexItem] = {
    loadQueryDoc(sc, path, numFeatures, sc.defaultMinPartitions)
  }

  def loadQueryDoc(
      sc: SparkContext,
      path: String,
      numFeatures: Int,
      minPartitions: Int): RDD[IndexItem] = {
    val parsed = sc.textFile(path, minPartitions)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map { line =>
        val items = line.split(' ')
        val y = items(0).toInt
        val qid = items(1).substring(4)
        val rank = items(2).toInt
        val (indices, values) = items.slice(3, items.length).filter(_.nonEmpty).map { item =>
          val indexAndValue = item.split(':')
          val index = indexAndValue(0).toInt - 1 // Convert 1-based indices to 0-based.
          val value = indexAndValue(1).toDouble
          (index, value)
        }.unzip
        (y, qid, rank, indices.toArray, values.toArray)
      }

    // Determine number of features.
    val d = if (numFeatures > 0) {
      numFeatures
    } else {
      parsed.persist(StorageLevel.MEMORY_ONLY)
      parsed.map { case (y, qid, rank, indices, values) =>
        indices.lastOption.getOrElse(0)
      }.reduce(math.max) + 1
    }

    parsed.map { case (y, qid, rank, indices, values) =>
      IndexItem(qid, rank, Vectors.sparse(d, indices, values), y)
    }
  }
}