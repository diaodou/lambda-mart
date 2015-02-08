package com.hyperdata.ltr

import java.io.{FileWriter, PrintWriter}

import collection.mutable
import scala.io.Source

object IndexDataset {
  def main (args: Array[String]) {
    val q2idx = new mutable.HashMap[String, Int]().withDefaultValue(0)
    val pw = new PrintWriter(new FileWriter("dataset/ltr.dat"))
    Source.fromFile("dataset/mslr-train1.txt").getLines().foreach { line =>
      val idx1 = line.indexOf("qid:")
      if(idx1 != -1) {
        val idx2 = line.indexOf(' ', idx1)
        if(idx2 != -1) {
          val qid = line.substring(idx1, idx2)
          val idx = q2idx(qid)
          if(idx == 0 && q2idx.size < 100 || idx > 0) {
            q2idx.update(qid, idx + 1)
            pw.println(line.substring(0, idx2) + " " + idx + line.substring(idx2))
          }
        }
      }
    }
    pw.close
  }
}