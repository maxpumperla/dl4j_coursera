//import org.apache.spark.SparkConf
//import org.apache.spark.api.java.JavaSparkContext
//import org.deeplearning4j.spark.api.TrainingMaster
//import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
//import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

import java.io.File
import java.io.IOException
import java.util


object SparkApp {
    def main( args: Array[String] ) {

//        val sparkConf = new SparkConf()
//        sparkConf.setMaster("local[*]")
//        sparkConf.setAppName("Keras import and Spark training")
//        val sc = new JavaSparkContext(sparkConf)

        val modelPath = args(0)
        val network: MultiLayerNetwork = KerasModelImport.importKerasSequentialModelAndWeights(modelPath);


//        val tm = new ParameterAveragingTrainingMaster.Builder(40)
//          .averagingFrequency(5)
//          .workerPrefetchNumBatches(2)
//          .batchSizePerWorker(40)
//          .build
//
//        val sparkNet = new SparkDl4jMultiLayer(sc, network, tm)

//          ... train, eval etc.

    }
}
