// import org.apache.spark.SparkConf
// import org.apache.spark.SparkContext
import org.deeplearning4j.spark.api.TrainingMaster
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

import java.io.File
import java.io.IOException
import java.util


object SparkApp {
    def main( args: Array[String] ) {

      // val conf = new SparkConf().setAppName("Keras import")
      // val sc = new SparkContext(conf)

        val modelPath = args(0)
        val network: MultiLayerNetwork = KerasModelImport.importKerasSequentialModelAndWeights(modelPath);

       val tm = new ParameterAveragingTrainingMaster.Builder(40)
         .averagingFrequency(5)
         .workerPrefetchNumBatches(2)
         .batchSizePerWorker(40)
         .build

       val sparkNet = new SparkDl4jMultiLayer(sc, network, tm)

    }
}
