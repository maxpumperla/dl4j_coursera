import org.deeplearning4j.nn.modelimport.keras.KerasModelImport
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork


object SparkApp {
  
    def main( args: Array[String] ) {

        val modelPath = args(0)
        val network: MultiLayerNetwork = KerasModelImport.importKerasSequentialModelAndWeights(modelPath);

    }
}
