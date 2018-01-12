package skymind.dsx;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;


/**
 * Generic Keras model import runner. Takes a CSV file in a format that CSVRecordReader understands
 * and a Keras model file in HDF5.
 *
 * Example usage:
 *  $SPARK_HOME/bin/spark-submit \
 *    --class skymind.dsx.KerasImportCSVSparkRunner \
 *    --master $MASTER \
 *    --files iris.txt irisModel.h5 \
 *    dl4j.jar
 *      -batchSizePerWorker 150 \
 *      -indexLabel 4 \
 *      -numClasses 3 \
 *      -modelFileName model.h5 \
 *      -dataFileName iris.txt
 *
 *
 * @author Max Pumperla
 */
public class KerasImportCSVSparkRunner {

    private static final Logger log = LoggerFactory.getLogger(KerasImportCSVSparkRunner.class);

    @Parameter(names = "-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private boolean useSparkLocal = false;

    @Parameter(names = "-loadData", description = "Whether to load data or not ", arity = 1)
    private boolean loadData = true;

    @Parameter(names = "-train", description = "Whether to train model or not ", arity = 1)
    private boolean train = false;

    @Parameter(names = "-batchSizePerWorker", description = "Number of examples to fit each worker with")
    private int batchSizePerWorker = 10;

    @Parameter(names = "-indexLabel", description = "Index of column that has labels")
    private int indexLabel = -1;

    @Parameter(names = "-modelFileName", description = "Name of the keras model file")
    private String modelFileName = "model.h5";

    @Parameter(names = "-dataFileName", description = "Name of the CSV file")
    private String dataFileName = "data.csv";

    @Parameter(names = "-numClasses", description = "Number of output classes")
    private int numClasses = -1;

    @Parameter(names = "-epochs", description = "Number of epochs to train")
    private int epochs = 10;

    public static void main(String[] args) throws Exception {
        new KerasImportCSVSparkRunner().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {

        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            jcmdr.usage();
            try { Thread.sleep(500); } catch (Exception e2) { }
            throw e;
        }

        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("DL4J Keras model import runner");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        // Load Keras model
        MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights(modelFileName, train);

        // Build training master
        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(40)
                .averagingFrequency(5)
                .workerPrefetchNumBatches(2)
                .batchSizePerWorker(batchSizePerWorker)
                .build();

        // Init Spark net
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, network, tm);

        if (loadData) {
            // Assume no header and comma separation
            int numLinesToSkip = 0;
            char delimiter = ',';
            RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
            recordReader.initialize(new FileSplit(new File(dataFileName)));

            DataSetIterator iterator = new RecordReaderDataSetIterator(
                    recordReader, batchSizePerWorker, indexLabel, numClasses);

            List<DataSet> dataList = new ArrayList<>();
            while (iterator.hasNext()) {
                dataList.add(iterator.next());
            }
            JavaRDD<DataSet> data = sc.parallelize(dataList);

            if (train) {
                for (int i = 0; i < epochs; i++) {
                    sparkNet.fit(data);
                    log.info("Completed Epoch {}", i);
                }
            }

            // Evaluation
            Evaluation evaluation = sparkNet.evaluate(data);
            log.info(evaluation.stats());
            log.info("***** Example Complete *****");
        } else {
            log.info("***** Model import complete");
        }
    }
}
