# dl4j_coursera

Run `mvn clean install` to build jar with dependencies.

Full example of importing a serialized keras 2 model, instantiating a distributed DL4J
model with Spark and evaluating model performance on data read from a CSV file:

```
$SPARK_HOME/bin/spark-submit \
     --class skymind.dsx.KerasImportCSVSparkRunner \
     --files iris.txt,model.h5 \
     target/dl4j-keras-runner-0.1-jar-with-dependencies.jar \
       -useSparkLocal true \
       -batchSizePerWorker 15 \
       -indexLabel 4 \
       -numClasses 3 \
       -modelFileName model.h5 \
       -dataFileName iris.txt
```

For a keras 1 model, try `model_keras_1.h5` instead. To run this on a cluster you
need to specify the Spark master `$MASTER` and remove `useSparkLocal`, i.e. run:

```
$SPARK_HOME/bin/spark-submit \
     --class skymind.dsx.KerasImportCSVSparkRunner \
     --files iris.txt,model.h5 \
     --master $MASTER \
     target/dl4j-keras-runner-0.1-jar-with-dependencies.jar \
       -batchSizePerWorker 15 \
       -indexLabel 4 \
       -numClasses 3 \
       -modelFileName model.h5 \
       -dataFileName iris.txt
```
