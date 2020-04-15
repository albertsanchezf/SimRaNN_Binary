package org.asanchezf.SimRaNN_Binary;

import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

public class Classifier {

    static Logger log = Logger.getLogger(Classifier.class);

    public static void main(String[] args) throws  Exception {

        String configFilename = System.getProperty("user.dir")
                + File.separator + "log4j.properties";
        PropertyConfigurator.configure(configFilename);

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        char delimiter = ',';

        String datasetTrain = "/Users/AlbertSanchez/Desktop/Post/Tests/DatasetSplit/dataset_train.csv"; //DS
        String datasetTest = "/Users/AlbertSanchez/Desktop/Post/Tests/DatasetSplit/dataset_test.csv"; //DS

        File earlyStoppingModelFile = new File(obtainFilename(System.getProperty("user.dir") + "/resources/","EarlyStoppingDSNet"));
        LocalFileModelSaver lfms = new LocalFileModelSaver(earlyStoppingModelFile);

        int numClasses = 1;  //1 class (types of incidents). 0 - No incident | 1 - Incident
        int batchSize = 512;

        // Load Train DS
        RecordReader recordReader1 = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader1.initialize(new FileSplit(new File(datasetTrain)));
        // Load Test DS
        RecordReader recordReader2 = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader2.initialize(new FileSplit(new File(datasetTest)));

        // Build a Input Schema
        Schema inputDataSchema = new Schema.Builder()
                .addColumnsFloat("speed","mean_acc_x","mean_acc_y","mean_acc_z","std_acc_x","std_acc_y","std_acc_z")
                .addColumnDouble("sma")
                .addColumnFloat("mean_svm")
                .addColumnsDouble("entropyX","entropyY","entropyZ")
                .addColumnsInteger("bike_type","phone_location","incident_type")
                .build();

        // Made the necessary transformations
        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .integerToOneHot("bike_type",0,8)
                .integerToOneHot("phone_location",0,6)
                .build();

        // Get output schema
        Schema outputSchema = tp.getFinalSchema();

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = outputSchema.getColumnNames().size() - 1;     //15 values in each row of the dataset.csv; CSV: 14 input features followed by an integer label (class) index. Labels are the 15th value (index 14) in each row

        TransformProcessRecordReader transformProcessRecordReader1 = new TransformProcessRecordReader(recordReader1,tp);
        TransformProcessRecordReader transformProcessRecordReader2 = new TransformProcessRecordReader(recordReader2,tp);
        DataSetIterator trainingData = new RecordReaderDataSetIterator(transformProcessRecordReader1,batchSize,labelIndex,numClasses);
        DataSetIterator testData = new RecordReaderDataSetIterator(transformProcessRecordReader2,batchSize,labelIndex,numClasses);

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data

        while(trainingData.hasNext())
            normalizer.transform(trainingData.next());     //Apply normalization to the training data
        while (testData.hasNext())
            normalizer.transform(testData.next());         //Apply normalization to the test data. This is using statistics calculated from the *training* set

        trainingData.reset();
        testData.reset();

        log.info("Build model....");
        MultiLayerConfiguration conf;
        conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l2(1e-4)
                .updater(new Sgd(0.1))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(labelIndex)
                        .nOut(2000)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(2000)
                        .nOut(2000)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(2000)
                        .nOut(2000)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(2000)
                        .nOut(2000)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(2000)
                        .nOut(2000)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(2000)
                        .nOut(numClasses)
                        .activation(Activation.SIGMOID)
                        .lossFunction(LossFunctions.LossFunction.XENT)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        /* EARLY STOPPING */

        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(5000))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(60, TimeUnit.MINUTES))
                .scoreCalculator(new DataSetLossCalculator(testData, true))
                .evaluateEveryNEpochs(1)
                .modelSaver(lfms)
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,model,trainingData);

        EarlyStoppingResult result = trainer.fit();

        //Print out the results:
        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        MultiLayerNetwork bestModel = (MultiLayerNetwork) result.getBestModel();

        /* END EARLY STOPPING SECTION */

        testData.reset();

        // Evaluate the final model on the test set
        Evaluation eval1 = new Evaluation(numClasses);
        DataSet nextTestData;
        while(testData.hasNext())
        {
            nextTestData = testData.next();
            INDArray output = model.output(nextTestData.getFeatures());
            eval1.eval(nextTestData.getLabels(),output);
        }
        log.info(eval1.stats(true));
        //System.out.println(eval.stats());

        testData.reset();
        // Evaluate the model obtained from early stopping on the test set
        Evaluation eval2 = new Evaluation(numClasses);
        while(testData.hasNext())
        {
            nextTestData = testData.next();
            INDArray output = bestModel.output(nextTestData.getFeatures());
            eval2.eval(nextTestData.getLabels(),output);
        }
        log.info(eval2.stats(true));

        // Save the trained model
        File locationToSave; //Where to save the network. Note: the file is in .zip format - can be opened externally

        locationToSave = new File(obtainFilename(System.getProperty("user.dir") + "/resources/","DSNet"));

        ModelSerializer.writeModel(bestModel,locationToSave, false);
        System.out.println("Model trained saved in: " + locationToSave.toString());

        // Save the statistics for the DS
        ModelSerializer.addNormalizerToModel(locationToSave,normalizer);
        System.out.println("Normalizer statistics saved in the model");

    }

    public static String obtainFilename(String path, String name)
    {
        String[] s, s1;
        String filename = "";
        boolean dsFile = false;
        int maxFile = 0;

        // To ensure any file is overwrited
        File f = new File(path);
        String[] files = f.list();
        for(int i=0; i < files.length; i++)
        {
            if(new File(path + files[i]).isFile())
            {
                s = files[i].split(name);
                if(s.length==2)
                {
                    if(s[1].equals(".zip"))
                        dsFile = true;
                    else
                    {
                        s1 = s[1].split(".zip");
                        if (s1[0].matches("^[0-9]*$"))
                            if(Integer.valueOf(s1[0]) > maxFile)
                                maxFile = Integer.valueOf(s1[0]);
                    }
                }
            }
        }

        maxFile++;

        filename = path + name + maxFile + ".zip";

        return filename;
    }




}

