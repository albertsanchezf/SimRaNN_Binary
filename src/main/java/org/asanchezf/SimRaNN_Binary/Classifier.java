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

        //String datasetPath = "/Users/AlbertSanchez/Desktop/Post/DatasetStatistics_v1/1/dataset.csv"; //DS
        String datasetTrain = "/Users/AlbertSanchez/Desktop/Post/DatasetStatistics500/3/dataset1_train.csv"; //DS
        String datasetTest = "/Users/AlbertSanchez/Desktop/Post/DatasetStatistics500/3/dataset1_test.csv"; //DS

        //File earlyStoppingModelFile = new File("resources/earlyStoppingModels/DSNet.zip");

        int numClasses = 1;  //1 class (types of incidents). 0 - No incident | 1 - Incident
        int batchSize = 512;

        //RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        //recordReader.initialize(new FileSplit(new File(datasetPath)));

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

        /*
        TransformProcessRecordReader transformProcessRecordReader = new TransformProcessRecordReader(recordReader,tp);
        DataSetIterator iterator = new RecordReaderDataSetIterator(transformProcessRecordReader,batchSize,labelIndex,numClasses);
        DataSet allData = iterator.next();
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.7);  //Use 70% of data for training

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();
        */

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
                .updater(new Sgd(0.01))
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
        /*
        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(30))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
                .scoreCalculator(new DataSetLossCalculator(testData, true))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver(earlyStoppingModelFile))
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,model,trainData);

        EarlyStoppingResult result = trainer.fit();

        //Print out the results:
        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        MultiLayerNetwork bestModel = result.getBestModel();
        */
        /* END EARLY STOPPING SECTION */

        for(int i=0; i<5000; i++) {
            model.fit(trainingData);
        }

        // Evaluate the model on the test set
        /*
        Evaluation eval = new Evaluation(numClasses);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);
        log.info(eval.stats(true));
        System.out.println(eval.stats());
        */

        Evaluation eval = new Evaluation(numClasses);
        DataSet nextTestData;
        while(testData.hasNext())
        {
            nextTestData = testData.next();
            INDArray output = model.output(nextTestData.getFeatures());
            eval.eval(nextTestData.getLabels(),output);
        }
        log.info(eval.stats(true));
        //System.out.println(eval.stats());

        // Save the trained model
        renameDSNetLastFile(numClasses);
        File locationToSave; //Where to save the network. Note: the file is in .zip format - can be opened externally

        //locationToSave = new File("resources/trainedNN/binaryDSNet_DS.zip"); //DS
        locationToSave = new File("resources/trainedNN/DSNet.zip"); //512

        ModelSerializer.writeModel(model,locationToSave, false);
        System.out.println("Model trained saved in: " + locationToSave.toString());

        // Save the statistics for the DS
        ModelSerializer.addNormalizerToModel(locationToSave,normalizer);
        System.out.println("Normalizer statistics saved in the model");

    }

    public static void renameDSNetLastFile(int numClasses)
    {
        File f;
        String[] s, s1;
        String filename, path = System.getProperty("user.dir") + "/resources/trainedNN/";
        boolean dsFile = false;
        int maxFile = 0;
        filename = "DSNet"; //All

        f = new File(path + filename + ".zip");

        if (f.exists())
        {
            f = new File(path);
            // To ensure any file is overwritten
            String[] files = f.list();
            for (String file : files)
            {
                if(new File(path+file).isFile())
                {
                    s = file.split(filename+"_");
                    if(s.length==2)
                    {
                        if(!s[1].equals(".zip"))
                        {
                            s1 = s[1].split(".zip");
                            if (s1[0].matches("^[0-9]*$"))
                                if(Integer.valueOf(s1[0]) > maxFile) maxFile = Integer.valueOf(s1[0]);
                        }
                    }
                }
            }
            maxFile++;
            new File(path + filename + ".zip").renameTo(new File(path + filename + "_" + String.valueOf(maxFile) + ".zip")); //All

        }

    }

    private static String textfile2String(String filePath)
    {
        StringBuilder contentBuilder = new StringBuilder();
        try (Stream<String> stream = Files.lines( Paths.get(filePath), StandardCharsets.US_ASCII))
        {
            stream.forEach(s -> contentBuilder.append(s).append("\n"));
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
        return contentBuilder.toString();
    }

}
