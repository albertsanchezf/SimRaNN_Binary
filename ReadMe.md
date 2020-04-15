# SimRaNN_Binary

This program defines a neural network model and train it using a dataset CSV file generate with AdaptDatabase program (https://github.com/albertsanchezf/AdaptDatabase). This neural network is configurated to be a binary classifier (*numClasses = 1*).

**Basic Configurations**
-*datasetPath*: The location of the Dataset CSV file. 

**Advanced Configurations**
-*Number of epochs*: defined in *nEpochs*
-*batchSize*: The batchSize used for the training phase.
-*Train and test dataset split ratio*: defined in trainPercentageAdd
-*Model configuration*: it is done under *conf = new NeuralNetConfiguration.Builder()*
  -*Layer configuration*: .layer(new DenseLayer.Builder() <- Type of layer
                            .nIn(2000)                    <- Number of inputs (If it is an intermediate layer nIn = nOut of the previous layer
                            .nOut(2000)                   <- Number of outputs (If it is an intermediate layer nOut = nIn of the next layer
                            .activation(Activation.RELU)  <- Activation function for the layer
                            .build())
  -*Learning rate*: defined in *.updater(new Sgd(0.1))*
  -*Type of Optimization algorhithm*: defined in .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
  


                            
