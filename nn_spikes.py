import scipy.special
import numpy as np
import math
import pandas as pd
import utilities

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, lr, activation_function='sigmoid', error_function='difference'):
        """
        Function called on creating a new instance of the class. Number of nodes at each section defined as well as learning rate.
        :param input_nodes: integer number of input nodes
        :param hidden_nodes: integer number of hidden nodes
        :param output_nodes: integer number of output nodes
        :param lr: float for learning rate
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = lr

        # Weight initialisation: with small, random floats for weights going from input to hidden and hidden to output
        self.wih = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.who = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        # Set the activation function as the logistic sigmoid
        self.activationFunc, self.activationDeriv = self.getActivationFuncs(activation_function)
        self.errorFunc, self.errorDeriv = self.getErrorFuncs(error_function)
        self.inputsDeriv = None

        # Declare attributes to persist neuron outputs
        self.hiddenActivations = None
        self.finalActivations = None

    def fit(self, inputs, targets):
        """
        Function calculates new weights for NN using back propagation of errors
        :param inputs: input vectors as list
        :param targets:
        :return: no return
        """

        # Forward propagate inputs through network
        self.forwardProp(inputs)

        # Calculate output error with selected error function, default == 'difference'
        output_errors = self.errorFunc(targets, self.finalActivations)

        # Backwards propagate
        self.backProp(inputs, output_errors)

        return output_errors

    def forwardProp(self, inputs):
        """
        Forward propagate the activations of each neuron from the input layer to the output layer
        :param inputs: array of input values
        :return: returns the weighted inputs to the final layer
        """
        # Calculate weighted input into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        # Calculates activation of hidden neurons
        self.hiddenActivations = self.activationFunc(hidden_inputs)

        # Calculates the weighted input into final layer
        final_inputs = np.dot(self.who, self.hiddenActivations)

        # Calculates activation of final (output) neurons
        self.finalActivations = self.activationFunc(final_inputs)

        return final_inputs

    def backProp(self, inputs_array, output_errors):
        """
        Back propagate the neuron activations from the error in the output layer to update the weights in the network
        :param inputs_array: array of input values
        :param output_errors: error in each neuron of the output layer
        :return: returns nothing as weights are updated and stored in the object's state
        """
        # Hidden layer errors are the output errors, split by the weights, recombined at the hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * self.finalActivations * (1.0 - self.finalActivations)),
                                     np.transpose(self.hiddenActivations))

        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * self.hiddenActivations * (1.0 - self.hiddenActivations)),
                                     np.transpose(inputs_array))

    def query(self, inputs):
        """
        Function used to query the neural network: pass one set of inputs and retrieve one prediction from the outputs.
        :param inputs: array-like of input values
        :return: returns max output from final layer of nodes
        """
        # Convert the inputs list into a 2D array and use to calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        # Calculate output from the hidden layer
        hidden_outputs = self.activationFunc(hidden_inputs)

        # Calculate signals into final layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # Calculate the outputs from the final layer, used to indicate the class of the spike waveform
        final_outputs = self.activationFunc(final_inputs)

        return np.argmax(final_outputs)

    def decayLR(self, epoch, lrInitial):
        """
        Function decays the learning rate by a drop rate based on the number of epochs since the last time of decay. This is useful to refine the
        updates made to weight during each training cycle and to increase the likelihood of finding the best possible local minima.
        :param epoch: current epoch
        :param lrInitial: initial learning rate set at start of training process
        :return: No return needed as learning rate is stored in the attribute of the NeuralNetwork object.
        """

        # Drop rate to decay the learning rate by
        drop = 0.6
        # Number of epochs to pass before learning rate is decayed
        epochsDrop = 6.0

        # Set new learning rate using an exponential decay of learning rate
        self.lr = lrInitial * math.pow(drop, math.floor((1 + epoch) / epochsDrop))

        if epoch % epochsDrop == 0 and epoch > 0:
            print("LR has decayed after {} cycles. New lr = {}".format(epochsDrop, self.lr))

    @staticmethod
    def getActivationFuncs(activation_function):
        """
        Function used to generalise the network configuration to use a configurable activation function during training.
        :param activation_function: string indicating which activation function should be used
        :return: returns a lambda function which can be applied during training
        """
        if activation_function == 'sigmoid':
            func = lambda x: scipy.special.expit(x)
            funcPrime = lambda x: scipy.special.expit(x) * (1 - scipy.special.expit(x))
            return func, funcPrime

        elif activation_function == 'relu':
            func = lambda x: np.maximum(x, 0)
            funcPrime = []
            return func, funcPrime

        elif activation_function == 'tanh':
            func = lambda x: (2/(1 + np.exp(-2*x))) -1
            funcPrime = []
            return func, funcPrime
        else:
            raise ValueError("Please make sure you specify an activation function from the list.")

    @staticmethod
    def getErrorFuncs(error_function):
        """
        Function allows for the generalisation of error functions used to evaluate the network outputs
        :param error_function: string indicating which error function to use
        :return: returns lambda function that can be applied during training of the network
        """
        if error_function == 'difference':
            func = lambda x1, x2: np.subtract(x1, x2)
            funcPrime = None
            return func, funcPrime

        elif error_function == 'difference-squared':
            func = lambda x1, x2: np.subtract(x1, x2) #should be np.square(np.subtract(x1, x2)) / 2
            funcPrime = lambda x1, x2: np.subtract(x1, x2)
            return func, funcPrime

        elif error_function == 'mse':
            func = lambda x1, x2: np.square(np.subtract(x1, x2)).mean() # taking mean should be equivalent to /2
            funcPrime = None
            return func, funcPrime
        else:
            raise ValueError("Please make sure you specify an error function from the list.")

def batchTrain(data_training,
               data_validation,
               spikeIndexes_training,
               spikeIndexes_validation,
               nn,
               batchSize=None,
               epochs=20,
               patienceInitial=4):
    """
    Function trains the supplied neural network on the training data and generates a performance based on both training and
    validation datasets to produce learning curves for diagnostics. Adaptive learning rate is implemented, which decays after
    a number of cycles to more precisely locate the local minimum.
    :param data_training: training dataset
    :param data_validation: validation dataset
    :param spikeIndexes_training: list-like of spike locations within training dataset
    :param spikeIndexes_validation: list-like of spike locations within validation dataset
    :param nn: neural network to fit to data
    :param batchSize: not implemented, but could be used to implement mini-batch stochastic gradient descent to improve training time
    :param epochs: number of times to train the network for
    :param patienceInitial: initial patience used to specify for how many epochs the network will be trained after validation performance
    has decreased below the minimum allowable level of variance. useful to help trigger a further decay of LR and improve max performance achievable
    :return: returns the neural network and the training and validation curves used for diagnostics
    """

    # Assert the datasets are of the desired data type: pandas dataframes
    assert isinstance(data_training, pd.DataFrame) and isinstance(data_validation, pd.DataFrame)

    # Create empty lists to store training and validation curve data
    trainingCurve = []
    validationCurve = []

    # Store the initial learning rate in the neural network attribute and create a new patience variable that will be changed
    lrInitial = getattr(nn, 'lr')
    patience = patienceInitial

    # Allocate batch sizes for batch training. If no batch size specified, take full dataset as single batch (stochastic gradient descent)
    batchSize = utilities.getBatchSize(batchSize, data_training.shape[0])

    # Train for n training cycles, where n = number of epochs
    for epoch in range(epochs):
        print("epoch: ", epoch)

        # Set the initial batch to start at the beginning of the dataset. The batch ends at the index determined by the desired batch size
        batchStart = 0
        batchEnd = batchSize - 1

        # Train the network using chunks of the full training dataset, where each chunk is of size batchSize
        while batchEnd <= data_training.shape[0] - 1:

            # Create a batch (subset) of the data
            batch = data_training.iloc[batchStart:batchEnd]

            # Train the network for each row in the batch
            for index in spikeIndexes_training:

                # Retrieve the inputs (spike waveforms) and target vectors (spike classes) to the network for the given spike
                inputs, targets = getInputsAndTargets(data_training.loc[index, 'waveform'], nn.output_nodes, int(batch.loc[index, 'assignedKnownClass']))

                # Complete one cycle of forward propagation, error calculation and back propagation to update the network weights
                nn.fit(inputs, targets)

            # Update the batch subset of data
            batchStart = batchEnd + 1
            batchEnd += batchSize

        # Collect performance on training and validation datasets for each epoch
        successTraining = test(data_training, spikeIndexes_training, nn)
        successValidation = test(data_validation, spikeIndexes_validation, nn)

        trainingCurve.append(successTraining)
        validationCurve.append(successValidation)

        # Check for early stopping opportunity by evaluating if the increase in performance has stagnated
        earlyStopCheck, patience = checkEarlyStop(validationCurve, epoch, patience, patienceInitial)
        if earlyStopCheck:
            break
        else:
            # Step-decay learning rate for given number of epochs
            nn.decayLR(epoch, lrInitial)

    return nn, trainingCurve, validationCurve

def test(data, spikeIndexes, nn):
    """
    Function evaluates the performance of the network classification, and returns a performance score.
    :param data: data to classify
    :param spikeIndexes: spike indexes to locate spikes in the data
    :param nn: neural network used for classification
    :return: performance of classification as precision of classification
    """

    # Ensure data is of type pandas dataframe
    assert isinstance(data, pd.DataFrame)

    # Create an empty string to accumulate the count of correct predictions
    scorecard = []

    # Iterate over each spike and query the trained neural network
    for index in spikeIndexes:

        # Identify the known class of the spike, which has been assigned based on the labelled data
        knownClass = data.loc[index, 'assignedKnownClass']

        # Retrieve only the inputs (spike waveforms) to the network
        inputs, _ = getInputsAndTargets(data.loc[index, 'waveform'], nn.output_nodes, knownClass)

        # Query the network to identify the predicted output for teh given spike waveform
        prediction = nn.query(inputs)

        # Add to scorecard if classification is correct
        if prediction == knownClass:
            scorecard.append(1)
        else:
            scorecard.append(0)

    # Calculate accuracy of network to correctly classify detected spikes (precision)
    scorecard = np.asarray(scorecard)
    successRate = (scorecard.sum() / scorecard.size) * 100

    return successRate

def checkEarlyStop(performances, epoch, patience, patienceInitial, window=5, tolerance=0.05):
    """
    Function evaluates the variance of the network over the trailing window of performance
    :param performances: performance to detect variance in
    :param epoch: current epoch
    :param patience: current value of patience - this decreases between consecutive epochs below the performance variance threshold
    :param patienceInitial: initial patience to reset to when variance rises above threshold
    :param window: window of performance values to evaluate variance over
    :param tolerance: variance tolerance
    :return: returns boolean indicating if training should be stopped and the correct patience value
    """
    # Check for opportunity for early stopping
    if len(performances) >= window:
        mean = sum(performances[-window:]) / window
        variance = sum((i - mean) ** 2 for i in performances[-window:]) / window

        if variance < tolerance and (performances[-1] - performances[-2]) < tolerance:
            # Drop patience when below variance tolerance
            patience -= 1
            if patience == 0:
                print("Stopped early at epoch number {} with final variance of {}.\n".format(epoch, variance))
                return True, patience
            else:
                print("We could stop early at epoch number {} with variance of {} but we can be patient for {} more training cycles.\n".format(epoch, variance, patience))
                return False, patience
        else:
            # Reset patience when variance increases above threshold.
            patience = patienceInitial
            print("Variance of {} in last {} training cycles. Continuing training and resetting patience to {}.".format(variance, window, patienceInitial))
            return False, patience
    elif 0 < len(performances) < window:
        return False, patience
    else:
        raise Exception("Something wrong happened in the early stop checker.")

def getInputsAndTargets(waveform, output_nodes, knownClass):
    """
    Function takes in spike waveform extract and generates input array to network. Also generates target array for output network
    based on the known class of the spike.
    :param knownClass: window of class values around detected spike in labelled data used to assign correct label to spike
    :param waveform: spike waveform signal values - mV extracellular recordings from electrode at 25kHz frequency
    :param output_nodes: number of output nodes for network
    :return: input array to network and target array for output nodes
    """
    # Assert function parameters are of expected values and data types
    assert knownClass in [0,1,2,3], "Known class should be in [0,1,2,3]. INFO:\n Waveform[:5]: {}, \nknownClass{}".format(waveform[:5], knownClass)
    assert isinstance(waveform, pd.Series), "Waveform extract should have been stored as a pandas Series object."

    # Create the target output values (all 0.01, except the desired label which is 0.99)
    targets = np.zeros(output_nodes) + 0.01

    # Store near-one value for target node, correlating to known spike class
    targets[knownClass] = 0.99

    # Cast and reshape inputs and targets array into standard format suitable for network
    inputs = np.array(waveform.tolist(), ndmin=2).T
    targets = np.array(targets.tolist(), ndmin=2).T

    return inputs, targets