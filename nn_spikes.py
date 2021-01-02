import scipy.special
import numpy as np
import matplotlib.pyplot as plt
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
        final_inputs = self.forwardProp(inputs)

        # Calculate output error with selected error function, default == 'difference'
        output_errors = self.errorFunc(targets, self.finalActivations)

        # Backwards propagate
        self.backProp(inputs, output_errors, final_inputs, targets)

        return output_errors

    def forwardProp(self, inputs):
        # Calculate weighted input into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        # Calculates activation of hidden neurons
        self.hiddenActivations = self.activationFunc(hidden_inputs)

        # Calculates the weighted input into final layer
        final_inputs = np.dot(self.who, self.hiddenActivations)

        # Calculates activation of final (output) neurons
        self.finalActivations = self.activationFunc(final_inputs)

        return final_inputs

    def backProp(self, inputs_array, output_errors, final_inputs, targets):
        # Hidden layer errors are the output errors, split by the weights, recombined at the hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # # Derivatives for output layer weights updates
        # activationDeriv = self.activationDeriv(final_inputs)
        # errorDeriv = self.errorDeriv(self.finalActivations, targets)
        # inputsDeriv = np.transpose(self.hiddenActivations)
        #
        # # Combine all the derivatives for the links between the hidden and output layers
        # self.who += self.lr * np.dot(errorDeriv, activationDeriv, inputsDeriv)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * self.finalActivations * (1.0 - self.finalActivations)),
                                     np.transpose(self.hiddenActivations))

        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * self.hiddenActivations * (1.0 - self.hiddenActivations)),
                                     np.transpose(inputs_array))

    def query(self, inputs):
        """
        Function used to query the neural network: pass one set of inputs and retrieve one set of predictions from the outputs.
        :param inputs: List of input values
        :return: Returns outputs from final layer of nodes
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
               plotCurves=True,
               patienceInitial=4):
    """

    :param data_training:
    :param data_validation:
    :param spikeIndexes_training:
    :param spikeIndexes_validation:
    :param nn:
    :param batchSize:
    :param epochs:
    :param plotCurves:
    :param patienceInitial:
    :return:
    """

    assert isinstance(data_training, pd.DataFrame) and isinstance(data_validation, pd.DataFrame)

    trainingCurve = []
    validationCurve = []
    lrInitial = getattr(nn, 'lr')
    patience = patienceInitial

    # Allocate batch sizes for batch training. If no batch size specified, take full dataset as single batch (batch gradient descent)
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

                # Retrieve the inputs (spike waveforms) and target vectors (spike classes) to the network
                inputs, targets, _ = getInputsAndTargets(data_training.loc[index, 'waveform'], nn.output_nodes, batch.loc[index-10:index+5, 'knownClass'])

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

        # Plot learning curves showing training vs validation performance, useful during development and debugging
        if plotCurves and epoch > 0:
            plotLearningCurve(epoch+1, trainingCurve, validationCurve)
        else:
            pass

        # Check for early stopping opportunity by evaluating if the increase in performance has stagnated
        earlyStopCheck, patience = checkEarlyStop(validationCurve, epoch, patience, patienceInitial)
        if earlyStopCheck:
            break
        else:
            # Step-decay learning rate for given number of epochs
            nn.decayLR(epoch, lrInitial)

    return nn, trainingCurve, validationCurve

def test(data, spikeIndexes, nn):

    # Ensure data is of type pandas dataframe
    assert isinstance(data, pd.DataFrame)

    # Create an empty string to accumulate the count of correct predictions
    scorecard = []

    # Train the network for each row in the batch
    for index in spikeIndexes:
        # Retrieve the inputs (spike waveforms) and target vectors (spike classes) to the network
        inputs, _, label = getInputsAndTargets(data.loc[index, 'waveform'], nn.output_nodes, data.loc[index-10:index+5, 'knownClass'])

        # Query the network to identify the predicted output
        prediction = nn.query(inputs)

        # Add to scorecard
        if prediction == label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    scorecard = np.asarray(scorecard)
    successRate = (scorecard.sum() / scorecard.size) * 100

    return successRate

def checkEarlyStop(performances, epoch, patience, patienceInitial, window=5, tolerance=0.05):
    """
    Function evaluates the variance of the network over the trailing window
    :param performances:
    :param epoch:
    :param patience:
    :param patienceInitial:
    :param window:
    :param tolerance:
    :return:
    """
    # TODO: Implement gradient check of performances to stop early if validation performance starts to decrease.
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

def plotLearningCurve(epoch, trainingCurve, validationCurve):
    try:
        assert not len(trainingCurve) == 0 and not len(validationCurve) == 0
    except AssertionError:
        raise("Training curve data: {}\nValidation curve data: {}\n".format(str(trainingCurve), str(validationCurve)))

    plt.plot(range(epoch), trainingCurve, label='trainingCurve')
    plt.plot(range(epoch), validationCurve, label='validationCurve')
    plt.ylabel('performance')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()

def getInputsAndTargets(waveform, output_nodes, knownClasses):
    """
    Function simply converts row of pixel data (plus first item is label) from MNIST .csv file into np array
    :param knownClasses: Window of class values around detected spike in labelled data used to assign correct label to spike
    :param waveform: list of comma separated pixel values
    :param output_nodes: number of output nodes for network
    :return: returns numpy array of (28*28=) 784 input values and 10 target output values (for digits 0-9)
    """
    assert isinstance(knownClasses, pd.Series)

    # Retrieve non-zero spike labels from list of known spike labels in window either side of detected spike
    possibleLabels = knownClasses[4:-2][knownClasses != 0].values

    # If no non-zero spike labels are detected, extend window range and try again
    if len(possibleLabels) == 0:
        possibleLabels = knownClasses[knownClasses != 0].values

        # If still no non-zero spike labels are detected, raise an error because the spike detected could be a false positive
        if len(possibleLabels) == 0:
            raise Warning("No labels detectable for detected spike with index {}. label window: {}".format(knownClasses.index, knownClasses))

    # If more than one non-zero spike labels are detected within the window, assert they are all the same, raise error if not
    if len(possibleLabels) > 1:
        try:
            assert (possibleLabels[0] == possibleLabels).all()
        except AssertionError:
            # More than two knownClass labels for a single spike at: 54412, 87433, 165493, 232479, 299250, 312319, 339791, 472193, 980407
            # raise Warning("All spikes labelled with more than one class in the given training dataset should have been removed.")
            #TODO: remove those spikes from training set
            pass

    # Retrieve the target label and account for non-zero count
    label = possibleLabels[0] - 1

    # Retrieve waveform points as inputs array
    inputs = waveform

    # Create the target output values (all 0.01, except the desired label which is 0.99)
    targets = np.zeros(output_nodes) + 0.01

    # pixelValues[0] is the target label for this record
    targets[label] = 0.99

    inputs = np.array(inputs.tolist(), ndmin=2).T
    targets = np.array(targets.tolist(), ndmin=2).T

    return inputs, targets, label