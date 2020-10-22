import scipy.special
import numpy as np
import matplotlib.pyplot as plt

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
        self.activationFunc = self.getActivationFunc(activation_function)
        self.activationDeriv = self.getActivationDeriv(activation_function)
        self.errorFunc = self.getErrorFunction(error_function)

        # Declare attributes to persist neuron outputs
        self.hidden_outputs = None
        self.final_outputs = None

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
        output_errors = self.errorFunc(targets, self.final_outputs)

        # Backwards propagate
        self.backProp(inputs, output_errors)

        return output_errors

    def forwardProp(self, inputs_array):
        # Calculate weighted input into hidden layer
        hidden_inputs = np.dot(self.wih, inputs_array)

        # Calculates activation of hidden neurons
        self.hidden_outputs = self.activationFunc(hidden_inputs)

        # Calculates the weighted input into final layer
        final_inputs = np.dot(self.who, self.hidden_outputs)

        # Calculates activation of final (output) neurons
        self.final_outputs = self.activationFunc(final_inputs)

    def backProp(self, inputs_array, output_errors):
        # Hidden layer errors are the output errors, split by the weights, recombined at the hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * self.final_outputs * (1.0 - self.final_outputs)),
                                     np.transpose(self.hidden_outputs))

        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * self.hidden_outputs * (1.0 - self.hidden_outputs)),
                                     np.transpose(inputs_array))

    def query(self, inputs):
        """
        Function to query the neural network.
        :param inputs: List of inputs
        :return: Returns outputs from final layer of nodes
        """
        try:
            assert isinstance(inputs, list)
        except AssertionError:
            raise TypeError("NN query inputs must be type list.")

        # Convert the inputs list into a 2D array and use to calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, np.array(inputs, ndmin=2).T)

        # Calculate output from the hidden layer
        hidden_outputs = self.activationFunc(hidden_inputs)

        # Calculate signals into final layer
        final_inputs = np.dot(self.who, hidden_outputs)

        final_outputs = self.activationFunc(final_inputs)

        return final_outputs

    @staticmethod
    def getActivationFunc(activation_function):
        if activation_function == 'sigmoid':
            return lambda x: scipy.special.expit(x)
        elif activation_function == 'relu':
            return lambda x: np.maximum(x, 0)
        elif activation_function == 'tanh':
            return lambda x: (2/(1 + np.exp(-2*x))) -1
        else:
            raise ValueError("Please make sure you specify an activation function from the list.")

    @staticmethod
    def getErrorFunction(error_function):
        if error_function == 'difference':
            return lambda x1, x2: np.subtract(x1, x2)
        elif error_function == 'mse':
            return lambda x1, x2: np.square(np.subtract(x1, x2)).mean() # should be equivalent to /2
        else:
            raise ValueError("Please make sure you specify an error function from the list.")

    @staticmethod
    def getActivationDeriv(activation_function):
        if activation_function == 'sigmoid':
            return lambda x: scipy.special.expit(x)
        elif activation_function == 'relu':
            return lambda x: np.maximum(x, 0)
        elif activation_function == 'tanh':
            return lambda x: (2/(1 + np.exp(-2*x))) -1
        else:
            raise ValueError("Please make sure you specify an activation function from the list.")

def batchTrain(data_training,
               data_validation,
               nn,
               batchSize=None,
               epochs=20,
               plotCurves=True):

    trainingCurve = []
    validationCurve = []

    # Allocate batch sizes for batch training. If no batch size specified, take full dataset as single batch (batch gradient descent)
    if not batchSize:
        batchSize = len(data_training)
    else:
        # Only allow batch sizes that don't discard any data (can improve later but low priority)
        assert len(data_training) % batchSize == 0
        pass

    # Train for n training cycles, where n = number of epochs
    for epoch in range(epochs):
        # Set initial batch
        batchStart = 0
        batchEnd = batchSize - 1

        # Train using chunks of full training dataset, where each chunk = batch size
        while batchEnd <= len(data_training) - 1:

            data_batch = data_training[batchStart:batchEnd]

            # Train network for each row in batch,
            for row in data_batch:
                inputs, targets = imageToTrainingInputs(row, nn.output_nodes)
                nn.fit(inputs, targets)

            batchStart = batchEnd + 1
            batchEnd += batchSize

        # Collect performance on training and validation datasets for each epoch
        trainingCurve.append(test(data_training, nn))
        validationCurve.append(test(data_validation, nn))

        # Plot learning curves showing training vs validation performance
        if plotCurves and epoch > 0:
            plotLearningCurve(epoch+1, trainingCurve, validationCurve)
        else:
            pass

        # Check for early stopping opportunity
        if checkStagnation(validationCurve, epoch):
            break
        else:
            pass

    return nn, trainingCurve, validationCurve

def test(data, nn):
    scorecard = []
    for record in data:
        # Split the record by commas
        pixelValues = record.split(',')

        # Correct label is the first value
        correct_label = int(pixelValues.pop(0))

        # Scale and shift the inputs
        inputs = (np.asfarray(pixelValues) / 255.0 * 0.99) + 0.01

        # Query the network
        outputs = nn.query(inputs.tolist())

        # Identify predicted label
        prediction = np.argmax(outputs)

        # Add to scorecard
        if prediction == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    scorecard = np.asarray(scorecard)
    successRate = (scorecard.sum() / scorecard.size) * 100

    return successRate

def checkStagnation(performances, epoch, window=5, tolerance=0.05):
    # Check for opportunity for early stopping
    if len(performances) > window:
        mean = sum(performances[-window:]) / window
        variance = sum((i - mean) ** 2 for i in performances[-window:]) / window

        if variance < tolerance and (performances[-1] - performances[-2]) < tolerance:
            print("Stopped early at epoch number {} with final variance of {}.\n".format(epoch, variance))
            return True
        else:
            print("Variance of {} in last {} training cycles. Continuing training.".format(variance, window))
            return False
    else:
        return False

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

def imageToTrainingInputs(row, output_nodes):
    """
    Function simply converts row of pixel data (plus first item is label) from MNIST .csv file into np array
    :param row: list of comma separated pixel values
    :param output_nodes: number of output nodes for network
    :return: returns numpy array of (28*28=) 784 input values and 10 target output values (for digits 0-9)
    """
    # Split the record by the commas
    pixelValues = row.split(',')
    label = pixelValues.pop(0)

    # Scale and shift the inputs from 0..255 to 0.01..1
    inputs = (np.asfarray(pixelValues) / 255.0 * 0.99) + 0.01

    # Create the target output values (all 0.01, except the desired label which is 0.99)
    targets = np.zeros(output_nodes) + 0.01

    # pixelValues[0] is the target label for this record
    targets[int(label)] = 0.99

    inputs = np.array(inputs.tolist(), ndmin=2).T
    targets = np.array(targets.tolist(), ndmin=2).T

    return inputs, targets
