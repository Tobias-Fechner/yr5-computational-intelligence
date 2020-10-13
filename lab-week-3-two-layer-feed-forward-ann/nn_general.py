import scipy.special
import numpy as np

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, lr):
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

        # Create weight matrix for input to hidden with size set by number of input and hidden nodes
        self.wih = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        # Create weight matrix for hidden to output with size set by number of hidden and output nodes
        self.who = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        # Set the activation function as the logistic sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs, targets):
        """
        Function calculates new weights for NN using back propagation of errors
        :param inputs: input vectors as list
        :param targets:
        :return: no return
        """
        try:
            assert isinstance(inputs, list)
            assert isinstance(targets, list)
        except TypeError:
            raise TypeError("NN training inputs and targets must both be type list. (two separate lists)")

        # Convert the inputs list into a 2D array and use to calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, np.array(inputs, ndmin=2).T)

        # Calculates the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculates the signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # Calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Calculate current error as = target - actual
        output_errors = np.array(targets, ndmin=2).T - final_outputs

        # Hidden layer errors are the output errors, split by the weights, recombined at the hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0-final_outputs)),
                                     np.transpose(hidden_outputs))

        # Update the weights for the links between the input and hidden layers
        # TODO: Change second dot product input to np.array(inputs, ndmin=2) after NN is working
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     np.transpose(np.array(inputs, ndmin=2).T))

        return None

    def queryNN(self, inputs):
        """
        Function to query the neural network.
        :param inputs: List of inputs
        :return: Returns outputs from final layer of nodes
        """
        try:
            assert isinstance(inputs, list)
        except TypeError:
            raise TypeError("NN query inputs must be type list.")

        # Convert the inputs list into a 2D array and use to calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, np.array(inputs, ndmin=2).T)

        # Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final layer
        final_inputs = np.dot(self.wih, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)

        return final_outputs