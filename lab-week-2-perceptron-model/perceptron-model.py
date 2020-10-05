# Import the NumPy library for matrix math
import numpy
from itertools import permutations

# A single perceptron function - poor practice to include datatype in variable name, therefore omitted.
def perceptron(ins, wts, b):
    # Convert the inputs and weights (should be list) into a numpy array
    inputs = numpy.array(ins)
    weights = numpy.array(wts)

    # Calculate the dot product and add bias
    summed = numpy.dot(inputs, weights) + b

    # Calculate the output
    output = 1 if summed > 0 else 0

    return output

#### Exercises
## Ex1.1
# Test perceptron
inputs = [[0,0],
          [0,1],
          [1,0],
          [1,1]]
weights = [1.0, 1.0]
bias = -1

file = open("ex1.txt", "w")

for item in inputs:
    file.write("Ex1.1\nInputs: {0}\nWeights: {1}\nBias: {2}\nResult: {3}\n\n".format(item, weights, bias, perceptron(item, weights, bias)))
file.close()

## Ex1.2
# Interpreting this information as a truth table, what logical function is being performed?
file = open("ex1.txt", "a")
file.write("Ex1.2\nQ: Interpret as truth table, what logical function is it?\nA: AND\n\n")
file.close()

## Ex1.3
# What are the weight vectors and the bias for the logical functions: AND,OR,NAND, NOR and XOR?
file = open("ex1.txt", "a")
file.write("Ex1.3\nOR: bias = 0, weights [[0.5,0.5], [1,1]]\nNAND: requires NOT operator\nNOR: requires NOT operator\nXOR: requires NOT operator.\n "
           "These truth tables are not linearly separable.\n\n")

#### Further Exercises
