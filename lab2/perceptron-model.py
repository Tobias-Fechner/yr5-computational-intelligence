# Import the NumPy library for matrix math
import numpy
import matplotlib.pyplot as plt

# Parameters
inputs = [[0,0],
          [0,1],
          [1,0],
          [1,1]]
weights = [1.0, 1.0]
bias = 1
lr = 0.05   # learning rate

def unitStep(x):
    return 1 if x > 0 else 0

def  sigmoid(x):
    return 1/(1+numpy.exp(-x))

def sigmoidDer(x):
    return sigmoid(x)*(1-sigmoid(x))

def hypTan(x):
    return (numpy.exp(x)-numpy.exp(-x))/(numpy.exp(x)+numpy.exp(-x))

def reLU(x):
    if x <= 0:
        return 0
    else:
        return x

# A single perceptron function - poor practice to include datatype in variable name, therefore omitted.
def perceptron(ins, wts, b, activationFunction=unitStep):
    # Convert the inputs and weights (should be list) into a numpy array
    inputs = numpy.array(ins)
    weights = numpy.array(wts)

    # Calculate the dot product and add bias
    summed = numpy.dot(inputs, weights) + b

    return activationFunction(summed)

def meanSquareError(targets, outputs):
    # Very sketchy, probably wrong - need to change to sum for all input sets - not guaranteeing its lined up with this
    if not isinstance(targets, list) or not isinstance(outputs, list):
        raise TypeError("Targets and outputs need to be passed in as a list")
    else:
        error = 1/2 * numpy.sum(abs(numpy.array(targets) - numpy.array(outputs)) ** 2)
        return error

# TODO: Delete when no longer in use

# #### Exercises
# ## Ex1.1
# file = open("ex1.txt", "w")
#
# for item in inputs:
#     file.write("Ex1.1\nInputs: {0}\nWeights: {1}\nBias: {2}\nResult: {3}\n\n".format(item, weights, bias, perceptron(item, weights, bias)))
# file.close()
#
# ## Ex1.2
# # Interpreting this information as a truth table, what logical function is being performed?
# file = open("ex1.txt", "a")
# file.write("Ex1.2\nQ: Interpret as truth table, what logical function is it?\nA: AND\n\n")
# file.close()
#
# ## Ex1.3
# # What are the weight vectors and the bias for the logical functions: AND,OR,NAND, NOR and XOR?
# file = open("ex1.txt", "a")
# file.write("Ex1.3\nOR: bias = 0, weights [[0.5,0.5], [1,1]]\nNAND: requires NOT operator\nNOR: requires NOT operator\nXOR: requires NOT operator.\n "
#            "These truth tables are not linearly separable.\n\n")

# #### Further Exercises - plotting
#
# fig = plt.xkcd()
#
# for item in inputs:
#     # Conditional color selection based on perceptron output
#     color = "green" if perceptron(item, weights, bias) else "red"
#
#     # Add to scatter plot
#     plt.scatter(item[0], item[1], s=50, color=color, zorder=3)
#
#     # Calculate linear separator
#     m = -weights[0]/ weights[1]
#     c = -bias/weights[1]
#     print("Line equation: x2 = {}x1 + {}".format(m, c))
#
# plt.xlim(-2,2)
# plt.ylim(-2,2)
#
# plt.xlabel("Input 1")
# plt.ylabel("Input 2")
# plt.title("State space of input vector")
#
# plt.grid(True, linewidth=1, linestyle=':')
# plt.tight_layout()
#
# plt.show()
#
# print("completed")