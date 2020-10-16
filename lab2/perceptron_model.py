# Import the np library for matrix math
import numpy as np
import matplotlib.pyplot as plt

def unitStep(x):
    return 1 if x > 0 else 0

def  sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidDer(x):
    return sigmoid(x)*(1-sigmoid(x))

def hypTan(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def reLU(x):
    if x <= 0:
        return 0
    else:
        return x

# A single perceptron function - poor practice to include datatype in variable name, therefore omitted.
def perceptron(inputBits, nodeWeights, bias, activationFunction=unitStep):
    # Convert the inputs and weights (should be list) into a np array
    inputs = np.array(inputBits)
    weights = np.array(nodeWeights)

    # Calculate the dot product and add bias
    summed = np.dot(inputs, weights) + bias

    return activationFunction(summed)

def plotResults(inputBitPairs, weights, bias, logicFunction):
    # Create array to collect perceptron outputs purely for use in generating XOR graph in specific notebook
    xorInputs = []

    for inputBits in inputBitPairs:
        # Conditional color selection based on perceptron output
        result = perceptron(inputBits, weights, bias)
        color = "green" if result else "red"

        # Add to scatter plot
        plt.scatter(inputBits[0], inputBits[1], s=50, color=color, zorder=3)

        xorInputs.append(result)

    plt.xlim(-0.5, 2)
    plt.ylim(-0.5, 2)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("State space for {}".format(logicFunction))

    plt.grid(True, linewidth=1, linestyle=':')

    # Calculate linear separator
    m = -weights[0] / weights[1]
    c = - bias / weights[1]

    x = np.linspace(-5, 5, 4)
    y = (m * x) + c

    label = "x2 = {}x1 + {}".format(m, c)
    plt.plot(x, y, linewidth=2, label = label)
    plt.legend(loc = 'upper left')
    plt.tight_layout()

    plt.show()

    assert isinstance(xorInputs, list)
    assert len(xorInputs) == 4
    return xorInputs

def plotXORResults(originalInputs,
                   originalWeights,
                   originalBias,
                   inputsFromAND,
                   inputsFromOR,
                   weights,
                   bias,
                   logicFunction="XOR"):
    for x1, x2, originalInput in zip(inputsFromAND, inputsFromOR, originalInputs):
        # Conditional color selection based on perceptron output
        result = perceptron([x1,x2], weights, bias)
        color = "green" if result else "red"

        # Add to scatter plot
        plt.scatter(originalInput[0], originalInput[1], s=50, color=color, zorder=3)

    plt.xlim(-0.5, 2)
    plt.ylim(-0.5, 2)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("State space of {}".format(logicFunction))

    plt.grid(True, linewidth=1, linestyle=':')

    # Calculate AND linear separator
    m1 = -originalWeights[0][0] / originalWeights[0][1]
    c1 = - originalBias[0] / originalWeights[0][1]
    x1 = np.linspace(-5, 5, 4)
    y1 = (m1 * x1) + c1

    # Calculate OR linear separator
    m2 = -originalWeights[1][0] / originalWeights[1][1]
    c2 = - originalBias[1] / originalWeights[1][1]
    x2 = np.linspace(-5, 5, 4)
    y2 = (m2 * x2) + c2

    label1 = "x2 = {}x1 + {}".format(m1, c1)
    label2 = "x2 = {}x1 + {}".format(m2, c2)

    plt.plot(x1, y1, linewidth=2, label = label1)
    plt.plot(x2, y2, linewidth=2, label=label2)
    plt.legend(loc = 'upper left')
    plt.tight_layout()

    plt.show()

def meanSquareError(targets, outputs):
    # Very sketchy, probably wrong - need to change to sum for all input sets - not guaranteeing its lined up with this
    if not isinstance(targets, list) or not isinstance(outputs, list):
        raise TypeError("Targets and outputs need to be passed in as a list")
    else:
        error = 1/2 * np.sum(abs(np.array(targets) - np.array(outputs)) ** 2)
        return error

# TODO: Delete below when no longer in use

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
