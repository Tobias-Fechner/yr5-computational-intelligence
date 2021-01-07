import numpy as np
from nn_spikes import NeuralNetwork, batchTrain
from spike_tools import dataPreProcess, getSpikeWaveforms, splitData
import random
import pandas as pd

# Simulated annealing function
def anneal(solution, spikeLocations, iterations=5, alpha=0.6,
           demand=99.9, variation=0.3, T = 1.0, T_min = 0.1, epochs=25):
    """
    Function to perform simulated annealing.
    :param epochs:
    :param spikeLocations:
    :param solution:
    :param alpha:
    :param iterations:
    :param demand:
    :param variation:
    :param T:
    :param T_min:
    :return:
    """

    data = pd.read_csv('./datasources/spikes/training_data.csv')
    data, predictedSpikeIndexes = dataPreProcess(data, spikeLocations, waveformWindow=solution[0])

    data_training, data_validation, spikeIndexes_training, spikeIndexes_validation = splitData(data, predictedSpikeIndexes)

    # Create new list to store cost values
    results = []
    i=1

    # Generate and append cost of first solution parameters
    oldError = __getError(solution, data_training, data_validation, spikeIndexes_training,
                          spikeIndexes_validation, demand=demand, epochs=epochs)
    results.append((T, i, solution, oldError))

    # Loop until temp is below min allowable temp
    while T > T_min:
        i = 1
        # Loop until iteration number is above or equal to max allowable number of iterations
        while i <= iterations:

            # Get new set of solution parameters and generate new cost value using this classification solution
            newSolution = __getNeighbour(solution, variation=variation)

            print(results[-1])

            data['waveform'] = getSpikeWaveforms(data['signalSavgol'], predictedSpikeIndexes, window=newSolution[0])
            data_training, data_validation, spikeIndexes_training, spikeIndexes_validation = splitData(data, predictedSpikeIndexes)

            newError = __getError(newSolution, data_training, data_validation, spikeIndexes_training,
                                  spikeIndexes_validation, demand=demand, epochs=epochs)

            # Calculate the acceptance probability
            pA = __acceptanceProbability(oldError, newError, T)

            # If the acceptance probability is above the randomly generated float in the range [0.0,1.0), use the new
            # solution as the active solution going forwards and store the cost value
            if pA > random.random():
                solution = newSolution
                oldError = newError

            results.append((T, i, solution, oldError))
            i += 1

        # Decay (cool) the temperature and return to the top
        T = T * alpha

    return pd.DataFrame(results, columns=['Temperature', 'Iteration', 'Solution', 'Error'])

def __acceptanceProbability(oldError, newError, T):
    """
    Calculate the acceptance porbability based on an exponentially decaying relationship to the temperature
    :param oldError: 
    :param newError: 
    :param T: 
    :return: 
    """
    return np.exp((oldError - newError) / T)

# Cost function
def __getError(supply, data_training, data_validation, spikeIndexes_training, spikeIndexes_validation, demand=99.9, epochs=25):
    """
    Function finds error between target performance and achieved performance of latest solution classification
    :param supply: input parameters
    :param demand: target performance
    :return: return error
    """

    # Extract next set of parameters
    assert isinstance(supply[0], int)
    assert isinstance(supply[1], int)
    assert isinstance(supply[2], float)

    hidden_nodes = supply[1]
    lr = supply[2]

    # Train network with new parameters
    nn = NeuralNetwork(input_nodes=len(data_training.loc[spikeIndexes_training[0], 'waveform']),
                       hidden_nodes=hidden_nodes,
                       output_nodes=4,
                       lr=lr,
                       error_function='difference-squared')

    _, _, validationCurve = batchTrain(data_training=data_training,
                                       data_validation=data_validation,
                                       spikeIndexes_training=spikeIndexes_training,
                                       spikeIndexes_validation=spikeIndexes_validation,
                                       nn=nn,
                                       epochs=epochs)

    score = validationCurve[-1]

    # Return the new error value
    return demand - score

def __getNeighbour(solution, variation=0.2):
    """
    Function to select next parameter iterations for set of parameters given
    :param solution: parameter set
    :param variation:
    :return:
    """
    # Create 3x1 array of random floats within range [0.0, 1.0)
    delta = np.random.random((3, 1))

    # Create 3x1 array with each element equal to twice the variation
    scale = np.full((3, 1), 2 * variation)

    # Create 3x1 array with each element equal to 1 - variation
    offset = np.full((3, 1), 1.0 - variation)

    # Calculate array of new variation value by multiplying delta and scale arrays and add the offset
    a = np.multiply(delta, scale)
    a = np.add(a, offset)

    newSolution = np.multiply(solution, a.flatten())

    return [int(newSolution[0]), int(newSolution[1]), float(newSolution[2])]