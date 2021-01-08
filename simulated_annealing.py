import numpy as np
from nn_spikes import NeuralNetwork, batchTrain
from spike_tools import dataPreProcess, getSpikeWaveforms, splitData
import random
import pandas as pd

def anneal(solution, spikeLocations, iterations=5, alpha=0.6,
           demand=99.9, variation=0.3, T = 1.0, T_min = 0.1, epochs=25):
    """
    Function to perform simulated annealing. waveform window size and number of hidden nodes form the parameter space searched
    by consecutive candidate solutions, with the cooling schedule intended to converge towards the global optimum solution.
    :param epochs: number of epochs to train each network for. This remains constant
    :param spikeLocations: list-like indicating known spike locations
    :param solution: list containing values used to configure a given candidate neural network configuration
    :param alpha: parameter governs the rate of cooling
    :param iterations: number of different solutions to evaluate at a given temperature
    :param demand: target performance value of near-perfect 99.9%
    :param variation: degree to which solution parameters are varied to generate a neighbouring solution
    :param T: initial temperature that is decayed every x-number of solutions (determined by parameter iterations)
    :param T_min: minimum temperature that determines when to stop the annealing algorithm
    :return: returns a dataframe containing information about the run and the solutions
    """

    # Load data from disk and apply pre-processing steps in an equivalent manner as when the network was trained, using the initial window size
    data = pd.read_csv('./datasources/spikes/training_data.csv')
    data, predictedSpikeIndexes = dataPreProcess(data, spikeLocations, waveformWindow=solution[0])

    # Split the data into training and validation sets
    data_training, data_validation, spikeIndexes_training, spikeIndexes_validation = splitData(data, predictedSpikeIndexes)

    # Create new list to store error values and instantiate the iteration count at 1
    results = []
    i=1

    # Use the initial solution to train a network, classify the data, and calculate the errror
    error = __getError(solution, data_training, data_validation, spikeIndexes_training,
                          spikeIndexes_validation, demand=demand, epochs=epochs)
    results.append((T, i, solution, error))

    # Loop until temp is below the minimum allowable temperature
    while T > T_min:
        i = 1
        # Loop until iteration number is above or equal to max allowable number of iterations
        while i <= iterations:

            # Get new set of solution parameters and generate new error value using this classification solution
            newSolution = __getNeighbour(solution, variation=variation)

            print(results[-1])

            # Extract the putative spike waveforms using the new window size and create new training and validation datasets
            data['waveform'] = getSpikeWaveforms(data['signalSavgol'], predictedSpikeIndexes, window=newSolution[0])
            data_training, data_validation, spikeIndexes_training, spikeIndexes_validation = splitData(data, predictedSpikeIndexes)

            # Use the new solution to train a network, classify the data, and calculate the error
            newError = __getError(newSolution, data_training, data_validation, spikeIndexes_training,
                                  spikeIndexes_validation, demand=demand, epochs=epochs)

            # Calculate the acceptance probability of the new solution
            pA = __acceptanceProbability(error, newError, T)

            # If the acceptance probability is above the randomly generated float in the range [0.0,1.0), use the new
            # solution as the active solution going forwards and store the error value
            if pA > random.random():
                solution = newSolution
                error = newError
            else:
                pass

            # Append the set of run info to the results store
            results.append((T, i, solution, error))
            i += 1

        # Decay (cool) the temperature and return to the top
        T = T * alpha

    # Return a dataframe containing the run information and solutions
    return pd.DataFrame(results, columns=['Temperature', 'Iteration', 'Solution', 'Error'])

def __acceptanceProbability(oldError, newError, T):
    """
    Calculate the acceptance probability based on an exponentially decaying relationship to the temperature and the difference in
    consecutive solution performances
    :param oldError: error from classification with previous solution
    :param newError: error from classification with current solution
    :param T: current temperature
    :return: returns acceptance probability
    """
    return np.exp((oldError - newError) / T)

def __getError(solution, data_training, data_validation, spikeIndexes_training, spikeIndexes_validation, demand=99.9, epochs=25):
    """
    Function finds error between target performance and achieved performance of latest solution classification
    :param solution: set of parameters used to configure the current solution
    :param data_training: training dataset containing all data needed to train the network
    :param data_validation: validation dataset used to validate performance of classification with trained network
    :param spikeIndexes_training: list-like containing spike locations in training dataset
    :param spikeIndexes_validation: list-like containing spike locations in validation dataset
    :param demand: target performance value
    :param epochs: number of epochs to train the network for
    :return: returns the performance of the current solution
    """

    # Assert solution parameters are of the correct type
    assert isinstance(solution[0], int)
    assert isinstance(solution[1], int)
    assert isinstance(solution[2], float)

    # Extract next set of parameters
    hidden_nodes = solution[1]
    lr = solution[2]

    # Instantiate new neural network with current solution parameters
    nn = NeuralNetwork(input_nodes=len(data_training.loc[spikeIndexes_training[0], 'waveform']),
                       hidden_nodes=hidden_nodes,
                       output_nodes=4,
                       lr=lr,
                       error_function='difference-squared')

    # Train the network and store the validation curve used to extract the final performance of the network
    _, _, validationCurve = batchTrain(data_training=data_training,
                                       data_validation=data_validation,
                                       spikeIndexes_training=spikeIndexes_training,
                                       spikeIndexes_validation=spikeIndexes_validation,
                                       nn=nn,
                                       epochs=epochs)

    # Score found using the final performance of the classified validation dataset
    score = validationCurve[-1]

    # Return the new error value
    return demand - score

def __getNeighbour(solution, variation=0.2):
    """
    Function to select neighbouring solution in parameter space by introducing a variation to the current solution
    :param solution: set of parameters that define the current solution
    :param variation: degree of variation to introduce
    :return: returns new solution parameters
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

    # Create new solution parameters by multiplying the current solution with the generated transformation vector
    newSolution = np.multiply(solution, a.flatten())

    # Return new solution parameters, cast to the appropriate data type
    return [int(newSolution[0]), int(newSolution[1]), float(newSolution[2])]