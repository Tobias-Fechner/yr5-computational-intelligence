from random import randint, random, choice
from math import ceil
import logging
import numpy as np

logging.basicConfig()
logging.getLogger().setLevel(logging.WARN)

def createIndividual(min, max, length):
    """
    Function creates a single individual of the population using a random integer between the min/ max parameter inputs.
    :param length: number of random ints to generate
    :param min: min int of range
    :param max: max int of range
    :return: returns list of size 'length' random numbers within min/max range
    """
    return np.random.randint(min, max, length)

def createPopulation(count, min, max, length):
    """
    Creates a population of individuals.
    :param count: the number of individuals in the population
    :param length: the number of values per individual
    :param min: the minimum possible value in an individual's list of values
    :param max: the maximum possible value in an individual's list of values
    :return: returns a list of individuals of size 'length
    """
    return [createIndividual(min, max, length) for x in range(count)]

def generatePredictions(individual):

    # Extract polynomial coefficients from individual
    a1 = individual[0]
    a2 = individual[1]
    a3 = individual[2]
    a4 = individual[3]
    a5 = individual[4]
    c = individual[5]

    yTarget = np.array([])
    yPrediction = np.array([])

    # For a range of values of x, calculate the target and predicted value of y
    for x in np.linspace(-1,1, 20):
        yTarget = np.append(yTarget, 25*x**5 + 18*x**4 + 0 - 14*x**2 + 7*x -19)
        yPrediction = np.append(yPrediction, a1*x**5 + a2*x**4 + a3*x**3 + a4*x**2 + a5*x + c)

    return yTarget, yPrediction

def getMSE(individual):
    """
    Take in individual, which are effectively coefficient predictions, then use a range of x values to generate a corresponding y value
    with the same equation pattern as the polynomial we are trying to fit. Return the mean square error by comparing with y values
    generated from target polynomial.
    :param individual:
    :return:
    """

    yTarget, yPrediction = generatePredictions(individual)

    # Calculate error as difference between each y value prediction and corresponding target
    errors = np.subtract(yTarget, yPrediction)

    # Square each error term
    squaredErrors = np.power(errors, 2)

    # Calculate weighted average of the squared errors array
    meanSquaredErrors = np.average(squaredErrors)

    return meanSquaredErrors

def getAvgAbsError(individual):

    yTarget, yPrediction = generatePredictions(individual)

    errors = np.subtract(yTarget, yPrediction)
    absErrors = np.absolute(errors)

    return np.average(absErrors)

def errorsToAvgFitness(errors):

    # Identify least fit individual
    unfittest = max(errors)

    # Calculate raw fitness of population (convert error into a fitness value)
    rawFitnesses = np.subtract(unfittest, errors)

    #TODO: Implement scaling of fitnesses

    return np.average(rawFitnesses)

def getAvgPopulationFitness(population, error='mse'):
    """
    Determines the fitness of a given individual by calculating MSE between prediction and target over a range of values of x.
    :param population: population of individuals
    :param error: error calculation methodology
    :return: average fitness of population
    """
    if 'mse' in error:
        # Calculate MSE for each individual in population (pre-fitness)
        individualMSEs = [getMSE(individual) for individual in population]

        # Return weighted average of raw fitnesses as fitness for entire population
        return errorsToAvgFitness(individualMSEs)

    elif 'abs' in error:
        # Calculate average abs error for each individual in population
        individualErrors = [getAvgAbsError(individual) for individual in population]

        return np.average(individualErrors)

    else:
        raise ValueError("Please specify error calculation methodology that has been implemented.")

def rouletteSelection(population):

    fitnesses = np.array([])

    # Get total fitness of population (cumulative sum of each individual's error)
    for individual in population:

        # Get errors for each individual
        errors = getMSE(individual)

        # Get fitness, append to list of fitnesses
        fitness = errorsToAvgFitness(errors)
        fitnesses = np.append(fitnesses, fitness)

    # Calculate total fitness for population as sum of individuals' fitness
    totalFitness = np.sum(fitnesses)

    # Calculate selection probability for each individual
    selectProbabilities = np.divide(fitnesses, totalFitness)

    # Assert probabilities all are floats within range [0,1] and sum to 1
    assert 0 <= selectProbabilities.all() <= 1
    assert abs(np.sum(selectProbabilities) - 1) < 0.0001

    matingPool = np.array([])

    # Select individuals to add to mating pool using probability based on fitness of individual
    for individual, selectProbability in zip(population, selectProbabilities):
        if selectProbability > random():
            matingPool = np.append(matingPool, individual)
        else:
            pass

    # Return mating pool
    return matingPool


def randomSelection(population, random_select, retain):
    # Grade each individual within the population
    individualGrades = [(getAvgAbsError(individual), individual) for individual in population]

    # Rank population based on individual grades
    graded = [x[1] for x in sorted(individualGrades)]

    # Calculate number of individuals to retain as matingPool of the next generation
    retain_length = ceil(len(graded)*retain)

    # Retain the desired number of elements from the leading side of the list as the matingPool of next generation
    matingPool = graded[:retain_length]

    # Add other individuals to matingPool group to promote genetic diversity
    for individual in graded[retain_length:]:

        # Configured probability used to decide if each individual from remaining population is added to parent gene pool
        if random_select > random():
            matingPool.append(individual)

    return matingPool, individualGrades

def evolve(population, retain=0.2, random_select=0.05, mutate=0.01, femalePortion=0.5, select='roulette'):

    # Assert all fractions between 0-1
    assert 0 <= femalePortion <= 1
    assert 0 <= random_select <= 1
    assert 0 <= mutate <= 1
    assert 0 <= retain <= 1

    if 'roulette' in select:
        raise NotImplementedError
    elif 'random' in select:
        parents, individualGrades = randomSelection(population, random_select, retain)
    else:
        raise ValueError("Please specify a selection methodology that has been implemented.")

    # Mutate some individuals
    for individual in parents:

        # Configured probability used to decide if each individual from parent gene pool is mutated
        if mutate > random():

            # Generate value to randomly index individual's contents
            randIndex = choice(range(len(individual)))

            # Generate a random integer between the individual's min/ max and store value using random index
            individual[randIndex] = randint(min(individual), max(individual))

    children = []
    lenParents = len(parents)
    desiredChildren = len(population) - lenParents

    # Create children from high performing parent group to maintain population size
    while len(children) < desiredChildren:

        # Get female and male index
        femaleIndex = randint(0, lenParents-1)
        maleIndex = randint(0, lenParents-1)

        if maleIndex == femaleIndex:
            pass

        else:

            # Get female and male values
            female = parents[femaleIndex]
            male = parents[maleIndex]

            # Determine proportion of female chromosomes to
            split = int(len(female)*femalePortion)
            logging.info("Split at {}. Parent chromosomes: {}; Child gets {} from female and {} from male.".format(
                split, len(male), len(female[:split]), len(male[split:])))

            # Split individual chromosomes based on desired female portion of chromosomes
            # Create child with half of male chromosomes and other half of female's chromosomes
            child = np.append(male[:split], female[split:])
            logging.info("Child made with {} chromosomes.".format(len(child)))

            # Add child to list
            children.append(child)

    # Return list of children and parent individuals to be next generation
    parents.extend(children)

    return parents, individualGrades
